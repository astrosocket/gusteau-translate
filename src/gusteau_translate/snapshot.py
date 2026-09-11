import json
import logging
import os
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from traceback import TracebackException
from typing import Literal, cast

import h5py
import unyt.dimensions as ud
from natsort import natsorted
from unyt.array import unyt_quantity
from unyt.unit_registry import UnitRegistry
from unyt.unit_systems import UnitSystem

from .config import (
    LOGGER_NAME,
    TranslationOptions,
    ValidationOptions,
    _default_translation_options,
    _default_validation_options,
)
from .optional_steps.packcubes import CubesTranslator
from .pipeline import ErrorHandler
from .specification import (
    ValidationError,
    get_spec_from_version,
    h5type,
    visit_all,
)
from .translation_base import (
    Translation,
    TranslationContext,
    dummy_field,
    translation_error_handler,
)
from .translations import GroupTranslation

LOGGER = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class _SubDataset:
    filename: Path
    shape: tuple[int, ...]


def _create_virtual_dataset(
    virtual: h5py.File, dataset_path: str, file_list: list[Path]
):
    """Create a virtual dataset in virtual from the dataset_path in each file

    Parameters
    ----------
    virtual: h5py.File
        The file that will contain the virtual dataset
    dataset_path: Path
        The path to the dataset in each file.
        Must be identical for all files in the list.
    file_list: list[Path]
        List of paths to the hdf5 files that contain the subdatasets to be combined.
    """
    # Process has 5 steps
    # 1. Determine the final dimensions and data type of the virtual dataset
    # 2. Create a VirtualLayout object representing the virtual dataset
    # 3. Create VirtualSource objects representing each of the subdatasets
    # 4. Map the VirtualSource objects into slices in the VirtualLayout
    # 5. Convert the VirtualLayout into a virtual dataset
    # We will actually do the steps out of order because we don't know a priori
    # the dimensions of the virtual dataset. What we do know is that all of the
    # datasets are effectively vertically stacked, i.e. in the 0th dimension.
    # So we'll start with step 3, reading the metadata of every subdataset to
    # determine that subdatasets shape and the final dataset's size
    subdatasets = []
    dtype = float
    for filename in file_list:
        with h5py.File(str(filename), "r") as subfile:
            shape = subfile[dataset_path].shape
            subdatasets.append(_SubDataset(filename, shape))
            dtype = subfile[dataset_path].dtype
    total_length = sum(s.shape[0] for s in subdatasets)
    shape = (total_length, *subdatasets[0].shape[1:])
    LOGGER.debug(
        f"Creating virtual layout for {dataset_path} with {dtype=} and {shape=}"
    )
    layout = h5py.VirtualLayout(
        shape,
        dtype,
    )
    offset = 0
    for subds in subdatasets:
        layout[offset : offset + subds.shape[0], ...] = h5py.VirtualSource(
            subds.filename, dataset_path, shape=subds.shape
        )
    LOGGER.info(f"Creating virtual dataset {dataset_path}")
    virtual.create_virtual_dataset(dataset_path, layout)


def _build_virtual_input_dataset(snapshots: list[Path], virtual_file: Path) -> Path:
    """Build a virtual snapshot to combine multipart files"""
    LOGGER.debug("Setting up multipart virtual input file")
    # need file to obtain all of the datasets, groups, and attributes
    infile_path = snapshots[0]
    with (
        h5py.File(infile_path, "r", libver="latest") as infile,
        h5py.File(virtual_file, "w", libver="latest") as virtual,
    ):
        for obj in visit_all(infile):
            match obj.type:
                case h5type.ATTRIBUTE:
                    LOGGER.debug(f"Adding attribute {obj.name} to {obj.path.stem}")
                    virtual[obj.path.stem].attrs[obj.name] = obj.obj
                case h5type.GROUP:
                    LOGGER.debug(f"Adding group {obj.path!s}")
                    virtual.require_group(str(obj.path))
                case h5type.DATASET:
                    _create_virtual_dataset(virtual, str(obj.path), snapshots)
                case _:
                    raise ValueError(f"Unknown h5 object type: {obj.type}!")
    return virtual_file


def _check_snapshot_glob(
    snapshot_path: str | Path,
) -> tuple[Path, list[Path]]:
    snapshot_path = Path(snapshot_path)
    if h5py.is_hdf5(snapshot_path):
        return snapshot_path, []
    parent = snapshot_path.parent
    # Note: we are explicitly using the third-party library here.
    # While we could use a simpler regex based method, (e.g. from
    # https://stackoverflow.com/a/11150413), it would be potentially quite
    # fragile
    snapshots = natsorted(parent.glob(f"{snapshot_path.name}*"))
    if not snapshots:
        raise ValueError(f"Glob pattern {snapshot_path} produced no snapshots!")
    if not all(h5py.is_hdf5(snapshot) for snapshot in snapshots):
        bad_files = [snapshot for snapshot in snapshots if not h5py.is_hdf5(snapshot)]
        raise ValueError(
            f"Snapshot glob {snapshot_path} is not specific enough."
            f" It produces non-hdf5 files {bad_files}"
        )
    base_name = Path(os.path.commonprefix(snapshots))
    virtual_file = Path(f"{base_name.stem}_virtual.hdf5")
    return virtual_file, snapshots


def _create_field_metadata(outfile: h5py.File):

    metadata = {}

    def visit_item(
        name: str,
        obj: h5py.Group
        | h5py.Dataset
        | h5py.HardLink
        | h5py.SoftLink
        | h5py.ExternalLink,
    ):
        d = metadata
        for n in name.split("/"):
            if n not in d:
                break
            d = d[n]
        # assuming lexicographic traversal!
        # so /Group comes before /Group/Dataset
        if isinstance(obj, h5py.HardLink):
            return
        if isinstance(obj, h5py.SoftLink | h5py.ExternalLink):
            shape = []
            attrs = {}
            type_ = "Link"
            path = obj.path
        else:
            if isinstance(obj, h5py.Group):
                shape = [len(obj.keys())]
                type_ = "Group"
            else:
                shape = obj.shape
                type_ = f"Dataset{obj.dtype}"
            path = name
            attrs = {
                str(a_name): {
                    "-shape": [] if isinstance(a, str) else a.shape,
                    "-type": type(a).__name__,
                    "-attrs": {},
                    "-path": f"{name}.{a_name}",
                }
                for a_name, a in obj.attrs.items()
            }
        d[n] = {
            "-shape": shape,
            "-type": type_,
            "-attrs": attrs,
            "-path": path,
        }

    outfile.visititems(visit_item)
    outfile.visititems_links(visit_item)

    outfile.attrs.create("Metadata", json.dumps(metadata, separators=(",", ":")))


def create_unit_registry(context: TranslationContext):
    """Create a unit registry and add it to the context"""
    quantities = context.quantities
    registry = UnitRegistry(unit_system="cgs")
    dimensions = {
        "U_I": (ud.current_mks, "A", "A"),
        "U_L": (ud.length, "cm", "m"),
        "U_M": (ud.mass, "g", "kg"),
        "U_T": (ud.temperature, "K", "K"),
        "U_t": (ud.time, "s", "s"),
    }
    for uname, (dimension, cgs_unit, mks_unit) in dimensions.items():
        mks = unyt_quantity(quantities.get(uname), cgs_unit).to(mks_unit)
        registry.add(uname, base_value=float(mks.v), dimensions=dimension)
    LOGGER.debug(f"Created unit registry with lut: {registry.lut}")
    context.quantities["registry"] = registry
    snapshot_units = UnitSystem(
        name="snapshot",
        length_unit="U_L",
        mass_unit="U_M",
        time_unit="U_t",
        current_mks_unit="U_I",
        temperature_unit="U_T",
        registry=registry,
    )
    LOGGER.debug(f"Created snapshot unit system: {snapshot_units}")


def set_cosmo_quantities(context: TranslationContext):
    """Set scale-factor and little h if not already present"""
    quantities = context.quantities
    if "a" not in quantities:
        # z is a required header variable (as redshift) so it should exist
        # but if it doesn't simply assume z=now
        redshift = quantities.get("z", 0)
        quantities["a"] = 1 / (1 + redshift)
    if "h" not in quantities:
        quantities["h"] = 1


def initialize_particle_groups(
    particle_groups: list[str], context: TranslationContext, error_handler: ErrorHandler
):
    translation = context.translation
    LOGGER.debug(f"Initializing the {particle_groups} particle groups for packingcubes")
    for pg in particle_groups:
        output_path = f"/{pg}"
        field = translation.get_field_from_output(output_path)
        if field is None:
            LOGGER.warning(
                f"Particle group {pg} does not have a translation defined."
                " Creating one using default factory"
            )
            field = cast(
                GroupTranslation,
                translation._default_translation_factory(
                    "group",
                    output_path,
                ),
            )
            field.has_particles = True

        translation.apply_translation(field, context, error_handler)


def _set_cubes_translator(translation: Translation, context: TranslationContext):
    gusteau = translation.gusteau
    cubes_translator_name = gusteau.get("cubes_translator")
    if cubes_translator_name:
        LOGGER.debug(f"Trying cubes translator: {cubes_translator_name}")
        cubes_translator = CubesTranslator._subclasses.get(cubes_translator_name)
        if cubes_translator:
            LOGGER.debug(f"Setting cubes translator: {cubes_translator_name}")
            context.cubes_translator = cubes_translator()  # type: ignore


def _process_packingcubes(
    context: TranslationContext,
    error_handler: ErrorHandler,
    options: TranslationOptions,
):
    particle_groups = options.packingcubes
    particle_groups = (
        context.quantities["Particle_names"]
        if particle_groups[0] == "*"
        else particle_groups
    )
    # ensure particle groups are present
    # (note that this might require translations)
    translation = context.translation
    initialize_particle_groups(particle_groups, context, error_handler)
    _set_cubes_translator(translation, context)
    LOGGER.debug(
        f"Prioritizing the Coordinates dataset for the {particle_groups}"
        " groups for packingcubes processing"
    )
    translation.set_priority_fields(
        **{f"/{pg}/Coordinates": "ParticleDatasetTranslation" for pg in particle_groups}
    )


class Snapshot:
    path: Path
    """Path to this snapshot file"""

    directory: set
    """List of all output attributes/datasets ever visited"""

    multipart_file: list[Path]

    def __init__(self, snapshot_path: str | Path):

        self.path, self.multipart_file = _check_snapshot_glob(snapshot_path)

        self.directory: set[str] = set()

    def translate(
        self,
        translation: Translation,
        new_snapshot_path: str | Path,
        *,
        options: TranslationOptions | None = None,
        **kwargs,
    ) -> Literal[True]:
        if options is None:
            options = _default_translation_options()

        if h5py.is_hdf5(new_snapshot_path) and not options.force_overwrite:
            raise FileExistsError(new_snapshot_path)

        if self.multipart_file:
            self.path = _build_virtual_input_dataset(
                self.multipart_file, Path(new_snapshot_path).parent / self.path
            )

        with (
            h5py.File(self.path) as infile,
            h5py.File(
                new_snapshot_path, "w", libver="v200", track_order=True
            ) as outfile,
            tempfile.TemporaryDirectory() as tempdir,
        ):
            # Start with header, since that's most important
            LOGGER.debug("Initializing context")
            context = TranslationContext(
                infile, outfile, tempdir, translation, dummy_field, options=options
            )

            LOGGER.debug("Initializing error handler")
            error_handler = translation_error_handler

            LOGGER.debug("Creating required groups in outfile")
            translation.initialize_required_groups(outfile)

            # Order of translations:
            #  1. Explicit transformations
            #    a. Header
            #    b. Remaining required groups (Code, RunInfo, etc)
            #    c. Remaining translations in style file
            #  2. Remaining fields in input file
            #  3. Any unused translations

            LOGGER.info(
                "Translating header, required groups, and explicit translations"
            )

            # Note that we **need** the header done first to be able to do any
            # of the remaining steps, because it sets a number of quantities
            # that other steps might need
            # We also need the units step done so we can do any unit conversions
            self.directory.update(
                translation.apply_translation(
                    translation.get_required_translations(), context, error_handler
                )
            )

            # Now that we have the required group Units, we can make the registry
            create_unit_registry(context)
            set_cosmo_quantities(context)

            if options.packingcubes:
                _process_packingcubes(
                    context,
                    error_handler,
                    options,
                )

            self.directory.update(
                translation.apply_explicit_translations(
                    context,
                    error_handler,
                )
            )

            # do remainder of input fields
            def visit_attribute(
                name: str,
                obj: h5py.Group | h5py.Dataset,
                directory: set[str],
                context: TranslationContext,
                error_handler: ErrorHandler,
            ):
                full_name = str(obj.name) + "." + name
                LOGGER.debug(f"Visiting attribute {full_name}")
                if full_name in directory:
                    return
                LOGGER.info(f"Translating {full_name}")
                self.directory.update(
                    context.translation.apply_translation(
                        full_name, context, error_handler, obj="attribute"
                    )
                )

            def visit_field(
                name: str,
                obj: h5py.Group | h5py.Dataset,
                directory: set[str],
                context: TranslationContext,
                error_handler: ErrorHandler,
            ):
                full_name = str(obj.name)
                LOGGER.debug(f"Visiting {full_name}")
                if full_name in directory:
                    return
                self.directory.update(
                    context.translation.apply_translation(
                        full_name, context, error_handler, obj=obj
                    )
                )

                for a in obj.attrs:
                    visit_attribute(a, obj, directory, context, error_handler)

            LOGGER.info("Translating remaining fields from input file")
            infile.visititems(
                partial(
                    visit_field,
                    directory=self.directory,
                    context=context,
                    error_handler=error_handler,
                )
            )

            LOGGER.info("Performing any remaining translations")
            translation.apply_unused_translations(context, error_handler)

            LOGGER.info("Writing /.Metadata attribute")
            _create_field_metadata(outfile)

        return True

    def validate(
        self,
        *,
        options: ValidationOptions | None = None,
    ) -> bool:
        if options is None:
            options = _default_validation_options()

        spec = get_spec_from_version(options.gusteau_version)

        validation_result = False
        with h5py.File(self.path) as h5_file:
            try:
                validation_result = spec.validate(h5_file, options=options)
            except ValidationError as ve:
                cause = ve.__cause__ if ve.__cause__ is not None else ve
                traceback = TracebackException.from_exception(
                    cause, capture_locals=True, lookup_lines=True
                )
                msg = (
                    f"\n  Validation failed:\n  {ve}\n  {'\n'.join(traceback.format())}"
                )
                LOGGER.warning(msg)

        return validation_result

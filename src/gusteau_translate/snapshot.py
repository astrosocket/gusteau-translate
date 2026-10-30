import json
import logging
import os
import tempfile
from functools import partial
from pathlib import Path
from traceback import TracebackException
from typing import Literal, cast

import h5py
import numpy as np
import unyt.dimensions as ud
from natsort import natsorted
from unyt.array import unyt_quantity
from unyt.unit_registry import UnitRegistry
from unyt.unit_systems import UnitSystem

from gusteau_translate.translations.misc_translation import SkipTranslation

from .cells import (
    _create_virtual_dataset,
    create_partfiles,
    generate_cells,
)
from .config import (
    LOGGER_NAME,
    TranslationOptions,
    ValidationOptions,
    _default_translation_options,
    _default_validation_options,
)
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
    base_name = Path(os.path.commonpath(snapshots))
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


def _process_particle_groups(
    context: TranslationContext,
    error_handler: ErrorHandler,
    options: TranslationOptions,
):
    particle_groups = context.quantities["Particle_names"]

    # ensure particle groups are present
    # (note that this might require translations)
    translation = context.translation
    initialize_particle_groups(particle_groups, context, error_handler)
    # _set_cubes_translator(translation, context)
    LOGGER.debug(
        f"Prioritizing the Coordinates dataset for the {particle_groups}"
        " groups for cells, packingcubes processing"
    )
    translation.set_priority_fields(
        **{f"/{pg}/Coordinates": "ParticleDatasetTranslation" for pg in particle_groups}
    )


def _add_cell_translations(translation: Translation, options: TranslationOptions):
    """Add the translations for the cells group structure"""

    factory = translation._default_translation_factory

    translation.add_field(factory("GroupTranslation", "/Cells/Meta-data"))

    cells_attrs = [
        ".Grid_box",
        ".Number_of_grid_cells",
        ".Number_of_nongrid_cells",
        "/Meta-data.dimension",
        "/Meta-data.nr_cells",
        "/Meta-data.size",
    ]

    for attr in cells_attrs:
        name = attr.split(".")[1]
        field = factory(
            "CellAttributeTranslation",
            path=f"/Cells{attr}",
            input_description="Generated from Cells structure",
            cell_field=name,
        )
        # Since Meta-data needs to keep the hyphen (ANNOYING!)
        # we need to manually override the normalization
        field.output_path = f"/Cells{attr}"
        translation.add_field(field)

    cells_datasets = ["BoundingBoxes", "Centres"]
    for ds in cells_datasets:
        translation.add_field(
            factory(
                "CellDatasetTranslation",
                path=f"/Cells/{ds}",
                input_description="Generated from Cells structure",
                cell_field=ds,
            )
        )

    cells_groups = [
        "Counts",
        "Files",
        "Indices",
        "MaxPositions",
        "MinPositions",
        "OffsetsInFile",
    ]

    descriptions = [
        "The number of particles of the particular type",
        (
            "The number of the actual file containing the cell's particles."
            " Only relevant if /Header.Num_files_per_snapshot > 1"
        ),
        "The overall cell index into the virtual dataset.",
        (
            "Maximum coordinates of the particular particle type for that cell in"
            " simulation units"
        ),
        (
            "Minimum coordinates of the particular particle type for that cell in"
            " simulation units"
        ),
        "Index to the start of the cell in the file it's stored in",
    ]

    # packingcubes handles the Trees group
    translation.add_field(
        SkipTranslation(output_path="/Cells/Trees", input_description="", input_path="")
    )
    translation.add_field(
        SkipTranslation(
            output_path="/Cells/Trees.Description", input_description="", input_path=""
        )
    )

    for grp, description in zip(cells_groups, descriptions, strict=True):
        translation.add_field(factory("GroupTranslation", f"/Cells/{grp}"))
        translation.add_field(
            factory(
                "ConstantAttributeTranslation",
                f"/Cells/{grp}.Description",
                constant=description,
            )
        )


def _fix_header_numbers(outfile: h5py.File, options: TranslationOptions):
    """Update the NumPart fields and Num_files_per_snapshot"""
    header = outfile["/Header"].attrs

    header["Num_files_per_snapshot"] = np.int32(options.num_partfiles)

    pt_map = header["Part_type_mapping"]
    particle_numbers = header["Particle_numbers"]

    NumPart_Total = np.zeros((6,), dtype=np.int32)
    NumPart_Total_HW = np.zeros_like(NumPart_Total)

    for i, pt in enumerate(pt_map):
        if pt < 0:
            continue
        NumPart_Total[i] = particle_numbers[pt] & 0xFFFF_FFFF
        NumPart_Total_HW[i] = particle_numbers[pt] >> 32

    header["NumPart_Total"] = NumPart_Total
    header["NumPart_Total_HW"] = NumPart_Total_HW
    header["NumPart_ThisFile"] = NumPart_Total.astype(np.uint32)


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

        new_snapshot_path = Path(new_snapshot_path)
        partfile_paths = create_partfiles(new_snapshot_path, options)

        with (
            h5py.File(self.path) as infile,
            h5py.File(
                new_snapshot_path, "w", libver="v200", track_order=True
            ) as outfile,
            tempfile.TemporaryDirectory() as tempdir,
        ):
            LOGGER.debug("Initializing cell structure")
            cells_meta = generate_cells(infile, options)
            _add_cell_translations(translation, options)

            LOGGER.debug("Initializing context")
            context = TranslationContext(
                infile,
                outfile,
                tempdir,
                translation,
                dummy_field,
                cells_meta,
                partfile_paths,
                options=options,
            )

            LOGGER.debug("Initializing error handler")
            error_handler = translation_error_handler

            LOGGER.debug("Creating required groups in outfile")
            translation.initialize_required_groups(
                outfile, partfile_paths if options.num_partfiles > 1 else []
            )

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

            # Now we have the header, we need to update it to fix NumPart_ThisFile,
            # NumPart_Total, NumPart_Total_HW, and Num_files_per_snapshot
            _fix_header_numbers(outfile, options)

            # Now that we have the required group Units, we can make the registry
            create_unit_registry(context)
            set_cosmo_quantities(context)

            # Prioritize the Coordinates field for all particle groups
            _process_particle_groups(
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

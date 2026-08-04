import json
import logging
import tempfile
from functools import partial
from pathlib import Path
from traceback import TracebackException
from typing import Literal, cast

import h5py
import unyt.dimensions as ud
from unyt.array import unyt_quantity
from unyt.unit_registry import UnitRegistry

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
)
from .translation_base import (
    Translation,
    TranslationContext,
    dummy_field,
    translation_error_handler,
)
from .translations import GroupTranslation

LOGGER = logging.getLogger(LOGGER_NAME)


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


def _process_packingcubes(
    context: TranslationContext,
    error_handler: ErrorHandler,
    options: TranslationOptions,
):
    particle_groups = options.packingcubes
    particle_groups = (
        context.quantities["ParticleNames"]
        if particle_groups[0] == "*"
        else particle_groups
    )
    # ensure particle groups are present
    # (note that this might require translations)
    translation = context.translation
    initialize_particle_groups(particle_groups, context, error_handler)
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

    def __init__(self, snapshot_path: str | Path):

        if not h5py.is_hdf5(snapshot_path):
            raise ValueError("Must provide valid path to snapshot")

        self.path = Path(snapshot_path)
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

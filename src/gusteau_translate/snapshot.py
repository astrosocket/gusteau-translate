import json
import logging
from functools import partial
from pathlib import Path
from traceback import TracebackException
from typing import Literal

import h5py

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
        ):
            # Start with header, since that's most important
            LOGGER.debug("Initializing context")
            context = TranslationContext(
                infile, outfile, translation, dummy_field, options=options
            )

            LOGGER.debug("Initializing error handler")
            error_handler = translation_error_handler

            LOGGER.debug("Creating required groups in outfile")
            translation.initialize_required_groups(outfile)

            # Order of translations:
            #  1. Explicit transformations
            #    a. Header
            #    b. Remaining required groups (Code, RunInfo, etc)
            #    c. out2in
            #  2. Remaining fields in input file
            #  3. Any unused translations

            LOGGER.info(
                "Translating header, required groups, and explicit translations"
            )
            self.directory.update(
                translation.apply_explicit_translations(context, error_handler)
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

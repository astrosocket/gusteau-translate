import argparse
import contextlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Literal

__lazy_modules__ = ["packingcubes"]
import ruamel.yaml as pyyaml

import gusteau_translate

from .config import (
    _ALL_PARTICLE_GROUPS,
    DDEBUG,
    LOGGER_NAME,
    DatasetDefaultOptions,
    PackingCubesOptions,
    TranslationOptions,
    ValidationOptions,
)
from .snapshot import DiagnosticWarning, Snapshot
from .styles import parse_modular_style_file
from .translation_base import (
    GUSTEAU_Options,
    Translation,
)
from .translations.translation_loader import (
    default_translation_factory,
    from_dict,
)

LOGGER = logging.getLogger(LOGGER_NAME)


def _add_packingcubes_subarguments(parser: argparse.ArgumentParser):
    """Add packingcubes' Cubes command arguments

    Add the CLI arguments available for packingcubes' Cubes command
    to the CLI. Arguments will be available with `pc-` prepended to
    their names. For example, Cubes' `particle_threshold` will
    become `--pc-particle-threshold PARTICLE_THRESHOLD`. Note that
    only the long-form names will be included.

    Parameters
    ----------
    parser:
        The parser to add CLI arguments to
    """
    if not importlib.util.find_spec("packingcubes"):
        LOGGER.info("Module packingcubes not found, disallowing CLI options.")
        return
    parser.add_argument(
        "-p",
        "--packing-cubes",
        action="store_true",
        help="""
        Run packingcubes on particle types.
        """,
        dest="packingcubes",
    )

    # We only want a few of the available parameters:
    # particle_threshold, side-length, particle-types, and the bounding-box
    pc_args = parser.add_argument_group(
        "Packingcubes options",
        """
        Options for the packingcubes step. Only relevant for translation.
        """,
    )
    pc_args.add_argument(
        "--pc-override-cells",
        help="""
        Override the cell structure already present in the source file.
        Does nothing if no such structure is present.
        """,
        action="store_true",
    )
    pc_args.add_argument(
        "--pc-particle-threshold",
        help="Set the leaf node particle threshold",
        type=int,
        metavar="Particle Threshold",
    )
    pc_args.add_argument(
        "--pc-side-length",
        help="""
        Set the number of cells per dimension.
        Value must be between 3 and 32, default %(default)s.

        **Ignored** if the top-level cell structure is already provided in the
        snapshot and --pc-override-cells was not specified.
        """,
        type=int,
        default=PackingCubesOptions.side_length,
        choices=range(3, 32),
        metavar="N",
    )
    pc_args.add_argument(
        "--pc-particle-types",
        help="""
        Particle types to run packingcubes on as a comma-separated string. 
        Remaining types will not be run through packingcubes, but will be split
        according to the top-level grid if multi-file output is selected.
        """,
        type=str,
        metavar='"PartType0,PartType1,etc."',
        default=_ALL_PARTICLE_GROUPS,
    )
    pc_args.add_argument(
        "--pc-box",
        help="""
        Snapshot bounding box to use instead of the bounding box in
        /Header.Bounding_box when generating the cell grid. 
        
        **Ignored** if the top-level group structure is already provided in the
        snapshot and --pc-override-cells was not specified.

        This can also be used in conjuction with the --pc-autobox option:
        --pc-autobox will set the boundaries of the cell-grid while this option
        sets the boundaries of the outer/overflow cell.
        """,
        nargs=6,
        type=float,
        metavar=("x", "y", "z", "dx", "dy", "dz"),
    )
    pc_args.add_argument(
        "--pc-autobox",
        help="""
        Particle type to use for cell grid auto-sizing (for a zoom-in simulation
        for example). Will find the xyz extents of the specified (output!)
        particle type and use those to create the grid cells' bounding box.

        **Ignored** if the top-level group structure is already provided in the
        snapshot and --pc-override-cells was not specified.

        Note that this can be used in conjunction with --pc-box: --pc-box will
        set the boundaries of the outer/overflow cell while this option sets the
        cell-grid.
        """,
        type=str,
        metavar="output_particle_type",
    )


def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse CLI arguments using argparse

    ```bash exec="on" result="bash"
    alchemize --help
    ```

    """
    if argv is None:
        # need to skip caller or it's picked up as the first positional argument
        argv = sys.argv[1:]

    description = """
    Transform an HDF5 file into a GUSTEAU-type snapshot
    """

    parser = argparse.ArgumentParser(
        description=description,
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase output verbosity",
    )

    parser.add_argument(
        "-q", "--quiet", action="count", help="decrease output verbosity", default=0
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"gusteau-translate: {gusteau_translate.__version__}",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        help="""Specify the name of the logfile to use. Default %(default)s""",
        default="alchemize.log",
    )

    parser.add_argument(
        "input",
        help="Path to the input snapshot file",
        type=str,
        metavar="input-snapshot",
    )

    # We can't use subcommands for this next part because we want input-snapshot
    # to remain positional

    parser.set_defaults(subcommand=translate_and_validate)

    parser.add_argument(
        "--to",
        "--translate-only",
        help="Only run the translation part of the process",
        dest="subcommand",
        action="store_const",
        const=translate,
    )

    parser.add_argument(
        "--vo",
        "--validate-only",
        help="Only run the validation part of the process",
        dest="subcommand",
        action="store_const",
        const=validate,
    )

    parser_trans = parser.add_argument_group(
        "Translation Options",
        "Arguments used by the translation step only",
    )

    parser_valid = parser.add_argument_group(
        "Validation Options",
        "Arguments used by the validation step only",
    )

    parser_trans.add_argument(
        "output",
        help="Path to the (primary) output snapshot file",
        type=str,
        metavar="output-snapshot",
        nargs="?",
    )

    parser_trans.add_argument(
        "-f",
        "--force-overwrite",
        help="Overwrite output snapshot file if it exists",
        action="store_true",
    )

    parser_trans.add_argument(
        "--diagnostic",
        action="store_true",
        help="""
        Skip loading/saving entire datasets and just load the first row.
        Use this option when checking the output of a style.
        """,
    )

    parser_trans.add_argument(
        "-t",
        "--translation",
        type=str,
        help="Path of the translation style file to use",
    )

    default_compression = DatasetDefaultOptions.model_fields["compression"].default
    parser_trans.add_argument(
        "-c",
        "--compression",
        type=str,
        help=f"""
        Override the style-provided default compression option
        (usually {default_compression}). Compression options cannot be overridden
        from the command line. For more control, specify in a style file.
        """,
    )

    _add_packingcubes_subarguments(parser)

    parser_trans.add_argument(
        "--num-partfiles",
        help="""
        Specify the number of partfiles to use. For example, --num-partfiles 3
        will produce 4 files: output-snapshot.hdf5, and output-snapshot_0.hdf5,
        output-snapshot_1.hdf5, and output-snapshot_2.hdf5. output-snapshot.hdf5
        will contain virtual datasets comprised of the partial datasets in each
        of the partfiles. NOTE: requires the --packingcubes option
        """,
        type=int,
        default=1,
    )

    parser_valid.add_argument(
        "--gusteau",
        help="GUSTEAU version specification to use",
        type=str,
    )

    args = parser.parse_args(argv)

    verbosity = args.verbose - args.quiet

    loglvl = LOGGER.level
    if verbosity >= 2:
        loglvl = DDEBUG
    elif verbosity >= 1:
        loglvl = logging.DEBUG
    elif verbosity >= 0:
        loglvl = logging.INFO
    elif verbosity >= -1:
        loglvl = logging.WARNING
    elif verbosity < -1:
        loglvl = logging.ERROR
    LOGGER.setLevel(loglvl)

    return args


def get_translation(
    translation_file: str | Path, options: TranslationOptions | None
) -> Translation:
    translation_file = Path(translation_file)
    yaml = pyyaml.YAML(typ="safe", pure=True)
    try:
        doc, gusteau_dict = parse_modular_style_file(translation_file, yaml=yaml)
    except FileNotFoundError as fnferr:
        raise FileNotFoundError(
            f"Could not find style file {translation_file} as file or as built in."
        ) from fnferr

    gusteau = GUSTEAU_Options(**gusteau_dict)

    if options and options.compression is not None:
        gusteau.dataset_defaults.compression = options.compression

    if options and options.packingcubes:
        # Add dataset sorting step to all particle datasets

        gusteau.pipeline_class_additions.setdefault(
            "ParticleDatasetTranslation", []
        ).append("SortDatasetStep")

        # Add dataset cubing step to all particle groups included
        for pg in options.packingcubes:
            gusteau.pipeline_path_additions.setdefault(f"/{pg}/Coordinates", []).append(
                "PackCubesStep"
            )

        from .translations.attribute_translation import SaveQuantityStep

        # Add saving the bounding box
        gusteau.pipeline_path_additions.setdefault("/Header.Bounding_box", []).append(
            SaveQuantityStep(name="Bounding_box")
        )

    return from_dict(
        doc, default_factory=default_translation_factory(), gusteau=gusteau
    )


def translate(snapshot: Snapshot, args: argparse.Namespace) -> Literal[True]:
    """Translate the provided snapshot

    Parameters
    ----------
    snapshot:
        The snapshot to translate

    args:
        Command-line arguments for the process. See [parse_arguments][parse_arguments]

    Returns
    -------
    :
        If the snapshot was translated without errors

    Raises
    ------
    TranslationError
        If an error occured during the translation process

    """
    if not hasattr(args, "output"):
        sys.exit("the following arguments are required: output-snapshot")

    if args.num_partfiles > 1 and not args.packingcubes:
        sys.exit(
            "Snapshots can only be split into multiple partfiles if"
            " the packingcubes option is selected."
        )

    pc_options = PackingCubesOptions(
        particle_type_str=getattr(
            args, "pc_particle_types", PackingCubesOptions.particle_type_str
        ),
        particle_threshold=getattr(
            args, "pc_particle_threshold", PackingCubesOptions.particle_threshold
        ),
        override_cells=getattr(
            args, "pc_override_cells", PackingCubesOptions.override_cells
        ),
        side_length=getattr(args, "pc_side_length", PackingCubesOptions.side_length),
        box=getattr(args, "pc_box", PackingCubesOptions.box),
        autobox=getattr(args, "pc_autobox", PackingCubesOptions.autobox),
    )

    options = TranslationOptions(
        packingcubes=pc_options.particle_types if args.packingcubes else [],
        force_overwrite=bool(args.force_overwrite),
        diagnostic=bool(args.diagnostic),
        num_partfiles=args.num_partfiles,
        pc_options=pc_options,
        compression=args.compression,
    )
    LOGGER.info(f"Loaded translation options: {options}")

    translation = get_translation(args.translation, options)

    with contextlib.suppress(DiagnosticWarning):
        return snapshot.translate(
            translation,
            args.output,
            options=options,
        )


def validate(snapshot: Snapshot, args: argparse.Namespace) -> bool:
    """Validate the provided snapshot

    Parameters
    ----------
    snapshot:
        The snapshot to validate

    args:
        Command-line arguments for the process. See [parse_arguments][parse_arguments]

    Returns
    -------
    :
        `True` if the snapshot correctly conforms to the GUSTEAU specification


    """
    options = ValidationOptions(diagnostic=args.diagnostic)
    LOGGER.info(f"Loaded validation options: {options}")

    return snapshot.validate(options=options)


def translate_and_validate(snapshot: Snapshot, args: argparse.Namespace) -> bool:
    translate(snapshot, args)
    out_snapshot = Snapshot(args.output)
    return validate(out_snapshot, args)


def cli(argv: list[str] | None = None) -> bool:
    """Run the command-line interface

    See [the User's Guide](..#command-line-interface) for more details

    Parameters
    ----------
    argv:
        List of command line arguments in [sys.argv][sys.argv] format.

    Returns
    -------
    :
        `True` if translation and/or validation are successful

    """
    logging.basicConfig(stream=sys.stdout)

    args = parse_arguments(argv=argv)

    file_handler = logging.FileHandler(filename=args.log_file, mode="w")
    logging.getLogger().addHandler(file_handler)

    snapshot = Snapshot(args.input)

    return args.subcommand(snapshot, args)


if __name__ == "__main__":
    sys.exit(cli())

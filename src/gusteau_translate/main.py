import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Literal, cast

__lazy_modules__ = ["packingcubes"]
import ruamel.yaml as pyyaml

import gusteau_translate

from .config import (
    _ALL_PARTICLE_GROUPS,
    LOGGER_NAME,
    PackingCubesOptions,
    TranslationOptions,
    ValidationOptions,
)
from .snapshot import Snapshot
from .styles import parse_style_file
from .translation_base import (
    Translation,
    TranslationStep,
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
        "--pc-particle-threshold",
        help="Set the leaf node particle threshold",
        type=int,
        metavar="Particle Threshold",
    )
    pc_args.add_argument(
        "--pc-side-length",
        help="""
        Set the number of cells per dimension.
        Value must be between 3 and 32, default 8.

        **Ignored** if the top-level cell structure is already provided in the
        snapshot.
        """,
        type=int,
        default=8,
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
        /Header.Bounding_box. 
        
        **Ignored** if the top-level group structure is already provided in the
        snapshot.
        """,
        nargs=6,
        type=float,
        metavar=("x", "y", "z", "dx", "dy", "dz"),
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
        "--skip-data",
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
    if verbosity >= 1:
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
        doc, gusteau = parse_style_file(translation_file, yaml=yaml)
    except FileNotFoundError:
        LOGGER.warning(
            f"Could not find style file {translation_file} as file or as built in. "
            "Using identity"
        )
        return Translation(default_factory=default_translation_factory())
    if options and options.packingcubes:
        # Add dataset sorting step to all particle datasets
        pipeline_class_additions = cast(
            dict[str, list[str]] | None, gusteau.get("pipeline_class_additions")
        )
        pipeline_class_additions = (
            {} if pipeline_class_additions is None else pipeline_class_additions
        )
        pipeline_class_additions.setdefault("ParticleDatasetTranslation", []).append(
            "SortDatasetStep"
        )
        gusteau["pipeline_class_additions"] = pipeline_class_additions

        # Add dataset cubing step to all particle groups included
        pipeline_path_additions = cast(
            dict[str, list[str | TranslationStep]] | None,
            gusteau.get("pipeline_path_additions"),
        )
        pipeline_path_additions = (
            {} if pipeline_path_additions is None else pipeline_path_additions
        )
        for pg in options.packingcubes:
            pipeline_path_additions.setdefault(f"/{pg}/Coordinates", []).append(
                "PackCubesStep"
            )

        from .translations.attribute_translation import SaveQuantityStep

        # Add saving the bounding box
        pipeline_path_additions.setdefault("/Header.Bounding_box", []).append(
            SaveQuantityStep(name="Bounding_box")
        )
        gusteau["pipeline_path_additions"] = pipeline_path_additions
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
        **{k: v for k, v in vars(args).items() if "pc-" in k}
    )

    options = TranslationOptions(
        packingcubes=pc_options.particle_types if args.packingcubes else [],
        force_overwrite=bool(args.force_overwrite),
        skip_data=bool(args.skip_data),
        pc_options=pc_options,
    )
    LOGGER.info(f"Loaded translation options: {options}")

    translation = get_translation(args.translation, options)

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
    options = ValidationOptions(skip_data=args.skip_data)
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
    file_handler = logging.FileHandler(filename="alchemize.log", mode="w")
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(handlers=[file_handler, stdout_handler])

    args = parse_arguments(argv=argv)
    snapshot = Snapshot(args.input)

    return args.subcommand(snapshot, args)


if __name__ == "__main__":
    sys.exit(cli())

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal, cast

import ruamel.yaml as pyyaml

import gusteau_translate

from .config import LOGGER_NAME, TranslationOptions, ValidationOptions
from .snapshot import Snapshot
from .styles import parse_style_file
from .translation_base import (
    Translation,
)
from .translations.translation_loader import (
    default_translation_factory,
    from_dict,
)

LOGGER = logging.getLogger(LOGGER_NAME)


def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse CLI arguments using argparse"""
    if argv is None:
        # need to skip caller or it's picked up as the first positional argument
        argv = sys.argv[1:]

    description = """
    Translate an HDF5 file into a GUSTEAU-type snapshot
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
        help="Path to the output snapshot file",
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

    parser_trans.add_argument(
        "-p",
        "--packing-cubes",
        action="extend",
        nargs="*",
        help="""
        Run packingcubes on the specified particle types.
        If no particle types are specified, defaults to all.
        """,
        dest="packingcubes",
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


def get_translation(translation_file: str | Path) -> Translation:
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
    return from_dict(
        doc, default_factory=default_translation_factory(), gusteau=gusteau
    )


def translate(snapshot: Snapshot, args: argparse.Namespace) -> Literal[True]:
    if not hasattr(args, "output"):
        sys.exit("the following arguments are required: output-snapshot")
    options = TranslationOptions(
        packingcubes=cast(list[str], args.packingcubes),
        force_overwrite=bool(args.force_overwrite),
        skip_data=bool(args.skip_data),
    )
    LOGGER.info(f"Loaded translation options: {options}")

    translation = get_translation(args.translation)

    return snapshot.translate(
        translation,
        args.output,
        options=options,
    )


def validate(snapshot: Snapshot, args: argparse.Namespace) -> bool:
    options = ValidationOptions(skip_data=args.skip_data)
    LOGGER.info(f"Loaded validation options: {options}")

    return snapshot.validate(options=options)


def translate_and_validate(snapshot: Snapshot, args: argparse.Namespace) -> bool:
    translate(snapshot, args)
    out_snapshot = Snapshot(args.output)
    return validate(out_snapshot, args)


def cli(argv=None) -> bool:
    file_handler = logging.FileHandler(filename="alchemize.log", mode="w")
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(handlers=[file_handler, stdout_handler])

    args = parse_arguments(argv=argv)
    snapshot = Snapshot(args.input)

    return args.subcommand(snapshot, args)


if __name__ == "__main__":
    sys.exit(cli())

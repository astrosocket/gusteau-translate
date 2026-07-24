"""
Module for locating and reading style files
"""

import importlib.resources
import logging
from pathlib import Path
from typing import Any

import ruamel.yaml as pyyaml
import tomlkit

LOGGER = logging.getLogger("gusteau_translate")


def _load_yaml_file(
    style_str: str, *, yaml: pyyaml.YAML | None = None, **kwargs
) -> dict:
    if yaml is None:
        yaml = pyyaml.YAML(typ="safe", pure=True)
    return yaml.load(style_str)


def _load_toml_file(style_str: str, **kwargs) -> dict:
    return tomlkit.parse(style_str).unwrap()


def _load_style_file(path: Path, text: str, **kwargs) -> dict:
    match path.suffix:
        case ".yml" | ".yaml" | ".translation":
            return _load_yaml_file(text, **kwargs)
        case ".toml" | ".tml":
            return _load_toml_file(text, **kwargs)
        case _:
            raise NotImplementedError(f"The {path.suffix} file type is not understood!")


def read_style_file(style_file: str | Path, **kwargs) -> tuple[str, str]:
    """
    Read in the provided style file

    Parameters
    ----------
    style_file: str | Path
        The path to the specified style file.
        Can be just the file stem if using one of the built-ins (e.g.
        "swift-eagle").
        File must be utf-8 encoded.

    **kwargs:
        Passed to individual file type readers

    Returns
    -------
    text: str
        The text contents of the file

    filename: str
        The actual location of the file.
        If the file is a built-in style, will be in the form of
        "{built-in}/name"

    Raises
    ------
    FileNotFound
        If the file doesn't exist as named and cannot be found in the built-in
        styles.
    """
    style_file = Path(style_file)
    if not style_file.exists():
        # Look in the provided styles
        anchor = "gusteau_translate"

        if not style_file.suffix:
            style_file = style_file.with_suffix(".yml")
        style_path = importlib.resources.files(anchor).joinpath("styles", style_file)
        if not style_path.is_file():
            raise FileNotFoundError(f"Requested style ({style_file.stem}) not found.")
        with importlib.resources.as_file(style_path) as path:
            LOGGER.debug(f"Loading style file {style_file} from built-ins")
            return path.read_text("utf-8"), "{built-ins}/" + str(style_file)
    LOGGER.debug(f"Loaded provided style file {style_file}")
    return style_file.read_text("utf-8"), str(style_file)


def parse_style_file(style_file: str | Path, **kwargs) -> tuple[dict, dict[str, Any]]:
    """
    Parse the text of a style file as dictionary

    Parameters
    ----------
    style_file:
        The path or name of the style file.
        Will first search for a file with the exact name, then look for the name
        in the provided style files.

    kwargs:
        Pass additional options to the actual text parser (e.g. yaml or toml)

    Returns
    -------
    style:
        Style file in dictionary form

    gusteau:
        Settings specific to this style
    """
    text, path = read_style_file(style_file, **kwargs)
    style = _load_style_file(Path(path), text=text, **kwargs)
    top_level = {k.lower(): k for k in style}
    gusteau = style.pop(top_level["gusteau"], {})
    return style, gusteau

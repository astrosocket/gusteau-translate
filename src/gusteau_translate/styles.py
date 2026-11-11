"""
Module for locating and reading style files
"""

import importlib.resources
import logging
from collections import deque
from pathlib import Path
from typing import Any

import ruamel.yaml as pyyaml
import tomlkit

from .config import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


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


def _validate_depends_on(depends_on: Any) -> deque[str]:
    """Private function to validate the gusteau depends-on field"""
    depends_on = [depends_on] if isinstance(depends_on, str) else depends_on

    if not isinstance(depends_on, list) or not all(
        isinstance(s, str) for s in depends_on
    ):
        raise ValueError(
            "Only strings or lists of strings are valid for"
            " the gusteau depends-on field."
        )
    return deque(depends_on)


def parse_modular_style_file(
    style_file: str | Path, **kwargs
) -> tuple[dict, dict[str, Any]]:
    """
    Parse the text of style_file as dictionary, populating missing values from subfiles

    Calls [parse_style_file][gusteau_translate.styles.parse_style_file] but then checks
    the gusteau dictionary for the "depends-on" field. Processes the value(s) as modular
    style files (recursing as needed) and concatenating the loaded results into the
    final style.

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

    Notes
    -----
    Note that values from ancestor/depended-on style files do not overwrite values
    in the top-level, so this can be used as a specialization mechanism. For example,
    one could write a simulation-code.style_file (like gizmo.yml) and a
    simulation.style_file (like simba.yml) where most of the translations would be in
    the simulation-code.style_file and simulation.style_file would contain only the
    translations that differ from the baseline, like paths to parameter files or
    chemical abundance redefinitions.

    Operates on a queue, so given example styles
    ```yml
    # simba.yml
    gusteau:
        depends-on: [gizmo, simba-sub]
    Header:
        Name: Simba
    # gizmo.yml
    gusteau:
        depends-on: [base]
    Header:
        Name: Gizmo
        Type: GIZMO
    # simba-sub.yml
    Header:
        Type: Simba-sub
        Other: Simba-sub
    # base.yml
    Header:
        Other: Base
    ```
    the equivalent style would be
    ```yml
    Header:
        Name: Simba
        Type: GIZMO
        Other: Simba-sub
    ```
    """
    LOGGER.info(f"Parsing modular style file {style_file}")
    style, gusteau = parse_style_file(style_file, **kwargs)

    to_visit = _validate_depends_on(gusteau.pop("depends-on", []))

    visited = {str(style_file)}

    while to_visit:
        sf = Path(to_visit.popleft())
        if sf in visited:
            continue
        visited.add(str(sf))
        base_style = style
        base_gusteau = gusteau
        style, gusteau = parse_style_file(sf, **kwargs)
        style |= base_style
        additional_to_visit = _validate_depends_on(gusteau.pop("depends-on", []))
        to_visit.extend(additional_to_visit)
        gusteau |= base_gusteau
    LOGGER.info("Built modular style file from")
    for sf in visited:
        LOGGER.info(f"\t{sf}")
    gusteau["depends-on"] = visited

    return style, gusteau

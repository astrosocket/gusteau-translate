"""
The gusteau_translate package

`gusteau_translate` is a tool for converting HDF5 particle data snapshots into
and verifying that snapshots match the
[GUSTEAU specification](gusteau-spec.readthedocs.io).

It is generally intended to be used from the command line (see the
[User's Guide](#Users-Guide) or run `alchemize --help`) but can also be accessed
programmatically.
"""

import logging
from importlib.metadata import PackageNotFoundError, version

from . import optional_steps as optional_steps
from .main import cli as cli
from .main import parse_arguments as parse_arguments
from .main import translate as translate
from .main import validate as validate
from .snapshot import Snapshot as Snapshot

try:
    __version__ = version("gusteau_translate")
except PackageNotFoundError:
    __version__ = "Not Found"

__all__ = ["Snapshot", "cli", "parse_arguments", "translate", "validate"]

LOGGER = logging.getLogger(__name__)

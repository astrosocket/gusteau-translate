import logging
from importlib.metadata import PackageNotFoundError, version

from gusteau_translate import optional_steps as optional_steps
from gusteau_translate.main import cli as cli
from gusteau_translate.main import translate as translate
from gusteau_translate.main import validate as validate

try:
    __version__ = version("gusteau_translate")
except PackageNotFoundError:
    __version__ = "Not Found"

__all__ = ["cli", "translate", "validate"]

LOGGER = logging.getLogger(__name__)

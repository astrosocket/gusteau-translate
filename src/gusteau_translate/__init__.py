import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gusteau_translate")
except PackageNotFoundError:
    __version__ = "Not Found"

LOGGER = logging.getLogger(__name__)

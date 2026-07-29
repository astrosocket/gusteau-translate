import dataclasses
from dataclasses import KW_ONLY, dataclass

LOGGER_NAME = "alchemize"


@dataclass
class TranslationOptions:
    _: KW_ONLY
    force_overwrite: bool = False
    skip_data: bool = False
    packingcubes: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        self.packingcubes = [] if self.packingcubes is None else self.packingcubes


def _default_translation_options() -> TranslationOptions:
    return TranslationOptions()


@dataclass
class ValidationOptions:
    _: KW_ONLY
    gusteau_version: str | None = None
    skip_data: bool = False


def _default_validation_options() -> ValidationOptions:
    return ValidationOptions()


@dataclass
class Options:
    _: KW_ONLY
    trans_options: TranslationOptions = dataclasses.field(
        default_factory=_default_translation_options
    )
    valid_options: ValidationOptions = dataclasses.field(
        default_factory=_default_validation_options
    )

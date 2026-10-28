import dataclasses
from dataclasses import KW_ONLY, InitVar, dataclass

import numpy as np
from pydantic import PositiveInt

LOGGER_NAME = "alchemize"
_ALL_PARTICLE_GROUPS = "*"


@dataclass
class PackingCubesOptions:
    _: KW_ONLY
    particle_threshold: int | None = None
    side_length: int = 8
    particle_type_str: InitVar[str] = _ALL_PARTICLE_GROUPS
    particle_types: list[str] = dataclasses.field(init=False, default_factory=list[str])
    box: np.ndarray | None = None

    def __post_init__(self, particle_type_str: str):
        self.particle_types = [s.strip() for s in particle_type_str.split(",")]


@dataclass
class TranslationOptions:
    _: KW_ONLY
    force_overwrite: bool = False
    skip_data: bool = False
    packingcubes: list[str] = dataclasses.field(default_factory=list)
    num_partfiles: PositiveInt = 1
    pc_options: PackingCubesOptions = dataclasses.field(
        default_factory=PackingCubesOptions
    )

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

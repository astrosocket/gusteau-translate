import dataclasses
import logging
from dataclasses import KW_ONLY, InitVar, dataclass
from typing import Annotated, Literal, Self

import numpy as np
from annotated_types import MaxLen, MultipleOf
from pydantic import BaseModel, NonNegativeInt, PositiveInt, model_validator

LOGGER_NAME = "alchemize"
DDEBUG = logging.DEBUG - 1
logging.addLevelName(DDEBUG, "DEVDEBUG")
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


class FilterInfo(BaseModel):
    filter_code: int
    flags: NonNegativeInt
    aux: Annotated[tuple[NonNegativeInt, ...], MaxLen(256)]
    name: Annotated[str, MaxLen(256)]


type compression_type = (
    Literal["gzip", "lzf", "szip"]
    | str
    | int
    | tuple[int, ...]
    | tuple[FilterInfo, ...]
)


class DatasetDefaultOptions(BaseModel):
    _: KW_ONLY
    compression: compression_type | None = "gzip"
    """Generic dataset compression. Defaults to gzip"""
    compression_opts: (
        NonNegativeInt
        | tuple[
            Literal["ec", "nn"],
            Annotated[NonNegativeInt, MultipleOf(2)],
        ]
        | None
    ) = None
    """Generic dataset compression options. Defaults to None"""
    scale_offset_filter: NonNegativeInt | bool | None = None
    """Whether to apply the scale offset filter to generic datasets
    Default is to use the source dataset's scale_offset value
    """
    shuffle_filter: bool | None = None
    """Whether to apply the shuffle filter to generic datasets
    Default is to use the source dataset's shuffle_filter flag
    """
    fletcher32_filter: bool | None = None
    """Whether to apply the shuffle filter to generic dataset
    Default is to use the source dataset's fletcher32_filter flag
    """
    chunks: Literal[True] | None = True
    """
    Whether chunking should be used for generic datasets. 
    Default is to use the source dataset's chunks, accounting for column specification,
    or, if not specified, `True` to have h5py compute them automatically.
    Note that if `compression` is specified, chunks **must** be `True`.
    """

    @model_validator(mode="after")
    def check_chunks_with_compression(self) -> Self:
        if self.compression is not None and self.chunks is None:
            raise ValueError(
                "Chunks cannot be None if a compression method is specified"
            )
        return self


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
    compression: str | None = None

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

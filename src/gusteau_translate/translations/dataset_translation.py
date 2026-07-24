import logging
import numbers
import typing
from dataclasses import KW_ONLY, InitVar, dataclass, field
from pathlib import Path
from typing import Literal, cast

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..base_steps.base import LoadStep, SaveStep
from ..pipeline import NextStep
from ..translation_base import (
    FieldTranslation,
    TranslationClassError,
    TranslationContext,
    TranslationError,
    TranslationPipeline,
    TranslationStep,
)
from .misc_translation import SkipTranslation

LOGGER = logging.getLogger("gusteau_translate")


def _validate_columns(columns: ArrayLike | None) -> int | NDArray[np.int_] | None:
    if columns is None:
        return None
    columns = np.array(columns).astype(np.int_)
    return int(columns) if not len(columns.shape) else columns


@dataclass
class DatasetTranslation(FieldTranslation):
    _: KW_ONLY
    columns: int | list[int] | NDArray[np.int_] | None = None
    """
    Index or list of indices representing the column in the 2D dataset to load
    1D datasets or datasets that use all the columns should specify None
    """
    compression: str | Literal["gzip", "lzf", "szip"] | int | None = None
    """Compression used by this dataset"""
    compression_opts: int | dict[str, str | int | float] | None = None
    """Compression options used by this dataset"""
    chunks: tuple[int, ...] | bool | None = None
    """
    Chunk division used by dataset. 
    Set to `True` to have h5py compute them automatically
    """

    name_mapping: dict[str, str | Literal["[skip]"]] = field(default_factory=dict)
    """
    Mapping of names to outcomes. 
    Attribute names that are not included are mapped identically
    Alternatively, map to a valid GUSTEAU attribute name (only containing 
    A-Za-z0-9_) to change the attribute name or "[skip]" to skip translating
    the attribute.
    """

    def __post_init__(
        self,
    ):
        self.columns = _validate_columns(self.columns)
        self.pipeline.append(DatasetLoadStep())
        self.pipeline.append(DatasetSaveStep())
        if self.input_path != self.output_path or self.name_mapping:
            self.pipeline.append(MoveDatasetAttributesStep())

    @property
    def name(self) -> str:
        return Path(self.output_path).stem


@dataclass
class ParticleDatasetTranslation(DatasetTranslation):
    _: KW_ONLY
    units: str | None = None

    def __post_init__(
        self,
    ):
        self.name_mapping["Lossy compression filter"] = "LossyCompression"
        super().__post_init__()
        # Shouldn't use these if dataset already has units...
        # self.pipeline.append(DoCosmoUnytsStep())


@dataclass
class ConstantParticleDatasetTranslation(ParticleDatasetTranslation):
    """Dataset with constant value and copied attributes

    Use `like_dataset` (alias for `input_path`) to set the location for attribute,
    compression, chunking, etc. copying. Value will always be array of same
    length/units/etc. but equal to `constant`
    """

    _: KW_ONLY
    constant: numbers.Number
    like_dataset: InitVar[str | None] = None
    """Alias for input_path (overrides input path)"""

    def __post_init__(self, like_dataset: str | None = None):
        super().__post_init__()
        self.input_path = like_dataset if like_dataset else self.input_path

    @property
    def value(self):
        return self._value * 0 + self.constant

    @value.setter
    def value(self, new_value):
        self._value = new_value


@dataclass
class DatasetLoadStep(LoadStep):
    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, DatasetTranslation):
            raise TranslationClassError(step=self, field=field)
        LOGGER.debug(f"Loading dataset {field.input_path}")
        indata = cast(h5py.Dataset, context.infile[field.input_path])
        if field.columns is not None and len(indata.shape) < 2:
            raise TranslationError("Attempting to load 1D dataset as 2D")
        max_index = 1 if context.options.skip_data else indata.shape[0] + 1
        new_type = indata.dtype if field.dtype is None else field.dtype
        field.value = np.array(
            indata[:max_index]
            if field.columns is None
            else indata[:max_index, field.columns],
            dtype=new_type,
        )
        field.compression = (
            cast(str | None, indata.compression)
            if field.compression is None
            else field.compression
        )
        field.compression_opts = (
            cast(type(field.compression_opts), indata.compression_opts)
            if field.compression_opts is None
            else field.compression_opts
        )
        field.chunks = (
            cast(tuple[int, ...] | None, indata.chunks)
            if field.chunks is None
            else field.chunks
        )
        if isinstance(field.chunks, tuple):
            chunk_height = 1 if context.options.skip_data else field.chunks[0]
            if field.columns is not None:
                field.chunks = (
                    (chunk_height,)
                    if isinstance(field.columns, int)
                    else (chunk_height, len(field.columns))
                )
            else:
                field.chunks = (chunk_height, *field.chunks[1:])
        next_step(context)


@dataclass
class DatasetSaveStep(SaveStep):
    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, DatasetTranslation):
            raise TranslationClassError(step=self, field=field)
        LOGGER.debug(f"Saving dataset {field.output_path}")
        match field.compression:
            case "":
                context.outfile.create_dataset(name=field.output_path, data=field.value)
            case "lzf" | "szip":
                context.outfile.create_dataset(
                    name=field.output_path,
                    data=field.value,
                    compression=field.compression,
                    chunks=field.chunks,
                )
            case "gzip":
                context.outfile.create_dataset(
                    name=field.output_path,
                    data=field.value,
                    compression="gzip",
                    compression_opts=field.compression_opts,
                    chunks=field.chunks,
                )
            case _:
                if isinstance(field.compression_opts, int):
                    field.compression_opts = {
                        "compression_opts": field.compression_opts
                    }
                elif field.compression_opts is None:
                    field.compression_opts = {}
                context.outfile.create_dataset(
                    name=field.output_path,
                    data=field.value,
                    **field.compression_opts,
                    chunks=field.chunks,
                )
        # add source description attribute
        context.outfile[field.output_path].attrs["Source_Description"] = (
            field.input_description
        )
        field.value = None
        next_step(context)


@dataclass
class DoCosmoUnytsStep(LoadStep):
    """Load and apply cosmological and unit information"""

    priority: int = LoadStep.priority + 10
    _: KW_ONLY
    is_comoving: bool | None = None
    U_I: typing.SupportsFloat = 0
    U_L: typing.SupportsFloat = 0
    U_M: typing.SupportsFloat = 0
    U_T: typing.SupportsFloat = 0
    U_t: typing.SupportsFloat = 0
    a_scale: typing.SupportsFloat = 0
    h_scale: typing.SupportsFloat = 0

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, ParticleDatasetTranslation):
            raise TranslationClassError(step=self, field=field)

        # context.translation.add_field(
        #     ConstantAttributeTranslation(
        #         output_path=f"{field.output_path}.Units",
        #         constant=expression,
        #         pipeline=TranslationPipeline(),
        #     )
        # )

        next_step(context)


@dataclass
class MoveDatasetAttributesStep(TranslationStep):
    """
    Create translations to move all the dataset attributes
    """

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, DatasetTranslation):
            raise TranslationClassError(step=self, field=field)
        LOGGER.debug(f"Moving {field.input_path} attributes to {field.output_path}")

        dataset = cast(h5py.Dataset, context.infile[field.input_path])

        translation = context.translation

        def visit_attr(path: str, name: str):
            LOGGER.debug(f"Visiting {path}.{name}")
            full_name = f"{path}.{name}"
            mapping = field.name_mapping.get(name)
            output_path = f"{field.output_path}.{name if mapping is None else mapping}"
            # check if attribute already has explicit translation
            if (
                output_path in translation.out2in
                and translation.out2in[output_path]._explicit
            ):
                LOGGER.debug(f"Skipping already defined field {name}")
                return
            # We only try checking for an already defined output to support
            # single input/multiple output patterns

            if mapping == "[skip]":
                attr_trans = SkipTranslation(
                    input_path=full_name,
                    input_description="",
                    output_path=output_path,
                    pipeline=TranslationPipeline(),
                    _explicit=False,
                )
            else:
                attr_trans = translation._default_translation_factory(
                    obj="attribute",
                    path=full_name,
                    input_path=full_name,
                    output_path=output_path,
                )

            translation.add_field(attr_trans)

        path = dataset.name
        for name in dataset.attrs:
            visit_attr(path=path, name=name)
        next_step(context)

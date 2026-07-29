from __future__ import annotations

import logging
from dataclasses import KW_ONLY, InitVar, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..base_steps.base import LoadStep, SaveStep
from ..config import LOGGER_NAME
from ..pipeline import NextStep
from ..styles import read_style_file
from ..translation_base import (
    FieldTranslation,
    TranslationClassError,
    TranslationContext,
    TranslationError,
    TranslationStep,
    dict_info_type,
)

LOGGER = logging.getLogger(LOGGER_NAME)


@dataclass
class AttributeTranslation(FieldTranslation):
    _: KW_ONLY
    force_scalar: bool = False

    def __post_init__(self):
        self.pipeline.append(AttributeSaveStep())
        self.pipeline.append(AttributeLoadStep())
        path = Path(self.input_path)
        if self.input_path and not path.suffix:
            self.input_path = str(path.parent) + "." + path.stem
        path = Path(self.output_path)
        if self.output_path and not path.suffix:
            self.output_path = str(path.parent) + "." + path.stem

    @property
    def name(self) -> str:
        return Path(self.output_path).suffix[1:]


@dataclass
class ConstantAttributeTranslation(AttributeTranslation):
    _: KW_ONLY
    input_path: str = ""
    input_description: str = "Generated from constant"
    constant: InitVar[Any]

    def __post_init__(self, constant: Any):
        super().__post_init__()
        self.input_path = ""
        # don't need to load anything
        self.pipeline.remove("AttributeLoadStep")
        if np.array(constant).dtype == np.object_:
            raise TranslationError(
                "Constant must be convertible to a non-object numpy array"
            )
        if self.force_scalar and len(np.array(constant)) > 1:
            raise TranslationError("Attempting to force an array to a scalar")
        self._value = constant
        # Apply dtype if present and not a string
        if self.dtype is not None:
            if isinstance(constant, str):
                raise TranslationError("dtype cannot be applied to str input")
            self._value = np.array(self._value, dtype=self.dtype)

    def to_dict(self, docpart: dict[str, dict_info_type]) -> dict[str, dict_info_type]:
        table = super().to_dict(docpart)
        table["constant"] = self.value
        return table


@dataclass
class FileContentsTranslation(AttributeTranslation):
    _: KW_ONLY
    file_path: str
    input_description: str = "Generated from file"

    def __post_init__(self):
        super().__post_init__()
        # don't need to load anything
        self.pipeline.remove("AttributeLoadStep")
        self.input_description = f"Generated from {self.input_path}"

    def _load_file_contents(self) -> str:
        text, _ = read_style_file(self.file_path)
        return text

    @property
    def value(self):
        if self._value is None:
            self._value = self._load_file_contents()
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value


def process_attribute(
    attr: Any,
    *,
    dtype: np.typing.DTypeLike | None = None,
    force_scalar: bool | None = None,
) -> Any:
    """Process an attribute into a more useful type

    Convert an attribute in a few ways:
        1. Change the dtype of an array/number
        2. Convert fixed-length strings to variable-length
        3. Coerce values to scalars (eg np.array([1], dtype=np.uint32) -> np.uint32(1))

    Parameters
    ----------
    attr:
        The attribute to process

    dtype: DTypeLike, optional
        The new type of the value. If None (default), do nothing.

    force_scalar: bool, optional
        Flag to attempt scalar-coercing. If None (default), do nothing.

    Returns
    -------
    :
        The processed attribute

    Raises
    ------
    TranslationError
        Attribute was an array with more than one element and `force_scale=True`
    """
    # attributes are either strings (if a variable length string) or numpy
    # arrays. We want to convert fixed-length strings to variable-length and
    # all non-str values to field.dtype, if provided
    if not isinstance(attr, str):
        # _value is numpy array

        # convert types if requested
        attr = attr if dtype is None else attr.astype(dtype)

        # check if using an old-style fixed-length string and convert
        # attr = str(attr) if isinstance(attr, np.bytes_) else attr
        if isinstance(attr, np.bytes_):
            return str(attr)

        if force_scalar:
            if len(np.array(attr)) == 1:
                arr_value = np.array(attr)
                attr = attr if arr_value.ndim == 0 else arr_value[0]
            else:
                raise TranslationError(
                    f"Attempting to force an array {attr} to a scalar!"
                )
    return attr


@dataclass
class AttributeLoadStep(LoadStep):
    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, AttributeTranslation):
            raise TranslationClassError(step=self, field=field)
        # Note: can't use Path here, because non-Gusteau names may have spaces
        # in them
        inpath = field.input_path.split(".")
        path = inpath[0]
        name = inpath[1]
        LOGGER.debug(f"Loading attribute {name} on {path}")
        field._value = process_attribute(
            context.infile[path].attrs[name],
            dtype=field.dtype,
            force_scalar=field.force_scalar,
        )

        next_step(context)


@dataclass
class AttributeSaveStep(SaveStep):
    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        outpath = field.output_path.split(".")
        path = outpath[0]
        name = outpath[1]
        LOGGER.debug(f"Saving attribute {name}={field.value} on {path}")
        context.outfile[path].attrs[name] = field.value
        field.value = None
        next_step(context)


@dataclass
class SaveQuantityStep(TranslationStep):
    _: KW_ONLY
    name: str
    """The name of the quantity to store in the pipeline context"""

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, AttributeTranslation):
            raise TranslationClassError(step=self, field=field)
        context.quantities[self.name] = field.value
        next_step(context)

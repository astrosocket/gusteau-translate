from __future__ import annotations

import datetime
import logging
from dataclasses import KW_ONLY, InitVar, dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
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


def _process_constant(constant, *, force_scalar: bool, dtype: str | None):
    """Process a provided constant as an attribute"""
    if np.array(constant).dtype == np.object_:
        raise TranslationError(
            "Constant must be convertible to a non-object numpy array"
        )
    if force_scalar and len(np.array(constant)) > 1:
        raise TranslationError("Attempting to force an array to a scalar")
    # Apply dtype if present and not a string
    if dtype is not None:
        if isinstance(constant, str):
            raise TranslationError("dtype cannot be applied to str input")
        constant = np.array(constant, dtype=dtype)
    return constant


@dataclass
class ConstantAttributeTranslation(AttributeTranslation):
    """Attribute provided by a hard-coded constant value"""

    _: KW_ONLY
    input_path: str = ""
    input_description: str = "Generated from constant"
    constant: InitVar[Any]

    def __post_init__(self, constant: Any):
        super().__post_init__()
        self.input_path = ""
        # don't need to load anything
        self.pipeline.remove("AttributeLoadStep")
        self._value = _process_constant(
            constant, force_scalar=self.force_scalar, dtype=self.dtype
        )

    def to_dict(self, docpart: dict[str, dict_info_type]) -> dict[str, dict_info_type]:
        table = super().to_dict(docpart)
        table["constant"] = self.value
        return table


@dataclass
class ConditionalRequiredAttributeTranslation(AttributeTranslation):
    """
    Attributes that are only present if another attribute is true

    If the attribute is not present, fall back to constant value
    """

    _: KW_ONLY
    depends_on: str
    constant: InitVar[Any]

    def __post_init__(self, constant: Any):
        super().__post_init__()
        self.pipeline.remove("AttributeLoadStep")
        self.pipeline.append(ConditionalLoadStep())
        self.input_description = f"Dependent on {self.depends_on}, otherwise {constant}"


@dataclass
class FileContentsTranslation(AttributeTranslation):
    """Attribute provided by file contents"""

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


@dataclass
class SnapshotAttributeTranslation(AttributeTranslation):
    """Attribute provided by a snapshot file attribute"""

    _: KW_ONLY
    attribute: Literal["timestamp"] | str
    """Name of the file attribute to use (obtained through Path.stat). 
    Note that timestamp will return an ISO8601 formatted version of st_mtime
    """

    def __post_init__(self):
        super().__post_init__()
        # don't need to load anything
        self.pipeline.remove("AttributeLoadStep")
        self.pipeline.append(SnapshotAttributeLoadStep(attribute=self.attribute))
        self.input_description = f"Generated from {self.input_path}"


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
        if dtype == "str":
            return str(attr)
        attr = attr if dtype is None else attr.astype(dtype)

        # check if using an old-style fixed-length string and convert
        # attr = str(attr) if isinstance(attr, np.bytes_) else attr
        if isinstance(attr, np.bytes_):
            return attr.decode("utf8")

        if force_scalar:
            a = np.array(attr)
            if not a.ndim or len(a) == 1:
                arr_value = np.array(attr)
                attr = attr if arr_value.ndim == 0 else arr_value[0]
            else:
                raise TranslationError(
                    f"Attempting to force an array {attr} to a scalar!"
                )
    return attr


def _load_attribute(
    infile: h5py.File,
    input_path,
    *,
    dtype: str | None = None,
    force_scalar: bool,
    required: bool = False,
):
    # Note: can't use Path here, because non-Gusteau names may have spaces
    # in them
    inpath = input_path.split(".")
    path = inpath[0]
    name = inpath[1]
    if not (path in infile and name in infile[path].attrs):
        if required:
            raise TranslationError(f"Required attribute {input_path} not present!")
        LOGGER.debug(f"Attribute {name} on {path} not present. Nothing to do")
        return None
    LOGGER.debug(f"Loading attribute {name} on {path}")
    return process_attribute(
        infile[path].attrs[name],
        dtype=dtype,
        force_scalar=force_scalar,
    )


@dataclass
class AttributeLoadStep(LoadStep):
    """Load the value from field.input_path

    Note
    ----
    Will call [process_attribute][process_attribute] on the attribute as part
        of the loading process.
    """

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, AttributeTranslation):
            raise TranslationClassError(step=self, field=field)
        field._value = _load_attribute(
            context.infile,
            field.input_path,
            dtype=field.dtype,
            force_scalar=field.force_scalar,
            required=field._required,
        )
        if field._value is None:
            return

        next_step(context)


@dataclass
class ConditionalLoadStep(AttributeLoadStep):
    """Load the value from field.input_path iff depended-on field is True

    Note
    ----
    Will call [process_attribute][process_attribute] on the attribute as part
        of the loading process.
    """

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, ConditionalRequiredAttributeTranslation):
            raise TranslationClassError(step=self, field=field)

        dep_field = _load_attribute(
            context.infile,
            field.depends_on,
            force_scalar=True,
        )

        if dep_field:
            field._value = _load_attribute(
                context.infile,
                field.input_path,
                dtype=field.dtype,
                force_scalar=field.force_scalar,
                required=field._required,
            )
        # if dep_field is not present, we don't need to do anything because we
        # already set value on initialization

        if field._value is None:
            return

        next_step(context)


def _save_attribute(outfile: h5py.File, field: FieldTranslation) -> tuple[str, str]:
    outpath = field.output_path.split(".")
    path = outpath[0]
    name = outpath[1]
    if path not in outfile:
        msg = f"{path} not present. Cannot save attribute on nonexistent object"
        if field._required:
            raise TranslationError(msg)
        LOGGER.warning(msg)
        return path, name
    outfile[path].attrs[name] = field.value
    return path, name


@dataclass
class AttributeSaveStep(SaveStep):
    """Save the value to field.output_path"""

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field

        path, name = _save_attribute(context.outfile, field)
        LOGGER.debug(f"Saving attribute {name}={field.value} on {path}")

        if (
            context.options.num_partfiles > 1
            and field.output_path != "/Header.File_index"
        ):
            # only do this if we actually have more than one output file and this
            # isn't the file index
            # We skip the file index because it's already been written out
            for partfile_path in context.partfile_paths:
                with h5py.File(partfile_path, "a") as outfile:
                    _save_attribute(outfile, field)

        field.value = None
        next_step(context)


@dataclass
class SaveQuantityStep(TranslationStep):
    """Save the attribute value to a named quantity

    Attributes
    ----------
    name: str
        The name of the quantity to store in the pipeline context
    """

    _: KW_ONLY
    # need it before SaveStep because we delete field.value in SaveStep
    priority: int = SaveStep.priority - 1
    name: str

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, AttributeTranslation):
            raise TranslationClassError(step=self, field=field)
        context.quantities[self.name] = field.value
        next_step(context)


@dataclass
class SnapshotAttributeLoadStep(LoadStep):
    """Load attribute value from snapshot file information

    Attributes
    ----------
    attribute: Literal["timestamp"] | str
        Name of the file attribute to use (obtained through Path.stat).
        Note that `"timestamp"` will return an ISO8601 formatted version of `st_mtime`
    """

    _: KW_ONLY
    attribute: Literal["timestamp"] | str

    def _load_file_attributes(self, file_path) -> str:
        stat_result = Path(file_path).stat()
        if self.attribute == "timestamp":
            attribute = str(
                datetime.datetime.fromtimestamp(stat_result.st_mtime, datetime.UTC)
            )
        else:
            attribute = str(stat_result.__dict__.get(self.attribute, "unknown"))
        return attribute

    def __call__(self, context: TranslationContext, next_step: NextStep):
        infile = context.infile
        context.field.value = self._load_file_attributes(infile.filename)
        next_step(context)

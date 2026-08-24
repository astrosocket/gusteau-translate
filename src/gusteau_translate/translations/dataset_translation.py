import logging
import numbers
import typing
from dataclasses import KW_ONLY, InitVar, dataclass, field
from pathlib import Path
from typing import Annotated, Literal, cast

import h5py
import h5py._hl.filters
import numpy as np
from annotated_types import MaxLen, MultipleOf
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, NonNegativeInt, PositiveInt
from unyt.array import unyt_quantity
from unyt.unit_registry import UnitRegistry

from ..base_steps.base import LoadStep, SaveStep
from ..config import LOGGER_NAME
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

LOGGER = logging.getLogger(LOGGER_NAME)


def _validate_columns(columns: ArrayLike | None) -> int | NDArray[np.int_] | None:
    if columns is None:
        return None
    columns = np.array(columns).astype(np.int_)
    return int(columns) if not len(columns.shape) else columns


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


def _add_options_to_proplist(  # noqa: C901
    *,
    plist: h5py.h5p.PropDCID | None = None,
    shape,
    dtype,
    chunks,
    maxshape,
    external,
    fill_time=None,
    compression: compression_type | None = None,
    compression_opts: NonNegativeInt
    | tuple[
        Literal["ec", "nn"],
        Annotated[PositiveInt, MultipleOf(2)],
    ]
    | None = None,
    scaleoffset: NonNegativeInt | bool | None = None,
    shuffle: bool | None = None,
    fletcher32: bool | None = None,
) -> h5py.h5p.PropDCID:
    """Add options to property list

    Custom implementation of fill_dcpl to fill the filter pipeline of a proplist.
    Necessary due to h5py's inability to use both fletcher32 and scale_offset
    filters in the high-level API

    Undocumented and subject to change without warning
    """
    plist = h5py._hl.filters.fill_dcpl(
        # arguments we will pass to the original
        plist=plist or h5py.h5p.create(h5py.h5p.DATASET_CREATE),
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        maxshape=maxshape,
        external=external,
        fill_time=fill_time,
        # arguments we'll override manually
        compression=None,
        compression_opts=None,
        shuffle=None,
        fletcher32=None,
        scaleoffset=None,
    )

    if scaleoffset:
        if dtype.kind == "f":
            if scaleoffset is True:
                raise ValueError(
                    "Integer scale_offset_filter must be provided"
                    " for floating point types"
                )
            plist.set_scaleoffset(h5py.h5z.SO_FLOAT_DSCALE, scaleoffset)
        elif dtype.kind in ("u", "i"):
            scaleoffset = (
                h5py.h5z.SO_INT_MINBITS_DEFAULT if scaleoffset is True else scaleoffset
            )
            plist.set_scaleoffset(h5py.h5z.SO_INT, scaleoffset)
        else:
            raise TypeError(
                "Scale offset filter is only supported for integer/floating point types"
            )

    if shuffle:
        plist.set_shuffle()

    match compression:
        case "gzip":
            if isinstance(compression_opts, tuple):
                raise ValueError(
                    "Only integer compression options or None are supported for gzip"
                )
            gzip_level = (
                h5py._hl.filters.DEFAULT_GZIP
                if compression_opts is None
                else (compression_opts if compression_opts in range(10) else -1)
            )
            if gzip_level < 0:
                raise ValueError(
                    "Only integers within 0-9 are allowed"
                    f" (provided {compression_opts})"
                )
            plist.set_deflate(gzip_level)
        case "lzf":
            if compression_opts is not None:
                raise ValueError("LZF accepts no compression options")
            plist.set_filter(h5py.h5z.FILTER_LZF, h5py.h5z.FLAG_OPTIONAL)
        case "szip":
            compression_opts = (
                h5py._hl.filters.DEFAULT_SZIP
                if compression_opts is None
                else compression_opts
            )
            err_msg = (
                "SZIP options must be a 2-tuple ('ec'|'nn', even integer in (0, 32])"
            )
            if not isinstance(compression_opts, tuple):
                raise TypeError(err_msg)
            szmethod, szpix = compression_opts
            if szmethod not in ("ec", "nn") or not (0 < szpix <= 32 and szpix % 2 == 0):
                raise ValueError(err_msg)
            szmethod = (
                h5py.h5z.SZIP_EC_OPTION_MASK
                if szmethod == "ec"
                else h5py.h5z.SZIP_NN_OPTION_MASK
            )
            plist.set_filter(szmethod, szpix)
        case None:
            pass
        case _:
            # strictly disallow unknown filters
            if isinstance(compression, str):
                raise ValueError(f"Unknown compression filter: {compression}")
            compression = (
                (compression,) if not isinstance(compression, tuple) else compression
            )
            for comp in compression:
                if isinstance(comp, int):
                    if not h5py.h5z.filter_avail(comp):
                        raise ValueError(f"Unknown compression filter: {comp}")
                    if not isinstance(compression_opts, tuple):
                        raise ValueError(
                            f"compression_opts for unknown filter {comp}"
                            " must be a tuple of UINTs"
                        )
                    comp = FilterInfo(
                        filter_code=comp,
                        flags=h5py.h5z.FLAG_OPTIONAL,
                        aux=compression_opts,
                        name="unknown",
                    )
                plist.set_filter(comp.filter_code, comp.flags, comp.aux)

    if fletcher32:
        plist.set_fletcher32()

    return plist


@dataclass
class DatasetTranslation(FieldTranslation):
    _: KW_ONLY
    columns: int | list[int] | NDArray[np.int_] | None = None
    """
    Index or list of indices representing the column in the 2D dataset to load
    1D datasets or datasets that use all the columns should specify None
    """
    compression: compression_type | None = None
    """Compression used by this dataset"""
    compression_opts: (
        NonNegativeInt
        | tuple[
            Literal["ec", "nn"],
            Annotated[NonNegativeInt, MultipleOf(2)],
        ]
        | None
    ) = None
    """Compression options used by this dataset"""
    scale_offset_filter: NonNegativeInt | bool | None = None
    """Whether to apply the scale offset filter to this dataset
    Default is to use the source dataset's scale_offset value
    """
    shuffle_filter: bool | None = None
    """Whether to apply the shuffle filter to this dataset
    Default is to use the source dataset's shuffle_filter flag
    """
    fletcher32_filter: bool | None = None
    """Whether to apply the shuffle filter to this dataset
    Default is to use the source dataset's fletcher32_filter flag
    """
    chunks: tuple[int, ...] | bool | None = None
    """
    Chunk division used by dataset. 
    Default is to use the source dataset's chunks, accounting for self.columns.
    Set to `True` to have h5py compute them automatically
    """
    maxshape: tuple[PositiveInt, ...] | None = None
    """
    maxshape used by dataset.
    Default is to use the source dataset's maxshape
    """

    name_mapping: dict[str, str | Literal["[skip]"] | Literal["[ignore]"]] = field(
        default_factory=dict
    )
    """
    Mapping of names to outcomes. 
    Attribute names that are not included are mapped identically
    Alternatively, map to a valid GUSTEAU attribute name (only containing 
    A-Za-z0-9_) to change the attribute name or "[skip]" to skip translating
    the attribute or "[ignore]" to skip processing the attribute.
    """

    def __post_init__(
        self,
    ):
        self.columns = _validate_columns(self.columns)
        if self.columns is not None and "[" not in self.input_description:
            self.input_description += f"[:, {self.columns}]"
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
    description: str | None = None
    units: str | None = None
    is_comoving: bool | None = None
    conversion_factor_cgs: typing.SupportsFloat | None = None
    conversion_factor_physcgs: typing.SupportsFloat | None = None
    phys_CGS_expression: str | None = None
    lossy_compression: str | None = None
    U_I: typing.SupportsFloat | None = None
    U_L: typing.SupportsFloat | None = None
    U_M: typing.SupportsFloat | None = None
    U_T: typing.SupportsFloat | None = None
    U_t: typing.SupportsFloat | None = None
    a_scale: typing.SupportsFloat | None = None
    h_scale: typing.SupportsFloat | None = None

    def __post_init__(
        self,
    ):
        super().__post_init__()
        # Shouldn't use these if dataset already has units...
        self.pipeline.append(MakeParticleAttributesStep())


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

    def __post_init__(self, like_dataset: str | None):
        super().__post_init__()
        old_input = self.input_path
        self.input_path = like_dataset if like_dataset else self.input_path
        if self.input_description == f"From {old_input}":
            self.input_description = (
                f"Constructed a la {like_dataset} with constant {self.constant}"
            )

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
        if field.input_path not in context.infile:
            if field._required:
                raise TranslationError(
                    f"Required dataset {field.input_path} not found!"
                )
            LOGGER.debug(
                f"Optional dataset {field.input_path} not present. Nothing to translate"
            )
            return
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
        if field.compression == "unknown":
            # need to read the filter ids/filter_names field
            field.compression = tuple(
                i
                for i, name in zip(indata.filter_ids, indata.filter_names, strict=True)
                if name not in ["scaleoffset", "shuffle", "deflate", "fletcher32"]
            )
        field.compression_opts = (
            cast(type(field.compression_opts), indata.compression_opts)
            if field.compression_opts is None
            else field.compression_opts
        )
        field.scale_offset_filter = (
            cast(int | None, indata.scaleoffset)
            if field.scale_offset_filter is None
            else field.scale_offset_filter
        )
        field.shuffle_filter = (
            cast(bool | None, indata.shuffle)
            if field.shuffle_filter is None
            else field.shuffle_filter
        )
        field.fletcher32_filter = (
            cast(bool | None, indata.fletcher32)
            if field.fletcher32_filter is None
            else field.fletcher32_filter
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

        # Special case: the maxshape property always exists, but if we pass it
        # to create_dataset, the new dataset will automatically get chunked
        # layout. So we copy it only if it is different from shape.
        field.maxshape = (
            indata.maxshape
            if field.maxshape is None and indata.maxshape != indata.shape
            else field.maxshape
        )
        next_step(context)


@dataclass
class DatasetSaveStep(SaveStep):
    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, DatasetTranslation):
            raise TranslationClassError(step=self, field=field)
        LOGGER.debug(f"Saving dataset {field.output_path}")
        dcpl = _add_options_to_proplist(
            shape=field.value.shape,
            dtype=field.value.dtype,
            chunks=field.chunks,
            maxshape=field.maxshape,
            external=None,
            compression=field.compression,
            compression_opts=field.compression_opts,
            scaleoffset=field.scale_offset_filter,
            shuffle=field.shuffle_filter,
            fletcher32=field.fletcher32_filter,
        )
        context.outfile.create_dataset(
            name=field.output_path, data=field.value, dcpl=dcpl
        )

        # add source description attribute
        context.outfile[field.output_path].attrs["Source_Description"] = (
            field.input_description
        )
        field.value = None
        next_step(context)


@dataclass
class MakeParticleAttributesStep(LoadStep):
    """Load and apply cosmological and unit information"""

    priority: int = SaveStep.priority + 1
    write_out: bool = False
    _: KW_ONLY

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, ParticleDatasetTranslation):
            raise TranslationClassError(step=self, field=field)

        if field.description is None:
            raise TranslationError(f"{field.name} is missing a description!")

        unames = [f"U_{u}" for u in "ILMTt"]

        # we could do something clever here with get/setattr, but this is fine
        field.U_I = 0 if field.U_I is None else field.U_I
        field.U_L = 0 if field.U_L is None else field.U_L
        field.U_M = 0 if field.U_M is None else field.U_M
        field.U_T = 0 if field.U_T is None else field.U_T
        field.U_t = 0 if field.U_t is None else field.U_t
        field.a_scale = 0 if field.a_scale is None else field.a_scale
        field.h_scale = 0 if field.h_scale is None else field.h_scale

        terms = [*unames, "a_scale", "h_scale"]
        a = float(context.quantities.get("a", 1))
        h = float(context.quantities.get("h", 1))
        exponents = {term: float(getattr(field, term)) for term in terms}

        registry = cast(
            UnitRegistry,
            context.quantities.get("registry", UnitRegistry(unit_system="cgs")),
        )

        conversion_factor_cgs = unyt_quantity(1, "dimensionless", registry=registry)
        for uname in unames:
            conversion_factor_cgs *= (
                unyt_quantity(1, units=uname, registry=registry) ** exponents[uname]
            )
        conversion_factor_cgs /= unyt_quantity(1, "dimensionless", registry=registry)

        conversion_factor_physcgs = cast(
            unyt_quantity,
            conversion_factor_cgs
            * a ** exponents["a_scale"]
            * h ** exponents["h_scale"],
        )

        field.conversion_factor_cgs = (
            conversion_factor_cgs.v
            if field.conversion_factor_cgs is None
            else field.conversion_factor_cgs
        )

        field.conversion_factor_physcgs = (
            conversion_factor_physcgs.v
            if field.conversion_factor_physcgs is None
            else field.conversion_factor_physcgs
        )

        names: dict[str, str] = {uname: uname for uname in unames}
        names["a_scale"] = "a"
        names["h_scale"] = "h"

        unit_str = str(conversion_factor_physcgs.units.get_cgs_equivalent())

        if field.phys_CGS_expression is None:
            field.phys_CGS_expression = " ".join(
                f"{names[term]}^{exponents[term]}" for term in terms
            )

            field.phys_CGS_expression += f" [{unit_str}]"

        field.units = unit_str if field.units is None else field.units
        field.units = "" if field.units == "-" else field.units

        field.lossy_compression = (
            "" if field.lossy_compression is None else field.lossy_compression
        )

        attributes = {
            "Units": field.units,
            "Conversion_factor_CGS": field.conversion_factor_cgs,
            "Conversion_factor_PhysCGS": field.conversion_factor_physcgs,
            "Description": field.description,
            "Phys_CGS_expression": field.phys_CGS_expression,
            "Lossy_compression": field.lossy_compression,
            "U_I_exponent": field.U_I,
            "U_L_exponent": field.U_L,
            "U_M_exponent": field.U_M,
            "U_T_exponent": field.U_T,
            "U_t_exponent": field.U_t,
            "a_scale_exponent": field.a_scale,
            "h_scale_exponent": field.h_scale,
        }
        translation = context.translation
        for name, attr in attributes.items():
            translation.add_field(
                translation._default_translation_factory(
                    obj="ConstantAttributeTranslation",
                    path="",
                    input_path="",
                    output_path=f"{field.output_path}.{name}",
                    constant=attr,
                    _explicit=True,
                )
            )

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

    write_out: bool = False

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, DatasetTranslation):
            raise TranslationClassError(step=self, field=field)
        LOGGER.debug(f"Moving {field.input_path} attributes to {field.output_path}")

        dataset = cast(h5py.Dataset, context.infile[field.input_path])

        translation = context.translation

        def visit_attr(path: str, name: str):
            full_name = f"{path}.{name}"
            mapping = field.name_mapping.get(name)
            if mapping == "[ignore]":
                LOGGER.debug(f"Ignoring {full_name}")
                return
            LOGGER.debug(f"Visiting {full_name}")
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

from dataclasses import KW_ONLY, InitVar, dataclass, field
from typing import LiteralString

import h5py
import numpy as np

from ..pipeline import NextStep
from ..translation_base import (
    TranslationContext,
    TranslationError,
    TranslationStep,
)


@dataclass
class MathStep(TranslationStep):
    """
    Perform basic mathematical expressions on the field value

    This step uses `eval` to evaluate a 1-variable expression, i.e., something
    like `x**2` or `sin(x)`

    !!! warning
        We perform no sanitization of the inputs. Use at own risk!

    Attributes
    ----------
    equation: LiteralString
        A string literal defining the mathematical expression.
        Can use any numpy expression and the value `x`
    """

    _: KW_ONLY
    equation: LiteralString

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        # expr = sympify(self.equation)
        # x = symbols("x")
        # field.value = np.array(expr.subs(x, unyt_array(field.value).v)).astype(
        #     field.dtype
        # )
        x = field.value
        field.value = np.asarray(eval(self.equation)).astype(field.dtype)
        next_step(context)


def _load_external_field(
    external_path: str,
    external_columns: np.ndarray | None,
    infile: h5py.File,
    *,
    skip_data: bool = False,
) -> np.ndarray:
    if "." in external_path:
        # read as attribute
        path, name = external_path.split(".")
        external = infile[path].attrs[name]
    else:
        external_ds = infile[external_path]
        shape = (1 if skip_data else external_ds.shape[0], *external_ds.shape[1:])
        external = np.empty(shape, dtype=external_ds.dtype)
        slice_ = np.s_[: shape[0], ...]
        external_ds.read_direct(external, source_sel=slice_)

    return (
        external
        if external_columns is None
        else np.atleast_2d(external)[:, external_columns]
    )


@dataclass
class Math2Step(MathStep):
    """
    Perform basic mathematical expressions on the field value

    This step uses `eval` to evaluate an 2+-variable expression, i.e., something
    like `x**2 * y` or `sin(x) / y`. The external parameter(s) map to a field
    (dataset or attribute) in the input file, the external_columns specifies
    columns to pick out.

    !!! warning
        We perform no sanitization of the inputs. Use at own risk!

    Attributes
    ----------
    external: str | list[str]
        The path to the external variable (`y`) in standard GUSTEAU field notation.
        If multiple external variables are given, they correspond to additional
        `y_i` terms in the expression, e.g. `x**2 * y * y1`
    external_columns: NDArray | list[NDArray | None], optional
        The column(s) to extract.
        For 2D arrays, this is equivalent to `y=external[:, external_columns]`.
        For 1D arrays, this is equivalent to `y=external[external_columns]`.

        If multiple external variables are present, this parameter is evaluated
        in one of two ways. If given as a single NDArray, it is equivalent to
        `y=external[0][:, external_columns], y1=external[1][:, external_columns]`.
        If given as a list of NDArrays, it is equivalent to
        `y=external[0][:, external_columns[0]],
        y1=external[1][:, external_columns[1]]`.
        `external_columns[i] is None` is equivalent to all columns for `external[i]`.
    force_scalar: bool, optional
        Attempt to make output a scalar quantity instead of an array
    """

    _: KW_ONLY
    external: InitVar[str | list[str]]
    external_columns: InitVar[np.ndarray | list[np.ndarray | None] | None] = None
    force_scalar: bool = False

    _external: list[str] = field(init=False)
    _columns: list[np.ndarray | None] = field(init=False)

    def __post_init__(
        self,
        external: str | list[str],
        external_columns: np.ndarray | list[np.ndarray | None] | None,
    ):
        self._external = [external] if isinstance(external, str) else external

        if external_columns:
            self._columns = (
                external_columns
                if isinstance(external_columns, list)
                else [external_columns]
            )
            self._columns = [
                np.array(ec).squeeze() if ec else ec for ec in self._columns
            ]
        else:
            self._columns = [None] * len(self._external)

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        x = field.value
        locals_ = locals()
        for i, (external, columns) in enumerate(
            zip(self._external, self._columns, strict=True)
        ):
            locals_[f"y{i if i else ''}"] = _load_external_field(
                external,
                columns,
                context.infile,
                skip_data=context.options.skip_data,
            )
        field.value = np.array(eval(self.equation, locals=locals_)).astype(field.dtype)
        if self.force_scalar:
            if len(field.value) > 1:
                raise TranslationError("Attempting to force an array into a scalar!")
            field.value = field.value[0]
        next_step(context)


@dataclass
class ReorderColumnStep(TranslationStep):
    """
    Reorder columns in a 1 or 2D array

    Attributes
    ----------
    new_width: int
        New length(width) in 1(2)D array
    col_indices: list[int]
        Column index in original array to be used for this column.
        Use -1 to fill with the fill_value
    fill_value: number | str
        Fill value to use for the unmapped columns
    """

    _: KW_ONLY
    new_width: int
    col_indices: list[int]
    fill_value: float | int | str = 0

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        old_value = np.atleast_1d(field.value)
        shape = old_value.shape
        shape = (self.new_width,) if len(shape) <= 1 else (shape[0], self.new_width)
        value = np.full_like(old_value, shape=shape, fill_value=self.fill_value)
        for i, col in enumerate(self.col_indices):
            if col < 0:
                continue
            if len(shape) == 1:
                value[i] = old_value[col]
            else:
                value[:, i] = old_value[:, col]
        field.value = value
        next_step(context)

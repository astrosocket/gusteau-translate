from dataclasses import KW_ONLY, dataclass
from typing import LiteralString

import h5py
import numpy as np

from ..pipeline import NextStep
from ..translation_base import (
    TranslationContext,
    TranslationStep,
)


@dataclass
class MathStep(TranslationStep):
    """
    Perform basic mathematical expressions on the field value

    This step uses sympy to evaluate a 1-variable expression, i.e., something
    like `x**2` or `sin(x)`

    **WARNING**: this uses sympy's `sympify` function which contains a bare
    `eval` and we perform no sanitization of the inputs. Use at own risk!
    """

    _: KW_ONLY
    equation: LiteralString
    """A string literal defining the mathematical expression"""

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

    This step uses sympy to evaluate a 2-variable expression, i.e., something
    like `x**2 * y` or `sin(x) / y`. The external parameter maps to a field
    (dataset or attribute) in the input file, the external_columns specifies
    columns to pick out.

    **WARNING**: this uses sympy's `sympify` function which contains a bare
    `eval` and we perform no sanitization of the inputs. Use at own risk!
    """

    _: KW_ONLY
    external: str
    """The path to the external variable in standard GUSTEAU field notation"""
    external_columns: np.ndarray | None = None
    """The column(s) to extract. 
    For 2D arrays, this is equivalent to y=external[:, external_columns].
    For 1D arrays, this is equivalent to y=external[external_columns].
    """

    def __post_init__(self):
        self.external_columns = (
            None
            if self.external_columns is None
            else np.array(self.external_columns).squeeze()
        )

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        x = field.value
        y = _load_external_field(
            self.external,
            self.external_columns,
            context.infile,
            skip_data=context.options.skip_data,
        )
        field.value = np.array(eval(self.equation)).astype(field.dtype)
        next_step(context)


@dataclass
class ReorderColumnStep(TranslationStep):
    """
    Reorder columns in a 1 or 2D array
    """

    _: KW_ONLY
    new_width: int
    """New length(width) in 1(2)D array"""
    col_indices: list[int]
    """
    Column index in original array to be used for this column.
    Use -1 to fill with the fill_value
    """
    fill_value = 0
    """Fill value to use for unmapped columns"""

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

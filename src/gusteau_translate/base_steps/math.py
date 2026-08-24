from dataclasses import KW_ONLY, dataclass
from typing import LiteralString

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

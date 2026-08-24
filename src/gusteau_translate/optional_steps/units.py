import re
from dataclasses import dataclass

import numpy as np
from unyt.array import unyt_quantity

from ..pipeline import NextStep
from ..translation_base import (
    TranslationClassError,
    TranslationContext,
    TranslationError,
    TranslationStep,
)
from ..translations.dataset_translation import ParticleDatasetTranslation


@dataclass
class ParseUnitsStep(TranslationStep):
    """Parse the units field to determine U_X exponents

    Note that we assume the scale factor ($a$), and little-h ($h$) are
    specified separately (as `a_scale` and `h_scale`) and that 'a' and 'h' terms
    in the units field should be parsed as units/constants (e.g. Planck's
    constant)
    """

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, ParticleDatasetTranslation):
            raise TranslationClassError(step=self, field=field)

        if not field.units:
            # Nothing to do if units is empty
            next_step(context)
            return

        # convert pi if it's on it's own
        units = re.sub(r"\bpi\b", str(np.pi), field.units)

        registry = context.quantities["registry"]

        test_quantity = unyt_quantity(1, units, registry=registry)

        powers_dict = test_quantity.in_base(
            "snapshot"
        ).units.dimensions.as_powers_dict()

        terms = {
            "mass": "U_M",
            "length": "U_L",
            "time": "U_t",
            "temperature": "U_T",
            "current_mks": "U_I",
        }
        for dimension, a in powers_dict.items():
            # assume dimension is in the form (dimension)
            dimension = str(dimension)[1:-1]
            if dimension in terms:
                setattr(field, terms[dimension], float(a))
            else:
                raise TranslationError(f"Don't understand dimension {dimension}!")

        next_step(context)

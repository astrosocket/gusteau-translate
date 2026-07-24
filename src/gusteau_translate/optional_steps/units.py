import logging
import re
from dataclasses import dataclass
from typing import cast

import h5py
import unyt as u
from unyt.array import unyt_quantity

from gusteau_translate.translations import ConstantAttributeTranslation

from ..base_steps.base import LoadStep
from ..pipeline import NextStep
from ..translation_base import (
    TranslationClassError,
    TranslationContext,
    TranslationPipeline,
    TranslationStep,
)
from ..translations.dataset_translation import ParticleDatasetTranslation

LOGGER = logging.getLogger("gusteau_translate")


def _create_unit_registry(context: TranslationContext) -> dict[str, unyt_quantity]:
    unit_base = {
        "I": u.ampere,
        "L": u.cm,
        "M": u.g,
        "T": u.kelvin,
        "t": u.second,
    }
    registry = {}
    for unit_type, unit in unit_base.items():
        unit_name = f"U_{unit_type}"
        registry[unit_name] = float(context.quantities.get(unit_name, 1)) * unit
    return registry


@dataclass
class LoadSwiftLikeUnitsStep(TranslationStep):
    """
    Load units from a Swift-like setup

    A Swift-like setup has the following attributes:
        - Conversion factor to CGS (not including cosmological corrections)
        - Conversion factor to physical CGS (including cosmological corrections)
        - Expression for physical CGS units
        - U_I exponent
        - U_L exponent
        - U_M exponent
        - U_T exponent
        - U_t exponent
        - a-scale exponent
        - h-scale exponent
    While we _technically_ only care about the exponent terms, the other terms
    are just the equivalent name mapping, so we might as well use them as is
    """

    priority: int = LoadStep.priority + 1

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, ParticleDatasetTranslation):
            raise TranslationClassError(step=self, field=field)
        LOGGER.debug("Loading Swift-style units")

        attribute_mapping = {
            "Conversion factor to CGS (not including cosmological corrections)": "ConversionFactorCGS",  # noqa: E501
            "Conversion factor to physical CGS (including cosmological corrections)": "ConversionFactor_PhysCGS",  # noqa: E501
            "Expression for physical CGS units": "Expression_for_physical_CGS",
            "U_I exponent": "U_I_exponent",
            "U_L exponent": "U_L_exponent",
            "U_M exponent": "U_M_exponent",
            "U_T exponent": "U_T_exponent",
            "U_t exponent": "U_t_exponent",
            "a-scale exponent": "a_scale_exponent",
            "h-scale exponent": "h_scale_exponent",
        }
        # add to name_mapping
        for old, new in attribute_mapping.items():
            field.name_mapping[old] = new
        # we likely don't have the "Units" attribute, so generate it if necessary
        if "Units" not in cast(h5py.Dataset, context.infile[field.input_path]).attrs:
            expression = str(
                context.infile[field.input_path].attrs[
                    "Expression for physical CGS units"
                ]
            )
            expression = re.search(r"\[(.+)\]", expression)
            expression = expression.group(1) if expression else ""
            context.translation.add_field(
                ConstantAttributeTranslation(
                    output_path=f"{field.output_path}.Units",
                    constant=expression,
                    pipeline=TranslationPipeline(),
                )
            )
        next_step(context)

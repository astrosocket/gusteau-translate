import logging
import re
from dataclasses import dataclass

import unyt.dimensions as ud
from unyt.array import unyt_quantity
from unyt.unit_registry import UnitRegistry

from ..base_steps.base import LoadStep
from ..pipeline import NextStep
from ..translation_base import (
    TranslationClassError,
    TranslationContext,
    TranslationStep,
)
from ..translations.attribute_translation import process_attribute
from ..translations.dataset_translation import ParticleDatasetTranslation

LOGGER = logging.getLogger("gusteau_translate")


@dataclass
class CreateUnitRegistryStep(TranslationStep):
    """Create a unit registry and add it to the context"""

    def __call__(self, context: TranslationContext, next_step: NextStep):
        quantities = context.quantities
        registry = UnitRegistry(unit_system="cgs")
        dimensions = {
            "U_I": (ud.current_mks, "A", "A"),
            "U_L": (ud.length, "cm", "m"),
            "U_M": (ud.mass, "g", "kg"),
            "U_T": (ud.temperature, "K", "K"),
            "U_t": (ud.time, "s", "s"),
        }
        for uname, (dimension, cgs_unit, mks_unit) in dimensions.items():
            mks = unyt_quantity(quantities.get(uname), cgs_unit).to(mks_unit)
            registry.add(uname, base_value=float(mks.v), dimensions=dimension)
        LOGGER.debug(registry.lut)
        context.quantities["registry"] = registry
        next_step(context)


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
            "conversion_factor_cgs": (
                "Conversion factor to CGS (not including cosmological corrections)",
                "ConversionFactorCGS",
            ),
            "conversion_factor_physcgs": (
                "Conversion factor to physical CGS"
                " (including cosmological corrections)",
                "ConversionFactor_PhysCGS",
            ),
            "description": ("Description", "Description"),
            "expression_for_physical_CGS": (
                "Expression for physical CGS units",
                "Expression_for_physical_CGS",
            ),
            "u_I": ("U_I exponent", "U_I_exponent"),
            "u_L": ("U_L exponent", "U_L_exponent"),
            "u_M": ("U_M exponent", "U_M_exponent"),
            "u_T": ("U_T exponent", "U_T_exponent"),
            "u_t": ("U_t exponent", "U_t_exponent"),
            "a_scale": ("a-scale exponent", "a_scale_exponent"),
            "h_scale": ("h-scale exponent", "h_scale_exponent"),
            "lossy_compression": ("Lossy compression filter", "LossyCompression"),
        }

        infile = context.infile
        attrs = infile[field.input_path].attrs
        for param, (swift_attr_name, gus_attr_name) in attribute_mapping.items():
            field_value = getattr(field, param, None)
            attr = attrs.get(swift_attr_name)
            attr = attr if attr is None else process_attribute(attr, force_scalar=True)
            setattr(field, param, attr if field_value is None else field_value)
            field.name_mapping[swift_attr_name] = gus_attr_name

        expression = (
            ""
            if field.expression_for_physical_CGS is None
            else field.expression_for_physical_CGS
        )
        expression = re.search(r"\[(.+)\]", expression)
        field.units = expression.group(1) if expression else ""
        field.units = "" if field.units == "-" else field.units

        next_step(context)

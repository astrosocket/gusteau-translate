import logging
from dataclasses import dataclass

import unyt.dimensions as ud
from unyt.array import unyt_quantity
from unyt.unit_registry import UnitRegistry

from ..pipeline import NextStep
from ..translation_base import TranslationContext, TranslationStep

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

import logging
from dataclasses import dataclass

from ..base_steps.base import LoadStep
from ..config import LOGGER_NAME
from ..pipeline import NextStep
from ..translation_base import (
    TranslationClassError,
    TranslationContext,
    TranslationStep,
)
from ..translations.attribute_translation import process_attribute
from ..translations.dataset_translation import ParticleDatasetTranslation

LOGGER = logging.getLogger(LOGGER_NAME)


@dataclass
class LoadSwiftLikeUnitsStep(TranslationStep):
    """
    Load units from a Swift-like setup

    A Swift-like setup has the following attributes on every particle dataset:

      - `Conversion factor to CGS (not including cosmological corrections)`
      - `Conversion factor to physical CGS (including cosmological corrections)`
      - `Expression for physical CGS units`
      - `U_I exponent`
      - `U_L exponent`
      - `U_M exponent`
      - `U_T exponent`
      - `U_t exponent`
      - `a-scale exponent`
      - `h-scale exponent`

    While we _technically_ only care about the exponent terms, the other terms
    are just the equivalent name mapping, so we might as well use them as is.
    """

    priority: int = LoadStep.priority + 1

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, ParticleDatasetTranslation):
            raise TranslationClassError(step=self, field=field)
        LOGGER.debug("Loading Swift-style units")

        attribute_mapping = {
            "conversion_factor_cgs": (
                "Conversion factor to CGS (not including cosmological corrections)"
            ),
            "conversion_factor_physcgs": (
                "Conversion factor to physical CGS (including cosmological corrections)"
            ),
            "description": ("Description"),
            "phys_CGS_expression": ("Expression for physical CGS units"),
            "U_I": ("U_I exponent"),
            "U_L": ("U_L exponent"),
            "U_M": ("U_M exponent"),
            "U_T": ("U_T exponent"),
            "U_t": ("U_t exponent"),
            "a_scale": ("a-scale exponent"),
            "h_scale": ("h-scale exponent"),
            "lossy_compression": ("Lossy compression filter"),
        }

        infile = context.infile
        attrs = infile[field.input_path].attrs
        translation = context.translation
        for param, swift_attr_name in attribute_mapping.items():
            field_value = getattr(field, param, None)
            attr = attrs.get(swift_attr_name)
            attr = attr if attr is None else process_attribute(attr, force_scalar=True)
            setattr(field, param, attr if field_value is None else field_value)
            field.name_mapping[swift_attr_name] = "[skip]"
            fields = translation.get_fields_from_input(swift_attr_name)
            for attr_field in fields:
                if not attr_field._explicit:
                    translation.remove_field(attr_field)
                    LOGGER.debug(
                        f"Removing dynamically generated field {attr_field.name}"
                    )
                else:
                    LOGGER.debug(f"Explicit translation for {attr_field.name}")

        if field.units is None:
            expression = ""
            terms = [f"U_{x}" for x in "ILMTt"]
            for term in terms:
                p = getattr(field, term, 0)
                expression += (
                    f"{' * ' if expression else ''}{term}{'' if p == 1 else f'**{p}'}"
                    if p
                    else ""
                )

            field.units = expression

        next_step(context)

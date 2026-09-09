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
from ..translations.dataset_translation import ParticleDatasetTranslation

LOGGER = logging.getLogger(LOGGER_NAME)


@dataclass
class HandleIllustrisUnitsStep(TranslationStep):
    """
    Handle units from IllustrisTNG-like setup

    IllustrisTNG has the following attributes on **some** particle datasets:

      - `a_scaling`
      - `h_scaling`
      - `length_scaling`
      - `mass_scaling`
      - `to_cgs`
      - `velocity_scaling`

    Which datasets have these seems to be arbitrary; sometimes it's datasets
    with units, but not always.

    Plus, there does not appear to be exact definitions for these terms, and,
    while we can guess, it seems safer to have them manually entered.
    Thus, these attributes will be **ignored** when encountered.
    """

    priority: int = LoadStep.priority + 1

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, ParticleDatasetTranslation):
            raise TranslationClassError(step=self, field=field)
        LOGGER.debug("Ignoring IllustrisTNG-style units")

        skippable_attributes = [
            "a_scaling",
            "h_scaling",
            "length_scaling",
            "mass_scaling",
            "to_cgs",
            "velocity_scaling",
        ]

        for attr_name in skippable_attributes:
            field.name_mapping[attr_name] = "[skip]"

        next_step(context)

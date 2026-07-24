import logging
from dataclasses import KW_ONLY, dataclass

import h5py

from gusteau_translate.pipeline import NextStep
from gusteau_translate.translation_base import (
    TranslationContext,
    TranslationStep,
)

LOGGER = logging.getLogger("gusteau_translate")


@dataclass
class LoadStep(TranslationStep):
    priority: int = -100
    write_out: bool = False


@dataclass
class SaveStep(TranslationStep):
    priority: int = 100
    write_out: bool = False


@dataclass
class SkipStep(TranslationStep):
    def __call__(self, context: TranslationContext, next_step: NextStep):
        LOGGER.debug(
            f"Skipping remainder of pipeline for field {context.field.input_path}"
        )
        return


@dataclass
class LinkStep(TranslationStep):
    priority: int = 200
    _: KW_ONLY
    path: str

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        outfile = context.outfile
        outfile[self.path] = h5py.SoftLink(field.output_path)
        next_step(context)

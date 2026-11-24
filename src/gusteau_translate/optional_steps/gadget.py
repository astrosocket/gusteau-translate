import logging
from dataclasses import KW_ONLY, dataclass
from functools import cache

import h5py
import numpy as np
from numpy.typing import ArrayLike

from ..base_steps.math import _load_external_field
from ..config import DDEBUG, LOGGER_NAME
from ..pipeline import NextStep
from ..translation_base import (
    TranslationClassError,
    TranslationContext,
    TranslationError,
    TranslationStep,
)
from ..translations import AttributeTranslation

LOGGER = logging.getLogger(LOGGER_NAME)


@cache
def _get_gadget_particle_number(infile: h5py.File) -> np.ndarray:
    """Compute the total particle number for the 6 Gadget-like particle types"""
    numpart_total = _load_external_field(
        "/Header.NumPart_Total",
        None,
        infile,
    )
    numpart_total_hw = _load_external_field(
        "/Header.NumPart_Total_HighWord", None, infile
    )

    gadget_particle_number = np.uint64(numpart_total) + (numpart_total_hw << 32)
    LOGGER.log(DDEBUG, f"Gadget particle numbers: {gadget_particle_number}")
    return gadget_particle_number


@dataclass
class MaskParticlesStep(TranslationStep):
    """Mask which particles are present using the NumPart_Total* fields

    Useful for fields like Particle_names, Particle_numbers, etc. Effectively a
    pre-provided version of a Math2Step.
    """

    _: KW_ONLY
    ignore: ArrayLike | None = None
    """
    Optional indices into the NumPart_Total* fields to treat as 0.
    E.g. Given NumPart_Total = [10, 11, 12, 13, 14, 15] setting ignore=4 is
    equivalent to NumPart_Total = [10, 11, 12, 13, 0, 15].
    Multiple indices can be provided at the same time.
    Values must be valid integer indices or an array of booleans
    """

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, AttributeTranslation):
            raise TranslationClassError(step=self, field=field)

        try:
            len(field.value)
        except AttributeError:
            raise TranslationError(
                f"Field {field.name} should be an enumerable object!"
                f" (has type {type(field.value)})"
            ) from None

        gadget_particle_number = _get_gadget_particle_number(context.infile)
        if self.ignore:
            gadget_particle_number[np.array(self.ignore)] = 0

        if len(field.value) != len(gadget_particle_number):
            raise TranslationError(
                f"Field {field.name} should have {len(gadget_particle_number)} values"
                f" (has {len(field.value)})"
            )

        if isinstance(field.value, list):
            field.value = [
                f
                for f, t in zip(field.value, gadget_particle_number > 0, strict=True)
                if t
            ]
        else:
            field.value = field.value[gadget_particle_number > 0]

        next_step(context)


@dataclass
class MapParticlesStep(TranslationStep):
    """Map which particles are present using the NumPart_Total* fields

    Specifically useful for Part_type_mapping. The field value should be a
    6-element array mapping the particle names found in Particle_names to the 6
    gadget particle types.
    """

    _: KW_ONLY
    ignore: ArrayLike | None = None
    """
    Optional indices into the NumPart_Total* fields to treat as 0.
    E.g. Given NumPart_Total = [10, 11, 12, 13, 14, 15] setting ignore=4 is
    equivalent to NumPart_Total = [10, 11, 12, 13, 0, 15].
    Multiple indices can be provided at the same time.
    Values must be valid integer indices or an array of booleans
    """

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, AttributeTranslation):
            raise TranslationClassError(step=self, field=field)

        try:
            len(field.value)
        except AttributeError:
            raise TranslationError(
                f"Field {field.name} should be an enumerable object!"
                f" (has type {type(field.value)})"
            ) from None

        gadget_particle_number = _get_gadget_particle_number(context.infile)
        if self.ignore:
            gadget_particle_number[np.array(self.ignore)] = 0

        if len(field.value) != 6:
            raise TranslationError(
                f"Field {field.name} should have 6 values matching the original"
                f" gadget species (has {len(field.value)})"
            )

        is_present = gadget_particle_number > 0
        value = field.value
        masked_indices = np.arange(len(is_present))[is_present]
        new_value = -1 * np.ones_like(value)
        for i, m_orig in enumerate(value):
            if m_orig < 0:
                continue
            m_new = np.where(masked_indices == m_orig)[0]
            m_new = m_new[0] if len(m_new) else -1
            new_value[i] = m_new
        field.value = new_value

        next_step(context)

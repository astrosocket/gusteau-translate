import datetime
import logging
import time
from dataclasses import KW_ONLY, dataclass

import h5py

from ..config import LOGGER_NAME
from ..pipeline import NextStep
from ..translation_base import (
    TranslationContext,
    TranslationError,
    TranslationStep,
)

LOGGER = logging.getLogger(LOGGER_NAME)


@dataclass
class LoadStep(TranslationStep):
    """Abstract step to load something"""

    priority: int = -100
    write_out: bool = False


@dataclass
class SaveStep(TranslationStep):
    """Abstract step to save something"""

    priority: int = 100
    write_out: bool = False


@dataclass
class SkipStep(TranslationStep):
    """Step to skip remainder of pipeline"""

    def __call__(self, context: TranslationContext, next_step: NextStep):
        LOGGER.debug(
            f"Skipping remainder of pipeline for field {context.field.input_path}"
        )
        return


@dataclass
class LinkStep(TranslationStep):
    """Soft link this field to a path

    Essentially, `path` will be linked to `field.output_path`.

    Attributes
    ----------
    path: str
        Link name of the soft link
    """

    priority: int = 200
    _: KW_ONLY
    path: str

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        outfile = context.outfile
        outfile[self.path] = h5py.SoftLink(field.output_path)
        next_step(context)


@dataclass
class ConvertDateStep(TranslationStep):
    """Convert date/timestamp field values to ISO8601 using time/datetime

    Attributes
    ----------
    format: str
        Format string to pass to time.strptime, optional.
        If None, will attempt to parse the field value as ISO8601
    """

    _: KW_ONLY
    format: str | None = None

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field.value, str):
            raise TranslationError(
                f"Can only convert string inputs, given {type(field.value)}"
            )
        if self.format is None:
            when = datetime.datetime.fromisoformat(field.value)
        else:
            stime = time.strptime(field.value, self.format)
            tz = datetime.timezone(
                datetime.timedelta(0 if stime.tm_gmtoff is None else stime.tm_gmtoff),
                stime.tm_zone,
            )
            when = datetime.datetime(
                year=stime.tm_year,
                month=stime.tm_mon,
                day=stime.tm_mday,
                hour=stime.tm_hour,
                minute=stime.tm_min,
                second=stime.tm_sec,
                tzinfo=tz,
            )
        field.value = when.isoformat()

        next_step(context)

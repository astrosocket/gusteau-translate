from pathlib import Path

__lazy_modules__ = ["packingcubes"]

import logging
from dataclasses import dataclass

import numpy as np
import packingcubes

from ..config import LOGGER_NAME
from ..pipeline import NextStep
from ..translation_base import (
    TranslationClassError,
    TranslationContext,
    TranslationStep,
)
from ..translations import DatasetTranslation

LOGGER = logging.getLogger(LOGGER_NAME)


def particle_group_from_output_path(output_path: str) -> str:
    return output_path[1:].split("/")[0]


@dataclass
class PackCubesStep(TranslationStep):
    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, DatasetTranslation):
            raise TranslationClassError(step=self, field=field)

        particle_group = particle_group_from_output_path(field.output_path)
        LOGGER.debug(f"Doing initial cubing on {particle_group}")

        bounding_box = context.quantities["BoundingBox"]
        dataset = packingcubes.InMemory(
            positions=field.value, bounding_box=bounding_box
        )

        cubes = packingcubes.Cubes(dataset)

        cubes.save(
            dataset=context.outfile, particle_type=particle_group, force_overwrite=True
        )

        index_file_name = Path(context.tempdir) / f"{particle_group}.index"
        LOGGER.debug(f"Dumping index to {index_file_name}")
        dataset.index.tofile(index_file_name)

        # remove the sort datastep since we've already sorted the Coordinates
        field.pipeline.remove("SortDatasetStep")

        next_step(context)


@dataclass
class SortDatasetStep(TranslationStep):
    priority: int = PackCubesStep.priority + 10

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, DatasetTranslation):
            raise TranslationClassError(step=self, field=field)
        particle_group = particle_group_from_output_path(field.output_path)
        LOGGER.debug(
            f"Sorting {context.field.output_path} dataset from"
            f" {particle_group}'s packingcubes indices"
        )
        index_file = Path(context.tempdir) / f"{particle_group}.index"
        index = np.memmap(index_file, dtype=np.int64)

        packingcubes.data_objects._sort_field_in_place(index, field.value)

        next_step(context)

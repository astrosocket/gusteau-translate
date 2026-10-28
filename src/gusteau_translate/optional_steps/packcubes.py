from __future__ import annotations

__lazy_modules__ = ["packingcubes"]
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import packingcubes
import packingcubes.bounding_box
import packingcubes.cubes.cubes_numba
from numpy.typing import NDArray

from ..cells import Cells
from ..config import LOGGER_NAME
from ..pipeline import NextStep
from ..translation_base import (
    TranslationClassError,
    TranslationContext,
    TranslationStep,
)
from ..translations.dataset_translation import (
    DatasetTranslation,
    particle_group_from_output_path,
)

LOGGER = logging.getLogger(LOGGER_NAME)


@dataclass
class CubesParts:
    cube_indices: np.ndarray
    cube_boxes: list[packingcubes.bounding_box.BoundingBox]
    cube_trees: list[np.ndarray | packingcubes.PackedTree] | None = None


def _compute_cell_indices(
    dataset: packingcubes.data_objects.MultiParticleDataset, cells: Cells
) -> NDArray:
    """Use packingcubes to generate cell indices and sort particles"""

    data_container = dataset.data_container
    box = packingcubes.bounding_box.make_bounding_box(cells.Grid_box)
    cube_indices = packingcubes.cubes.cubes_numba.cube(
        data_container, cells.dimension[0], box
    )
    # check cube sizes
    if np.any(np.diff(cube_indices[:-1]) >= 2**32):
        raise ValueError(
            f"Requested number of cubes is insufficient for {dataset.particle_type}."
            " At least one cube has more than 2**32 particles, the max allowed for"
            " a packed tree."
        )
    return cube_indices


@dataclass
class PackCubesStep(TranslationStep):
    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, DatasetTranslation):
            raise TranslationClassError(step=self, field=field)

        particle_group = particle_group_from_output_path(field.output_path)
        LOGGER.debug(f"Doing initial cubing on {particle_group}")

        bounding_box = context.quantities["Bounding_box"]
        dataset = packingcubes.InMemory(
            positions=field.value,
            bounding_box=bounding_box,
            particle_type=particle_group,
        )

        cells = context.cells

        orig_particle_group = particle_group_from_output_path(field.input_path)
        cell_indices = cells.Indices.get(orig_particle_group)
        if cell_indices is None:
            cell_indices = _compute_cell_indices(dataset, cells)
            cells.Indices[particle_group] = cell_indices

        cubes_dict = {
            "cube_indices": cell_indices,
            "cube_boxes": [
                packingcubes.bounding_box.make_bounding_box(b)
                for b in cells.BoundingBoxes
            ],
            "_typing": [],  # needed for typing
        }
        cubes = packingcubes.Cubes(dataset, cubes_dict=cubes_dict)

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

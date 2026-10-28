from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import ClassVar, cast

__lazy_modules__ = ["packingcubes"]

import logging
from dataclasses import dataclass

import numpy as np
import packingcubes
import packingcubes.bounding_box

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


class CubesTranslator:
    """Class to translate pre-defined cubes structures into the packingcubes system"""

    _subclasses: ClassVar[dict[str, type[CubesTranslator]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        if name in cls._subclasses:
            raise Exception(
                f"{name} has already been registered as a CubesTranslation. "
                "Overwriting is forbidden"
            )
        cls._subclasses[name] = cls

    @abstractmethod
    def __call__(
        self, particle_group: str, context: TranslationContext
    ) -> CubesParts: ...


class SwiftCubesTranslator(CubesTranslator):
    """Convert Swift-like Cells structure to packingcubes cubes"""

    def __call__(self, particle_group: str, context: TranslationContext) -> CubesParts:
        # we don't actually care how the cells are laid out, we just need to
        # know their offsets and bounding boxes
        LOGGER.debug(
            "Loading cube indices and boxes from SWIFT-like"
            f" Cell structure for {particle_group}"
        )
        infile = context.infile

        counts_ds = infile[f"/Cells/Counts/{particle_group}"]
        counts = np.empty(counts_ds.shape, dtype=counts_ds.dtype)
        counts_ds.read_direct(counts)

        offsets = infile[f"/Cells/OffsetsInFile/{particle_group}"]
        indices = np.empty(offsets.shape, dtype=offsets.dtype)
        offsets.read_direct(indices)

        min_pos_ds = infile[f"/Cells/MinPositions/{particle_group}"]
        min_pos = np.empty(min_pos_ds.shape, dtype=min_pos_ds.dtype)
        min_pos_ds.read_direct(min_pos)

        max_pos_ds = infile[f"/Cells/MaxPositions/{particle_group}"]
        max_pos = np.empty(max_pos_ds.shape, dtype=max_pos_ds.dtype)
        max_pos_ds.read_direct(max_pos)

        min_size = infile["/Cells/Meta-data"].attrs["size"]

        # min_pos and max_pos are unset if no particles are present in the box
        # (counts==0). So we need to skip those boxes
        nonempty = counts > 0
        min_pos = min_pos[nonempty]
        max_pos = max_pos[nonempty]
        indices = indices[nonempty]
        counts = counts[nonempty]

        box_sizes = max_pos - min_pos
        box_sizes[counts == 1, :] = min_size
        boxes = np.hstack((min_pos, box_sizes))

        bboxes = [packingcubes.bounding_box.make_bounding_box(box) for box in boxes]

        return CubesParts(cube_indices=indices, cube_boxes=bboxes)


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
            positions=field.value, bounding_box=bounding_box
        )

        cubes_dict = None
        if hasattr(context, "cubes_translator"):
            original_particle_group = particle_group_from_output_path(field.input_path)
            ct = cast(CubesTranslator, context.cubes_translator)
            cd = ct(particle_group=original_particle_group, context=context)
            cubes_dict = {
                "cube_indices": cd.cube_indices,
                "cube_boxes": cd.cube_boxes,
            }
        cubes = packingcubes.Cubes(dataset, cubes_dict=cubes_dict)  # type: ignore

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

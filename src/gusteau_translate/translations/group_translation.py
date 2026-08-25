import logging
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import cast

import h5py

from ..base_steps.base import LoadStep, SaveStep
from ..config import LOGGER_NAME
from ..pipeline import NextStep
from ..translation_base import (
    FieldTranslation,
    TranslationClassError,
    TranslationContext,
    TranslationError,
    TranslationStep,
)

LOGGER = logging.getLogger(LOGGER_NAME)


@dataclass
class GroupTranslation(FieldTranslation):
    _: KW_ONLY
    has_particles: bool = False
    description: str | None = None
    tags: list[str] | None = None
    """Flag denoting whether any datasets under this group are particle datasets"""

    def __post_init__(self):
        self.pipeline.append(GroupSaveStep())
        self.pipeline.append(GroupLoadStep())
        if self.input_path:
            self.pipeline.append(ProcessGroupChildrenStep())
        if self.has_particles and (self.description is None or self.tags is None):
            raise TranslationError(
                f"Particle Group {self.name} must specify"
                " the `description` and `tags` fields!"
            )

    @property
    def name(self) -> str:
        return Path(self.output_path).stem


@dataclass
class CreateGroupTranslation(GroupTranslation):
    def __post_init__(self):
        self.input_path = ""
        super().__post_init__()
        self.pipeline.remove("GroupLoadStep")


@dataclass
class GroupLoadStep(LoadStep):
    """'Load' the group at field.input_path"""

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = cast(GroupTranslation, context.field)
        if field.input_path not in context.infile:
            if field._required:
                raise TranslationError(
                    f"Required group {field.input_path} not present!"
                )
            LOGGER.debug(
                f"Optional group {field.input_path} not present. Nothing to translate"
            )
            return
        LOGGER.debug(f"Loading group {field.input_path}")
        next_step(context)


@dataclass
class GroupSaveStep(SaveStep):
    """Create the group at field.output_path"""

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = cast(GroupTranslation, context.field)
        LOGGER.debug(f"Saving group {field.output_path}")
        # using require group instead of create group since it might already
        # exist
        group = context.outfile.require_group(
            field.output_path
        )  # add source description attribute
        group.attrs["Source_description"] = field.input_description
        if field.description is not None:
            group.attrs["Description"] = field.description
        if field.tags is not None:
            group.attrs["Tags"] = field.tags
        next_step(context)


@dataclass
class ProcessGroupChildrenStep(TranslationStep):
    """Recursively process children to update paths"""

    priority: int = SaveStep.priority + 50
    write_out: bool = False

    def __call__(self, context: TranslationContext, next_step: NextStep):
        field = context.field
        if not isinstance(field, GroupTranslation):
            raise TranslationClassError(step=self, field=field)
        LOGGER.debug(
            f"Processing children of {field.input_path} --> {field.output_path}"
        )

        group = cast(h5py.Group, context.infile[field.input_path])

        translation = context.translation

        def visit_object(
            name: str, out_name: str, obj: h5py.Group | h5py.Dataset | str
        ):
            LOGGER.debug(f"Visiting {name}")
            child_fields = translation.get_fields_from_input(name)
            child_field = translation.get_field_from_output(out_name)
            child_fields.extend([] if child_field is None else [child_field])
            if child_fields:
                LOGGER.debug(f"Skipping already defined field {name}")
                return
            obj = (
                "particles"
                if isinstance(obj, h5py.Dataset) and field.has_particles
                else obj
            )
            subfield = translation._default_translation_factory(
                obj, name, out_replace=(field.input_path, field.output_path)
            )
            subfield._explicit = False
            if isinstance(subfield, GroupTranslation):
                subfield.has_particles = field.has_particles
            # LOGGER.debug(
            #     f"Adding translation {subfield.input_path} to {subfield.output_path}"
            # )
            translation.add_field(subfield)

        def update_translations(_: str, obj: h5py.Group | h5py.Dataset) -> None:
            name = str(obj.name)
            out_name = name.replace(field.input_path, field.output_path)
            visit_object(name, out_name, obj)
            if isinstance(obj, h5py.Dataset):
                # datasets handle their own attributes
                return
            for attr in obj.attrs:
                visit_object(f"{name}.{attr}", f"{out_name}.{attr}", "")
            return

        group.visititems(func=update_translations)
        for attr in group.attrs:
            visit_object(f"{group.name}.{attr}", f"{field.output_path}.{attr}", "")

        next_step(context)

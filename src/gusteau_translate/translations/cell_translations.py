from dataclasses import KW_ONLY, dataclass

from gusteau_translate.pipeline import NextStep

from ..base_steps.base import LoadStep
from ..cells import update_cells
from ..translation_base import (
    TranslationClassError,
    TranslationContext,
    TranslationStep,
)
from .attribute_translation import AttributeTranslation
from .dataset_translation import DatasetTranslation


@dataclass
class UpdateFromCellsStep(TranslationStep):
    _: KW_ONLY
    priority = LoadStep.priority

    def __call__(self, context: TranslationContext, next_step: NextStep):

        field = context.field
        if not isinstance(field, (CellAttributeTranslation, CellDatasetTranslation)):
            raise TranslationClassError(step=self, field=field)

        cells = context.cells
        update_cells(cells, context.outfile)
        field.value = getattr(cells, field.cell_field)

        next_step(context)


@dataclass
class CellAttributeTranslation(AttributeTranslation):
    """Cells-group attribute translation

    Effectively a normal attribute translation, but automatically includes the
    UpdateFromCellsStep and does not load anything.
    """

    _: KW_ONLY
    input_path: str = ""
    input_description: str = "Generated from cells structure"
    cell_field: str

    def __post_init__(self):
        super().__post_init__()
        self.pipeline.remove("AttributeLoadStep")
        self.pipeline.append(UpdateFromCellsStep())


@dataclass
class CellDatasetTranslation(DatasetTranslation):
    """Cells-group dataset translation

    Effectively a normal dataset translation, but automatically includes the
    UpdateFromCellsStep and does not load anything.
    """

    _: KW_ONLY
    input_path: str = ""
    input_description: str = "Generated from cells structure"
    cell_field: str

    def __post_init__(self):
        super().__post_init__()
        self.pipeline.remove("DatasetLoadStep")
        self.pipeline.append(UpdateFromCellsStep())

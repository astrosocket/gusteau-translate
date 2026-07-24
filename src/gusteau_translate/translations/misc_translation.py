from dataclasses import KW_ONLY, dataclass, field

from ..base_steps.base import SkipStep
from ..translation_base import FieldTranslation, TranslationPipeline


@dataclass
class SkipTranslation(FieldTranslation):
    """
    Throw this field away before even loading data
    """

    _: KW_ONLY
    pipeline: TranslationPipeline = field(default_factory=TranslationPipeline)

    def __post_init__(
        self,
    ):
        self.pipeline.append(SkipStep(priority=-1000))

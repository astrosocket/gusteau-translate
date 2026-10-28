from __future__ import annotations

import dataclasses
import itertools
import logging
import re
from abc import abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import KW_ONLY, dataclass
from enum import Enum, auto
from pathlib import Path
from traceback import TracebackException
from typing import Any, ClassVar, Protocol, cast

import h5py

from .cells import Cells
from .config import LOGGER_NAME, TranslationOptions, _default_translation_options
from .pipeline import ErrorHandler, NextStep, Pipeline, PipelineError, PipelineStep
from .specification import GUSTEAU, DatasetSpec, GroupSpec

LOGGER = logging.getLogger(LOGGER_NAME)

type dict_info_type = (
    int | float | str | list[dict_info_type] | dict[str, dict_info_type]
)
step_info = dict_info_type.__value__


class TranslationError(PipelineError): ...


class TranslationClassError(TranslationError):
    """Error for attempting to run a TranslationStep on unsupported FieldTranslation"""

    def __init__(self, *args, step: TranslationStep, field: FieldTranslation, **kwargs):
        super().__init__(*args)
        self.step = step
        self.field = field

    def __str__(self):
        return (
            f"Attempted to apply a {self.step.__class__.__name__} "
            f"to a {self.field.__class__.__name__}!"
        )


@dataclass
class TranslationContext:
    infile: h5py.File
    outfile: h5py.File
    tempdir: str
    translation: Translation
    field: FieldTranslation
    cells: Cells
    quantities: dict[str, Any] = dataclasses.field(default_factory=dict)
    options: TranslationOptions = dataclasses.field(
        default_factory=_default_translation_options
    )


@dataclass
class TranslationStep[TranslationContext](PipelineStep):
    """Abstract base class for steps in the translation pipeline

    Attributes
    ----------
    priority:
        Priority of this step in the pipeline.
        Lower values come first. Default is `1`.
    write_out:
        Whether to include this step when writing out a translation
        Not needed for generic steps, like attribute loading.
        Default is `True`.

    """

    priority: int = 1
    write_out: bool = True

    _subclasses: ClassVar[dict[str, type[TranslationStep]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        if name in cls._subclasses:
            raise Exception(
                f"{name} has already been registered as a TranslationStep. "
                "Overwriting is forbidden"
            )
        cls._subclasses[name] = cls

    @abstractmethod
    def __call__(self, context: TranslationContext, next_step: NextStep): ...

    def to_dict(
        self, pipeline_dict: dict[str, dict_info_type]
    ) -> dict[str, dict_info_type]:
        step_table: dict[str, dict_info_type] = {}
        for key, value in self.__dict__.items():
            if key == "write_out":
                continue
            if key == "priority" and value == self.__class__.priority:
                continue
            if not isinstance(value, step_info):
                continue
            step_table[key] = value
        pipeline_dict[self.__class__.__name__] = step_table
        return step_table


type TranslationPipelineQueue[Context] = list[TranslationStep[Context]]


class TranslationPipeline[TranslationContext](Pipeline):
    queue: TranslationPipelineQueue

    def __init__(self, *steps: TranslationStep[TranslationContext]):
        super().__init__(*steps)

    def append(self, step: PipelineStep[TranslationContext]):
        if not isinstance(step, TranslationStep):
            raise ValueError("Only TranslationSteps are supported")
        super().append(step)

    def extend(self, steps: Iterable[PipelineStep[TranslationContext]]):
        if not all(isinstance(step, TranslationStep) for step in steps):
            raise ValueError("Only TranslationSteps are supported")
        super().extend(steps)

    def remove(self, step_name: str) -> TranslationStep[TranslationContext] | None:
        return cast(TranslationStep | None, super().remove(step_name))

    def __iter__(self) -> Iterator[TranslationStep[TranslationContext]]:
        return iter(self.queue)


class TranslationStatus(Enum):
    UNUSED = auto()
    """Translation is newly created"""
    IN_PROGRESS = auto()
    """Translation pipeline in progress"""
    USED = auto()
    """Translation pipeline complete"""
    RESET = auto()
    """Translation pipeline reset"""


@dataclass
class FieldTranslation[TranslationContext]:
    """
    Translation of an individual field (attribute/dataset/group)
    """

    input_path: str
    """Path to the input field in the original file"""
    input_description: str
    """Description of the input field in the original file"""
    output_path: str
    """Path to the output field in the GUSTEAU snapshot"""
    pipeline: TranslationPipeline[TranslationContext]
    """Collection of transformations to be run on this field"""

    _value: Any = dataclasses.field(init=False, default=None, repr=False)
    """The actual field data corresponding to this translation"""

    _status: TranslationStatus = dataclasses.field(
        init=False, default=TranslationStatus.UNUSED
    )
    """Whether this translation has already been used"""

    _: KW_ONLY
    dtype: str | None = None
    """The dtype of the field (e.g. uint32, str)"""
    _required: bool = False
    """
    Whether the field should raise an error on invalid input_path/output_path
    Effectively, if required, field must be "loadable" and "saveable" for
    whatever those words mean in the pipeline steps.
    """
    _explicit: bool = True
    """Whether this translation was explicitly described in a style file"""

    _subclasses: ClassVar[dict[str, type[FieldTranslation]]] = {}
    """Registry of FieldTranslations"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value

    @property
    def name(self) -> str:
        return self.log_name

    def __hash__(self) -> int:
        return hash(f"{self.input_path}->{self.output_path}")

    @property
    def log_name(self) -> str:
        return self.input_path if self.input_path else self.output_path

    def to_dict(self, docpart: dict[str, dict_info_type]) -> dict[str, dict_info_type]:
        field_table: dict[str, dict_info_type] = {}
        field_table["type"] = self.__class__.__name__
        if self.input_path != self.output_path:
            field_table["input_path"] = self.input_path
            field_table["output_path"] = self.output_path
        if self.input_description:
            field_table["input_description"] = self.input_description
        pipeline_dict: dict[str, dict_info_type] = {}
        for step in self.pipeline:
            if step.write_out:
                step.to_dict(pipeline_dict)

        # Need to not include pipeline if it's only load/save steps
        # ask pipeline step if writeable/readable?
        if pipeline_dict:
            field_table["pipeline"] = pipeline_dict

        docpart[self.name] = field_table
        return field_table


def normalize_output_path(output_path: str) -> str:
    """Normalize an output_path string"""
    output_path = re.sub(pattern=r"[^a-zA-Z0-9_/.]+", repl="_", string=output_path)
    if "." in output_path:
        output_path_parts = output_path.split(".")
        output_path = "_".join(output_path_parts[:-1]) + "." + output_path_parts[-1]
    output_path = re.sub(pattern=r"__+", repl="_", string=output_path)
    return output_path[:-1] if output_path[-1] == "_" else output_path


def translation_error_handler(
    ex: Exception,
    ctx: TranslationContext,
    current_step: PipelineStep[TranslationContext],
    next_step: NextStep,
):
    cause = ex.__cause__ if ex.__cause__ is not None else ex
    traceback = TracebackException.from_exception(
        cause, capture_locals=True, lookup_lines=True
    )
    msg = f"""
    Translation failed on step {current_step.__class__.__name__}
    in the pipeline for {ctx.field.input_path}
    {"\n  ".join(traceback.format())}
    """
    LOGGER.warning(msg)
    raise TranslationError(ex) from cause


def add_field_to_dict(
    field: FieldTranslation,
    doc: dict[str, dict_info_type],
    *,
    prefer_input: bool | None = None,
):
    # input and output should look like /group/subgroup/name or
    # /group/subgroup/name.attribute_name
    inpath = Path(field.input_path[1:].replace(".", "/")).parts
    outpath = Path(field.output_path[1:].replace(".", "/")).parts
    if prefer_input is None:
        # find which has already been specified more
        in_len = out_len = 0
        for i, (length, path) in enumerate(
            zip([in_len, out_len], [inpath, outpath], strict=True)
        ):
            docpart = doc
            while path[length] in docpart and length + 1 < len(path):
                if not isinstance(docpart[path[length]], dict):
                    break
                docpart = cast(dict[str, dict_info_type], docpart[path[length]])
                length += 1
            if length + 1 == len(path):
                # we've specified the entire path except this name
                prefer_input = bool(i)
                break
        # if we still haven't decided, use longer specification
        prefer_input = in_len > out_len if prefer_input is None else prefer_input

    path = inpath if prefer_input else outpath
    docpart = doc
    for group_name in path[:-1]:
        if group_name not in docpart:
            group: dict[str, dict_info_type] = {}
            docpart[group_name] = group
        docpart = cast(dict[str, dict_info_type], docpart[group_name])

    field.to_dict(docpart)


dummy_field = FieldTranslation("", "", "", TranslationPipeline[TranslationContext]())


class TranslationFactory(Protocol):
    def __call__(
        self,
        obj: Any,
        path: str,
        *args,
        input_path: str | None = None,
        input_description: str | None = None,
        output_path: str | None = None,
        pipeline: TranslationPipeline | None = None,
        **kwargs,
    ) -> FieldTranslation[TranslationContext]: ...


class Translation:
    """
    Translation between a style of snapshot output and the GUSTEAU specification
    """

    out2in: dict[str, FieldTranslation]
    in2out: dict[str, list[FieldTranslation]]
    gusteau: dict[str, Any]
    _SPEC: ClassVar[GUSTEAU] = GUSTEAU()
    _default_translation_factory: TranslationFactory
    _pipeline_class_additions: dict[str, list[TranslationStep]]
    _pipeline_path_additions: dict[str, list[TranslationStep]]
    _priority_fields: list[FieldTranslation]

    def __init__(
        self,
        *,
        default_factory: TranslationFactory,
        gusteau: dict[str, Any] | None = None,
    ):
        self.out2in: dict[str, FieldTranslation] = {}
        self.in2out: dict[str, list[FieldTranslation]] = {}
        self._default_translation_factory = default_factory

        self._priority_fields: list[FieldTranslation] = []

        self._process_gusteau(gusteau)

    def _process_gusteau(self, gusteau: dict[str, Any] | None = None):
        self.gusteau = {} if gusteau is None else gusteau
        pipeline_class_additions = cast(
            dict[str, list[str | TranslationStep]],
            self.gusteau.get("pipeline_class_additions", {}),
        )
        self._pipeline_class_additions = {
            translation_class: [
                TranslationStep._subclasses[step]() if isinstance(step, str) else step
                for step in steps
            ]
            for translation_class, steps in pipeline_class_additions.items()
            if steps
        }
        self._pipeline_class_additions["FieldTranslation"] = (
            self._pipeline_class_additions.get("FieldTranslation", [])
        )
        self._pipeline_class_additions["FieldTranslation"].extend(
            self._pipeline_class_additions.get("any", [])
        )
        self._pipeline_class_additions["FieldTranslation"].extend(
            self._pipeline_class_additions.get("all", [])
        )

        pipeline_path_additions = cast(
            dict[str, list[str | TranslationStep]] | None,
            self.gusteau.get("pipeline_path_additions", {}),
        )
        pipeline_path_additions = (
            {} if pipeline_path_additions is None else pipeline_path_additions
        )
        self._pipeline_path_additions = {
            path_pattern: [
                TranslationStep._subclasses[step]() if isinstance(step, str) else step
                for step in step_names
            ]
            for path_pattern, step_names in pipeline_path_additions.items()
        }

    def _iterate_fields_only(
        self, group: GroupSpec, map_: dict[str, FieldTranslation], *, start_path=""
    ) -> Iterator[FieldTranslation]:
        for a in group.attrs:
            path = f"{start_path}.{a.name}"
            if path not in map_:
                LOGGER.debug(f"Generating field {path} from group spec")
            field = map_.get(path, self._default_translation_factory("attribute", path))
            field._required |= not a.optional
            yield field
        for d in group.datagroups:
            if isinstance(d, DatasetSpec):
                path = f"{start_path}/{d.name}"
                field = map_.get(
                    path, self._default_translation_factory("dataset", path)
                )
                field._required |= not d.optional
                yield field
            elif isinstance(d, GroupSpec):
                yield from self._iterate_fields_only(
                    d, map_, start_path=f"{start_path}/{d.name}"
                )

    def add_field(self, field: FieldTranslation):
        LOGGER.debug(f"Adding {field.__class__.__name__} {field.log_name}")

        # add pipeline additions via class
        class_ = field.__class__
        while class_:
            steps = self._pipeline_class_additions.get(class_.__name__, [])
            field.pipeline.extend([] if steps is None else steps)
            class_ = class_.__base__

        # add pipeline additions via output path
        output_path = Path(field.output_path)
        for pattern, steps in self._pipeline_path_additions.items():
            if output_path.full_match(pattern, case_sensitive=True):
                field.pipeline.extend([] if steps is None else steps)

        self.in2out.setdefault(field.input_path, []).append(field)
        self.out2in[field.output_path] = field

    def get_field_from_output(self, output_path: str) -> FieldTranslation | None:
        return self.out2in.get(output_path)

    def get_fields_from_input(self, input_path: str) -> list[FieldTranslation]:
        return self.in2out.get(input_path, [])

    def remove_field(self, field: FieldTranslation) -> FieldTranslation | None:
        if field.input_path in self.in2out:
            self.in2out[field.input_path].remove(field)
        return self.out2in.pop(field.output_path, None)

    def remove_field_from_output(self, output_path: str) -> FieldTranslation | None:
        field = self.get_field_from_output(output_path)
        return None if field is None else self.remove_field(field)

    def remove_fields_from_input(self, input_path: str) -> list[FieldTranslation]:
        """Remove all fields matching input path, returning removed fields"""
        fields = self.in2out.get(input_path, [])
        for field in fields:
            self.out2in.pop(field.output_path)
        return fields

    def set_priority_fields(self, *field_list: str, **field_dict: str):
        """
        Set ordered list of translations to apply prior to explicit but post required

        Parameters
        ----------
        *field_list: str, ...
            List of output paths that need to be prioritized. Note that if the
            translation does not exist, an error will be raised.

        **field_dict: str->str, ...
            Dict of output paths->field types that need to be prioritized. If
            the translation does not exist, will use the
            _default_translation factory to create a translation of type
            field_type.
            If finer-grained control on the constructor is required, use an
            explicit translation.

        Raises
        ------
        ValueError:
            If an element of field list is not defined already

        """

        field_dict |= dict.fromkeys(field_list, "")
        for field_name, obj in field_dict.items():
            if field_name not in self.out2in:
                if not obj:
                    raise ValueError(f"Translation {field_name} is not defined!")
                field = self._default_translation_factory(obj, field_name)
                self.add_field(field)
            else:
                field = self.out2in[field_name]
            self._priority_fields.append(field)

    def apply_translation(
        self,
        field: str | FieldTranslation | Iterable[FieldTranslation],
        context: TranslationContext,
        error_handler: ErrorHandler,
        *,
        obj: Any = None,
    ) -> list[str]:
        out_fields = []

        if isinstance(field, FieldTranslation):
            field_list = [field]
        elif isinstance(field, str):
            field_list = self.in2out.setdefault(
                field,
                [
                    self._default_translation_factory(
                        obj,
                        field,
                    )
                ],
            )
        else:
            field_list = field

        for field in field_list:
            if field._status != TranslationStatus.UNUSED:
                continue
            LOGGER.info(f"Translating {field.name}")
            field._status = TranslationStatus.IN_PROGRESS
            context.field = field
            field.pipeline(context=context, error_handler=error_handler)
            field._status = TranslationStatus.USED
            if field.input_path:
                out_fields.append(field.input_path)
        return out_fields

    def initialize_required_groups(self, outfile: h5py.File):
        reqd = [(g, g.name) for g in Translation._SPEC.required_tlgs.values()]
        while reqd:
            group = reqd.pop()
            outfile.create_group(name=group[1], track_order=True)
            reqd.extend(
                (g, f"{group[1]}/{g.name}")
                for g in group[0].datagroups
                if isinstance(g, GroupSpec) and not g.optional
            )

    def get_header_translations(self) -> Iterator[FieldTranslation]:
        header = Translation._SPEC.required_tlgs["Header"]
        yield from self._iterate_fields_only(header, self.out2in, start_path="/Header")

    def get_required_translations(self) -> Iterator[FieldTranslation]:
        """Get the required translations from the GUSTEAU spec

        Note
        ----
        This is required to return the Header translations first

        See Also
        --------
        get_header_translations, get_explicit_translations
        """
        reqd = Translation._SPEC.required_tlgs.values()
        for group in reqd:
            yield from self._iterate_fields_only(
                group, self.out2in, start_path="/" + group.name
            )

    def get_explicit_translations(
        self,
        *,
        skip_priority: bool = False,
        skip_input: bool = False,
        skip_output: bool = False,
    ) -> Iterable[FieldTranslation]:
        in2out_lists = (
            [] if skip_input else list(itertools.chain(*self.in2out.values()))
        )
        return itertools.chain(
            [] if skip_priority else self._priority_fields,
            [] if skip_output else list(self.out2in.values()),
            in2out_lists,
        )

    def apply_explicit_translations(
        self,
        context: TranslationContext,
        error_handler: ErrorHandler,
        *,
        skip_priority: bool = False,
        skip_input: bool = False,
        skip_output: bool = False,
    ) -> list[str]:
        return self.apply_translation(
            self.get_explicit_translations(
                skip_priority=skip_priority,
                skip_output=skip_output,
                skip_input=skip_input,
            ),
            context,
            error_handler,
        )

    def apply_unused_translations(
        self, context: TranslationContext, error_handler: ErrorHandler
    ) -> list[str]:
        # Note that we only include fields in out2in because we don't care about
        # fields that don't have an output
        return self.apply_translation(self.out2in.values(), context, error_handler)

    def to_dict(self, *, prefer_input: bool | None = None) -> dict[str, dict_info_type]:
        doc: dict[str, dict_info_type] = {}
        # reset field usage
        for field in itertools.chain(self.out2in.values(), *self.in2out.values()):
            field._status = TranslationStatus.RESET

        for field in itertools.chain(
            self.get_required_translations(),
            self.out2in.values(),
            *self.in2out.values(),
        ):
            if field._status == TranslationStatus.USED:
                continue
            field._status = TranslationStatus.USED
            add_field_to_dict(field, doc, prefer_input=prefer_input)

        return doc

import logging
import re
from typing import Any

import h5py

from ..translation_base import (
    FieldTranslation,
    Translation,
    TranslationContext,
    TranslationFactory,
    TranslationPipeline,
    TranslationStep,
    dict_info_type,
)
from ..translations import (
    AttributeTranslation,
    DatasetTranslation,
    GroupTranslation,
)
from ..translations.dataset_translation import (
    ParticleDatasetTranslation,
)

LOGGER = logging.getLogger("gusteau_translate")


class TranslationParseError(Exception): ...


class default_translation_factory(TranslationFactory):
    def __call__(
        self,
        obj: Any,
        path: str,
        *,
        input_path: str | None = None,
        input_description: str | None = None,
        output_path: str | None = None,
        pipeline: TranslationPipeline | None = None,
        in_replace: tuple[str, str] | None = None,
        out_replace: tuple[str, str] | None = None,
        **kwargs,
    ) -> FieldTranslation[TranslationContext]:
        obj_name = obj.lower() if isinstance(obj, str) else obj
        if isinstance(obj, h5py.Group) or obj_name == "group":
            translation_class = GroupTranslation
        elif isinstance(obj, h5py.Dataset) or obj_name == "dataset":
            translation_class = DatasetTranslation
        elif obj_name == "particles":
            translation_class = ParticleDatasetTranslation
        elif obj in FieldTranslation._subclasses:
            translation_class = FieldTranslation._subclasses[obj]
        else:
            translation_class = AttributeTranslation
        input_path = path if input_path is None else input_path
        output_path = path if output_path is None else output_path
        if in_replace is not None:
            input_path = input_path.replace(*in_replace)
        if out_replace is not None:
            output_path = output_path.replace(*out_replace)
        output_path = re.sub(pattern=r"[^a-zA-Z0-9_./]+", repl="_", string=output_path)
        output_path = output_path[:-1] if output_path[-1] == "_" else output_path
        input_description = path if input_description is None else input_description
        pipeline = TranslationPipeline() if pipeline is None else pipeline
        kwargs["_explicit"] = kwargs.get("_explicit", False)
        return translation_class(
            input_path=input_path,
            input_description=input_description,
            output_path=output_path,
            pipeline=pipeline,
            **kwargs,
        )


def _generate_name(
    *, path_field: str | None, provided_prefix: str, provided_name: str
) -> str:
    if path_field is None:
        return f"{provided_prefix}/{provided_name}"
    # check if relative path - does path_field start with "/"?
    if path_field[0] == "/":
        return path_field
    return path_field if path_field[0] == "/" else f"{provided_prefix}/{path_field}"


def _clean_field_translation(
    d: dict, *, name: str, start_path: str, out_path: str | None = None
) -> dict | None:
    # in_name = f"{start_path}/{name}"
    # out_path = start_path if out_path is None else out_path
    # out_name = f"{out_path}/{name}"
    # if "input_path" not in d:
    #     d["input_path"] = in_name
    # if "output_path" not in d:
    #     d["output_path"] = out_name

    in_name = d["input_path"] = _generate_name(
        path_field=d.get("input_path"), provided_prefix=start_path, provided_name=name
    )
    d["output_path"] = _generate_name(
        path_field=d.get("output_path"),
        provided_prefix=start_path if out_path is None else out_path,
        provided_name=name,
    )

    if "input_description" not in d:
        d["input_description"] = d.get("input_description", f"From {in_name}")

    d_pipeline = d.get("pipeline", {})
    if not isinstance(d_pipeline, dict):
        raise TranslationParseError(
            f"The `pipeline` field in {in_name} is not a dictionary"
        )

    pipeline: list[TranslationStep[TranslationContext]] = []
    for step_class, step in d_pipeline.items():
        pipeline.append(TranslationStep._subclasses[step_class](**step))
    d["pipeline"] = TranslationPipeline(*pipeline)

    children = d.pop("children", None)
    if children is None:
        return None

    if not isinstance(children, dict):
        raise TranslationParseError(
            f"The `children` field in {in_name} is not a dictionary"
        )

    return children


def recurse_groups(
    group: dict,
    translation: Translation,
    start_path: str = "",
    *,
    out_path: str | None = None,
    has_particles: bool = False,
):
    for key, value in group.items():
        if not isinstance(key, str):
            raise TranslationParseError("All keys must be strings")
        if isinstance(value, dict):
            type_ = value.pop("type", None)
            if type_ is not None:
                children = _clean_field_translation(
                    value,
                    name=key,
                    start_path=start_path,
                    out_path=out_path,
                )
                try:
                    field = FieldTranslation._subclasses[type_](**value)
                    translation.add_field(field)
                except Exception as ex:
                    LOGGER.warning(f"Failed on {key=}, {type_=}, {value=}")
                    raise TranslationParseError(ex) from ex

                if isinstance(field, GroupTranslation):
                    has_particles = has_particles or field.has_particles
                    field.has_particles = has_particles or field.has_particles

                if children:
                    recurse_groups(
                        children,
                        translation,
                        field.input_path,
                        out_path=field.output_path,
                        has_particles=has_particles,
                    )
            else:
                recurse_groups(
                    value,
                    translation,
                    f"{start_path}/{key}",
                    out_path=out_path,
                    has_particles=has_particles,
                )
        else:
            LOGGER.warning(
                f"Don't know how to load {key}:{value}! Perhaps a missing type?"
            )


def from_dict(
    doc: dict[str, dict_info_type],
    *,
    default_factory: TranslationFactory,
    gusteau: dict[str, Any] | None = None,
) -> Translation:
    translation = Translation(default_factory=default_factory, gusteau=gusteau)
    recurse_groups(doc, translation)
    LOGGER.debug("Parsed style file")
    return translation

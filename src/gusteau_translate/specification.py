from __future__ import annotations

import contextlib
import itertools
import json
import logging
from dataclasses import KW_ONLY, InitVar, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import h5py
import numpy as np

from .config import LOGGER_NAME, ValidationOptions

LOGGER = logging.getLogger(LOGGER_NAME)

type h5obj = h5py.Group | h5py.Dataset
type integer_type = int | np.int_ | np.int32 | np.uint32 | np.uint64
integer = integer_type.__value__
type number_type = float | integer | np.float32 | np.float64
number = number_type.__value__
type num_str_type = str | integer
num_str_array = (np.ndarray, num_str_type.__value__)


class RegexName(str):
    """Name that is vaguely regex-like"""

    def found_in(self, parent, validation_info):
        if "{pg}" in self:
            particle_names = validation_info.get("Particle_names", [])
            for pg in particle_names:
                if self.replace("{pg}", pg):
                    return parent.get(self.replace("{pg}", pg))
        return parent.get(self)


class ValidationError(Exception): ...


class AttributeMissing(ValidationError): ...


class DataGroupMissing(ValidationError): ...


class GroupMissing(ValidationError): ...


class TagMissing(ValidationError): ...


@dataclass
class ValidatedSpec:
    name: str
    """Name of the validateable object in the specification"""
    _: KW_ONLY
    optional: bool = False
    """Whether the object is optional"""

    _TYPE: ClassVar[str] = "N/A"

    def __post_init__(self): ...

    @classmethod
    def class_name(cls):
        return cls.__name__.replace("Spec", "")

    def check_presence(
        self,
        parent: h5obj,
        *,
        validation_info: dict[str, Any],
        prefix: str | None = None,
    ) -> h5obj | None:
        raise NotImplementedError("Only implemented in subclasses")

    def validate(
        self, obj: h5obj, *, validation_info: dict[str, Any], prefix: str | None = None
    ) -> Literal[True]:
        """Validate field"""
        prefix = "" if prefix is None else prefix
        LOGGER.debug(f"{prefix}Validating {self.class_name()}: {self.name}")
        return True


def _check_attribute_type(a: Any, attr_type: type | tuple[type, type]) -> str | None:
    parameterized_type = isinstance(attr_type, tuple)
    is_scalar = isinstance(a, str) | np.ndim(a) == 0

    if parameterized_type:
        if is_scalar:
            return f"Received {type(a)} but expected {attr_type[0]}[{attr_type[1]}]"
        if isinstance(a, attr_type[0]):
            return (
                None
                if len(a) == 0 or isinstance(a[0], attr_type[1])
                else (
                    f"Received {attr_type[0]} of {type(a[0])} instead of {attr_type[1]}"
                )
            )
        return f"Received {type(a)} instead of {attr_type[0]}"

    if is_scalar:
        return (
            None
            if isinstance(a, attr_type)
            else f"Received {type(a)} but expected {attr_type}"
        )

    return None


@dataclass
class AttributeSpec(ValidatedSpec):
    attr_type: type | tuple[type, type]
    """Type of the attribute in the specification"""
    attr_len: int | str = 1
    """Number of elements if intended to have a **set** number of elements"""

    _: KW_ONLY

    sets_info: str | None = None

    _TYPE: ClassVar[str] = "attribute"

    def check_presence(
        self,
        parent: h5obj,
        *,
        validation_info: dict[str, Any],
        prefix: str | None = None,
    ) -> h5obj | None:
        if self.name not in parent.attrs:
            if not self.optional:
                raise AttributeMissing(
                    f"Missing required attribute {self.name} from {parent}"
                )
            return None
        return parent

    def validate(
        self, obj: h5obj, *, validation_info: dict[str, Any], prefix: str | None = None
    ) -> Literal[True]:
        """Validate object has this attribute with correct type"""
        super().validate(obj, validation_info=validation_info, prefix=prefix)
        if self.name not in obj.attrs:
            if self.optional:
                return True
            raise AttributeMissing(
                f"Required attribute {self.name} not present in {obj}"
            )
        a = obj.attrs[self.name]

        err_msg = _check_attribute_type(a, self.attr_type)

        if err_msg is not None:
            raise ValidationError(f"Wrong attribute type for {self.name}. {err_msg}")

        attr_len = (
            int(validation_info[self.attr_len])
            if isinstance(self.attr_len, str)
            else self.attr_len
        )
        if attr_len > 1:
            with contextlib.suppress(TypeError):
                if len(a) != attr_len:
                    raise ValidationError(
                        f"{self.name} has wrong length. "
                        f"Expected {attr_len} elements, got {len(a)}"
                    )
        # if attr_len == 1, we don't need to match length, because that's
        # not a well defined operation. Example: string has length but only
        # want 1 string...
        if self.sets_info is not None:
            validation_info[self.sets_info] = a
        return True


@dataclass
class OptionalAttributeSpec(AttributeSpec):
    _: KW_ONLY
    optional: bool = True


@dataclass
class DatasetSpec(ValidatedSpec):
    m: int | str = 1
    """Size of the second dimension in the specification"""

    _TYPE: ClassVar[str] = "dataset"

    def check_presence(
        self,
        parent: h5obj,
        *,
        validation_info: dict[str, Any],
        prefix: str | None = None,
    ) -> h5obj | None:
        if not isinstance(parent, h5py.Group):
            raise ValidationError(
                f"Attempting to check for {self.name} dataset under {type(parent)}"
            )
        dobj = (
            self.name.found_in(parent, validation_info)
            if isinstance(self.name, RegexName)
            else parent.get(self.name)
        )
        if dobj is None:
            if not self.optional:
                raise DataGroupMissing(
                    f"Missing required {self._TYPE} {self.name} from {parent}"
                )
            return None
        return dobj

    def validate(
        self, obj: h5obj, *, validation_info: dict[str, Any], prefix: str | None = None
    ) -> Literal[True]:
        """Validate dataset in object"""
        super().validate(obj, validation_info=validation_info, prefix=prefix)
        if not isinstance(obj, h5py.Dataset):
            raise ValidationError(f"{obj} is not a Dataset")
        ndim = obj.ndim
        if ndim > 2:
            raise ValidationError(
                f"{obj} is an {ndim}-dimensional array. Only 1D & 2D arrays are allowed"
            )
        shape = obj.shape
        n = int(validation_info.get("N", shape[0]))
        if shape[0] != n:
            raise ValidationError(
                f"{obj} has the wrong length. Expected {n}, received {shape[0]}"
            )
        try:
            m = int(validation_info[self.m]) if isinstance(self.m, str) else self.m
        except TypeError as te:
            LOGGER.debug(f"{self.m=} {validation_info=}")
            LOGGER.debug(te)
            raise ValidationError(te) from te

        if (m > 1 and ndim < 2) or (m == 1 and ndim == 2):
            raise ValidationError(f"{obj} is a {ndim}D array and expected a {m}D array")
        if ndim >= 2 and shape[1] != m:
            raise ValidationError(f"{obj} has {shape[1]} columns. Expected {m}.")
        return True


def required_particle_dataset_attributes() -> list[AttributeSpec]:
    return [
        AttributeSpec("Units", str),
        AttributeSpec("Conversion_factor_CGS", number),
        AttributeSpec("Conversion_factor_PhysCGS", number),
        AttributeSpec("Description", str),
        AttributeSpec("Phys_CGS_expression", str),
        AttributeSpec("Lossy_compression", str),
        AttributeSpec("U_I_exponent", number),
        AttributeSpec("U_L_exponent", number),
        AttributeSpec("U_M_exponent", number),
        AttributeSpec("U_T_exponent", number),
        AttributeSpec("U_t_exponent", number),
        AttributeSpec("a_scale_exponent", number),
        AttributeSpec("h_scale_exponent", number),
    ]


@dataclass
class ParticleDatasetSpec(DatasetSpec):
    reqd_attrs: list[AttributeSpec] = field(default_factory=list[AttributeSpec])
    """Required dataset attributes in the specification"""

    def __post_init__(self):
        self.reqd_attrs.extend(required_particle_dataset_attributes())

    def validate(
        self, obj: h5obj, *, validation_info: dict[str, Any], prefix: str | None = None
    ) -> Literal[True]:
        """Validate, including required attributes"""
        prefix = " " if prefix is None else prefix
        super().validate(obj, validation_info=validation_info, prefix=prefix)
        all(
            a.validate(obj, validation_info=validation_info, prefix=prefix + " ")
            for a in self.reqd_attrs
        )
        return True


@dataclass
class OptionalParticleDatasetSpec(ParticleDatasetSpec):
    _: KW_ONLY
    optional: bool = True


@dataclass
class OptionalValidationGroup(ValidatedSpec):
    _: KW_ONLY
    num_required: int = 1
    num_max: int = field(default=1, init=False)
    num_maximum: InitVar[int | None] = None
    options: list[ValidatedSpec]

    _TYPE: ClassVar[str] = "OptionSet"

    def __post_init__(self, num_maximum: int | None):
        if self.num_required < 1:
            raise ValueError(
                "Need at least one required Dataset."
                " Otherwise just use the optional flag"
            )
        self.num_max = (
            len(self.options)
            if num_maximum is None
            else min(num_maximum, len(self.options))
        )

    def check_presence(
        self,
        parent: h5obj,
        *,
        validation_info: dict[str, Any],
        prefix: str | None = None,
    ) -> h5obj | None:
        LOGGER.debug("Called check_presence")
        if not isinstance(parent, h5py.Group):
            raise ValidationError(
                f"Attempting to check for {self.options} datasets under {type(parent)}"
            )
        num_found = 0
        for option in self.options:
            num_found += (
                option.name.found_in(parent, validation_info)
                if isinstance(option.name, RegexName)
                else option.name in parent
            )
        if num_found < self.num_required:
            raise DataGroupMissing(
                f"Missing more than {self.num_required} (minimum) in {self.name}"
            )
        if num_found > self.num_max:
            raise ValidationError(f"Too many datasets in {self.name} found")
        return parent

    def validate(
        self, obj: h5obj, *, validation_info: dict[str, Any], prefix: str | None = None
    ) -> Literal[True]:
        prefix = "" if prefix is None else prefix
        super().validate(obj, validation_info=validation_info, prefix=prefix)

        for option in self.options:
            o = None
            with contextlib.suppress(ValidationError):
                o = option.check_presence(
                    obj, validation_info=validation_info, prefix=prefix + " "
                )
            if o is not None:
                option.validate(o, validation_info=validation_info, prefix=prefix + " ")

        return True


@dataclass
class GroupSpec(ValidatedSpec):
    attrs: list[AttributeSpec] = field(default_factory=list[AttributeSpec])
    """Required group attributes"""

    datagroups: list[ValidatedSpec] = field(default_factory=list[ValidatedSpec])
    """Required datasets and subgroups"""

    _TYPE: ClassVar[str] = "group"

    def check_presence(
        self,
        parent: h5obj,
        *,
        validation_info: dict[str, Any],
        prefix: str | None = None,
    ) -> h5obj | None:
        if not isinstance(parent, h5py.Group):
            raise ValidationError(
                f"Attempting to check for {self.name} dataset under {type(parent)}"
            )
        dobj = (
            self.name.found_in(parent, validation_info)
            if isinstance(self.name, RegexName)
            else parent.get(self.name)
        )
        if dobj is None:
            if not self.optional:
                raise DataGroupMissing(
                    f"Missing required {self._TYPE} {self.name} from {parent}"
                )
            return None
        return dobj

    def validate(
        self, obj: h5obj, *, validation_info: dict[str, Any], prefix: str | None = None
    ) -> Literal[True]:
        """Validate group

        Checks that group is present in object with any specified attributes,
        datasets, or subgroups.

        Parameters
        ----------
        :
            The HDF5 object (Dataset or Group) to validate

        Returns
        -------
        :
            True if group is present and has all required fields and subgroups.
            False otherwise.

        Raises
        ------
        ValidationError
            if obj is not a group, group is not present, or some required
            fields or subgroups are missing
        """
        prefix = "" if prefix is None else prefix
        super().validate(obj, validation_info=validation_info, prefix=prefix)
        if not isinstance(obj, h5py.Group):
            raise ValidationError(f"{obj} is not a Group. (It's a {type(obj)})")
        # for a in self.attrs:
        #     aobj = a.check_presence(
        #         obj, validation_info=validation_info, prefix=prefix + " "
        #     )
        #     if aobj is None:
        #         continue
        #     a.validate(aobj, validation_info=validation_info, prefix=prefix + " ")

        # for d in self.datagroups:
        #     dobj = d.check_presence(
        #         obj, validation_info=validation_info, prefix=prefix + " "
        #     )
        #     if dobj is None:
        #         continue
        #     d.validate(dobj, validation_info=validation_info, prefix=prefix + " ")
        for o in itertools.chain(self.attrs, self.datagroups):
            pobj = o.check_presence(
                obj, validation_info=validation_info, prefix=prefix + " "
            )
            if pobj is None:
                continue
            o.validate(pobj, validation_info=validation_info, prefix=prefix + " ")
        LOGGER.debug(f"{prefix}Finished validating group {self.name}")
        return True


class INCLUDE_LEVEL(Enum):
    ABOVE = auto()
    EQUAL = auto()
    BELOW = auto()
    BELOP = auto()


ABOVE = INCLUDE_LEVEL.ABOVE
EQUAL = INCLUDE_LEVEL.EQUAL
BELOW = INCLUDE_LEVEL.BELOW
BELOP = INCLUDE_LEVEL.BELOP


def _check_below_tags(
    current_tag: str,
    included_tags: dict[str, INCLUDE_LEVEL],
    validation_info: dict[str, Any],
):
    below_tags = set()
    belop_tags = set()
    for tag, level in included_tags.items():
        match level:
            case INCLUDE_LEVEL.BELOW:
                below_tags.add(tag.lower())
            case INCLUDE_LEVEL.BELOP:
                belop_tags.add(tag.lower())
            case _:
                pass
    for t in below_tags:
        if t not in validation_info["tags"]:
            raise TagMissing(
                f"Tag {t} is required in a subgroup of {current_tag} but is missing"
            )

    if belop_tags and not any(t in validation_info["tags"] for t in belop_tags):
        raise TagMissing(
            f"Tag {current_tag} requires one of {belop_tags}, but none were present"
        )


# Tags
@dataclass
class TagSpec:
    name: str
    """Name of this tag"""

    _: KW_ONLY

    attrs: list[AttributeSpec] = field(default_factory=list[AttributeSpec])
    """Required group attributes"""

    datagroups: list[ValidatedSpec] = field(default_factory=list[ValidatedSpec])
    """Required datasets"""

    includes: dict[str, INCLUDE_LEVEL] = field(
        default_factory=lambda: {"particles": ABOVE}
    )
    """
    Any other tags that this tag includes mapped to the level where it needs to
    be included
    """

    def validate(
        self,
        obj: h5py.Group,
        *,
        validation_info: dict[str, Any],
        prefix: str | None = None,
    ) -> Literal[True]:
        """Validate that provided group has all of the specified fields and subgroups"""
        prefix = "" if prefix is None else prefix
        LOGGER.debug(f"{prefix}Validating tag {self.name}")
        # for a in self.attrs:
        #     if a.name not in obj.attrs:
        #         if not a.optional:
        #             raise AttributeMissing(
        #                 f"Missing required attribute {a.name} "
        #                 f"from {obj} for tag {self.name}"
        #             )
        #         continue
        #     a.validate(obj, validation_info=validation_info, prefix=prefix + " ")

        # for d in self.datagroups:
        #     dobj = obj.get(d.name)
        #     if dobj is None:
        #         if not d.optional:
        #             raise DataGroupMissing(
        #                 f"Missing required {d._TYPE} {d.name} "
        #                 f"from {obj} for tag {self.name}"
        #             )
        #         continue
        #     d.validate(dobj, validation_info=validation_info, prefix=prefix + " ")
        for o in itertools.chain(self.attrs, self.datagroups):
            pobj = o.check_presence(
                obj, validation_info=validation_info, prefix=prefix + " "
            )
            if pobj is None:
                continue
            o.validate(pobj, validation_info=validation_info, prefix=prefix + " ")

        _check_below_tags(self.name, self.includes, validation_info)

        LOGGER.debug(f"{prefix}Finished validating tag {self.name}")

        return True


def _get_attr_with_subgroups(obj: h5py.Group, name: str) -> h5obj | None:
    if name in obj.attrs:
        return obj
    for subgroup_name in ChemicalTagSpec._QUANTITIES:
        subgroup = obj.get(subgroup_name)
        if subgroup is not None and name in subgroup.attrs:
            return subgroup

    return None


def _get_obj_with_subgroups(obj: h5py.Group, name: str) -> h5py.Group | None:
    if name in obj:
        return obj[name]
    LOGGER.debug(f"{name} not at current level, checking subgroups of {obj.name}")
    for subgroup_name in ChemicalTagSpec._QUANTITIES:
        subgroup = obj.get(subgroup_name)
        LOGGER.debug(f"  Checking subgroup {subgroup_name} ({subgroup})")
        if subgroup is not None and name in subgroup:
            return subgroup[name]
    return None


@dataclass
class ChemicalTagSpec(TagSpec):
    """
    Tags for chemical quantities and types

    Chemical datasets are specified in the form `Chemical_qty/Chemical_type`,
    e.g. `Abundances/Hydrogen`, but the required datasets can be split across
    quantity types. So you can have `Abundances/Hydrogen/P` and
    `MassFractions/Hydrogen/E`, corresponding to the tags `["Abundance",
    "Hydrogen", "MassFraction"]`.

    So in a sense, the tags could be `["Hydrogen-Abundance",
    "Hydrogen-MassFraction"]` or tags on the subgroups, like
    `["Abundances"."Hydrogen", "MassFraction"."Hydrogen"]`, except that all of
    species datasets need to be present across all groups, and the tags are
    expected to be shared at the "top-level".

    So we need a special subclass and validation logic.
    """

    group: str
    """Group that this tag adds"""

    _: KW_ONLY

    extra_attrs: dict[str, list[AttributeSpec]] = field(
        default_factory=dict[str, list[AttributeSpec]]
    )
    """
    Extra required attributes that are only required if one of the
    combination tags matches
    """

    extra_datagroups: dict[str, list[ValidatedSpec]] = field(
        default_factory=dict[str, list[ValidatedSpec]]
    )
    """
    Extra required datasets/groups that are only required if one of the
    combination tags matches
    """

    _QUANTITIES: ClassVar[set[str]] = set()

    def validate(
        self,
        obj: h5py.Group,
        *,
        validation_info: dict[str, Any],
        prefix: str | None = None,
    ) -> Literal[True]:
        """Validate that provided group has all of the specified fields and subgroups"""
        prefix = "" if prefix is None else prefix
        LOGGER.debug(f"{prefix}Validating chemical tag {self.name} on {obj.name}")
        attrs = self.attrs
        datagroups = self.datagroups
        for s, a in self.extra_attrs.items():
            attrs.extend(a if s in obj else [])
        for s, d in self.extra_datagroups.items():
            datagroups.extend(d if s in obj else [])

        for a in attrs:
            full_name = f"{self.group}/{a.name}"
            aobj = _get_attr_with_subgroups(obj, full_name)
            if aobj is None:
                if not a.optional:
                    raise AttributeMissing(
                        f"Missing required attribute {full_name} "
                        f"from {aobj} for tag {self.name}"
                    )
                continue
            a.validate(aobj, validation_info=validation_info, prefix=prefix + " ")

        for d in datagroups:
            full_name = f"{self.group}/{d.name}"
            dobj = _get_obj_with_subgroups(obj, full_name)
            if dobj is None:
                if not d.optional:
                    raise DataGroupMissing(
                        f"Missing required {d._TYPE} {full_name} "
                        f"from {obj} for tag {self.name}"
                    )
                continue
            d.validate(dobj, validation_info=validation_info, prefix=prefix + " ")

        _check_below_tags(self.name, self.includes, validation_info)

        LOGGER.debug(f"{prefix}Finished validating chemical tag {self.name}")
        return True


particles = TagSpec(
    name="Particles",
    includes={},  # Particles needs no other tags
    datagroups=[
        DatasetSpec(name="ParticleIDs"),
        DatasetSpec(name="Coordinates", m="D"),
        DatasetSpec(name="Velocities", m="D"),
    ],
)

tag_defs = {
    "particles": particles,
}


def required_tag_group_attributes() -> list[AttributeSpec]:
    """Return the list of required attributes for tag groups"""
    return [
        AttributeSpec("Description", str),
        AttributeSpec("Tags", (np.ndarray, str)),
    ]


def _check_tags(
    tags: set[str], *, validation_info: dict[str, Any], prefix: str
) -> set[str]:
    included_tags: dict[str, None] = {}
    validation_tags = tags.union(validation_info["tags"])
    for tag in tags:
        tagspec = tag_defs.get(tag)
        if tagspec is None:
            LOGGER.debug(f"Cannot check unknown tag: {tag}")
            continue
        includes = tagspec.includes
        for intag, level in includes.items():
            intag = intag.lower()
            match level:
                case INCLUDE_LEVEL.ABOVE:
                    if intag not in validation_tags:
                        raise TagMissing(
                            f"Tag {tag} requires {intag} in this particle group,"
                            " but it's not included\n"
                            f"Tags above: {validation_tags}"
                        )
                case INCLUDE_LEVEL.EQUAL:
                    included_tags[intag] = None
                case _:
                    pass
        included_tags[tag] = None
    LOGGER.debug(f"{prefix}Including tags {set(included_tags) - tags}")
    return set(included_tags)


@dataclass
class TaggedGroupSpec(GroupSpec):
    """
    Group with the tags attribute

    This definition is recursive. All subgroups and contained datasets must be
    fulfil TaggedGroupSpec and ParticleDatasetSpec, respectively
    If a group has tags, we also need to verify that all of it's tags are valid
    """

    def __post_init__(self):
        super().__post_init__()
        self.attrs.extend(required_tag_group_attributes())

    def validate_tags(
        self,
        obj: h5py.Group,
        tags: set[str],
        *,
        validation_info: dict[str, Any],
        prefix: str | None = None,
    ) -> Literal[True]:
        """Validate any tags in the Tags attribute"""
        LOGGER.debug(
            f"{prefix}"
            + (f"Validating tags: {tags}" if tags else "No tags to validate")
        )
        for tag_str in tags:
            tag = tag_defs.get(tag_str)
            if tag is None:
                LOGGER.warning(f"Unknown tag {tag_str}, skipping!")
                validation_info["violations"].append(
                    f"Found unknown tag {tag_str} on {obj.name}"
                )
                continue
            tag_defs[tag_str].validate(
                obj, validation_info=validation_info, prefix=prefix
            )
        return True

    def validate(
        self, obj: h5obj, *, validation_info: dict[str, Any], prefix: str | None = None
    ) -> Literal[True]:
        prefix = "" if prefix is None else prefix
        super().validate(obj, validation_info=validation_info, prefix=prefix)
        # can't use assert_type below because logic is too complicated
        assert isinstance(obj, h5py.Group)
        # We've already validated the Tags attribute exists
        tags = {tag.lower() for tag in cast(list[str], obj.attrs["Tags"])}
        _check_tags(tags, validation_info=validation_info, prefix=prefix)
        validation_info["tags"].update(tags)
        for key, child in obj.items():
            if isinstance(child, h5py.Dataset):
                child_spec = ParticleDatasetSpec(
                    key,
                    # just use the actual size
                    child.shape[1] if child.ndim > 1 else 1,
                )

            else:
                child_spec = TaggedGroupSpec(key)
            child_spec.validate(
                child, validation_info=validation_info, prefix=prefix + " "
            )
        self.validate_tags(
            obj, tags, validation_info=validation_info, prefix=prefix + " "
        )
        LOGGER.debug(f"{prefix}Finished validating tagged group {self.name}")
        return True


# Required Groups
header = GroupSpec(
    name="Header",
    attrs=[
        AttributeSpec("Code", str),
        AttributeSpec("Source", str),
        AttributeSpec("Source_file", str),
        AttributeSpec("Particle_names", (np.ndarray, str), sets_info="Particle_names"),
        AttributeSpec(
            "Particle_numbers", (np.ndarray, np.uint64), sets_info="Particle_numbers"
        ),
        AttributeSpec("Initial_mass_table", (np.ndarray, number)),
        AttributeSpec("Bounding_box", np.ndarray, 6),
        AttributeSpec("Time", number),
        AttributeSpec("Run_name", str),
        AttributeSpec("Num_files_per_snapshot", np.int32),
        AttributeSpec("Dimension", integer, sets_info="D"),
        AttributeSpec("Part_type_mapping", num_str_array, 6),
        AttributeSpec("Snapshot_date", str),
        AttributeSpec("Redshift", number),
        # Gadget-2 header fields not already included
        AttributeSpec("NumPart_ThisFile", (np.ndarray, np.uint32), 6),
        AttributeSpec("MassTable", (np.ndarray, np.float64), 6),
        AttributeSpec("Time", float),
        AttributeSpec("Flag_Sfr", np.int32),
        AttributeSpec("Flag_Feedback", np.int32),
        AttributeSpec("NumPart_Total", (np.ndarray, np.int32), 6),
        AttributeSpec("Flag_Cooling", np.int32),
        AttributeSpec("BoxSize", float),
        AttributeSpec("Omega0", float),
        AttributeSpec("OmegaLambda", float),
        AttributeSpec("HubbleParam", float),
        AttributeSpec("Flag_StellarAge", np.int32),
        AttributeSpec("Flag_Metals", np.int32),
        AttributeSpec("NumPart_Total_HW", (np.ndarray, np.int32), 6),
        AttributeSpec("Flag_Entropy_ICs", np.int32),
    ],
)

code = GroupSpec(
    name="Code",
    attrs=[
        AttributeSpec("Code", str),
        AttributeSpec("Code_version", str),
        AttributeSpec("Code_hash", str),
        AttributeSpec("HDF5_version", str),
        AttributeSpec("VCS", str),
        AttributeSpec("VCS_id", str),
        AttributeSpec("VCS_repository", str),
        AttributeSpec("Library_versions", (np.ndarray, str)),
        AttributeSpec("Configuration_options", str),
    ],
)

# Note: this does not handle the case where Is_cosmological is True, in which
# case Omega_matter, Omega_lambda, h, Redshift, and Scale_factor are required
cosmo = GroupSpec(
    name="Cosmology",
    attrs=[
        AttributeSpec("Is_cosmological", number),
        OptionalAttributeSpec("Omega_matter", number),
        OptionalAttributeSpec("Omega_lambda", number),
        OptionalAttributeSpec("h", number),
        OptionalAttributeSpec("Redshift", number),
        OptionalAttributeSpec("Scale_factor", number),
        OptionalAttributeSpec("H", number),
        OptionalAttributeSpec("H0", number),
        OptionalAttributeSpec("Critical_density", number),
        OptionalAttributeSpec("Critical_density_at_z0", number),
        OptionalAttributeSpec("Hubble_time", number),
        OptionalAttributeSpec("Lookback_time", number),
        OptionalAttributeSpec("Mean_baryon_density", number),
        OptionalAttributeSpec("Mean_matter_density", number),
        OptionalAttributeSpec("N_eff", number),
        OptionalAttributeSpec("Omega_baryon", number),
        OptionalAttributeSpec("Omega_darkmatter", number),
        OptionalAttributeSpec("Omega_radiation", number),
        OptionalAttributeSpec("Omega_curvature", number),
        OptionalAttributeSpec("T_CMB_0_SI", number),
        OptionalAttributeSpec("T_CMB_0", number),
        OptionalAttributeSpec("Universe_age", number),
        OptionalAttributeSpec("a_begin", number),
        OptionalAttributeSpec("a_end", number),
        OptionalAttributeSpec("Time_begin", number),
        OptionalAttributeSpec("Time_end", number),
        OptionalAttributeSpec("w", number),
        OptionalAttributeSpec("w_0", number),
        OptionalAttributeSpec("w_a", number),
    ],
)

runinfo = GroupSpec(
    name="RunInfo",
    attrs=[
        AttributeSpec("System", str),
        AttributeSpec("Snapshot_date", str),
        AttributeSpec("Total_runtime", number),
        AttributeSpec("Session_runtime", number),
    ],
    datagroups=[
        GroupSpec(name="Parameters"),
        GroupSpec(
            name="ParameterFiles",
            attrs=[
                AttributeSpec("Parameter_file", str),
                AttributeSpec("Parameter_file_path", str),
            ],
        ),
    ],
)

units = GroupSpec(
    name="Units",
    attrs=[
        AttributeSpec("Unit_length_CGS", number),
        AttributeSpec("Unit_mass_CGS", number),
        AttributeSpec("Unit_time_CGS", number),
        AttributeSpec("Unit_temperature_CGS", number),
        AttributeSpec("Unit_current_CGS", number),
    ],
)

# Optional Groups
cubes = GroupSpec(
    name="cubes",
    # TODO: need a way to specify "name must match a particle group"
    optional=True,
)


# Particle Tags
accel = TagSpec(
    name="Accel",
    datagroups=[ParticleDatasetSpec(name="Accelerations", m="D")],
)

agb = TagSpec(name="AGB", datagroups=[TaggedGroupSpec(name="AGB")])

agn = TagSpec(name="AGN", datagroups=[TaggedGroupSpec(name="AGN")])

ags = TagSpec(
    name="AGS",
    includes={"Massive": ABOVE},
    datagroups=[ParticleDatasetSpec(name="AGS_Softenings")],
)

black_hole = TagSpec(
    name="Black-hole",
    includes={"Massive": ABOVE},
    datagroups=[
        TaggedGroupSpec(
            name="BH",
        )
    ],
)

feedback = TagSpec(
    name="Feedback",
    includes={"AGN": BELOP, "SNII": BELOP, "SNIa": BELOP},
    datagroups=[
        TaggedGroupSpec(
            name="Feedback",
        )
    ],
)

group = TagSpec(name="Group", datagroups=[ParticleDatasetSpec(name="GroupIDs")])

hydro = TagSpec(
    name="Hydro",
    includes={"Massive": ABOVE},
    datagroups=[
        ParticleDatasetSpec(name="InternalEnergies"),
        ParticleDatasetSpec(name="Densities"),
        ParticleDatasetSpec(name="KernelLengths"),
    ],
)

magnetic = TagSpec(
    name="Magnetic",
    datagroups=[
        ParticleDatasetSpec(name="MagneticFields", m="D"),
        ParticleDatasetSpec(name="MagneticFieldDivergences"),
    ],
)

massive = TagSpec(name="Massive", datagroups=[ParticleDatasetSpec(name="Masses")])

potential = TagSpec(
    name="Potential",
    includes={"Massive": ABOVE},
    datagroups=[ParticleDatasetSpec(name="Potentials")],
)

sink_stars = TagSpec(
    name="Sink-stars",
    includes={"Massive": EQUAL, "Merge-Split": BELOW, "Transform": EQUAL},
)

snia = TagSpec(name="SNIa", datagroups=[TaggedGroupSpec(name="SNIa")])

snii = TagSpec(name="SNII", datagroups=[TaggedGroupSpec(name="SNII")])

sph = TagSpec(
    name="SPH",
    includes={"Hydro": EQUAL},
    datagroups=[ParticleDatasetSpec(name="ArtificialViscosities")],
)

merge_split = TagSpec(
    name="Merge-split",
    datagroups=[
        ParticleDatasetSpec(name="SplitCounts"),
        ParticleDatasetSpec(name="SplitTrees"),
        ParticleDatasetSpec(name="ProgenitorParticleIDs"),
    ],
)

stellar_wind = TagSpec(
    name="StellarWind",
    includes={"AGN": BELOP, "SNII": BELOP, "SNIa": BELOP},
    datagroups=[TaggedGroupSpec(name="StellarWind")],
)

subfind = TagSpec(
    name="Subfind",
    includes={"Massive": ABOVE},
    attrs=[
        AttributeSpec(name="Num_nearest", attr_type=number, sets_info="NumNearest"),
        AttributeSpec(
            name="Dark_matter_species",
            attr_type=num_str_array,
            sets_info="Subfind_dark_matter_species",
        ),
    ],
    datagroups=[
        TaggedGroupSpec(
            name="Subfind",
            datagroups=[
                ParticleDatasetSpec(name="DMDensities"),
                ParticleDatasetSpec(name="Densities"),
                ParticleDatasetSpec(name="Hsml"),
                ParticleDatasetSpec(name="VelocityDispersions"),
            ],
        )
    ],
)

subgrid = TagSpec(name="Subgrid", datagroups=[TaggedGroupSpec(name="Subgrid")])

thermal = TagSpec(
    name="Thermal",
    datagroups=[
        TaggedGroupSpec(
            name="Thermal", datagroups=[ParticleDatasetSpec(name="Temperatures")]
        )
    ],
)

transform = TagSpec(
    name="Transform",
    datagroups=[
        TaggedGroupSpec(name=RegexName("FromParent_{pg}")),
        OptionalValidationGroup(
            name="TransformTimeGroup",
            num_required=1,
            options=[
                ParticleDatasetSpec(name="TransformTimes"),
                ParticleDatasetSpec(name="TransformScaleFactors", optional=True),
            ],
        ),
        ParticleDatasetSpec(
            name="PreviousParticleTypes",
        ),
    ],
)

## chemistry
number_density = TagSpec(
    name="NumberDensity",
    datagroups=[TaggedGroupSpec(name="NumberDensities")],
)

mass_density = TagSpec(
    name="MassDensity",
    includes={"Massive": ABOVE},
    datagroups=[TaggedGroupSpec(name="MassDensities")],
)

abundances = TagSpec(
    name="Abundance",
    datagroups=[TaggedGroupSpec(name="Abundances")],
)

mass_fraction = TagSpec(
    name="MassFraction",
    includes={"Massive": ABOVE},
    datagroups=[TaggedGroupSpec(name="MassFractions")],
)

hydrogen = ChemicalTagSpec(
    name="Hydrogen",
    group="Hydrogen",
    datagroups=[
        ParticleDatasetSpec(name="H"),
        ParticleDatasetSpec(name="Hp"),
        ParticleDatasetSpec(name="E"),
        ParticleDatasetSpec(name="H2"),
    ],
)

helium = ChemicalTagSpec(
    name="Helium",
    group="Helium",
    datagroups=[
        ParticleDatasetSpec(name="He"),
        ParticleDatasetSpec(name="Hep"),
        ParticleDatasetSpec(name="Hepp"),
    ],
)

hydrogen_extended = ChemicalTagSpec(
    name="ExtendedHydrogen",
    group="Hydrogen",
    datagroups=[
        ParticleDatasetSpec(name="Hm"),
        ParticleDatasetSpec(name="D"),
        ParticleDatasetSpec(name="Dp"),
        ParticleDatasetSpec(name="Dm"),
        ParticleDatasetSpec(name="H2p"),
        ParticleDatasetSpec(name="HD"),
        ParticleDatasetSpec(name="HDp"),
        ParticleDatasetSpec(name="HDm"),
        ParticleDatasetSpec(name="H3p"),
        ParticleDatasetSpec(name="H2Dp"),
    ],
)

metal_only_denom = AttributeSpec(
    name="Metal_only_denominator",
    attr_type=number,
)

metallic = ChemicalTagSpec(
    name="Metallic",
    group="Metals",
    extra_attrs={"abundance": [metal_only_denom], "massfraction": [metal_only_denom]},
    extra_datagroups={
        "abundance": [
            ParticleDatasetSpec(
                name="TotalAbundances",
                optional=True,
            )
        ],
        "massfraction": [ParticleDatasetSpec(name="TotalMasses", optional=True)],
    },
)

## tag_defs

tag_defs.update(
    {
        tag.name.lower(): tag
        for tag in [
            accel,
            agb,
            agn,
            ags,
            black_hole,
            feedback,
            group,
            hydro,
            magnetic,
            massive,
            potential,
            sink_stars,
            snia,
            snii,
            sph,
            merge_split,
            stellar_wind,
            subfind,
            subgrid,
            thermal,
            transform,
        ]
    }
)

chemical_quantity_tags = {
    tag.name: tag for tag in [abundances, mass_density, mass_fraction, number_density]
}
chemical_type_tags = {tag.name.lower(): tag for tag in [hydrogen, helium, metallic]}

tag_defs.update({k.lower(): v for k, v in chemical_quantity_tags.items()})
tag_defs.update(chemical_type_tags)

ChemicalTagSpec._QUANTITIES.update(
    [t.datagroups[0].name for _, t in chemical_quantity_tags.items()]
)

# Resume specification object definitions


# Particle Groups
def required_particle_group_attributes() -> list[AttributeSpec]:
    """Return the list of required attributes for particle groups"""
    return [
        AttributeSpec("Aliases", (np.ndarray, str)),
        AttributeSpec("Number_of_fields", integer),
        OptionalAttributeSpec("Number_of_particles", integer),
        # Already included by subclassing TaggedGroup
        # AttributeSpec("Tags", (np.ndarray, str)),
        # AttributeSpec("Description", str),
    ]


@dataclass
class ParticleGroupSpec(TaggedGroupSpec):
    """
    Particle Group definition

    Particle groups are a little unusual. They themselves can have an
    **arbitrary** name, _but_ they must be top-level.
    Plus, a valid particle group must have a tags attribute and every
    attribute/dataset/subgroup specified by a tag must be present
    """

    _: KW_ONLY

    n: int
    """The number of particles in this group"""

    def __post_init__(self):
        super().__post_init__()
        self.attrs.extend(required_particle_group_attributes())

    def validate(
        self, obj: h5obj, *, validation_info: dict[str, Any], prefix: str | None = None
    ) -> Literal[True]:
        """Validate that provided object is a valid particle group at the top level"""
        validation_info["N"] = self.n
        if obj.parent.name != "/":
            raise ValidationError("Particle groups must be top-level!")
        super().validate(obj, validation_info=validation_info, prefix=prefix)
        return True


def _log_violations(violations: list[str]):
    if violations:
        LOGGER.warning("Encountered non-critical spec violations:")
        for violation in violations:
            LOGGER.warning(f" - {violation}")


class GUSTEAU:
    required_tlgs: dict[str, GroupSpec]
    optional_tlgs: dict[str, GroupSpec]

    def __init__(self):
        self.required_tlgs = {
            tlg.name: tlg for tlg in [header, code, cosmo, runinfo, units]
        }
        self.optional_tlgs = {tlg.name: tlg for tlg in [cubes]}

    def _validate_metadata_attribute(
        self, h5_file: h5py.File, *args, validation_info: dict[str, Any], **kwargs
    ) -> Literal[True]:
        """
        Special function to validate the /.Metadata attribute

        Since .Metadata needs to be a JSON formatted string, we'll check that's the case
        In the future, we may also check the _contents_, since we shouldn't
        need to process anything for that
        """

        metadata_spec = AttributeSpec("Metadata", attr_type=str)
        metadata_spec.validate(h5_file, validation_info=validation_info, prefix="")

        metadata = h5_file.attrs["Metadata"]
        try:
            json.loads(metadata)
        except json.JSONDecodeError as je:
            raise ValidationError("Metadata is not valid JSON") from je

        return True

    def validate(
        self, h5_file: h5py.File, *, options: ValidationOptions | None = None
    ) -> Literal[True]:
        """
        Validate the provided h5py File as a GUSTEAU snapshot
        """
        options = ValidationOptions() if options is None else options
        validation_info: dict[str, Any] = {}
        validation_info["violations"] = []
        visited_groups: set[str] = set(self.required_tlgs.keys())
        visited_groups.update(self.optional_tlgs)

        # validate required groups
        LOGGER.info("Validating required groups")
        for tlg_name, tlg in self.required_tlgs.items():
            if tlg_name not in h5_file:
                raise GroupMissing(f"Missing required group {tlg_name}!")
            LOGGER.debug(f"Validating {tlg_name}")
            tlg.validate(h5_file[tlg_name], validation_info=validation_info)

        # Create the particle name and number mapping. Note that these fields are
        # set by the header, so if the header was validated correctly they must
        # exist
        particle_names = cast(list[str], validation_info.get("Particle_names"))
        particle_numbers = cast(list[int], validation_info.get("Particle_numbers"))
        particle_numbers = (
            [1 for _ in particle_numbers] if options.skip_data else particle_numbers
        )
        number_map = dict(zip(particle_names, particle_numbers, strict=True))

        # validate any mentioned particle groups
        LOGGER.info("Validating particle groups")
        for ptype in particle_names:
            if ptype not in h5_file:
                raise GroupMissing(
                    f"Defined particle type {ptype} but not found in file"
                )
            pgroup_spec = ParticleGroupSpec(ptype, n=number_map[ptype])
            validation_info["tags"] = set()
            pgroup_spec.validate(h5_file[ptype], validation_info=validation_info)
            visited_groups.add(ptype)

        # validate optional top-level groups
        LOGGER.info("Validating optional groups")
        for tlg_name, tlg in self.optional_tlgs.items():
            if tlg_name in h5_file:
                LOGGER.debug(f"Validating {tlg_name}")
                tlg.validate(h5_file[tlg_name], validation_info=validation_info)

        # Log groups that were not validated
        # skip links
        links: set[str] = set()

        def _count_duplicate_links(name: str, obj):
            if not isinstance(obj, h5py.HardLink):
                LOGGER.debug(f"{name} is a SoftLink")
                links.add(name)

        h5_file.visititems_links(_count_duplicate_links)
        LOGGER.debug(f"Total Soft/External Links: {len(links)}")
        not_visited = cast(set[str], h5_file.keys()) - visited_groups - links
        if not_visited:
            LOGGER.info(f"Did not validate: {not_visited}")

        self._validate_metadata_attribute(
            h5_file, validation_info=validation_info, options=options
        )

        _log_violations(validation_info["violations"])

        return True


def get_spec_from_version(version: str | None = None) -> GUSTEAU:
    return GUSTEAU()


def validate_spec(filepath: str | Path) -> Literal[True]:
    """
    Validate an HDF5 file matches the GUSTEAU specification

    Parameters
    ----------
    h5file:
        The path to the HDF5 file to validate

    Returns
    -------
    :
        True if and only if the specified file is a valid GUSTEAU snapshot

    Raises
    ------
    ValidationError
        If the specified file is not a valid GUSTEAU snapshot
    """
    spec = GUSTEAU()

    with h5py.File(
        filepath,
    ) as h5_file:
        return spec.validate(h5_file)

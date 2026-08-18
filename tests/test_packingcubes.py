"""
Module to compare the packed, GUSTEAU snapshot with the original
"""

import logging
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import packingcubes
import packingcubes.bounding_box

from gusteau_translate.config import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


class GUSTEAUDataset(packingcubes.data_objects.HDF5Dataset):
    """GUSTEAU Formatted HDF5 dataset

    Represents an HDF5 dataset with fields in the
    [GUSTEAU specification](gusteau-spec.readthedocs.io)
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        filepath: str | Path,
        particle_type: str | None = None,
        data_slices=None,
        **kwargs,
    ):
        """Initialize a GUSTEAU-type HDF5Dataset

        Parameters
        ----------
        filepath: str | Path
            The path to the file

        name: str, optional
            A name for this dataset. Defaults to filepath

        particle_type: str, optional
            Initial particle type to (eagerly) load. Defaults to the first
            particle type listed in /Header.Particle_names

        data_slices: np.s_ | dict[str, np.s_], optional
            A numpy slice object or dictionary of slice objects per particle
            type. This can be used to load only a portion of the dataset.
            Effectively, the dataset will be loaded as
            `data = data[data_slice[0]:data_slice[1]:data_slice[3]]`

        **kwargs:
            Additional arguments are discarded
        """
        super().__init__(
            name=name,
            filepath=filepath,
            particle_type=particle_type,
            data_slices=data_slices,
        )

    def _preload(
        self,
        sorted_filepath: str | Path | None,
        particle_type: str | None = None,
    ):

        self._positions_field = "Coordinates"
        with h5py.File(self.filepath) as file:
            self._header = dict(file["Header"].attrs)
            self._top_level_groups = [g.lower() for g in file]

        self._particle_types = self._header["Particle_names"]

        self._sorted_file_name = self.filepath

        if self._sorted_file_name.exists() and not h5py.is_hdf5(self._sorted_file_name):
            raise packingcubes.data_objects.DatasetError(
                f"{self._sorted_file_name} already exists but is not an hdf5 file!",
            )

        # set initial particle type
        self._particle_type = (
            self._particle_types[0] if particle_type is None else particle_type
        )
        _particle_numbers = self._header["Particle_numbers"]
        self._particle_numbers = dict(
            zip(self._particle_types, _particle_numbers, strict=True)
        )

    def _set_bounding_box(self):
        boundingbox = np.atleast_1d(self._header["Bounding_box"])
        match len(boundingbox):
            case 2:
                boxSize = boundingbox[1]
                box = [boundingbox[0], 0, 0, boxSize, boxSize, boxSize]
            case 4:
                boxSize = boundingbox[2]
                box = [
                    boundingbox[0],
                    boundingbox[1],
                    0,
                    boundingbox[2],
                    boundingbox[3],
                    boxSize,
                ]
            case 6:
                box = boundingbox
            case _:
                raise packingcubes.data_objects.DatasetError(
                    "Header field BoundingBox is in violation of the GUSTEAU spec!"
                )
        self._box = packingcubes.bounding_box.make_bounding_box(box)


def initialize_gusteau_ds(
    gusteau_file: str | Path, particle_type: str
) -> GUSTEAUDataset:
    LOGGER.debug("Initializing GUSTEAUDataset")
    ds = GUSTEAUDataset(filepath=gusteau_file, particle_type=particle_type)
    LOGGER.debug("Processing extra fields")
    ds.process_extra_fields(pids="ParticleIDs")
    return ds


def test_conversion(
    original_file: str | Path,
    gusteau_file: str | Path,
    *,
    coords_field: str | None = None,  # noqa: PT028
    id_field: str | None = None,  # noqa: PT028
    random_seed: int | None = None,  # noqa: PT028
    num_slices: int | None = None,  # noqa: PT028
    sample_percentage: int | None = None,  # noqa: PT028
):

    coords_field = "Coordinates" if coords_field is None else coords_field
    id_field = "ParticleIDs" if id_field is None else id_field

    rng = np.random.default_rng(0xBABBADE if random_seed is None else random_seed)
    num_slices = 10 if num_slices is None else num_slices
    sample_percentage = 1 if sample_percentage is None else sample_percentage

    gusteau_ds = None

    with h5py.File(original_file) as original, h5py.File(gusteau_file) as gusteau:
        pgs = cast(list[str], gusteau["/Header"].attrs["ParticleNames"])
        pns = cast(np.ndarray, gusteau["/Header"].attrs["ParticleNumbers"])

        for pg, pn in zip(pgs[:1], pns[:1], strict=True):
            LOGGER.debug(f"Checking {pg} with {pn} particles")
            gusteau_ds = (
                initialize_gusteau_ds(gusteau_file, pg)
                if gusteau_ds is None
                else gusteau_ds
            )
            LOGGER.debug("Loading cubes")
            cubes = packingcubes.Cubes(
                dataset=gusteau_ds,
                particle_type=pg,
            )

            # we will only check the particle ids field. nothing special
            # happens to it in processing, so if the packingcubes part
            # screws up, it should affect particle ids
            if not hasattr(gusteau_ds, "pids"):
                raise AttributeError("Failed to load gusteau ParticleIDs")

            gpids = cast(np.ndarray, gusteau_ds.pids)

            # assuming num_slices = 5, sample_percentage=20
            # want a slice no bigger than 4%, but at least 10 if possible
            slice_size = int(
                np.floor(
                    np.maximum(
                        (sample_percentage / 100) / num_slices * pn, np.minimum(10, pn)
                    )
                )
            )
            # slice_size = 100

            possible_starts = np.arange(
                0, pn - slice_size, step=slice_size, dtype=np.int_
            )
            slice_starts = rng.choice(possible_starts, size=num_slices, replace=False)
            del possible_starts

            opg = gusteau[f"/{pg}"].attrs["Source_Description"]

            break_everything = False

            for slice_start in slice_starts:
                # slice_start = 0
                LOGGER.debug(f"Checking slice {slice_start}:{slice_start + slice_size}")
                coords = original[f"/{opg}/{coords_field}"][
                    slice_start : slice_start + slice_size
                ]
                pids = original[f"/{opg}/{id_field}"][
                    slice_start : slice_start + slice_size
                ]

                for i, (coord, pid) in enumerate(zip(coords, pids, strict=True)):
                    dist, index = cubes.get_closest_particles(xyz=coord, k=1)
                    # dist = dist[0]
                    # index = index[0]
                    LOGGER.debug(
                        f"Closest particle to particle {i + slice_start}"
                        f" is {dist[0]:.4g} away at index {index[0]}"
                    )
                    if pid == gpids[index[0]]:
                        LOGGER.debug(f"PIDs {pid} matched")
                    else:
                        LOGGER.warning(
                            f"PID mismatch! original: {pid}, gusteau: {gpids[index[0]]}"
                        )

                        actual = np.where(gpids == pid)[0][0]
                        actual_coords = gusteau_ds.positions[actual, :]

                        actual_dist = np.sqrt(np.sum((coord - actual_coords) ** 2))
                        LOGGER.debug(f"{actual=} {actual_dist=}")

                        break_everything = True
                        break
                if break_everything:
                    break
            if break_everything:
                break


if __name__ == "__main__":
    logging.basicConfig()
    LOGGER.setLevel(logging.DEBUG)
    original_file = "tests/snapshot_090.hdf5"
    gusteau_file = "tests/gusteau.hdf5"
    test_conversion(
        original_file,
        gusteau_file,
    )

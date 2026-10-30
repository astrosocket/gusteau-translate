# Module for working with the cell structures and virtual datasets


import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import cast

import h5py
import numpy as np
from pydantic import NonNegativeInt, PositiveInt, computed_field

from .config import LOGGER_NAME, TranslationOptions

LOGGER = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class _SubDataset:
    filename: Path
    shape: tuple[int, ...]


def _create_virtual_dataset(
    virtual: h5py.File, dataset_path: str, file_list: list[Path]
):
    """Create a virtual dataset in virtual from the dataset_path in each file

    Parameters
    ----------
    virtual: h5py.File
        The file that will contain the virtual dataset
    dataset_path: Path
        The path to the dataset in each file.
        Must be identical for all files in the list.
    file_list: list[Path]
        List of paths to the hdf5 files that contain the subdatasets to be combined.
    """
    # Process has 5 steps
    # 1. Determine the final dimensions and data type of the virtual dataset
    # 2. Create a VirtualLayout object representing the virtual dataset
    # 3. Create VirtualSource objects representing each of the subdatasets
    # 4. Map the VirtualSource objects into slices in the VirtualLayout
    # 5. Convert the VirtualLayout into a virtual dataset
    # We will actually do the steps out of order because we don't know a priori
    # the dimensions of the virtual dataset. What we do know is that all of the
    # datasets are effectively vertically stacked, i.e. in the 0th dimension.
    # So we'll start with step 3, reading the metadata of every subdataset to
    # determine that subdatasets shape and the final dataset's size
    subdatasets = []
    dtype = float
    for filename in file_list:
        filename = filename.resolve()
        with h5py.File(str(filename), "r") as subfile:
            shape = subfile[dataset_path].shape
            subdatasets.append(_SubDataset(filename, shape))
            dtype = subfile[dataset_path].dtype
    total_length = sum(s.shape[0] for s in subdatasets)
    shape = (total_length, *subdatasets[0].shape[1:])
    LOGGER.debug(
        f"Creating virtual layout for {dataset_path} with {dtype=} and {shape=}"
    )
    layout = h5py.VirtualLayout(
        shape,
        dtype,
    )
    offset = 0
    for subds in subdatasets:
        layout[offset : offset + subds.shape[0], ...] = h5py.VirtualSource(
            subds.filename, dataset_path, shape=subds.shape
        )
        offset += subds.shape[0]
    LOGGER.info(f"Creating virtual dataset {dataset_path}")
    virtual.create_virtual_dataset(dataset_path, layout, fillvalue=-57)


class REPLACEMENT_TYPE(Enum):
    NONE = auto()
    """No changes are necessary"""
    SCALING = auto()
    """Cell grid is correct but needs to be scaled by bounding box"""
    OVERFLOW = auto()
    """Grid is manually provided, but the overflow cell needs to be updated"""
    FULL = auto()
    """The entire cells structure needs to be replaced"""


@dataclass
class Cells:
    Grid_box: np.ndarray
    """Bounding box of the grid"""
    BoundingBoxes: np.ndarray
    """Cx6 array describing the bounding boxes of each cell (including nongridded)"""
    Centres: np.ndarray
    """Cx3 array describing the centerpoints of each cell (including nongridded)"""
    dimension: np.ndarray
    """Number of cells in each dimension as D-length array"""
    size: np.ndarray
    """D-length array describing the size of a cube"""
    needs_updating: REPLACEMENT_TYPE
    """
    Whether we need to replace some or all of the cell structure
    """
    Indices: dict[str, np.ndarray]
    """
    Indices of each cell into the virtual data set (*not* OffsetsInFile!)
    """
    Number_of_grid_cells: PositiveInt = 1
    Number_of_nongrid_cells: NonNegativeInt = 0

    @computed_field
    @property
    def nr_cells(self) -> PositiveInt:
        """Total number of cells"""
        return self.Number_of_grid_cells + self.Number_of_nongrid_cells


def _get_presorted_indices(particle_group: str, infile: h5py.File) -> np.ndarray:
    """Get already present cell indices"""

    try:
        return infile[f"/Cells/Indices/{particle_group}"]
    except KeyError as ke:
        LOGGER.debug("Indices not present in input file. Building from Offsets")

    # We already have the cell indices in
    # infile["/Cells/OffsetsInFile/{particle_group}"], but they're not
    # usable directly (since that's on a per-file basis)
    # We need to convert to indices into the vds
    files = infile[f"/Cells/Files/{particle_group}"]
    offsets = infile[f"/Cells/OffsetsInFile/{particle_group}"]
    old_counts = infile[f"/Cells/Counts/{particle_group}"]

    num_files = max(files)

    counts = [[] for i in range(num_files)]
    for cell in range(len(offsets)):
        file = files[cell]
        counts[file].append(old_counts[cell])
    file_offset = [0, *(int(sum(c)) for c in counts[:-1])]
    return np.array(
        [
            offset + file_offset[file_num]
            for offset, file_num in zip(files, offsets, strict=True)
        ]
    )


def _parse_swift_cell_structure(infile: h5py.File) -> Cells | None:
    """Parse infile for a Swift-style cells structure

    A Swift-like cells structure has the following:
    ```
    cells/
      # a Cx3 array of cell centers
      Centres

      # Group containing the number of particles per particle type per cell
      Counts/
        PartType0
        PartType1
        ...

      # Group containing the file numbers of each cell
      Files/
        PartType0
        PartType1
        ...

      # Group containing the max position of particles per particle type per cell
      MaxPositions/

      # Group containing the min position of particles per particle type per cell
      MinPositions/

      # Group containing the file offsets of particles per particle type per cell
      OffsetsInfile/
    ```

    Parameters
    ----------
    infile:
        The file to check/parse for cell information

    Returns
    -------
    cells: Cells | None
        The cells structure of the file if present otherwise None


    """
    if "Cells" not in infile:
        return None

    cell_group = cast(h5py.Group, infile["Cells"])

    try:
        attrs = cast(h5py.AttributeManager, cell_group["Meta-data"].attrs)
        dimension = attrs["dimension"]
        nr_cells = attrs["nr_cells"]
        size = attrs["size"]
        Centres = cell_group["Centres"]
    except KeyError as ke:
        LOGGER.debug(ke)
        return None

    # check groups present
    for group in ["Counts", "Files", "MaxPositions", "MinPositions", "OffsetsInFile"]:
        if group not in cell_group:
            return None

    Grid_box = np.zeros((6,))
    Grid_box[3:] += dimension * size
    Number_of_grid_cells = np.prod(dimension)
    if Number_of_grid_cells != nr_cells:
        LOGGER.warning(
            "Mismatch between computed number of cells"
            f" ({Number_of_grid_cells}) and provided ({nr_cells})."
            "\nUsing computed."
        )
    Number_of_nongrid_cells = 0

    half_size = cast(np.ndarray, size / 2)
    BoundingBoxes = np.hstack((Centres, Centres))
    BoundingBoxes[:, :3] -= half_size
    BoundingBoxes[:, 3:] = size

    Indices = {str(k): _get_presorted_indices(k, infile) for k in cell_group["Counts"]}

    return Cells(
        Grid_box=Grid_box,
        Number_of_grid_cells=Number_of_grid_cells,
        Number_of_nongrid_cells=Number_of_nongrid_cells,
        BoundingBoxes=BoundingBoxes,
        Centres=Centres,
        dimension=dimension,
        size=size,
        needs_updating=REPLACEMENT_TYPE.NONE,
        Indices=Indices,
    )


# TODO: Add generic cell structure parser
# Can base it off of CubesTranslator structure:
# def _set_cubes_translator(translation: Translation, context: TranslationContext):
#     gusteau = translation.gusteau
#     cubes_translator_name = gusteau.get("cubes_translator")
#     if cubes_translator_name:
#         LOGGER.debug(f"Trying cubes translator: {cubes_translator_name}")
#         cubes_translator = CubesTranslator._subclasses.get(cubes_translator_name)
#         if cubes_translator:
#             LOGGER.debug(f"Setting cubes translator: {cubes_translator_name}")
#             context.cubes_translator = cubes_translator()


def parse_cell_structure(infile: h5py.File) -> Cells | None:
    """Attempt to parse the cell structure for various structure types"""
    for methods in [_parse_swift_cell_structure]:
        cells = methods(infile)
        if cells:
            return cells
    return None


def _make_single_cell() -> Cells:
    """Make cell structure that is only a single cell

    Generated cell structure is 3D.
    This may need to be fixed later.

    """

    Grid_box = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    Number_of_grid_cells = 1
    Number_of_nongrid_cells = 0

    BoundingBoxes = np.atleast_2d(Grid_box.copy())
    Centres = np.array([[0.5, 0.5, 0.5]])
    dimension = np.array([1, 1, 1])
    size = np.ones((3,), dtype=float)

    return Cells(
        Grid_box=Grid_box,
        Number_of_grid_cells=Number_of_grid_cells,
        Number_of_nongrid_cells=Number_of_nongrid_cells,
        BoundingBoxes=BoundingBoxes,
        Centres=Centres,
        dimension=dimension,
        size=size,
        needs_updating=REPLACEMENT_TYPE.SCALING,
        Indices={},  # Since there's only one cell anyway...
    )


def _make_packingcubes_cells(options: TranslationOptions) -> Cells:
    """Make cell structure that is cube with N cells on a side

    Uses the grid box if available, but note that generated cell structure is 3D.
    This may need to be fixed later.
    """

    if options.pc_options.box:
        pc_box = options.pc_options.box
        d_box = len(pc_box) // 2
        if d_box != len(pc_box) / 2:
            raise ValueError(f"Dimension {d_box} of packingcubes box makes no sense!")
        Grid_box = np.hstack(
            (
                np.pad(pc_box[:d_box], (0, 3 - d_box)),
                np.pad(pc_box[d_box:], (0, 3 - d_box), constant_values=1),
            )
        )
        needs_updating = REPLACEMENT_TYPE.OVERFLOW
    else:
        Grid_box = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        needs_updating = REPLACEMENT_TYPE.SCALING
    n = options.pc_options.side_length
    Number_of_grid_cells = n**3
    Number_of_nongrid_cells = 1
    nr_cells = Number_of_grid_cells + Number_of_nongrid_cells

    size = np.ones((3,), dtype=float) / n * Grid_box[3:]
    half_size = size / 2

    BoundingBoxes = np.empty((nr_cells, 6), dtype=float)
    Centres = np.empty((nr_cells, 3), dtype=float)
    index = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x = i * size[0] + Grid_box[0]
                y = j * size[1] + Grid_box[1]
                z = k * size[2] + Grid_box[2]
                BoundingBoxes[index, 0] = x
                BoundingBoxes[index, 1] = y
                BoundingBoxes[index, 2] = z
                BoundingBoxes[index, 3] = size[0]
                BoundingBoxes[index, 4] = size[1]
                BoundingBoxes[index, 5] = size[2]
                Centres[index, 0] = x + half_size[0]
                Centres[index, 1] = y + half_size[1]
                Centres[index, 2] = z + half_size[2]
                index += 1
    # Setup overflow box
    BoundingBoxes[-1, :] = [0, 0, 0, 1, 1, 1]
    Centres[-1, :] = [0.5, 0.5, 0.5]

    dimension = np.array([n, n, n])

    return Cells(
        Grid_box=Grid_box,
        Number_of_grid_cells=Number_of_grid_cells,
        Number_of_nongrid_cells=Number_of_nongrid_cells,
        BoundingBoxes=BoundingBoxes,
        Centres=Centres,
        dimension=dimension,
        size=size,
        needs_updating=needs_updating,
        Indices={},
    )


def generate_cells(infile: h5py.File, options: TranslationOptions) -> Cells:
    """Generate the cells structure"""
    cells = _parse_swift_cell_structure(infile)
    if cells and not options.packingcubes:
        return cells

    return (
        _make_packingcubes_cells(options)
        if options.packingcubes
        else _make_single_cell()
    )


def update_cells(cells: Cells, outfile: h5py.File):
    """Update the Cells object based on the value of needs_updating

    Note that this will fix the dimensionality issue for cases where
    `cells.needs_updating != REPLACEMENT_TYPE.NONE`
    """
    header = outfile["/Header"].attrs
    box = header["Bounding_box"]
    D = header["Dimension"]
    if len(box) > 2 * D:
        # Update box in case Bounding_box is 3D but dimension is <3
        D_box = len(box) // 2
        if len(box) / 2 != len(box) / 2:
            raise ValueError(f"Don't understand dimension {len(box) / 2}!")
        box = np.hstack((box[:D], box[D_box : D_box + D]))
    if D < 3 and cells.needs_updating != REPLACEMENT_TYPE.NONE:
        # Trim extra dimensions if needed
        cells.Grid_box = np.hstack((cells.Grid_box[:D], cells.Grid_box[3 : 3 + D]))
        cells.BoundingBoxes = np.hstack(
            (cells.BoundingBoxes[:, :D], cells.BoundingBoxes[:, 3 : 3 + D])
        )
        cells.Centres = np.hstack((cells.Centres[:, :D], cells.Centres[:, 3 : 3 + D]))
        cells.dimension = np.hstack((cells.dimension[:D], cells.dimension[3 : 3 + D]))
        cells.size = np.hstack((cells.size[:D], cells.size[3 : 3 + D]))
    match cells.needs_updating:
        case REPLACEMENT_TYPE.NONE:
            # Nothing to do
            return
        case REPLACEMENT_TYPE.SCALING:
            # Rescale unit grid by bounding box
            cells.Grid_box = box
            cells.BoundingBoxes[:, :D] = cells.BoundingBoxes[:, :D] * box[D:] + box[:D]
            cells.BoundingBoxes[:, D:] *= box[D:]
            cells.Centres[:, :D] = cells.Centres[:, :D] * box[D:] + box[:D]
            cells.size *= box[D:]
        case REPLACEMENT_TYPE.OVERFLOW:
            # Fix overflow cell using bounding box
            cells.BoundingBoxes[-1, :] = box
            cells.Centres[-1, :] = box[:D] + box[D:] / 2
        case REPLACEMENT_TYPE.FULL:
            raise NotImplementedError("Not yet implemented")

    cells.needs_updating = REPLACEMENT_TYPE.NONE


def write_cells(cells: Cells, outfile: h5py.File):
    """Write out the cells structure"""
    cell_group = outfile.require_group("/Cells")
    cell_attrs = cell_group.attrs
    cell_attrs["Grid_box"] = cells.Grid_box
    cell_attrs["Number_of_grid_cells"] = cells.Number_of_grid_cells
    cell_attrs["Number_of_nongrid_cells"] = cells.Number_of_nongrid_cells
    cell_group["BoundingBoxes"] = cells.BoundingBoxes
    cell_group["Centres"] = cells.Centres
    meta = cell_group.require_group("Meta-data")
    meta_attrs = meta.attrs
    meta_attrs["dimension"] = cells.dimension
    meta_attrs["nr_cells"] = cells.nr_cells
    meta_attrs["size"] = cells.size
    counts_grp = cell_group.require_group("Counts")
    counts_grp.attrs["Description"] = "The number of particles of the particular type"
    files_grp = cell_group.require_group("Files")
    files_grp.attrs["Description"] = (
        "The number of the actual file containing the cell's particles."
        " Only relevant if /Header.Num_files_per_snapshot > 1"
    )
    indices_grp = cell_group.require_group("Indices")
    indices_grp.attrs["Description"] = (
        "The overall cell index into the virtual dataset."
    )
    max_grp = cell_group.require_group("MaxPositions")
    max_grp.attrs["Description"] = (
        "Maximum coordinates of the particular particle type for that cell in"
        " simulation units"
    )
    min_grp = cell_group.require_group("MinPositions")
    min_grp.attrs["Description"] = (
        "Minimum coordinates of the particular particle type for that cell in"
        " simulation units"
    )
    off_grp = cell_group.require_group("OffsetsInFile")
    off_grp.attrs["Description"] = (
        "Index to the start of the cell in the file it's stored in"
    )
    if cells.nr_cells == 1:
        header = outfile["/Header"].attrs
        names = header["Particle_names"]
        for i, name in enumerate(names):
            counts_grp[name] = np.atleast_1d(header["Particle_numbers"][i])
            files_grp[name] = np.atleast_1d(0)
            indices_grp[name] = np.atleast_1d(0)
            off_grp[name] = np.atleast_1d(0)


def _gen_partfile_path(snapshot_path: Path, index: int, num_files: int) -> Path:
    """Generate the path for partfile with index"""
    if num_files == 1:
        return snapshot_path
    stem = snapshot_path.stem
    return snapshot_path.with_stem(
        f"{stem}_{index}".zfill(np.ceil(np.log10(num_files), dtype=int))
    )


def create_partfiles(snapshot_path: Path, options: TranslationOptions) -> list[Path]:
    """Create new partfiles"""
    partfile_paths = []
    for i in range(options.num_partfiles):
        partfile_path = _gen_partfile_path(snapshot_path, i, options.num_partfiles)
        with h5py.File(
            partfile_path, "a", libver="latest", track_order=True
        ) as partfile:
            header = partfile.require_group("Header")
            header.attrs["File_index"] = i if options.num_partfiles > 1 else -1
        partfile_paths.append(partfile_path)
    return partfile_paths

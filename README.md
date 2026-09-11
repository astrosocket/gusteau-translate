# GUSTEAU-Translate

[![PyPI](https://img.shields.io/pypi/v/gusteau-translate)](https://pypi.org/project/gusteau-translate)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/gusteau-translate)](https://pypi.org/project/gusteau-translate/)

<!--- Tests and style --->

[![Tested with Hypothesis](https://img.shields.io/badge/hypothesis-tested-brightgreen.svg)](https://hypothesis.readthedocs.io/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)

Demo translation software for Project GUSTEAU

View the documentation at [](https://gusteau-translate.readthedocs.io).

See the GUSTEAU specification at [](https://gusteau-spec.readthedocs.io).

## Requirements

<!-- --8<-- [start:requirements] -->

[![Supported Python Versions](https://img.shields.io/pypi/pyversions/gusteau-tramslate)](https://pypi.org/project/gusteau-translate/) is required.
Python versions outside this range may work, but their usage is not supported, and at least some features are >Python 3.13.

### Python packages

- `numpy` - required for core routines
- `h5py` - required for reading and saving snapshot data
- `swiftsimio` - required for cosmological unit conversion
- `tomlkit` - required for reading toml-based style files
- `ruamel-yaml` - required for reading yaml-based style files
- `pydantic` - required for validating translations
- `packingcubes` - required for sorting the data and generating the packingcubes structure.
  Note that `packingcubes` is lazily imported in python 3.15+, so while it is not yet an optional package, it will be in the future.

<!-- --8<-- [end:requirements] -->

## Example Usage

GUSTEAU-Translate has a CLI via the `alchemize` command[^alchemize]:

``` sh
alchemize -t style INPUT OUTPUT
```

which will translate the `INPUT` HDF5 snapshot using the specified `style`(see below), saving a GUSTEAU formatted snapshot to `OUTPUT`.
The output file will be then validated against the GUSTEAU specification.

You can optionally skip the validation or translation steps by passing `--to/--translate-only` or `--vo/--validate-only`, if, e.g., you are testing GUSTEAU creation directly.

As always, you can specify `--help` to get the available options.

### Including `packingcubes`

You can additionally run the [`packingcubes`](https://packingcubes.readthedocs.io) software as part of the translation pipeline via `-p/--packingcubes`.
This will spatially sort the data (currently all particle groups listed in `/Header.ParticleNames` in the output file) and save the `ParticleCubes`/`PackedTree` information at the same time it translates it.

This adds a little bit of time to the startup (for the initial cubing) and will use more memory while running.
The contribution to the final snapshot size should be pretty minimal though; an EAGLE snapshot was actually _smaller_ in GUSTEAU format.

## Available Styles

The current available styles are

- `swift-eagle` - for converting [SWIFT](https://swift.strw.leidenuniv.nl/)-style snapshots
- `illustrisTNG` - for converting [IllustrisTNG](www.tng-project.org/)-style snapshots

> [!NOTE]
> Multipart snapshots (i.e. where the snapshot is spread across multiple files, like snapshot_001.0.hdf5, snapshot_001.1.hdf5, etc) are not yet supported.
> This is a high-priority feature and is expected soon.

This style was built off of the snapshots from the EAGLE simulation as found at [CAMELS](https://camels.readthedocs.io) and so is based on SWIFT v1.0.0

[^alchemize]: `gusteau-alchemize` and `gusteau-translate` also map to the same command.

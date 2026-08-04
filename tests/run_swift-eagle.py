from gusteau_translate import cli

cli(
    [
        "-vvf",  # debug level verbosity + force overwrite
        # "--skip-data",
        "-t",
        "swift-eagle",
        "--packing-cubes",
        "tests/snapshot_090.hdf5",
        "tests/gusteau.hdf5",
    ]
)

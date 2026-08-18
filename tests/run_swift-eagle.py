from gusteau_translate import cli

cli(
    [
        "-vvf",  # debug level verbosity + force overwrite
        # "--skip-data",
        "-t",
        "swift-eagle",
        "--packing-cubes",
        "tests/swift-eagle.hdf5",
        "tests/gusteau.hdf5",
    ]
)

from gusteau_translate import cli

cli(
    [
        "-vvf",
        "--skip-data",
        "-t",
        "swift-eagle",
        "tests/snapshot_090.hdf5",
        "tests/gusteau.hdf5",
    ]
)

from gusteau_translate import cli

cli(
    [
        "-vvf",  # debug level verbosity + force overwrite
        # "--diagnostic",
        "-t",
        "illustrisTNG",
        "--packing-cubes",
        "tests/illustrisTNG.hdf5",
        "tests/gusteauTNG.hdf5",
    ]
)

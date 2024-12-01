import contextlib
import os
import os.path
import uuid
import random
import json
import polars as pl
from glob import glob
from tqdm import tqdm

from fen_parser import fen_to_vec


def interleave(*input_globs, train_frac=0.95, size_per_file_mb=100):

    input_filenames = []
    for input_glob in input_globs:
        input_filenames.extend(glob(input_glob))

    input_size_mb = sum(os.path.getsize(filename) for filename in input_filenames) / 1e6
    n_files = int(input_size_mb // size_per_file_mb)

    random.shuffle(input_filenames)
    with contextlib.ExitStack() as stack:
        training_files = [
            stack.enter_context(open(f"training/{uuid.uuid4()}.json", "w"))
            for _ in range(n_files)
        ]
        testing_files = [
            stack.enter_context(open(f"testing/{uuid.uuid4()}.json", "w"))
            for _ in range(n_files)
        ]

        for input_filename in tqdm(input_filenames):

            lines = (
                pl.read_csv(
                    input_filename,
                    has_header=False,
                    new_columns=["fen", "cp", "result"],
                )
                .cast({"cp": pl.Int64, "result": pl.Float64})
                .sample(fraction=1.0, shuffle=True)
                .rows()
            )

            for fen, cp, result in lines:
                indices = fen_to_vec(fen)

                position_str = json.dumps(
                    {"indices": indices, "cp": cp, "result": result}
                )

                if random.random() < train_frac:
                    random.choice(training_files).write(position_str + "\n")
                else:
                    random.choice(testing_files).write(position_str + "\n")

    for filename in tqdm(glob("training/*")):
        pl.read_ndjson(filename).write_parquet(filename.replace(".json", ".parquet"))
        os.remove(filename)

    for filename in tqdm(glob("testing/*")):
        pl.read_ndjson(filename).write_parquet(filename.replace(".json", ".parquet"))
        os.remove(filename)


if __name__ == "__main__":
    interleave("data/lichess_elite/scored/*", train_frac=0.95)

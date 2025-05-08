import chess
import contextlib
import os
import os.path
import uuid
import random
import json
import polars as pl
from glob import glob
from tqdm import tqdm

from fen_parser import fen_to_indices


def convert_to_parquet(directory):
    for filename in tqdm(glob(f"{directory}/*.json"), desc=f"Converting {directory}"):
        output_file = filename.replace(".json", ".parquet")
        pl.read_ndjson(filename).write_parquet(output_file)
        os.remove(filename)


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
                .filter(
                    pl.col("fen").str.split(" ").list.get(5).cast(pl.Int32) >= 14
                )
                .sample(fraction=1.0, shuffle=True)
                .rows()
            )

            for fen, cp, result in lines:
                white_move = "w" in fen

                stm_indices, nstm_indices = fen_to_indices(fen)

                position_str = json.dumps(
                    {"stm_indices": stm_indices, "nstm_indices": nstm_indices, "cp": cp if white_move else -cp, "result": result if white_move else 1.0 - result}
                )
                output_files = training_files if random.random() < train_frac else testing_files
                random.choice(output_files).write(position_str + "\n")

    convert_to_parquet("training")
    convert_to_parquet("testing")

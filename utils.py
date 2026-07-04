import contextlib

import numpy as np
import uuid
import random
import csv
from glob import glob
from pathlib import Path
from tqdm import tqdm

from fen_parser import fen_to_indices

MIN_FULLMOVE = 14


def _count_lines(path):
    with open(path, "rb") as file:
        return sum(
            chunk.count(b"\n") for chunk in iter(lambda: file.read(1 << 22), b"")
        )


def _read_and_shuffle_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        filtered_rows = [
            row for row in reader if int(row[0].split()[-1]) >= MIN_FULLMOVE
        ]

    random.shuffle(filtered_rows)

    for fen, cp_str, result_str in filtered_rows:
        yield fen, int(cp_str), float(result_str)


def interleave(*input_globs, train_frac=0.95, size_per_file_mb=100):

    input_filenames = []
    for input_glob in input_globs:
        input_filenames.extend(
            filename for filename in glob(input_glob) if filename.endswith(".csv")
        )

    input_size_mb = (
        sum(Path(filename).stat().st_size for filename in input_filenames) / 1e6
    )
    n_files = max(1, int(input_size_mb // size_per_file_mb))
    line_counts = {path: _count_lines(path) for path in input_filenames}

    random.shuffle(input_filenames)
    with (
        contextlib.ExitStack() as stack,
        tqdm(
            unit=" positions",
            bar_format="{l_bar}{bar}| {n:,}/{total:,} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            total=sum(line_counts.values()),
        ) as pbar,
    ):
        training_files = [
            stack.enter_context(open(Path("training") / f"{uuid.uuid4()}.bin", "wb"))
            for _ in range(n_files)
        ]
        testing_files = [
            stack.enter_context(open(Path("testing") / f"{uuid.uuid4()}.bin", "wb"))
            for _ in range(n_files)
        ]

        for input_filename in input_filenames:
            file_start = pbar.n
            for fen, cp, result in _read_and_shuffle_csv(input_filename):
                white_move = "w" in fen

                stm_indices, nstm_indices = fen_to_indices(fen)

                output_files = (
                    training_files if random.random() < train_frac else testing_files
                )
                file = random.choice(output_files)
                file.write(np.array(stm_indices, dtype=np.int16).tobytes())
                file.write(np.array(nstm_indices, dtype=np.int16).tobytes())
                file.write(
                    np.array([cp if white_move else -cp], dtype=np.int16).tobytes()
                )
                file.write(
                    np.array(
                        [result if white_move else 1.0 - result], dtype=np.float16
                    ).tobytes()
                )

                pbar.update(1)

            # Rows dropped by the fullmove filter still count as progress.
            pbar.update(file_start + line_counts[input_filename] - pbar.n)


if __name__ == "__main__":
    interleave("data/lichess_elite/scored/*.csv")

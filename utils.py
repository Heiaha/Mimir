import numpy as np
import contextlib
import uuid
import random
import csv
from glob import glob
from pathlib import Path
from tqdm import tqdm

from fen_parser import fen_to_indices



def _read_and_shuffle_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        filtered_rows = list(reader)

    random.shuffle(filtered_rows)

    for fen, cp_str, result_str in filtered_rows:
        yield fen, int(cp_str), float(result_str)


def interleave(*input_globs, train_frac=0.95, size_per_file_mb=100):

    input_filenames = []
    for input_glob in input_globs:
        input_filenames.extend(
            filename for filename in glob(input_glob) if filename.endswith(".csv")
        )

    input_size_mb = sum(Path(filename).stat().st_size for filename in input_filenames) / 1e6
    n_files = int(input_size_mb // size_per_file_mb)

    random.shuffle(input_filenames)
    with contextlib.ExitStack() as stack:
        training_files = [
            stack.enter_context(open(Path("training") / f"{uuid.uuid4()}.bin", "wb"))
            for _ in range(n_files)
        ]
        testing_files = [
            stack.enter_context(open(Path("testing") / f"{uuid.uuid4()}.bin", "wb"))
            for _ in range(n_files)
        ]

        for input_filename in tqdm(input_filenames):
            for fen, cp, result in _read_and_shuffle_csv(input_filename):

                white_move = "w" in fen

                stm_indices, nstm_indices = fen_to_indices(fen)

                output_files = training_files if random.random() < train_frac else testing_files
                file = random.choice(output_files)
                file.write(np.array(stm_indices, dtype=np.int16).tobytes())
                file.write(np.array(nstm_indices, dtype=np.int16).tobytes())
                file.write(np.array([cp if white_move else -cp], dtype=np.int16).tobytes())
                file.write(np.array([result if white_move else 1.0 - result], dtype=np.float16).tobytes())

if __name__ == "__main__":
    interleave("data/selfplay/*.csv")

import numpy as np
import contextlib
import os
import uuid
import random
import csv
import chess
from glob import glob
from tqdm import tqdm

from fen_parser import fen_to_indices


def make_indices(fen):
    board = chess.Board(fen)

    mirror_white = chess.square_file(board.king(chess.WHITE)) > 3
    mirror_black = chess.square_file(board.king(chess.BLACK)) > 3

    stm_indices, nstm_indices = [], []

    for sq, piece in board.piece_map().items():
        h = hash(piece)

        white_idx = 64 * h + (sq ^ 7 if mirror_white else sq)

        mirror_sq = chess.square_mirror(sq)
        black_idx = 64 * ((h + 6) % 12) + (mirror_sq ^ 7 if mirror_black else mirror_sq)

        if board.turn == chess.WHITE:
            stm_indices.append(white_idx)
            nstm_indices.append(black_idx)
        else:
            stm_indices.append(black_idx)
            nstm_indices.append(white_idx)

    stm_indices.extend([768] * (32 - len(stm_indices)))
    nstm_indices.extend([768] * (32 - len(nstm_indices)))

    return stm_indices, nstm_indices




def _read_and_shuffle_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)

        filtered_rows = [
            row for row in reader if int(row[0].split()[-1]) >= 14
        ]

    random.shuffle(filtered_rows)

    for fen, cp_str, result_str in filtered_rows:
        yield fen, int(cp_str), float(result_str)


def interleave(*input_globs, train_frac=0.95, size_per_file_mb=100):

    input_filenames = []
    for input_glob in input_globs:
        input_filenames.extend(glob(input_glob))

    input_size_mb = sum(os.path.getsize(filename) for filename in input_filenames) / 1e6
    n_files = int(input_size_mb // size_per_file_mb)

    random.shuffle(input_filenames)
    with contextlib.ExitStack() as stack:
        training_files = [
            stack.enter_context(open(f"training/{uuid.uuid4()}.bin", "wb"))
            for _ in range(n_files)
        ]
        testing_files = [
            stack.enter_context(open(f"testing/{uuid.uuid4()}.bin", "wb"))
            for _ in range(n_files)
        ]

        for input_filename in tqdm(input_filenames[10:]):
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
    interleave("data/lichess_elite/scored/*")

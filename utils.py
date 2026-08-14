import contextlib

import numpy as np
import polars as pl
import uuid
import random
from glob import glob
from pathlib import Path
from tqdm import tqdm

from fen_parser import fen_to_indices
from scaling import Hist

MIN_FULLMOVE = 0


def _count_rows(path):
    return pl.scan_parquet(path).select(pl.len()).collect().item()


def _read_and_shuffle(path):
    rows = (
        pl.scan_parquet(path)
        .select("fen", "cp", "outcome")
        # This predicate is pushed into the row-group statistics, where parquet
        # truncates strings at 64 bytes, leaving mins with no fullmove field.
        .filter(
            pl.col("fen").str.split(" ").list.last().cast(pl.Int32, strict=False)
            >= MIN_FULLMOVE
        )
        .collect(engine="streaming")
        .sample(fraction=1.0, shuffle=True)
        .rows()
    )
    return rows


def _fit_file_scaling(path):
    labels = (
        pl.scan_parquet(path).select("cp", "outcome").collect(engine="streaming")
    )
    hist = Hist(-1000, 1000, 50)
    hist.fill_batch(labels["cp"].to_numpy(), (1 + labels["outcome"].to_numpy()) / 2)
    return 1 / hist.fit()


def interleave(*input_globs, train_frac=0.95, size_per_file_mb=100, canonical_scaling=None):

    input_filenames = []
    for input_glob in input_globs:
        input_filenames.extend(
            filename for filename in glob(input_glob) if filename.endswith(".parquet")
        )

    ratio_by_file = dict.fromkeys(input_filenames, 1.0)
    if canonical_scaling is not None:
        for path in tqdm(input_filenames, desc="calibrating"):
            ratio_by_file[path] = canonical_scaling / _fit_file_scaling(path)

        ratios = np.array(list(ratio_by_file.values()))
        print(
            f"rescale ratios: median {np.median(ratios):.3f}, "
            f"range [{ratios.min():.3f}, {ratios.max():.3f}]"
        )
        assert ratios.min() > 0.5 and ratios.max() < 2.0, "implausible scaling fit"

    input_size_mb = (
        sum(Path(filename).stat().st_size for filename in input_filenames) / 1e6
    )
    n_files = max(1, int(input_size_mb // size_per_file_mb))
    row_counts = {path: _count_rows(path) for path in input_filenames}

    random.shuffle(input_filenames)
    with (
        contextlib.ExitStack() as stack,
        tqdm(
            unit=" positions",
            bar_format="{l_bar}{bar}| {n:,}/{total:,} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            total=sum(row_counts.values()),
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
            ratio = ratio_by_file[input_filename]
            for fen, cp, result in _read_and_shuffle(input_filename):
                white_move = "w" in fen
                if ratio != 1.0:
                    cp = int(np.clip(round(cp * ratio), -32767, 32767))

                # Only stm is stored; dataset.derive_nstm rebuilds the rest.
                stm_indices = fen_to_indices(fen)

                output_files = (
                    training_files if random.random() < train_frac else testing_files
                )
                file = random.choice(output_files)
                file.write(np.array(stm_indices, dtype=np.int16).tobytes())
                file.write(
                    np.array([cp if white_move else -cp], dtype=np.int16).tobytes()
                )
                # White-POV {-1, 0, 1} in, stm-POV {0, 0.5, 1} out.
                file.write(
                    np.array(
                        [(1 + result) / 2 if white_move else (1 - result) / 2],
                        dtype=np.float16,
                    ).tobytes()
                )

                pbar.update(1)

            # Filtered-out rows still count as progress.
            pbar.update(file_start + row_counts[input_filename] - pbar.n)


if __name__ == "__main__":
    interleave(
        "data/selfplay_rescored/*.parquet",
        "data/selfplay_v0_rescored/*.parquet",
        canonical_scaling=120.0,
    )

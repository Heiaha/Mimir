import itertools
import numpy as np
import torch
from pathlib import Path

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"The given NumPy array is not writable, and PyTorch does not support non-writable tensors.*",
)


from torch.utils.data import IterableDataset

from model import NNUE


N_FILES = 8
N_SQUARES = 64
N_PIECE_CODES = 12
COLOR_FLIP = N_PIECE_CODES // 2
FILE_MIRROR = N_FILES - 1
RANK_FLIP = 56
ENEMY_KING = 11

BUCKET_BY_RANK = np.array([0, 1, 2, 2, 3, 3, 3, 3], dtype=np.int32)


class PositionVectorIterableDataset(IterableDataset):
    DTYPE = np.dtype(
        [
            ("stm_indices", np.int16, 32),
            ("cp", np.int16, 1),
            ("result", np.float16, 1),
        ]
    )

    def __init__(self, filenames, batch_size, steps=None, workers=1, verify_nstm=False):
        super().__init__()
        self.filenames = filenames
        self.batch_size = batch_size
        self.workers = max(1, workers)
        self.verify_nstm = verify_nstm

        full_pass = sum(
            Path(filename).stat().st_size // self.DTYPE.itemsize // batch_size
            for filename in filenames
        )
        self.per_worker = min(steps or full_pass, full_pass) // self.workers

    @staticmethod
    def derive_nstm(stm: np.ndarray) -> np.ndarray:
        # Padding slots yield garbage; _batch drops them with the stm mask.
        piece = (stm % NNUE.N_BASE_FEATURES) // N_SQUARES
        square = stm % N_SQUARES

        enemy_king = np.where(piece == ENEMY_KING, square, 0).max(axis=1)
        king_rel = enemy_king ^ RANK_FLIP
        mirror = np.where(
            king_rel % N_FILES >= N_FILES // 2, FILE_MIRROR, 0
        ).astype(np.int32)
        bucket = BUCKET_BY_RANK[(king_rel ^ mirror) // N_FILES]

        return (
            bucket[:, np.newaxis] * NNUE.N_BASE_FEATURES
            + N_SQUARES * ((piece + COLOR_FLIP) % N_PIECE_CODES)
            + (square ^ RANK_FLIP ^ mirror[:, np.newaxis])
        )

    def _batch(self, chunk, filepath):
        stm = chunk["stm_indices"].astype(np.int32)
        keep = stm != NNUE.N_FEATURES
        nstm = self.derive_nstm(stm)

        if self.verify_nstm:
            back = self.derive_nstm(np.where(keep, nstm, NNUE.N_FEATURES))
            assert (back[keep] == stm[keep]).all(), (
                f"{filepath}: derive_nstm involution failed"
            )

        counts = keep.sum(axis=1)
        offsets = np.cumsum(counts) - counts

        return {
            "stm_flat": torch.from_numpy(stm[keep].astype(np.int64)),
            "nstm_flat": torch.from_numpy(nstm[keep].astype(np.int64)),
            "offsets": torch.from_numpy(offsets.astype(np.int64)),
            "counts": torch.from_numpy(counts.astype(np.int64)),
            "cp": torch.from_numpy(np.ascontiguousarray(chunk["cp"])),
            "result": torch.from_numpy(np.ascontiguousarray(chunk["result"])),
        }

    def _stream_file(self, filepath):
        rows = np.fromfile(filepath, dtype=self.DTYPE)
        assert rows["stm_indices"].max() <= NNUE.N_FEATURES, (
            f"{filepath}: indices out of range"
        )
        np.random.shuffle(rows)
        n = len(rows) - len(rows) % self.batch_size
        for i in range(0, n, self.batch_size):
            yield self._batch(rows[i : i + self.batch_size], filepath)

    def __iter__(self):
        # Seeded from the per-worker torch seed, which train.py varies by
        # epoch, so the slice differs each epoch and resume stays reproducible.
        np.random.seed(torch.initial_seed() % (2**32))

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            assert len(self.filenames) >= worker_info.num_workers, (
                "Number of files must be larger than the number of workers."
            )
            filenames = self.filenames[worker_info.id :: worker_info.num_workers]
        else:
            filenames = self.filenames

        np.random.shuffle(filenames)
        batches = itertools.chain.from_iterable(
            self._stream_file(filename) for filename in filenames
        )
        yield from itertools.islice(batches, self.per_worker)

    def __len__(self):
        # Must match what __iter__ yields: the scheduler sizes itself on this.
        return self.per_worker * self.workers

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

    def __init__(self, filenames, batch_size):
        super().__init__()
        self.filenames = filenames
        self.batch_size = batch_size

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

    @staticmethod
    def _batch(chunk):
        stm = chunk["stm_indices"].astype(np.int32)
        nstm = PositionVectorIterableDataset.derive_nstm(stm)

        keep = stm != NNUE.N_FEATURES
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
            yield self._batch(rows[i : i + self.batch_size])

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            assert len(self.filenames) >= worker_info.num_workers, (
                "Number of files must be larger than the number of workers."
            )
            filenames = self.filenames[worker_info.id :: worker_info.num_workers]
        else:
            filenames = self.filenames

        np.random.shuffle(filenames)
        for filename in filenames:
            yield from self._stream_file(filename)

    def __len__(self):
        # Must match what __iter__ yields: the scheduler sizes itself on this.
        return sum(
            Path(filename).stat().st_size // self.DTYPE.itemsize // self.batch_size
            for filename in self.filenames
        )

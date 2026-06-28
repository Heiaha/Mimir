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


class PositionVectorIterableDataset(IterableDataset):

    DTYPE = np.dtype([
        ("stm_indices", np.int16, 32),
        ("nstm_indices", np.int16, 32),
        ("cp", np.int16, 1),
        ("result", np.float16, 1),
    ])

    def __init__(self, filenames, batch_size):
        super().__init__()
        self.filenames = filenames
        self.batch_size = batch_size

    def _stream_file(self, filepath):
        rows = np.fromfile(filepath, dtype=self.DTYPE)
        np.random.shuffle(rows)
        n = len(rows) - len(rows) % self.batch_size  # drop last partial batch
        for i in range(0, n, self.batch_size):
            chunk = rows[i:i + self.batch_size]
            yield {
                key: torch.from_numpy(np.ascontiguousarray(chunk[key]))
                for key in self.DTYPE.names
            }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            assert (
                len(self.filenames) >= worker_info.num_workers
            ), "Number of files must be larger than the number of workers."
            filenames = self.filenames[worker_info.id :: worker_info.num_workers]
        else:
            filenames = self.filenames

        np.random.shuffle(filenames)
        for filename in filenames:
            yield from self._stream_file(filename)

    def __len__(self):
        # Number of (drop-last) batches across all files -- matches what __iter__
        # actually yields, so len(dataloader) * epochs is exact for the scheduler.
        return sum(
            Path(filename).stat().st_size // self.DTYPE.itemsize // self.batch_size
            for filename in self.filenames
        )

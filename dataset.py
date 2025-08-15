import numpy as np
import os
import torch

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

    def __init__(self, filenames):
        super().__init__()
        self.filenames = filenames

    def _stream_file(self, filepath):
        rows = np.fromfile(filepath, dtype=self.DTYPE)
        np.random.shuffle(rows)
        yield from rows

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
        total_bytes = sum(os.path.getsize(filename) for filename in self.filenames)
        assert total_bytes % self.DTYPE.itemsize == 0, "File size must be a multiple of row size."
        return total_bytes // self.DTYPE.itemsize

    @staticmethod
    def fast_collate(batch):
        return {
            key: torch.from_numpy(np.stack([sample[key] for sample in batch]))
            for key in PositionVectorIterableDataset.DTYPE.names
        }

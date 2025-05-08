import random
import polars as pl
import torch

from torch.utils.data import IterableDataset


class PositionVectorIterableDataset(IterableDataset):
    def __init__(self, filenames):
        super().__init__()
        self.filenames = filenames
        self._len = (
            pl.scan_parquet(
                filenames,
            )
            .select(pl.len())
            .collect(engine="streaming")
            .item()
        )

    def read_file(self, filename):
        data = pl.read_parquet(filename).sample(fraction=1, shuffle=True)
        for row in data.iter_rows(named=True):
            stm_indices = row["stm_indices"]
            nstm_indices = row["nstm_indices"]

            assert len(stm_indices) == len(nstm_indices)

            yield {
                "stm_indices": torch.tensor(stm_indices + [768] * (32 - len(stm_indices)), dtype=torch.long),
                "nstm_indices": torch.tensor(nstm_indices + [768] * (32 - len(nstm_indices)), dtype=torch.long),
                "cp": torch.tensor([row["cp"]], dtype=torch.float),
                "result": torch.tensor([row["result"]], dtype=torch.float),
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

        random.shuffle(filenames)
        for filename in filenames:
            yield from self.read_file(filename)

    def __len__(self):
        return self._len

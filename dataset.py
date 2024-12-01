import random
import time
from dataclasses import dataclass

import polars as pl
import torch

from torch.utils.data import IterableDataset


@dataclass
class Batch:
    x: torch.Tensor
    cp: torch.Tensor
    result: torch.Tensor

    def __init__(self, x, cp, result):
        self.x = x
        self.cp = cp
        self.result = result

    def to(self, device):
        self.x, self.cp, self.result = (
            self.x.to(device, non_blocking=True),
            self.cp.to(device, non_blocking=True),
            self.result.to(device, non_blocking=True),
        )
        return self


class PositionVectorIterableDataset(IterableDataset):
    def __init__(self, filenames):
        super().__init__()
        self.filenames = filenames
        self._len = (
            pl.scan_parquet(
                filenames,
            )
            .select(pl.len())
            .collect(streaming=True)
            .item()
        )

    @staticmethod
    def read_file(filename):
        data = (
            pl.read_parquet(
                filename,
            )
            .sample(fraction=1, shuffle=True)
            .rows()
        )
        for indices, cp, result in data:

            yield (
                torch.sparse_coo_tensor(
                    indices=torch.tensor([indices]),
                    values=torch.ones(len(indices)),
                    size=(768,),
                    is_coalesced=True,
                    dtype=torch.float,
                ).to_dense(),  # fastest is to read sparse format and then convert to dense before collation
                torch.tensor([cp], dtype=torch.float),
                torch.tensor([result], dtype=torch.float),
            )

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            assert (
                len(self.filenames) >= worker_info.num_workers
            ), "Number of files must be larger than the number of workers."
            filenames = self.filenames[worker_info.id :: worker_info.num_workers]
            time.sleep(worker_info.id)
        else:
            filenames = self.filenames

        random.shuffle(filenames)
        for filename in filenames:
            yield from self.read_file(filename)

    def __len__(self):
        return self._len

import random
from dataclasses import dataclass

import polars as pl
import torch

from fen_parser import fen_to_vec
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
            self.x.to(device),
            self.cp.to(device),
            self.result.to(device),
        )
        return self


class PositionVectorIterableDataset(IterableDataset):
    def __init__(self, filenames):
        super().__init__()
        self.filenames = filenames
        self._len = (
            pl.scan_csv(
                self.filenames,
                low_memory=True,
                cache=False,
                has_header=False,
                new_columns=["fen", "cp", "result"],
                dtypes={"fen": pl.String, "cp": pl.Int16, "result": pl.Float32},
            )
            .select(pl.len())
            .collect(streaming=True)
            .item()
        )

    @staticmethod
    def read_file(filename):
        data = (
            pl.read_csv(
                filename,
                has_header=False,
                new_columns=["fen", "cp", "result"],
                dtypes={"fen": pl.String, "cp": pl.Int16, "result": pl.Float32},
            )
            .sample(fraction=1, shuffle=True)
            .rows()
        )
        for fen, cp, result in data:
            yield (
                torch.tensor(fen_to_vec(fen), dtype=torch.float),
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
        else:
            filenames = self.filenames

        random.shuffle(filenames)
        for filename in filenames:
            yield from self.read_file(filename)

    def __len__(self):
        return self._len


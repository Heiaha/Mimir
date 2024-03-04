from pathlib import Path
import random
from dataclasses import dataclass

import polars as pl
import torch

import config
from fen_parser import fen_to_vec
from torch.utils.data import IterableDataset


@dataclass
class Batch:
    X: torch.Tensor
    cp: torch.Tensor
    result: torch.Tensor

    def __init__(self, X, cp, result):
        self.X = X.to(config.DEVICE)
        self.cp = cp.to(config.DEVICE)
        self.result = result.to(config.DEVICE)


class PositionVectorDataset(IterableDataset):
    def __init__(self, filenames):
        super().__init__()
        self.filenames = filenames

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
            data = pl.read_parquet(filename).sample(fraction=1).to_dicts()
            for line in data:
                yield (
                    torch.Tensor(fen_to_vec(line["fen"])),
                    torch.Tensor([line["cp"]]),
                    torch.Tensor([line["result"]]),
                )

    def __len__(self):
        return (
            pl.scan_parquet(self.filenames, low_memory=True, cache=False)
            .select(pl.len())
            .collect(streaming=True)
            .item()
        )


if __name__ == "__main__":

    filenames = Path("data_d8").glob("*")
    dataset = PositionVectorDataset(filenames)
    print(len(dataset))

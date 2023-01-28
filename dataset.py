from pathlib import Path
from dataclasses import dataclass

import random
import torch

import config
from fen_parser import fen_to_vec
from torch.utils.data import IterableDataset


@dataclass
class Batch:
    X: torch.Tensor
    score: torch.Tensor
    result: torch.Tensor

    def __init__(self, X, score, result):
        self.X = X.to(config.DEVICE)
        self.score = score.to(config.DEVICE)
        self.result = result.to(config.DEVICE)


class PositionVectorDataset(IterableDataset):
    def __init__(self, file_names):
        super().__init__()
        self.filenames = file_names

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            assert (
                    len(self.filenames) >= worker_info.num_workers
            ), "Number of files must be larger than the number of workers."
            file_names = self.filenames[worker_info.id::worker_info.num_workers]
        else:
            file_names = self.filenames

        random.shuffle(file_names)
        for filename in file_names:
            fens, scores, results = [], [], []
            with open(filename, "r") as file:
                lines = file.readlines()
            random.shuffle(lines)
            for line in lines:
                fen, score, result = line.split(",")
                scores.append(int(score))
                results.append(float(result))
                fens.append(fen)

            for fen, score, result in zip(fens, scores, results):
                vector = fen_to_vec(fen)

                yield torch.Tensor(vector), torch.Tensor([score]), torch.Tensor(
                    [result]
                )

    def __len__(self):
        total = 0
        for filename in self.filenames:
            with open(filename, "r") as file:
                total += sum(1 for _ in file)
        return total


if __name__ == "__main__":
    filenames = Path("data").glob("*")
    dataset = PositionVectorDataset(filenames)
    print(len(dataset))


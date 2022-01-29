import random
import torch
from fen_parser import fen_to_vec
from torch.utils.data import IterableDataset


class PositionVectorDataset(IterableDataset):
    def __init__(self, file_names):
        super().__init__()
        self.file_names = file_names

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            assert (
                len(self.file_names) >= worker_info.num_workers
            ), "Number of files must be larger than the number of workers."
            file_names = self.file_names[worker_info.id :: worker_info.num_workers]
        else:
            file_names = self.file_names

        random.shuffle(file_names)
        for file_name in file_names:
            fens, scores, results = [], [], []
            with open(file_name, "r") as file:
                lines = file.readlines()
            random.shuffle(lines)
            for line in lines:
                fen, score, result = line.split(",")
                scores.append(int(score))
                results.append(float(result))
                fens.append(fen)

            for fen, score, result in zip(fens, scores, results):
                yield torch.Tensor(fen_to_vec(fen)), \
                      torch.Tensor([score]), \
                      torch.Tensor([result])

import os
import random
from glob import glob

from collections import defaultdict


def quantize(param_dict):
    new_dict = defaultdict(list)

    for name, weights in param_dict.items():
        for value in weights:
            new_dict[name].append(round(value * 64))

    return dict(new_dict)


def train_test_split(input_dir, train_frac):
    for input_path in glob(f"{input_dir}/*"):
        with open(input_path, "r") as file:
            lines = file.readlines()
        random.shuffle(lines)

        basename = os.path.basename(input_path)

        with open(f"training/{basename}", "w") as training_file, open(
            f"testing/{basename}", "w"
        ) as testing_file:

            for line in lines:
                if random.random() < train_frac:
                    training_file.write(line)
                else:
                    testing_file.write(line)


if __name__ == "__main__":
    train_test_split("data", 0.98)

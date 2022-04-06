import os
import random
import numpy as np
from collections import defaultdict
from glob import glob


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


def find_best_scaling(input_dir):
    scores = []
    results = []
    for filename in glob(f"{input_dir}/*"):
        with open(filename, "r") as file:
            for line in file:
                fen, score, result = line.split(",")
                if result == 0:
                    continue
                scores.append(int(score))
                results.append(float(result))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def mse(k, s, r):
        return np.square(r - sigmoid(k * s)).mean()

    scores = np.array(scores)
    results = np.array(results)
    min_ = 0.0
    max_ = 0.1
    delta = 0.01
    best = mse(min_, scores, results)

    for _ in range(10):
        value = min_
        while value < max_:
            error = mse(value, scores, results)
            if error <= best:
                best = error
                min_ = value
            value += delta

        print(f"Scaling = {min_}, MSE = {best}")
        max_ = min_ + delta
        min_ = min_ - delta
        delta /= 10

    return min_


def count_fens(input_dir):
    total = 0
    for filename in glob(f"{input_dir}/*"):
        with open(filename, "r") as file:
            total += sum(1 for _ in file)
    return total


if __name__ == "__main__":
    train_test_split("data", 0.98)
    print(find_best_scaling("data"))
    # print(count_fens("data"))

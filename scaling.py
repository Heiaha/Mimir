import numpy as np
import seaborn.objects as so

from glob import glob


class Hist:

    def __init__(self, low: int | float, high: int | float, n_bins: int):
        self.low = low
        self.high = high
        self.n_bins = n_bins

        self.weights = np.zeros(n_bins)
        self.counts = np.zeros(n_bins)

    def fill(self, value: int | float, weight: int | float):

        if not self.low <= value < self.high:
            return

        bin_n = self.n_bins * (value - self.low) // (self.high - self.low)
        self.weights[bin_n] += weight
        self.counts[bin_n] += 1

    def fit(self):

        def mse(x, y, k):
            return np.square(y - self.sigmoid(k * x)).mean()

        x = np.linspace(self.low, self.high, self.n_bins)
        y = self.weights / self.counts

        min_ = 0.0
        max_ = 0.1
        delta = 0.01
        best = mse(x, y, min_)

        for _ in range(10):
            value = min_
            while value < max_:
                error = mse(x, y, value)
                if error <= best:
                    best = error
                    min_ = value
                value += delta

            max_ = min_ + delta
            min_ = min_ - delta
            delta /= 10

        return min_

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    hist = Hist(-1500, 1500, 500)

    for filename in glob(f"data/*"):
        with open(filename, "r") as file:
            for line in file:
                fen, score, result = line.split(",")
                score = int(score)
                result = float(result)
                hist.fill(score, result)

    print(hist.fit())



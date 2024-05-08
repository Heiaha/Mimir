import numpy as np
import polars as pl
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


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
        max_ = 10
        delta = 0.1
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

    def plot(self, scale):
        x = np.linspace(self.low, self.high, self.n_bins)
        sns.lineplot(x=x, y=self.sigmoid(scale * x))
        sns.lineplot(x=x, y=self.weights / self.counts)

        plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit(x, y):
    def mse(x, y, k):
        return np.square(y - sigmoid(k * x)).mean()

    min_ = 0.0
    max_ = 10
    delta = 0.1
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


if __name__ == "__main__":
    from tqdm import tqdm

    hist = Hist(-1500, 1500, 100)

    for filename in tqdm(glob(f"training/*")):
        data = (
            pl.scan_csv(
                filename,
                new_columns=["fen", "cp", "result"],
                dtypes={"fen": pl.String, "cp": pl.Int16, "result": pl.Float32},
            )
            .select(pl.col("cp", "result"))
            .collect()
            .to_dicts()
        )

        for line in data:
            hist.fill(line["cp"], line["result"])

    scaling = hist.fit()
    print(1 / scaling)
    hist.plot(scaling)

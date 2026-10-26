import numpy as np
from pathlib import Path

from dataset import PositionVectorIterableDataset


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

    def fill_batch(self, values, weights):
        values = np.asarray(values, dtype=np.float64).ravel()
        weights = np.asarray(weights, dtype=np.float64).ravel()

        mask = (values >= self.low) & (values < self.high)
        values, weights = values[mask], weights[mask]

        bins = ((values - self.low) / (self.high - self.low) * self.n_bins).astype(int)
        bins = np.clip(bins, 0, self.n_bins - 1)
        self.weights += np.bincount(bins, weights=weights, minlength=self.n_bins)
        self.counts += np.bincount(bins, minlength=self.n_bins)

    def fit(self):
        filled = self.counts > 0
        x = np.linspace(self.low, self.high, self.n_bins)[filled]
        y = self.weights[filled] / self.counts[filled]

        def mse(k):
            return np.square(y - self.sigmoid(k * x)).mean()

        low, high, n_steps = 0.0, 10.0, 100
        best_k = 0.0
        best = mse(best_k)

        # Each pass re-brackets around the incumbent, which is never discarded.
        for _ in range(10):
            for k in np.linspace(low, high, n_steps + 1):
                error = mse(k)
                if error < best:
                    best, best_k = error, k

            step = (high - low) / n_steps
            low, high = best_k - step, best_k + step

        return float(best_k)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -700, 700)))

    def plot(self, scale):
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set()
        x = np.linspace(self.low, self.high, self.n_bins)
        sns.lineplot(x=x, y=self.sigmoid(scale * x))
        sns.lineplot(x=x, y=self.weights / self.counts)

        plt.show()


if __name__ == "__main__":
    hist = Hist(-1000, 1000, 50)
    filenames = sorted(Path("training").glob("*"))

    dataset = PositionVectorIterableDataset(filenames, 8092)

    from tqdm import tqdm

    for data in tqdm(dataset):
        hist.fill_batch(data["cp"], data["result"])

    scaling = hist.fit()
    print(1 / scaling)
    hist.plot(scaling)

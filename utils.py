import contextlib
import os.path
import uuid
import random
from glob import glob
from tqdm import tqdm


def interleave(input_dir, train_frac, size_per_file_mb=100):

    input_files = glob(f"{input_dir}/*")
    input_size_mb = sum(os.path.getsize(filename) for filename in input_files) / 1e6
    n_files = int(input_size_mb // size_per_file_mb)

    random.shuffle(input_files)
    with contextlib.ExitStack() as stack:
        training_files = [
            stack.enter_context(open(f"training/{uuid.uuid4()}.csv", "w"))
            for _ in range(n_files)
        ]
        testing_files = [
            stack.enter_context(open(f"testing/{uuid.uuid4()}.csv", "w"))
            for _ in range(n_files)
        ]

        for input_path in tqdm(input_files):

            with open(input_path) as file:
                lines = [line.strip() for line in file]
                random.shuffle(lines)

            for line in lines:
                line = line + "\n"
                if random.random() < train_frac:
                    random.choice(training_files).write(line)
                else:
                    random.choice(testing_files).write(line)


if __name__ == "__main__":
    interleave("data_d8_v2", 0.99)

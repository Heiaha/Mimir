import uuid
import random
from glob import glob
import polars as pl


def train_test_split(input_dir, train_frac, file_length=1_000_000):

    train_data = []
    test_data = []
    for input_path in glob(f"{input_dir}/*"):
        with open(input_path, "r") as file:
            lines = file.readlines()
        random.shuffle(lines)
        for line in lines:
            fen, cp, result = line.strip().split(",")
            pos_dict = {"fen": fen, "cp": cp, "result": result}
            if random.random() < train_frac:
                train_data.append(pos_dict)
                if len(train_data) > file_length:
                    (
                        pl.DataFrame(
                            train_data,
                            schema_overrides={
                                "fen": pl.String,
                                "cp": pl.Int16,
                                "result": pl.Float32,
                            },
                        ).write_parquet(
                            f"training/{uuid.uuid4()}.parquet", compression="lz4"
                        )
                    )
                    train_data = []
            else:
                test_data.append(pos_dict)
                if len(test_data) > file_length:
                    (
                        pl.DataFrame(
                            test_data,
                            schema_overrides={
                                "fen": pl.String,
                                "cp": pl.Int16,
                                "result": pl.Float32,
                            },
                        ).write_parquet(
                            f"testing/{uuid.uuid4()}.parquet", compression="lz4"
                        )
                    )
                    test_data = []

    (
        pl.DataFrame(
            train_data,
            schema_overrides={"fen": pl.String, "cp": pl.Int16, "result": pl.Float32},
        ).write_parquet(f"training/{uuid.uuid4()}.parquet", compression="lz4")
    )
    (
        pl.DataFrame(
            test_data,
            schema_overrides={"fen": pl.String, "cp": pl.Int16, "result": pl.Float32},
        ).write_parquet(f"testing/{uuid.uuid4()}.parquet", compression="lz4")
    )


if __name__ == "__main__":
    train_test_split("data_d8", 0.9)

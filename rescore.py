import concurrent.futures
from glob import glob
from pathlib import Path

import chess
import chess.syzygy
import polars as pl
from tqdm import tqdm

INPUT_GLOB = "data/selfplay_v2/*.parquet"
OUTPUT_DIR = "data/selfplay_v2_rescored"
SYZYGY_PATH = "syzygy"
TB_MAX_MEN = 5
N_CONCURRENT = 12
PROBE_CHUNK = 50_000

SYZYGY_OUTCOMES = {-2: -1, -1: 0, 0: 0, 1: 0, 2: 1}

_tablebase = None


def probe_chunk(fens: list[str]) -> list[int | None]:
    """White-perspective WDL in {-1, 0, 1} per FEN, None where unprobeable."""
    global _tablebase
    if _tablebase is None:
        _tablebase = chess.syzygy.open_tablebase(SYZYGY_PATH)

    results = []
    for fen in fens:
        board = chess.Board(fen)
        try:
            wdl = _tablebase.probe_wdl(board)  # side-to-move perspective, -2..2
        except KeyError:
            results.append(None)  # castling rights or missing table
            continue
        score = SYZYGY_OUTCOMES[wdl]  # cursed/blessed -> draw
        results.append(score if board.turn == chess.WHITE else -score)
    return results


def probe_candidates(input_glob: str) -> pl.DataFrame:
    """(game_id, fen, tb) for every distinct probeable in-range position."""
    candidates = (
        pl.scan_parquet(input_glob)
        .filter(
            pl.col("fen").str.split(" ").list.first().str.count_matches(r"[a-zA-Z]")
            <= TB_MAX_MEN
        )
        .select("game_id", "fen")
        .unique()
        .collect()
    )

    fens = candidates["fen"].to_list()
    chunks = [fens[i : i + PROBE_CHUNK] for i in range(0, len(fens), PROBE_CHUNK)]

    probes: list[int | None] = []
    with (
        concurrent.futures.ProcessPoolExecutor(max_workers=N_CONCURRENT) as executor,
        tqdm(total=len(fens), unit=" probes", desc="Probing") as pbar,
    ):
        for chunk_results in executor.map(probe_chunk, chunks):
            probes.extend(chunk_results)
            pbar.update(len(chunk_results))

    return candidates.with_columns(tb=pl.Series(probes, dtype=pl.Int8)).drop_nulls(
        "tb"
    )


def main(input_glob: str = INPUT_GLOB, output_dir: str = OUTPUT_DIR) -> None:
    probed = probe_candidates(input_glob)

    # Ply comes from the FEN; row order across file boundaries is not reliable.
    ply = (pl.col("fen").str.split(" ").list.last().cast(pl.Int32) - 1) * 2 + (
        pl.col("fen").str.split(" ").list.get(1) == "b"
    ).cast(pl.Int32)
    games = probed.group_by("game_id").agg(tb_first=pl.col("tb").sort_by(ply).first())

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows = n_exact = n_propagated = n_changed = 0
    for filename in tqdm(sorted(glob(input_glob)), unit=" files", desc="Rescoring"):
        df = (
            pl.read_parquet(filename)
            .join(probed, on=["game_id", "fen"], how="left")
            .join(games, on="game_id", how="left")
            .with_columns(
                rescored=pl.coalesce(
                    pl.col("tb"), pl.col("tb_first"), pl.col("outcome")
                ).cast(pl.Int8)
            )
        )

        n_rows += len(df)
        n_exact += df["tb"].is_not_null().sum()
        n_propagated += (df["tb"].is_null() & df["tb_first"].is_not_null()).sum()
        n_changed += (df["rescored"] != df["outcome"]).sum()

        df.select("fen", "game_id", "cp", outcome=pl.col("rescored")).write_parquet(
            out_dir / Path(filename).name,
            compression="zstd",
            statistics="full",
            compression_level=22,
        )

    print(f"{n_rows:,} rows rescored into {out_dir}")
    print(f"  exact tb labels: {n_exact:,}")
    print(f"  propagated tb labels: {n_propagated:,}")
    print(f"  outcomes changed: {n_changed:,}")
    print(f"  games reaching the tablebases: {games.height:,}")


if __name__ == "__main__":
    main()

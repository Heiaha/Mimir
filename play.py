import concurrent.futures
import random
import uuid
from pathlib import Path

import chess
import chess.engine
import chess.syzygy
import polars as pl
from tqdm import tqdm

ENGINE = "Weiawaga.exe"
DEPTH = 8
ROWS_PER_FILE = 100_000
OPENING_PLIES = 8
OUTPUT_DIR = "data/selfplay"
N_CONCURRENT = 12

SYZYGY_OUTCOMES = {-2: -1, -1: 0, 0: 0, 1: 0, 2: 1}
OUTCOMES = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}

SYZYGY_PATH = "syzygy"
TB_MAX_MEN = 5

_tablebase = (
    chess.syzygy.open_tablebase(SYZYGY_PATH) if Path(SYZYGY_PATH).is_dir() else None
)


def is_loud(board: chess.Board, move: chess.Move) -> bool:
    return bool(
        move.promotion
        or board.is_capture(move)
        or board.is_en_passant(move)
        or board.is_check()
        or board.gives_check(move)
    )


def tb_outcome(board: chess.Board) -> float | None:
    """Exact White-perspective WDL in {-1, 0, 1} for <=TB_MAX_MEN positions, else None."""
    if _tablebase is None or chess.popcount(board.occupied) > TB_MAX_MEN:
        return None
    try:
        wdl = _tablebase.probe_wdl(board)  # side-to-move perspective, -2..2
    except KeyError:
        return None  # material not covered by the tables we have
    score = SYZYGY_OUTCOMES[wdl]  # cursed/blessed -> draw
    return score if board.turn == chess.WHITE else -score


def random_opening(plies: int) -> chess.Board:
    board = chess.Board()

    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break

        board.push(random.choice(moves))

    return board


def play_one_game() -> list[dict]:
    # Reseed per game so workers (forked or spawned) never share an opening stream.
    random.seed()
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE)

    try:
        board = random_opening(OPENING_PLIES)
        if board.is_game_over(claim_draw=True):
            return []

        game_id = uuid.uuid4().bytes
        samples: list[dict] = []
        outcome = None

        while not board.is_game_over(claim_draw=True):
            analysis = engine.play(
                board,
                limit=chess.engine.Limit(depth=DEPTH),
                info=chess.engine.INFO_SCORE,
            )

            move = analysis.move
            if move is None:
                raise RuntimeError(f"Engine returned no move for: {board.fen()}")

            score = analysis.info.get("score")

            if score is not None and score.is_mate():
                outcome = 1 if score.white().mate() > 0 else -1
                break

            if score is not None:
                cp = score.white().score()

                if not is_loud(board, move):
                    samples.append({"fen": board.fen(), "cp": cp})

                if (tb := tb_outcome(board)) is not None:
                    outcome = tb
                    break

            board.push(move)

        if outcome is None:
            outcome = OUTCOMES.get(board.result(claim_draw=True), 0)

        return [{**s, "game_id": game_id, "outcome": outcome} for s in samples]

    finally:
        engine.quit()


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer: list[dict] = []

    with (
        concurrent.futures.ProcessPoolExecutor(max_workers=N_CONCURRENT) as executor,
        tqdm(
            unit=" positions",
            bar_format="{n:,}{unit} [{elapsed}, {rate_fmt}{postfix}]",
        ) as pbar,
    ):
        pending = {executor.submit(play_one_game) for _ in range(N_CONCURRENT)}

        while pending:
            completed, pending = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for future in completed:
                try:
                    rows = future.result()
                except Exception as exc:
                    tqdm.write(f"Worker failed: {exc!r}")
                    rows = []

                pbar.update(len(rows))

                buffer.extend(rows)
                pending.add(executor.submit(play_one_game))

            while len(buffer) >= ROWS_PER_FILE:
                name = uuid.uuid4().hex
                output_path = output_dir / f"{name}.parquet"
                pl.DataFrame(
                    buffer[:ROWS_PER_FILE],
                    schema={
                        "fen": pl.String,
                        "game_id": pl.Binary,
                        "cp": pl.Int16,
                        "outcome": pl.Int8,
                    },
                ).write_parquet(output_path, compression="zstd")
                del buffer[:ROWS_PER_FILE]


if __name__ == "__main__":
    main()

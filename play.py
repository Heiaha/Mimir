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


DRAW_ADJ_SCORE = 15
DRAW_ADJ_PLIES = 8
DRAW_ADJ_MIN_PLY = 80

MAX_PLIES = 600

SYZYGY_OUTCOMES = {-2: 0.0, -1: 0.5, 0: 0.5, 1: 0.5, 2: 1.0}
OUTCOMES = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}

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
    """Exact White-perspective WDL in {0.0, 0.5, 1.0} for <=TB_MAX_MEN positions, else None."""
    if _tablebase is None or chess.popcount(board.occupied) > TB_MAX_MEN:
        return None
    try:
        wdl = _tablebase.probe_wdl(board)  # side-to-move perspective, -2..2
    except KeyError:
        return None  # material not covered by the tables we have
    score = SYZYGY_OUTCOMES[wdl]  # cursed/blessed -> draw
    return score if board.turn == chess.WHITE else 1.0 - score


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

        game_id = uuid.uuid4().hex  # tags every sample from this game, for per-game analysis
        samples: list[dict] = []
        outcome = None
        draw_streak = 0

        for _ in range(MAX_PLIES):
            if board.is_game_over(claim_draw=True):
                outcome = OUTCOMES.get(board.result(claim_draw=True), 0.5)
                break

            analysis = engine.play(
                board,
                limit=chess.engine.Limit(depth=DEPTH),
                info=chess.engine.INFO_SCORE,
            )

            move = analysis.move
            if move is None:
                break

            score = analysis.info.get("score")

            if score is not None and score.is_mate():
                outcome = 1.0 if score.white().mate() > 0 else 0.0
                break

            if score is not None:
                cp = score.white().score()

                if not is_loud(board, move):
                    samples.append({"fen": board.fen(), "cp": cp})

                if (tb := tb_outcome(board)) is not None:
                    outcome = tb
                    break

                draw_streak = draw_streak + 1 if abs(cp) <= DRAW_ADJ_SCORE else 0

                if draw_streak >= DRAW_ADJ_PLIES and board.ply() >= DRAW_ADJ_MIN_PLY:
                    outcome = 0.5
                    break

            board.push(move)

        if outcome is None:
            return []

        return [{**s, "game_id": game_id, "outcome": outcome} for s in samples]

    finally:
        engine.quit()


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer: list[dict] = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=N_CONCURRENT
    ) as executor, tqdm(unit=" positions") as pbar:
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

            # Flush whole shards; the trailing partial stays buffered for next time.
            while len(buffer) >= ROWS_PER_FILE:
                name = uuid.uuid4().hex
                output_path = output_dir / f"{name}.parquet"
                pl.DataFrame(buffer[:ROWS_PER_FILE]).write_parquet(output_path, compression="zstd")
                del buffer[:ROWS_PER_FILE]


if __name__ == "__main__":
    main()

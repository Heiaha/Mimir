import concurrent.futures
import random
import uuid
from pathlib import Path

import chess
import chess.engine
from tqdm import tqdm

ENGINE = "Weiawaga.exe"
DEPTH = 8
ROWS_PER_FILE = 100_000
OPENING_PLIES = 8
ADJ_SCORE = 1000
MAX_PLIES = 400
OUTPUT_DIR = "data/selfplay"
N_CONCURRENT = 12

RESULTS = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def is_loud(board: chess.Board, move: chess.Move) -> bool:
    return bool(
        move.promotion
        or board.is_capture(move)
        or board.is_en_passant(move)
        or board.is_check()
        or board.gives_check(move)
    )


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

        game = object()  # Makes python-chess send ucinewgame for this game.
        samples: list[tuple[str, int]] = []
        result = None

        for _ in range(MAX_PLIES):
            if board.is_game_over(claim_draw=True):
                break

            analysis = engine.play(
                board,
                limit=chess.engine.Limit(depth=DEPTH),
                info=chess.engine.INFO_SCORE,
                game=game,
            )

            move = analysis.move
            if move is None:
                break

            score = analysis.info.get("score")
            if score is not None and not score.is_mate():
                cp = score.white().score()

                if not is_loud(board, move):
                    samples.append((board.fen(), cp))

                # Adjudicate clearly decided games to stop shuffling.
                if abs(cp) >= ADJ_SCORE:
                    result = 1.0 if cp > 0 else 0.0
                    break

            board.push(move)

        # Unadjudicated games take their played-out result, defaulting to a draw
        # if they ran past MAX_PLIES without finishing.
        if result is None:
            result = RESULTS.get(board.result(claim_draw=True), 0.5)

        return [{"fen": fen, "cp": cp, "result": result} for fen, cp in samples]

    finally:
        engine.quit()


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for stale in output_dir.glob("*.tmp"):
        stale.unlink()

    buffer: list[str] = []

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

                buffer.extend(f"{r['fen']},{r['cp']},{r['result']}" for r in rows)
                pending.add(executor.submit(play_one_game))

            # Flush whole shards; the trailing partial stays buffered for next time.
            while len(buffer) >= ROWS_PER_FILE:
                name = uuid.uuid4().hex
                output_path = output_dir / f"{name}.csv"
                output_path.write_text("\n".join(buffer[:ROWS_PER_FILE]), encoding="utf-8")
                del buffer[:ROWS_PER_FILE]


if __name__ == "__main__":
    main()

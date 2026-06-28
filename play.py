import concurrent.futures
import random
import uuid
from pathlib import Path

import chess
import chess.engine
from tqdm import tqdm

ENGINE = "Weiawaga.exe"
DEPTH = 8
GAMES_PER_FILE = 500
OPENING_PLIES = 8
ADJ_SCORE = 1000
MAX_PLIES = 400
OUTPUT_DIR = "data/selfplay"
N_CONCURRENT = 12


def result_to_float(result: str) -> float | None:
    if result == "1-0":
        return 1.0

    if result == "0-1":
        return 0.0

    if result == "1/2-1/2":
        return 0.5

    return None


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


def play() -> Path:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = output_dir / f"{uuid.uuid4().hex}.csv"
    temporary_path = output_filename.with_suffix(".tmp")

    random.seed()
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE)

    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
            for _ in range(GAMES_PER_FILE):
                board = random_opening(OPENING_PLIES)

                if board.is_game_over(claim_draw=True):
                    continue

                game = object()  # Makes python-chess send ucinewgame per game.
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
                        cp_value = score.white().score()

                        if cp_value is not None:
                            if not is_loud(board, move):
                                samples.append((board.fen(), cp_value))

                            # Adjudicate clearly decided games to stop shuffling.
                            if abs(cp_value) >= ADJ_SCORE:
                                result = 1.0 if cp_value > 0 else 0.0
                                break

                    board.push(move)

                if result is None:
                    result = result_to_float(board.result(claim_draw=True))

                if result is None:
                    continue

                output_file.writelines(
                    f"{fen},{cp_value},{result}\n"
                    for fen, cp_value in samples
                )

        temporary_path.replace(output_filename)
        return output_filename

    finally:
        engine.quit()


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for stale in output_dir.glob("*.tmp"):
        stale.unlink()

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=N_CONCURRENT
    ) as executor, tqdm(desc="shards", unit=" file") as pbar:
        pending = {
            executor.submit(play)
            for _ in range(N_CONCURRENT)
        }

        while pending:
            completed, pending = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for future in completed:
                try:
                    filename = future.result()
                except Exception as exc:
                    tqdm.write(f"Worker failed: {exc!r}")
                else:
                    pbar.update(1)
                    pbar.set_postfix_str(filename.name)

                pending.add(executor.submit(play))


if __name__ == "__main__":
    main()

import concurrent.futures
import itertools
import uuid
from glob import glob
from pathlib import Path

import chess
import chess.engine
import chess.pgn
from tqdm import tqdm

ENGINE = "weiawaga_v7.exe"
DEPTH = 8
ROWS_PER_FILE = 1_000_000
INPUT_GLOB = "data/lichess_elite/split/*"
OUTPUT_DIR = "data/lichess_elite/scored"
N_CONCURRENT = 5

RESULTS = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def split_file(input_filename, output_dir, games_per_file=10_000):

    file_n = 0

    # Construct initial output filename
    base_output_filename = Path(input_filename).stem
    output_filename = Path(output_dir) / f"{base_output_filename}_{file_n}.pgn"

    with open(input_filename, "r") as input_file, open(
        output_filename, "w"
    ) as output_file:
        count = 0

        # Read and write games
        while game := chess.pgn.read_game(input_file):

            if game.errors:
                continue  # Skip games with errors

            output_file.write(f"{game}\n\n")
            count += 1

            # Check if it's time to split to a new file
            if count >= games_per_file:
                file_n += 1
                count = 0
                output_file.close()
                output_filename = Path(output_dir) / f"{base_output_filename}_{file_n}.pgn"
                output_file = open(output_filename, "w")


def is_loud(board: chess.Board, move: chess.Move) -> bool:
    return bool(
        move.promotion
        or board.is_capture(move)
        or board.is_en_passant(move)
        or board.is_check()
        or board.gives_check(move)
    )


def score_one_game(start_fen, moves_uci, result) -> list[dict]:
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE)

    try:
        board = chess.Board(start_fen)
        game = object()  # Makes python-chess send ucinewgame for this game.
        rows: list[dict] = []

        for uci in moves_uci:
            analysis = engine.play(
                board,
                limit=chess.engine.Limit(depth=DEPTH),
                info=chess.engine.INFO_SCORE,
                game=game,
            )
            engine.configure({"Clear Hash": None})

            score = analysis.info.get("score")
            if (
                score is not None
                and not score.is_mate()
                and not is_loud(board, analysis.move)
            ):
                rows.append(
                    {"fen": board.fen(), "cp": score.white().score(), "result": result}
                )

            board.push(chess.Move.from_uci(uci))

        return rows

    finally:
        engine.quit()


def read_games():
    filenames = [f for f in sorted(glob(INPUT_GLOB), reverse=True) if "2021" in f]

    for filename in filenames:
        with open(filename, "r") as input_file:
            while game := chess.pgn.read_game(input_file):
                if game.errors:
                    continue

                result = RESULTS.get(game.headers["Result"])
                moves_uci = [move.uci() for move in game.mainline_moves()]

                if result is None or not moves_uci:
                    continue

                yield game.board().fen(), moves_uci, result


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for stale in output_dir.glob("*.tmp"):
        stale.unlink()

    games = read_games()
    buffer: list[str] = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=N_CONCURRENT
    ) as executor, tqdm(desc="rows", unit=" row") as pbar:
        pending = {
            executor.submit(score_one_game, *game)
            for game in itertools.islice(games, N_CONCURRENT)
        }

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

                # Keep the window full while games remain (islice yields 0 or 1).
                for game in itertools.islice(games, 1):
                    pending.add(executor.submit(score_one_game, *game))

            # Flush whole shards; the trailing partial stays buffered for next time.
            while len(buffer) >= ROWS_PER_FILE:
                name = uuid.uuid4().hex
                output_path = output_dir / f"{name}.csv"
                output_path.write_text("\n".join(buffer[:ROWS_PER_FILE]), encoding="utf-8")
                del buffer[:ROWS_PER_FILE]

        # Input is exhausted -- flush the final partial shard so no rows are lost.
        if buffer:
            name = uuid.uuid4().hex
            output_path = output_dir / f"{name}.csv"
            output_path.write_text("".join(buffer[:ROWS_PER_FILE]), encoding="utf-8")
            del buffer[:ROWS_PER_FILE]


if __name__ == "__main__":
    main()

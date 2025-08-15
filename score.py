import os
import time
import concurrent.futures

import chess
import chess.engine
import chess.pgn

from glob import glob
from tqdm import tqdm


def split_file(input_filename, output_dir, games_per_file=10_000):

    file_n = 0

    # Construct initial output filename
    base_output_filename, _ = os.path.splitext(os.path.basename(input_filename))
    output_filename = os.path.join(output_dir, f"{base_output_filename}_{file_n}.pgn")

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
                output_filename = os.path.join(
                    output_dir, f"{base_output_filename}_{file_n}.pgn"
                )
                output_file = open(output_filename, "w")


def result_to_float(result):
    if result == "1-0":
        return 1.0

    if result == "0-1":
        return 0.0

    if result == "1/2-1/2":
        return 0.5

    return None


def is_loud(board: chess.Board, move: chess.Move):
    return (
        move.promotion
        or board.is_capture(move)
        or board.is_en_passant(move)
        or board.is_check()
        or board.gives_check(move)
    )


def score(input_filename, *, output_dir, engine, depth):
    engine = chess.engine.SimpleEngine.popen_uci(engine)
    output_basename = os.path.basename(input_filename).replace(".pgn", ".csv")
    output_filename = os.path.join(output_dir, output_basename)

    if os.path.exists(output_filename):
        return output_filename

    with open(input_filename, "r") as input_file, open(
        output_filename, "w"
    ) as output_file:

        while game := chess.pgn.read_game(input_file):

            if game.errors:
                continue

            result = result_to_float(game.headers["Result"])
            if result is None:
                continue

            board = game.board()
            for node in game.mainline():

                analysis = engine.play(
                    board,
                    limit=chess.engine.Limit(depth=depth),
                    info=chess.engine.Info.ALL,
                    game=game,
                )
                engine.configure({"Clear Hash": None})

                if (
                    (cp := analysis.info.get("score"))
                    and not cp.is_mate()
                    and not is_loud(board, analysis.move)
                ):
                    output_file.write(f"{board.fen()},{cp.white()},{result}\n")

                board.push(node.move)

    engine.quit()
    return output_filename


def main(func, *, input_glob, n_concurrent, func_kwargs: dict):
    filenames = [filename for filename in sorted(glob(input_glob), reverse=True) if "2021" in filename]

    with concurrent.futures.ProcessPoolExecutor() as executor, tqdm(
        total=len(filenames)
    ) as pbar:
        pending = set()
        for _ in range(n_concurrent):
            pending.add(
                executor.submit(func, input_filename=filenames.pop(), **func_kwargs)
            )

        while pending:
            completed, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            pbar.update()

            if filenames:
                pending.add(
                    executor.submit(func, input_filename=filenames.pop(), **func_kwargs)
                )


if __name__ == "__main__":
    main(
        score,
        input_glob="data/lichess_elite/split/*",
        n_concurrent=5,
        func_kwargs={
            "output_dir": "data/lichess_elite/scored",
            "depth": 8,
            "engine": "weiawaga_v7.exe",
        },
    )

import os
import asyncio

import chess
import chess.engine
import chess.pgn

from glob import glob


N_CONCURRENT = 5
ENGINE = "Weiawaga.exe"
OUTPUT_DIR = "data_d8_v2"
INPUT_GLOB = "lichess/*"
DEPTH = 8


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


async def score(input_path):
    transport, engine = await chess.engine.popen_uci(ENGINE)
    output_filename = os.path.basename(input_path).replace(".pgn", ".csv")

    with open(input_path, "r") as input_file, open(os.path.join(OUTPUT_DIR, output_filename), "w") as output_file:

        while game := chess.pgn.read_game(input_file):
            result = result_to_float(game.headers["Result"])
            if result is None:
                continue

            board = game.board()
            for node in game.mainline():

                analysis = await engine.play(
                    board, limit=chess.engine.Limit(depth=DEPTH), info=chess.engine.Info.ALL
                )

                if (
                        (cp := analysis.info.get("score"))
                        and not cp.is_mate()
                        and not is_loud(board, analysis.move)
                ):
                    output_file.write(
                        f"{board.fen()},{cp.white()},{result}\n"
                    )

                board.push(node.move)


async def main():

    filenames = sorted(glob(INPUT_GLOB))
    os.mkdir(OUTPUT_DIR)

    pending = set()
    for _ in range(N_CONCURRENT):
        pending.add(asyncio.create_task(score(filenames.pop())))

    while pending:
        completed, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED
        )

        if filenames:
            pending.add(
                asyncio.create_task(score(filenames.pop()))
            )

if __name__ == "__main__":
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(main())
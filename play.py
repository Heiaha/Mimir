import chess
import chess.engine
import chess.polyglot
import random
import asyncio
import csv
import datetime

from dataclasses import dataclass
from tqdm import tqdm


# Options
ENGINE_NAME = "Weiawaga.exe"
N_GAMES = 10_000
N_CONCURRENT = 10
MIN_DEPTH = 8
MAX_DEPTH = 10
MAX_PLY = 400
DRAW_SCORE = 10
DRAW_COUNT = 10
MIN_DRAW_PLY = 80

with open("filtered_openings.epd") as file:
    OPENINGS = file.readlines()

OUTCOME_MAP = {"1-0": 1.0, "1/2-1/2": 0.5, "*": 0.5, "0-1": 0.0}


def is_loud(board, move):
    return (
        move.promotion
        or board.is_capture(move)
        or board.is_check()
        or board.gives_check(move)
    )


async def play():

    transport, engine = await chess.engine.popen_uci(ENGINE_NAME)

    draw_count = 0

    fens = []
    scores = []
    zobrist_hashes = set()
    board = chess.Board(random.choice(OPENINGS))

    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLY:
        limit = chess.engine.Limit(depth=random.randint(MIN_DEPTH, MAX_DEPTH))
        result = await engine.play(board, limit=limit, info=chess.engine.Info.SCORE)
        if "score" not in result.info:
            board.push(result.move)
            continue

        score = result.info["score"].white()

        if score.is_mate() or is_loud(board, result.move):
            board.push(result.move)
            continue

        if (key := chess.polyglot.zobrist_hash(board)) in zobrist_hashes:
            board.push(result.move)
            continue
        else:
            zobrist_hashes.add(key)

        fens.append(board.fen())
        scores.append(score.score())
        board.push(result.move)

        if board.ply() > MIN_DRAW_PLY:
            if abs(score.score()) <= DRAW_SCORE:
                draw_count += 1
                if draw_count >= DRAW_COUNT:
                    break
            else:
                draw_count = 0

    await engine.quit()
    outcome = OUTCOME_MAP[board.result(claim_draw=True)]
    return zip(fens, scores, [outcome] * len(fens))


async def main():

    pending = {asyncio.create_task(play()) for _ in range(N_CONCURRENT)}
    finished_games = 0

    with open(
        f"./data_test/{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.csv",
        "w",
        newline="",
    ) as out_file, tqdm(total=N_GAMES) as pbar:

        csv_writer = csv.writer(out_file)
        while pending:

            # wait for a game to complete
            completed, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            # write to the output file
            for task in completed:
                # add another to the queue
                csv_writer.writerows(task.result())
                finished_games += 1
                pbar.update()
                if finished_games + len(pending) < N_GAMES:
                    pending.add(asyncio.create_task(play()))


if __name__ == "__main__":
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    while True:
        asyncio.run(main())

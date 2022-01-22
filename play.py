import chess
import chess.engine
import chess.polyglot
import random
import asyncio
import csv
import os
import datetime
from tqdm import tqdm

# Options
ENGINE_NAME = "Weiawaga.exe"
N_GAMES = 10_000
N_RANDOM = 8
N_CONCURRENT = 5
DEPTH = 10

OUTCOME_MAP = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}


def is_mate_score(score):
    return 2 * abs(score) >= 10_000


def is_loud(board, move):
    return (
        move.promotion
        or board.is_capture(move)
        or board.is_check()
        or board.gives_check(move)
    )


async def play():

    board = chess.Board()
    for _ in range(N_RANDOM):
        move = random.choice(list(board.legal_moves))
        board.push(move)
        if board.is_game_over(claim_draw=True):
            return None

    fens = []
    scores = []
    board_hashes = set()
    limit = chess.engine.Limit(depth=DEPTH)
    transport, engine = await chess.engine.popen_uci(ENGINE_NAME)

    while not board.is_game_over(claim_draw=True):
        result = await engine.play(board, limit=limit, info=chess.engine.Info.SCORE)
        score = result.info["score"].white().score()

        if is_mate_score(score) or is_loud(board, result.move):
            board.push(result.move)
            continue

        if key := board._transposition_key() in board_hashes:
            board.push(result.move)
            continue
        else:
            board_hashes.add(key)

        fens.append(board.fen())
        scores.append(score)
        board.push(result.move)

    await engine.quit()
    outcome = OUTCOME_MAP[board.outcome(claim_draw=True).result()]
    return zip(fens, scores, [outcome for _ in scores])


async def main():

    if not os.path.exists("./data/"):
        os.mkdir("./data/")

    pending = {asyncio.create_task(play()) for _ in range(N_CONCURRENT)}

    finished_games = 0
    with open(
        f"./data/{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.csv",
        "w",
        newline="",
    ) as out_file, tqdm(total=N_GAMES) as pbar:

        csv_writer = csv.writer(out_file)
        while finished_games < N_GAMES:

            # wait for a game to complete
            completed, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            # write to the output file
            for task in completed:
                # add another to the queue
                pending.add(asyncio.create_task(play()))
                if task.result() is not None:
                    finished_games += 1
                    pbar.update()
                    csv_writer.writerows(task.result())

    # Finish up the last few games.
    await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)


if __name__ == "__main__":
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(main())

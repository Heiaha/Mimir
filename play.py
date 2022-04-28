import chess
import chess.engine
import random
import asyncio
import csv
import datetime
from tqdm import tqdm

# Options
ENGINE_NAME = "Weiawaga.exe"
N_GAMES = 10_000
N_CONCURRENT = 10
MIN_DEPTH = 7
MAX_DEPTH = 10

with open("filtered_openings.epd") as file:
    OPENINGS = file.readlines()

OUTCOME_MAP = {chess.WHITE: 1.0, None: 0.5, chess.BLACK: 0.0}


def is_loud(board, move):
    return (
        move.promotion
        or board.is_capture(move)
        or board.is_check()
        or board.gives_check(move)
    )


async def play():

    transport, engine = await chess.engine.popen_uci(ENGINE_NAME)

    fens = []
    scores = []
    board_hashes = set()
    board = chess.Board(random.choice(OPENINGS))

    while not board.is_game_over(claim_draw=True):
        limit = chess.engine.Limit(depth=random.randint(MIN_DEPTH, MAX_DEPTH))
        result = await engine.play(board, limit=limit, info=chess.engine.Info.SCORE)
        if result.info.get("score") is None:
            board.push(result.move)
            continue

        score = result.info["score"].white()

        if score.is_mate() or is_loud(board, result.move):
            board.push(result.move)
            continue

        if (key := board._transposition_key()) in board_hashes:
            board.push(result.move)
            continue
        else:
            board_hashes.add(key)

        fens.append(board.fen())
        scores.append(score.score())
        board.push(result.move)

    await engine.quit()
    outcome = OUTCOME_MAP[board.outcome(claim_draw=True).winner]
    return zip(fens, scores, [outcome] * len(fens))


async def main():

    pending = {asyncio.create_task(play()) for _ in range(N_CONCURRENT)}
    finished_games = 0

    with open(
        f"./data/{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.csv",
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

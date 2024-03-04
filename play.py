import chess
import chess.engine
import chess.polyglot
import random
import asyncio
import csv
import datetime

from tqdm import tqdm


# Options
ENGINE_NAME = "Weiawaga.exe"
OUTPUT_DIR = "data"
N_GAMES = 10_000
N_CONCURRENT = 5
DEPTH = 8

RANDOM_MOVE_PROB = 0.05
MAX_PLY = 400

DRAW_SCORE = 20
DRAW_COUNT = 10
MIN_DRAW_PLY = 40

with open("noob_3moves.epd") as file:
    OPENINGS = file.readlines()


def is_loud(board: chess.Board, move: chess.Move):
    return (
        move.promotion
        or board.is_capture(move)
        or board.is_en_passant(move)
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
    while not board.is_game_over(claim_draw=True) and (ply := board.ply()) < MAX_PLY:

        results = await engine.analyse(
            board, limit=chess.engine.Limit(depth=DEPTH), info=chess.engine.Info.ALL, multipv=2
        )

        if random.random() < RANDOM_MOVE_PROB and len(results) > 1:
            result = results[1]
        else:
            result = results[0]

        move = result["pv"][0]

        if "score" not in result:
            board.push(move)
            continue

        score = result["score"].white()

        if score.is_mate() or is_loud(board, move):
            board.push(move)
            continue

        if (key := chess.polyglot.zobrist_hash(board)) in zobrist_hashes:
            board.push(move)
            continue
        else:
            zobrist_hashes.add(key)

        fens.append(board.fen())
        scores.append(score.score())

        if ply >= MIN_DRAW_PLY:
            if abs(score.score()) <= DRAW_SCORE:
                draw_count += 1
                if draw_count >= DRAW_COUNT:
                    break
            else:
                draw_count = 0

        board.push(move)

    if outcome := board.outcome(claim_draw=True):
        if outcome.winner == chess.WHITE:
            outcome_value = 1.0
        elif outcome.winner == chess.BLACK:
            outcome_value = 0.0
        else:
            outcome_value = 0.5
    else:
        outcome_value = 0.5

    await engine.quit()

    return zip(fens, scores, [outcome_value] * len(fens))


async def main():

    pending = {asyncio.create_task(play()) for _ in range(N_CONCURRENT)}
    finished_games = 0

    with open(
        f"./{OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.csv",
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

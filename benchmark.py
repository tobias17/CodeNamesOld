#!/usr/bin/env python
#!/usr/bin/env python

import argparse
import engine
import platform
import os

from tqdm import tqdm
import numpy as np

HISTOGRAM_WIDTH = 100


class ClueInfo:
    def __init__(self, clue_count):
        self.clue_dists = [
            [0 for i in range(HISTOGRAM_WIDTH)] for i in range(clue_count + 1)
        ]
        self.neutral_dist_max = [0 for i in range(HISTOGRAM_WIDTH)]
        self.neutral_dist_avg = [0 for i in range(HISTOGRAM_WIDTH)]
        self.negative_dist_max = [0 for i in range(HISTOGRAM_WIDTH)]
        self.negative_dist_avg = [0 for i in range(HISTOGRAM_WIDTH)]
        self.assassin_dist = [0 for i in range(HISTOGRAM_WIDTH)]


class Tracker:
    def __init__(self):
        self.word_counts = [0 for i in range(9)]
        self.clue_infos = [ClueInfo(i) for i in range(9)]

    def add(self, clue, words, e):
        index = len(words) - 1
        self.word_counts[index] += 1

        for i, sim in enumerate(get_sims(clue, words, e)):
            self.clue_infos[index].clue_dists[i][int(sim * HISTOGRAM_WIDTH)] += 1

        neutral_sims = get_sims(clue, [str(word)[2:-1] for word in e.neutral_words], e)
        self.clue_infos[index].neutral_dist_max[
            int(neutral_sims[0] * HISTOGRAM_WIDTH)
        ] += 1
        for sim in neutral_sims:
            self.clue_infos[index].neutral_dist_avg[int(sim * HISTOGRAM_WIDTH)] += 1

        negative_sims = get_sims(
            clue, [str(word)[2:-1] for word in e.opponent_words], e
        )
        self.clue_infos[index].negative_dist_max[
            int(negative_sims[0] * HISTOGRAM_WIDTH)
        ] += 1
        for sim in negative_sims:
            self.clue_infos[index].negative_dist_avg[int(sim * HISTOGRAM_WIDTH)] += 1

        assassin_sim = get_sims(clue, [str(e.assassin_word)[3:-2]], e)
        self.clue_infos[index].assassin_dist[
            int(assassin_sim[0] * HISTOGRAM_WIDTH)
        ] += 1

    def size(self):
        return sum(self.word_counts)

    def save_to_file(self, filename):
        with open(filename, "w+") as file:
            file.write(",".join(str(word) for word in self.word_counts) + "\n")
            for i in range(9):
                info = self.clue_infos[i]
                for j in range(i + 1):
                    file.write(
                        ",".join(str(word) for word in info.clue_dists[j]) + "\n"
                    )
                file.write(",".join(str(word) for word in info.neutral_dist_max) + "\n")
                file.write(",".join(str(word) for word in info.neutral_dist_avg) + "\n")
                file.write(
                    ",".join(str(word) for word in info.negative_dist_max) + "\n"
                )
                file.write(
                    ",".join(str(word) for word in info.negative_dist_avg) + "\n"
                )
                file.write(",".join(str(word) for word in info.assassin_dist) + "\n")

    def load_from_file(self, filename):
        with open(filename, "r") as file:
            lines = file.readlines()[::-1]
            self.word_counts = [int(item) for item in lines.pop().strip().split(",")]
            for i in range(9):
                info = ClueInfo(i)
                for j in range(i + 1):
                    info.clue_dists[j] = [
                        int(item) for item in lines.pop().strip().split(",")
                    ]
                info.neutral_dist_max = [
                    int(item) for item in lines.pop().strip().split(",")
                ]
                info.neutral_dist_avg = [
                    int(item) for item in lines.pop().strip().split(",")
                ]
                info.negative_dist_max = [
                    int(item) for item in lines.pop().strip().split(",")
                ]
                info.negative_dist_avg = [
                    int(item) for item in lines.pop().strip().split(",")
                ]
                info.assassin_dist = [
                    int(item) for item in lines.pop().strip().split(",")
                ]
                self.clue_infos[i] = info


def get_sims(clue, words, e):
    sims = []
    for word in words:
        sims.append(e.model.model.wv.similarity(clue, word))
    for i, sim in enumerate(sims):
        if sim >= 1.0:
            sims[i] = 0.995
        if sim < 0.0:
            sims[i] = 0.0
    sims.sort(reverse=True)
    return sims


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        sys.stdout.write(chr(27) + "[2J")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark an ai.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1000,
        help="How many games should be generated and tested",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed to have identical matches to benchmark",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="benchmarks/benchmark.txt",
        help="Name of file the benchmark gets saved to",
    )
    parser.add_argument(
        "--writealong", type=bool, default=False, help="Writes to file after every loop"
    )
    parser.add_argument(
        "--pickup",
        type=bool,
        default=False,
        help="Whether to open the file and pick up",
    )
    args = parser.parse_args()

    generator = np.random.RandomState(seed=args.seed)

    e = engine.GameEngine(seed=args.seed, display=False)

    tracker = Tracker()
    skip_index = 0
    if args.pickup:
        tracker.load_from_file(args.filename)
        skip_index = tracker.size()

    clear_screen()
    for i in tqdm(range(args.iters)):
        e.initialize_random_game()
        e.next_turn()
        if i % 2 == 0:
            e.next_turn()
        if i < skip_index:
            continue
        e.print_board(clear_screen=False, override=True)
        clue, words = e.play_computer_spymaster(give_words=True)
        tracker.add(str(clue)[2:-1], [str(word)[2:-1] for word in words], e)
        clear_screen()
        if args.writealong:
            tracker.save_to_file(args.filename)
    print(tracker.word_counts)
    print("Saving to file...")
    tracker.save_to_file(args.filename)


if __name__ == "__main__":
    main()

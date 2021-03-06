#!/usr/bin/env python
from __future__ import print_function, division

import argparse, os, sys
import re
import numpy as np
import cv2 as cv
import pytesseract
import platform
import itertools
from tqdm import tqdm

from config import config
from datetime import datetime

import engine


def say(message):
    sys.stdout.write(message + "\n")


def ask(message):
    try:
        return input(message)
    except KeyboardInterrupt:
        say("\nBye.")
        sys.exit(0)


def print_board(board, active, clear_screen=True, spymaster=True):

    if clear_screen:
        if platform.system() == "Windows":
            os.system("cls")
        else:
            sys.stdout.write(chr(27) + "[2J")

    for row in range(5):
        for col in range(5):
            index = row * 5 + col
            word = board[index]
            tag = " "
            if not active[word]:
                word = word + "-" * (11 - len(word))
                tag = "-"
            sys.stdout.write("{0}{1:11s} ".format(tag, word))
        sys.stdout.write("\n")

# TODO: this needs to be piped into engine, not copied and pasted

def play_computer_spymaster(
    engine,
    player_words,
    opponent_words,
    neutral_words,
    assassin_word,
    given_clues,
    gamma=1.0,
    verbose=True,
):

    say("Thinking...")
    sys.stdout.flush()

    # # Loop over all permutations of words.
    # num_words = len(player_words)
    # best_score, saved_clues = [], []
    # counts = range(num_words, 0, -1)
    # groups_and_count = []
    # for count in counts:
    #     for group in itertools.combinations(range(num_words), count):
    #         groups_and_count.append((group, count,))
    # for group, count in tqdm(groups_and_count):
    #     # Multiply similarity scores by this factor for any clue
    #     # corresponding to this many words.
    #     bonus_factor = count ** gamma
    #     # print(type(player_words), player_words)
    #     words = player_words[list(group)]
    #     clue, score = engine.model.get_clue(
    #         clue_words=words,
    #         pos_words=player_words,
    #         neg_words=np.concatenate((opponent_words, neutral_words)),
    #         veto_words=assassin_word,
    #     )
    #     if clue:
    #         best_score.append(score * bonus_factor)
    #         saved_clues.append((clue, words))
    # num_clues = len(saved_clues)
    # order = sorted(range(num_clues), key=lambda k: best_score[k], reverse=True)

    max_group = 2
    does_stretch = [2]
    num_words = len(player_words)
    best_score, saved_clues = [], []
    counts = range(min(num_words, max_group), 0, -1)
    groups_and_count = []
    for count in counts:
        for group in itertools.combinations(range(num_words), count):
            groups_and_count.append((group, count,))
    groups_and_count = tqdm(groups_and_count)
    for group, count in groups_and_count:
        # Multiply similarity scores by this factor for any clue
        # corresponding to this many words.
        bonus_factor = count ** gamma
        words = player_words[list(group)]
        clue, score, stretch = engine.model.get_clue_stretch(
            clue_words=words,
            pos_words=player_words,
            neg_words=opponent_words,
            neut_words=neutral_words,
            veto_words=assassin_word,
            given_clues=given_clues,
            give_stretch=(count in does_stretch),
        )
        if clue:
            best_score.append(score * bonus_factor)
            clue_words = words
            if stretch:
                clue_words = np.concatenate((words, np.asarray(stretch)))
            saved_clues.append((clue, clue_words))

    num_clues = len(saved_clues)
    order = sorted(range(num_clues), key=lambda k: best_score[k], reverse=True)

    if not os.path.exists(config.logs_folder):
        os.makedirs(config.logs_folder)
    with open(engine.logs_filename, 'a+') as f:
        for i in order[:10]:
            clue, words = saved_clues[i]
            f.write(
                u"{0:.3f} {2} {3} = {1}\n".format(
                    best_score[i],
                    " + ".join([str(w).upper()[2:-1] for w in words]),
                    str(clue)[2:-1],
                    len(words),
                )
            )
        f.write("\n")

    # print_board(board, active, spymaster=True)
    values = []
    for i in order[:10]:
        values.append(
            [saved_clues[i][0], saved_clues[i][1], best_score[i],]
        )
    return values


colors = [
    (92.5, 66.7, 66.7),
    (65.9, 84.7, 92.2),
    (97.3, 99.2, 100.0),
    (40.8, 42.4, 42.7),
]
colors = list([np.uint8(c * 2.55) for c in color[::-1]] for color in colors)
color_names = ["red", "blue", "white", "black"]


def main():
    parser = argparse.ArgumentParser(
        description="Play the CodeNames game.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nohide",
        type=bool,
        default=False,
        help="Shows only 1 clue, hides what the words for each clue",
    )
    args = parser.parse_args()

    e = engine.GameEngine()

    # manual init
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    )
    config = "-l eng --oem 1 --psm 3"
    e.size = 5
    pad = 20
    img = cv.imread("board.png")
    colors_loc = [[], [], [], []]
    word_color_map = {}
    board = []
    cx, cy = img.shape[1] // 5, img.shape[0] // 5
    for y in range(5):
        for x in range(5):
            img_sec = img[
                y * cy + pad : (y + 1) * cy - pad, x * cx + pad : (x + 1) * cx - pad
            ]
            img_gray = cv.cvtColor(img_sec, cv.COLOR_BGR2GRAY)
            avg = np.average(img_gray)
            mask1 = img_gray[:, :] < avg
            mask2 = img_gray[:, :] > avg
            img_gray[mask2] = 0
            img_gray[mask1] = 255
            kernel = np.ones((3, 3), np.uint8)
            erosion = cv.erode(img_gray, kernel, iterations=1)
            text = pytesseract.image_to_string(img_sec, config=config).replace(" ", "_")
            board.append(text)
            dev = [0, 0, 0, 0]
            for c in range(3):
                avg = np.median(img_sec[:, :, c])
                for i, color in enumerate(colors):
                    # print(avg, color[c])
                    dev[i] += (avg - color[c]) ** 2
            index, smol = 0, dev[0]
            if not args.nohide:
                print("({},{}) -> {}".format(x + 1, y + 1, text))
            else:
                print("({},{}) -> {} = {}".format(x + 1, y + 1, text, dev))
            for i in range(1, 4):
                if dev[i] < smol:
                    smol = dev[i]
                    index = i
            colors_loc[index].append((x, y,))
            word_color_map[text] = color_names[index]
            # cv.imshow('img', erosion)
            # cv.imshow('img_pre', img_sec)
            # cv.waitKey()

    # play the game
    active = {}
    for word in board:
        active[word] = True
    input("Loaded successfully! Hit enter to continue: ")
    print_board(board, active)
    values = None
    index = 0
    side = ""
    given_words = []
    while True:
        print_board(board, active)
        if values:
            if not args.nohide:
                clue, words, best_score = values[index]
                say(u"{3}{0:.3f} {1} {2}".format(best_score, str(clue)[2:-1], len(words), side))
            else:
                for clue, words, best_score in values:
                    say(
                        u"{0:.3f} {2} {3} = {1}".format(
                            best_score,
                            " + ".join([str(w).upper()[2:-1] for w in words]),
                            str(clue)[2:-1],
                            len(words),
                        )
                    )
        text = input().upper()
        if text in ("EXIT", "QUIT"):
            return
        if text in board:
            active[text] = not active[text]
        if text in ("USE", "USED"):
            given_words.append(clue)
        elif text == "RED" or text == "BLUE":
            red_words = np.asarray(
                [
                    word.lower().encode("utf8")
                    for word in board
                    if active[word] and word_color_map[word] == "red"
                ]
            )
            blue_words = np.asarray(
                [
                    word.lower().encode("utf8")
                    for word in board
                    if active[word] and word_color_map[word] == "blue"
                ]
            )
            neutral_words = np.asarray(
                [
                    word.lower().encode("utf8")
                    for word in board
                    if active[word] and word_color_map[word] == "white"
                ]
            )
            assassin_word = [
                word.lower().encode("utf8")
                for word in board
                if active[word] and word_color_map[word] == "black"
            ]
            if text == "RED":
                values = play_computer_spymaster(
                    e, red_words, blue_words, neutral_words, assassin_word, given_words
                )
                side = "red: "
            else:
                values = play_computer_spymaster(
                    e, blue_words, red_words, neutral_words, assassin_word, given_words
                )
                side = "blue: "
            index = 0
        elif text == "NEXT":
            index += 1
            if index >= len(values):
                index -= len(values)
        elif text == "PREV":
            index -= 1
            if index < 0:
                index += len(values)


if __name__ == "__main__":
    main()

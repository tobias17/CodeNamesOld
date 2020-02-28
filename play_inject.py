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

import engine


def say(message):
    sys.stdout.write(message + '\n')


def ask(message):
    try:
        return input(message)
    except KeyboardInterrupt:
        say('\nBye.')
        sys.exit(0)

def print_board(board, active, clear_screen=True, spymaster=True):

    if clear_screen:
        if platform.system() == 'Windows':
            os.system('cls')
        else:
            sys.stdout.write(chr(27) + '[2J')

    for row in range(5):
        for col in range(5):
            index = row*5 + col
            word = board[index]
            tag = ' '
            if not active[word]:
                word = word + '-' * (11 - len(word))
                tag = '-'
            sys.stdout.write('{0}{1:11s} '.format(tag, word))
        sys.stdout.write('\n')

def play_computer_spymaster(engine, player_words, opponent_words, neutral_words, assassin_word, gamma=1.0, verbose=True):

    say('Thinking...')
    sys.stdout.flush()

    # Loop over all permutations of words.
    num_words = len(player_words)
    best_score, saved_clues = [], []
    counts = range(num_words, 0, -1)
    groups_and_count = []
    for count in counts:
        for group in itertools.combinations(range(num_words), count):
            groups_and_count.append((group, count,))
    for group, count in tqdm(groups_and_count):
        # Multiply similarity scores by this factor for any clue
        # corresponding to this many words.
        bonus_factor = count ** gamma
        # print(type(player_words), player_words)
        words = player_words[list(group)]
        clue, score = engine.model.get_clue(clue_words=words,
                                          pos_words=player_words,
                                          neg_words=np.concatenate((opponent_words, neutral_words)),
                                          veto_words=assassin_word)
        if clue:
            best_score.append(score * bonus_factor)
            saved_clues.append((clue, words))
    num_clues = len(saved_clues)
    order = sorted(range(num_clues), key=lambda k: best_score[k], reverse=True)

    if not os.path.exists('logs'):
        os.makedirs('logs')
    with open('logs/clues.log', 'a+') as f:
        for i in order[:10]:
            clue, words = saved_clues[i]
            f.write(u'{0:.3f} {2} {3} = {1}\n'.format(best_score[i], ' + '.join([str(w).upper()[2:-1] for w in words]), str(clue)[2:-1], len(words)))
        f.write('\n')

    # print_board(board, active, spymaster=True)
    values = []
    for i in order[:10]:
        values.append([saved_clues[i][0], saved_clues[i][1], best_score[i],])
    return values

    # clue, words = saved_clues[order[0]]
    # self.unfound_words[self.player].update(words)
    # return clue, len(words)


colors = [(92.5, 66.7, 66.7), (65.9, 84.7, 92.2), (97.3, 99.2, 100.0), (40.8, 42.4, 42.7)]
colors = list([np.uint8(c * 2.55) for c in color[::-1]] for color in colors)
color_names = ['red', 'blue', 'white', 'black']

def main():
    parser = argparse.ArgumentParser(
        description='Play the CodeNames game.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default='CHCH',
                        help='Config <spy1><team1><spy2><team2> using C,H.')
    parser.add_argument('-x', '--expert', action='store_true',
                        help='Expert clues. For now implements \'unlimited\' only.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible games.')
    parser.add_argument('--init', type=str, default=None,
                        help='Initialize words ASSASSIN;TEAM1;TEAM2;NEUTRAL')
    parser.add_argument('--hide', type=bool, default=False,
                        help='Shows only 1 clue, hides what the words for each clue')
    args = parser.parse_args()

    if not re.match('^[CH]{4}$', args.config):
        print('Invalid configuration. Try HHHH or CHCH.')
        return -1

    d = dict(H='human', C='computer')
    spy1 = d[args.config[0]]
    team1 = d[args.config[1]]
    spy2 = d[args.config[2]]
    team2 = d[args.config[3]]

    e = engine.GameEngine(seed=args.seed, expert=args.expert)

    # with open('words.txt', encoding='utf-8') as words:
    #     with open('cluster.txt', 'w+', encoding='utf-8') as output:
    #         for word in tqdm(words):
    #             word = word.strip().lower().replace(' ', '_')
    #             output.write((word + '\n'))
    #             nearby_words = e.model.model.most_similar(positive=[word])
    #             for nearby_word in nearby_words:
    #                 output.write('{0}, {1:.4f}\n'.format(*nearby_word))
    #             output.write('\n')
    # return

    # manual init
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    config = ('-l eng --oem 1 --psm 3')
    e.size = 5
    pad = 20
    img = cv.imread('board.png')
    colors_loc = [ [], [], [], [] ]
    word_color_map = {}
    board = []
    cx, cy = img.shape[1] // 5, img.shape[0] // 5
    for y in range(5):
        for x in range(5):
            img_sec = img[y*cy+pad:(y+1)*cy-pad,x*cx+pad:(x+1)*cx-pad]
            img_gray = cv.cvtColor(img_sec, cv.COLOR_BGR2GRAY)
            avg = np.average(img_gray)
            mask1 = img_gray[:,:] < avg
            mask2 = img_gray[:,:] > avg
            img_gray[mask2] = 0
            img_gray[mask1] = 255
            kernel = np.ones((3,3),np.uint8)
            erosion = cv.erode(img_gray, kernel, iterations=1)
            text = pytesseract.image_to_string(img_sec, config=config).replace(' ', '_')
            board.append(text)
            dev = [0,0,0,0]
            for c in range(3):
                avg = np.median(img_sec[:,:,c])
                for i, color in enumerate(colors):
                    # print(avg, color[c])
                    dev[i] += (avg - color[c]) ** 2
            index, smol = 0, dev[0]
            print(x+1,y+1,dev,text)
            for i in range(1, 4):
                if dev[i] < smol:
                    smol = dev[i]
                    index = i
            colors_loc[index].append((x,y,))
            word_color_map[text] = color_names[index]
            # cv.imshow('img', erosion)
            # cv.imshow('img_pre', img_sec)
            # cv.waitKey()
    # print(word_color_map)
    # for i, color_loc in enumerate(colors_loc):
    #     for loc in color_loc:
    #         x,y = loc
    #         img[y*cy:(y+1)*cy,x*cx:(x+1)*cx] = colors[i]
    # cv.imshow('img', img)
    # cv.waitKey()
    # e.board = 
    active = {}
    for word in board:
        active[word] = True
    input("Loaded successfully! Hit enter to continue: ")
    print_board(board, active)
    values = None
    index = 0
    while True:
        print_board(board, active)
        if values:
            if args.hide:
                clue, words, best_score = values[index]
                say(u'{0:.3f} {1} {2}'.format(best_score, str(clue)[2:-1], len(words)))
            else:
                for clue, words, best_score in values:
                    say(u'{0:.3f} {2} {3} = {1}'.format(best_score, ' + '.join([str(w).upper()[2:-1] for w in words]), str(clue)[2:-1], len(words)))
        text = input().upper()
        if text in ('EXIT', 'QUIT'):
            return
        if text in board:
            active[text] = not active[text]
        elif text == 'RED' or text == 'BLUE':
            red_words = np.asarray([word.lower().encode('utf8') for word in board if active[word] and word_color_map[word] == 'red'])
            blue_words = np.asarray([word.lower().encode('utf8') for word in board if active[word] and word_color_map[word] == 'blue'])
            neutral_words = np.asarray([word.lower().encode('utf8') for word in board if active[word] and word_color_map[word] == 'white'])
            assassin_word = [word.lower().encode('utf8') for word in board if active[word] and word_color_map[word] == 'black']
            if text == 'RED':
                values = play_computer_spymaster(e, red_words, blue_words, neutral_words, assassin_word)
            else:
                values = play_computer_spymaster(e, blue_words, red_words, neutral_words, assassin_word)
            index = 0
        elif text == 'NEXT':
            index += 1
            if index >= len(values):
                index -= len(values)
        elif text == 'PREV':
            index -= 1
            if index < 0:
                index += len(values)




if __name__ == '__main__':
    main()

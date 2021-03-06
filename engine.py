from __future__ import print_function, division

import itertools
import re
import sys
import os
import platform

import numpy as np
from datetime import datetime

import model
from config import config
from tqdm import tqdm

CLUE_PATTERN = r"^([a-zA-Z]+) ({0})$"
UNLIMITED = "unlimited"

display_to_console = True

# noinspection PyAttributeOutsideInit
class GameEngine(object):
    def __init__(self, seed=None, expert=False, display=True):

        # Load our word list if necessary.
        # TODO: Max length of 11 is hardcoded here and in print_board()
        with open(config.word_list) as f:
            _words = [line.rstrip().lower().replace(" ", "_") for line in f.readlines()]
        self.words = np.array(_words, dtype="S11")

        # Initialize our word embedding model if necessary.
        self.model = model.WordEmbedding(config.embedding)
        self.logs_filename = '{}/{}.log'.format(config.logs_folder, str(datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-'))

        # Initialize random numbers.
        self.generator = np.random.RandomState(seed=seed)

        # Register expert mode
        self.expert = expert
        self.unfound_words = (set(), set())

        # If we want to display to console
        self.display = display
        global display_to_console
        display_to_console = display

        # Useful regular expressions.
        if self.expert:
            self.valid_clue = re.compile(CLUE_PATTERN.format("[0-9]|" + UNLIMITED))
        else:
            self.valid_clue = re.compile(CLUE_PATTERN.format("[0-9]"))

    def initialize_random_game(self, size=5):

        self.size = size

        # Shuffle the wordlist.
        shuffle = self.generator.choice(len(self.words), size * size, replace=False)
        self.board = self.words[shuffle]

        # Specify the layout for this game.
        assignments = self.generator.permutation(size * size)
        self.owner = np.empty(size * size, int)
        self.owner[assignments[0]] = 0  # assassin
        self.owner[assignments[1:10]] = 1  # first player: 9 words
        self.owner[assignments[10:18]] = 2  # second player: 8 words
        self.owner[assignments[18:]] = 3  # bystander: 7 words

        self.assassin_word = self.board[self.owner == 0]
        self.given_clues = []

        # All cards are initially visible.
        self.visible = np.ones_like(self.owner, dtype=bool)

        self.num_turns = -1

    def initialize_from_words(self, initial_words, size=5):
        """
        The initial_words parameter should be in the format:

            ASSASSIN;TEAM1;TEAM2;NEUTRAL

        where each group consists of comma-separated words from the word list.

        The total number of words must be <= size * size. Any missing words
        are considered to be already covered and neutral.
        """
        self.size = size

        word_groups = initial_words.split(";")
        if len(word_groups) != 4:
            raise ValueError("Expected 4 groups separated by semicolon.")

        board, owner, visible = [], [], []
        for group_index, word_group in enumerate(word_groups):
            words = word_group.split(",")
            for word in words:
                word = word.lower().replace(" ", "_")
                if word not in self.words:
                    raise ValueError('Invalid word "{0}".'.format(word))
                if word in board:
                    raise ValueError('Duplicate word "{0}".'.format(word))
                board.append(word)
                owner.append(group_index)
                visible.append(True)
        if len(board) > size * size:
            raise ValueError("Too many words. Expected <= {0}.".format(size * size))
        # Add dummy hidden words if necessary.
        while len(board) < size * size:
            board.append("---")
            owner.append(3)
            visible.append(False)

        self.board = np.array(board)
        self.owner = np.array(owner)
        self.visible = np.array(visible)

        # Perform a random shuffle of the board.
        shuffle = self.generator.permutation(size * size)
        self.board = self.board[shuffle]
        self.owner = self.owner[shuffle]
        self.visible = self.visible[shuffle]

        self.assassin_word = self.board[self.owner == 0]
        self.num_turns = -1

    def print_board(self, spymaster=False, clear_screen=True, override=False):

        # Check to see if we want to display, if not return
        if not self.display and not override:
            return

        if clear_screen:
            if platform.system() == "Windows":
                os.system("cls")
            else:
                sys.stdout.write(chr(27) + "[2J")

        board = self.board.reshape(self.size, self.size)
        owner = self.owner.reshape(self.size, self.size)
        visible = self.visible.reshape(self.size, self.size)

        sys.stdout.write(
            "Words left: {}, Opponent: {}\n".format(
                len(self.player_words), len(self.opponent_words)
            )
        )

        for row in range(self.size):
            for col in range(self.size):
                word = board[row, col]
                tag = "#<>-"[owner[row, col]]
                if not visible[row, col]:
                    word = tag * 11
                elif not spymaster:
                    tag = " "
                if not spymaster or owner[row, col] in (0, 1, 2):
                    word = word.upper()
                sys.stdout.write("{0}{1:11s} ".format(" ", str(word)[2:-1]))
            sys.stdout.write("\n")

    def play_computer_spymaster(self, gamma=1.0, verbose=True, give_words=False):

        say("Thinking...")
        sys.stdout.flush()

        saved_clues, best_score = self.get_clue_list_pair_stretch(gamma, verbose)

        num_clues = len(saved_clues)
        order = sorted(range(num_clues), key=lambda k: best_score[k], reverse=True)

        if not os.path.exists(config.logs_folder):
            os.makedirs(config.logs_folder)
        with open(self.logs_filename, 'a+') as f:
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

        if verbose or True:
            self.print_board(spymaster=True)
            for i in order[:10]:
                clue, words = saved_clues[i]
                say(
                    u"{0:.3f} {1} = {2}".format(
                        best_score[i], " + ".join([str(w).upper() for w in words]), clue
                    )
                )

        clue, words = saved_clues[order[0]]
        self.unfound_words[self.player].update(words)
        self.given_clues.append(clue)
        if give_words:
            return clue, words
        if self.expert and self._should_say_unlimited(nb_clue_words=len(words)):
            return clue, UNLIMITED
        else:
            return clue, len(words)


    def get_clue_list_pair_stretch(self, gamma, verbose, max_group=2, does_stretch=[2]):
        # Loop over all permutations of words.
        num_words = len(self.player_words)
        best_score, saved_clues = [], []
        counts = range(min(num_words, max_group), 0, -1)
        groups_and_count = []
        for count in counts:
            for group in itertools.combinations(range(num_words), count):
                groups_and_count.append((group, count,))
        if self.display or True:
            groups_and_count = tqdm(groups_and_count)
        for group, count in groups_and_count:
            # Multiply similarity scores by this factor for any clue
            # corresponding to this many words.
            bonus_factor = count ** gamma
            words = self.player_words[list(group)]
            clue, score, stretch = self.model.get_clue_stretch(
                clue_words=words,
                pos_words=self.player_words,
                neg_words=self.opponent_words,
                neut_words=self.neutral_words,
                veto_words=self.assassin_word,
                given_clues=self.given_clues,
                give_stretch=(count in does_stretch),
            )
            if clue:
                best_score.append(score * bonus_factor)
                clue_words = words
                if stretch:
                    clue_words = np.concatenate((words, np.asarray(stretch)))
                saved_clues.append((clue, clue_words))
        return saved_clues, best_score


    def get_clue_list_dkirkby(self, gamma, verbose):
        # Loop over all permutations of words.
        num_words = len(self.player_words)
        best_score, saved_clues = [], []
        counts = range(num_words, 0, -1)
        groups_and_count = []
        for count in counts:
            for group in itertools.combinations(range(num_words), count):
                groups_and_count.append((group, count,))
        if self.display or True:
            groups_and_count = tqdm(groups_and_count)
        for group, count in groups_and_count:
            # Multiply similarity scores by this factor for any clue
            # corresponding to this many words.
            bonus_factor = count ** gamma
            words = self.player_words[list(group)]
            clue, score = self.model.get_clue(
                clue_words=words,
                pos_words=self.player_words,
                neg_words=np.concatenate((self.opponent_words, self.neutral_words)),
                veto_words=self.assassin_word,
                given_clues=self.given_clues,
            )
            if clue:
                best_score.append(score * bonus_factor)
                saved_clues.append((clue, words))
        return saved_clues


    def _should_say_unlimited(self, nb_clue_words, threshold_opponent=2):
        """
        Announce "unlimited" if :
        (1) the opposing team risks winning with their next clue,
        (2) and our +1 guess isn't enough to catch up during this clue,
        (3) but all the words hinted by the current and previous clues
            are enough to catch up and win
        """
        return (
            len(self.opponent_words) <= threshold_opponent  # (1)
            and nb_clue_words + 1 < len(self.player_words)  # (2)
            and self.unfound_words[self.player] == set(self.player_words)
        )  # (3)

    def play_human_spymaster(self):

        self.print_board(spymaster=True)

        while True:
            clue = ask("{0} Enter your clue: ".format(self.player_label))
            matched = self.valid_clue.match(clue)
            if matched:
                word, count = matched.groups()
                if count != UNLIMITED:
                    count = int(count)
                return word, count
            say("Invalid clue, should be WORD COUNT.")

    def play_human_team(self, word, count):

        num_guesses = 0
        while (self.expert and count == UNLIMITED) or num_guesses < count + 1:
            self.print_board(clear_screen=(num_guesses == 0))
            say(
                u"{0} your clue is: {1} {2}".format(
                    self.player_label, str(word)[2:-1], count
                )
            )

            num_guesses += 1
            while True:
                guess = ask(
                    "{0} enter your guess #{1}: ".format(self.player_label, num_guesses)
                )
                guess = guess.strip().lower().replace(" ", "_")
                if guess == "":
                    # Team does not want to make any more guesses.
                    return True
                guess = guess.encode("utf8")
                if guess in self.board[self.visible]:
                    break
                say("Invalid guess, should be a visible word.")

            loc = np.where(self.board == guess)[0]
            self.visible[loc] = False

            if guess == self.assassin_word:
                say(
                    "{0} You guessed the assasin - game over!".format(self.player_label)
                )
                return False

            if guess in self.player_words:
                self.unfound_words[self.player].discard(guess)
                if num_guesses == len(self.player_words):
                    say("{0} You won!!!".format(self.player_label))
                    return False
                else:
                    ask(
                        "{0} Congratulations, keep going! (hit ENTER)\n".format(
                            self.player_label
                        )
                    )
            else:
                if guess in self.opponent_words:
                    ask(
                        "{0} Sorry, word from opposing team! (hit ENTER)\n".format(
                            self.player_label
                        )
                    )
                else:
                    ask("{0} Sorry, bystander! (hit ENTER)\n".format(self.player_label))
                break

        return True

    def next_turn(self):
        self.num_turns += 1

        self.player = self.num_turns % 2
        self.opponent = (self.player + 1) % 2

        self.player_label = "<>"[self.player] * 3
        self.player_words = self.board[(self.owner == self.player + 1) & self.visible]
        self.opponent_words = self.board[
            (self.owner == self.opponent + 1) & self.visible
        ]
        self.neutral_words = self.board[(self.owner == 3) & self.visible]

    def play_turn(self, spymaster="human", team="human"):

        self.next_turn()

        if spymaster == "human":
            word, count = self.play_human_spymaster()
        else:
            word, count = self.play_computer_spymaster()

        if team == "human":
            ongoing = self.play_human_team(word, count)
        else:
            raise NotImplementedError()

        return ongoing

    def play_game(
        self,
        spymaster1="human",
        team1="human",
        spymaster2="human",
        team2="human",
        init=None,
    ):

        if init is None:
            self.initialize_random_game()
        else:
            self.initialize_from_words(init)

        while True:
            if not self.play_turn(spymaster1, team1):
                break
            if not self.play_turn(spymaster2, team2):
                break


def say(message):
    if display_to_console:
        sys.stdout.write(message + "\n")


def ask(message):
    try:
        return input(message)
    except KeyboardInterrupt:
        say("\nBye.")
        sys.exit(0)

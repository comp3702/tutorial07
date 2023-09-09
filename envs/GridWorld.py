# Directions
from typing import Dict, Tuple, Optional

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTION_NAMES = {UP: '↑', DOWN: '↓', LEFT: '←', RIGHT: '→'}

EXIT_STATE = (-1, -1)

EPSILON = 0.0001

class Grid:
    def __init__(self, x_size: int = 4, y_size: int = 3, p: float = 0.8,
                 gamma: float = 0.9,
                 rewards: Optional[Dict[Tuple[int, int], int]] = None,
                 obstacles: Tuple[Tuple[int, int]] = ((1, 1),)):
        self.last_col = x_size - 1
        self.last_row = y_size - 1

        self.p = p
        self.alt_p = (1 - p) / 2

        self.actions = [UP, DOWN, LEFT, RIGHT]

        if rewards is None:
            self.rewards = {(3, 1): -100, (3, 2): 1}
        else:
            self.rewards = rewards

        self.gamma = gamma

        states = list( (x, y) for y in range(y_size) for x in range(x_size) )
        states.append(EXIT_STATE)
        for obstacle in obstacles:
            states.remove(obstacle)
        self.states = tuple(states)

        self.obstacles = obstacles

    def attempt_move(self, s: Tuple[int, int], a: int) -> Tuple[int, int]:
        """ Attempts to move the agent from state s via action a.

            Parameters:
                s: The current state.
                a: The *actual* action performed (as opposed to the chosen
                   action; i.e. you do not need to account for non-determinism
                   in this method).
            Returns: the state resulting from performing action a in state s.
        """
        x, y = s
        col, row = s

        # Check absorbing state
        if s in self.rewards:
            return EXIT_STATE

        if s == EXIT_STATE:
            return s

        # Check borders
        if a == RIGHT and x < self.last_col:
            x += 1
        elif a == LEFT and x > 0:
            x -= 1
        elif a == DOWN and y < self.last_row:
            y += 1
        elif a == UP and y > 0:
            y -= 1
        if a == RIGHT and col < self.last_col:
            col += 1
        elif a == LEFT and col > 0:
            col -= 1
        # indexed at bottom left!!!!! not top
        elif a == UP and row < self.last_row:
            row += 1
        elif a == DOWN and row > 0:
            row -= 1

        result = (x, y)
        result = (col, row)

        # Check obstacle cells
        if result in self.obstacles:
            result = s

        return result

    def stoch_action(self, a):
        """ Returns the probabilities with which each action will actually occur,
            given that action a was requested.

        Parameters:
            a: The action requested by the agent.

        Returns:
            The probability distribution over actual actions that may occur.
        """
        if a == RIGHT:
            return {RIGHT: self.p , UP: (1-self.p)/2, DOWN: (1-self.p)/2}
        elif a == UP:
            return {UP: self.p , LEFT: (1-self.p)/2, RIGHT: (1-self.p)/2}
        elif a == LEFT:
            return {LEFT: self.p , UP: (1-self.p)/2, DOWN: (1-self.p)/2}
        return {DOWN: self.p , LEFT: (1-self.p)/2, RIGHT: (1-self.p)/2}

    def get_transition_probabilities(self, s, a):
        """ Calculates the probability distribution over next states given
            action a is taken in state s.

        Parameters:
            s: The state the agent is in
            a: The action requested

        Returns:
            A map from the reachable next states to the probabilities of reaching
            those state; i.e. each item in the returned dictionary is of form
            s' : P(s'|s,a)
        """
        """
            TODO: Create and return a dictionary mapping each possible next state to the
            probability that that state will be reached by doing a in s.
        """

    def get_reward(self, s):
        """ Returns the reward for being in state s. """
        if s == EXIT_STATE:
            return 0

        return self.rewards.get(s, 0)


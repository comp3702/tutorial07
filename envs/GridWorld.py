# Directions
from typing import Dict, Tuple, Optional, Set

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
        col, row = s

        # Check absorbing state
        if self.check_absorbing_state(s):
            return EXIT_STATE

        # Check borders
        if a == RIGHT and col < self.last_col:
            col += 1
        elif a == LEFT and col > 0:
            col -= 1
        # indexed at top left!!!!! not top
        elif a == DOWN and row < self.last_row:
            row += 1
        elif a == UP and row > 0:
            row -= 1

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

    def check_absorbing_state(self, state: Tuple[int, int]) -> bool:
        return state in self.rewards or state == EXIT_STATE

    def get_reward(self, s, action=None):
        """ Returns the reward for being in state s. """
        if s == EXIT_STATE:
            return 0

        return self.rewards.get(s, 0)

class GridWithKey(Grid):
    def __init__(self, x_size: int = 4, y_size: int = 3, p: float = 0.8,
                 gamma: float = 0.9,
                 rewards: Optional[Dict[Tuple[int, int], int]] = None,
                 obstacles: Tuple[Tuple[int, int]] = ((1, 1),),
                 keys: Tuple[Tuple[int, int]] = ((2, 2),)):
        super().__init__(x_size, y_size, p, gamma, rewards, obstacles)
        self.keys = set(keys)

    def attempt_move(self, s: Tuple[int, int], a: int) -> Tuple[int, int]:
        next_state = super().attempt_move(s, a)
        if next_state in self.keys:
            self.keys.remove(next_state)

        return next_state

    def check_absorbing_state(self, state: Tuple[int, int]) -> bool:
        return not self.keys and state in self.rewards or state == EXIT_STATE

    def get_reward(self, s, action=None):
        if self.keys or s == EXIT_STATE:
            return 0

        return self.rewards.get(s, 0)

class GridWithKeyAndCosts(GridWithKey):
    def __init__(self, x_size: int = 4, y_size: int = 3, p: float = 0.8,
                 gamma: float = 0.9,
                 rewards: Optional[Dict[Tuple[int, int], int]] = None,
                 obstacles: Tuple[Tuple[int, int]] = ((1, 1),),
                 keys: Tuple[Tuple[int, int]] = ((2, 2),),
                 costs: Tuple = (0.1, 0.2, 0.3, 0.4)):
        super().__init__(x_size, y_size, p, gamma, rewards, obstacles, keys)
        self.costs = costs

    def get_reward(self, s, action=None):
        if self.keys or s == EXIT_STATE:
            return self.costs[action]

        return self.rewards.get(s, 0)
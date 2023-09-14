from typing import Optional, Dict, Tuple, List

from envs.GridWorldState import GridWorldState

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTION_NAMES = {UP: '↑', DOWN: '↓', LEFT: '←', RIGHT: '→'}

EPSILON = 0.0001
EXIT_STATE = GridWorldState(-1, -1, tuple())


class GridWorldWithKeys():
    def __init__(self, x_size: int = 4, y_size: int = 3, p: float = 0.8,
                 goal: Optional[Dict[Tuple[int, int], int]] = None,
                 hazards: Optional[Dict[Tuple[int, int], int]] = None,
                 obstacles: Tuple[Tuple[int, int]] = ((1, 1),),
                 keys: Tuple[Tuple[int, int], ...] = ((2, 2),)):
        self.last_col = x_size - 1
        self.last_row = y_size - 1

        self.p = p
        self.alt_p = (1 - p) / 2

        self.actions = [UP, DOWN, LEFT, RIGHT]

        if goal is None:
            self.goal = {(3, 0): 1}
        else:
            self.goal = goal

        if hazards is None:
            self.hazards = {(3, 1): -100, (3, 1): -100}
        else:
            self.hazards = hazards

        key_states = [tuple()]
        for key in keys:
            key_states.append(key_states[len(key_states) - 1] + (key,))

        states = list(GridWorldState(x, y, k) for y in range(y_size) for x in range(x_size) for k in key_states if
                      (x, y) not in obstacles)
        states.append(EXIT_STATE)
        self.states: Tuple[GridWorldState, ...] = tuple(states)

        self.obstacles = obstacles

    def attempt_move(self, state: GridWorldState, action: int) -> GridWorldState:
        x, y = state.x, state.y

        # Check borders
        if action == RIGHT and x < self.last_col:
            x += 1
        elif action == LEFT and x > 0:
            x -= 1
        # indexed at top left!!!!! not bottom left
        elif action == DOWN and y < self.last_row:
            y += 1
        elif action == UP and y > 0:
            y -= 1

        next_position = (x, y)

        if next_position in self.obstacles:
            return state.deepcopy()

        if next_position in state.key_state:
            new_keys = tuple(k for k in state.key_state if k != next_position)
        else:
            new_keys = state.key_state

        next_state = GridWorldState(next_position[0], next_position[1], new_keys)

        return next_state

    def stoch_action(self, action):
        if action == RIGHT:
            return {RIGHT: self.p , UP: (1-self.p)/2, DOWN: (1-self.p)/2}
        elif action == UP:
            return {UP: self.p , LEFT: (1-self.p)/2, RIGHT: (1-self.p)/2}
        elif action == LEFT:
            return {LEFT: self.p , UP: (1-self.p)/2, DOWN: (1-self.p)/2}
        return {DOWN: self.p , LEFT: (1-self.p)/2, RIGHT: (1-self.p)/2}

    def get_transition_outcomes(self, state: GridWorldState, action: int) -> List[Tuple[float, GridWorldState, float]]:
        if state == EXIT_STATE:
            return [(1.0, state, 0.0)]

        # handle terminal states
        if state.position() in self.hazards.keys():
            return [(1.0, EXIT_STATE, self.hazards[state.position()])]
        if not state.key_state and state.position() in self.goal:
            return [(1.0, EXIT_STATE, self.goal[state.position()])]

        # loop over all possible directions
        outcomes = []
        for actual_action, probability in self.stoch_action(action).items():
            next_state = self.attempt_move(state, actual_action)
            outcomes.append((probability, next_state, 0.0))
        return outcomes

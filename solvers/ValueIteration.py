from typing import Dict, Tuple

from envs.GridWorld import Grid, RIGHT, EPSILON, ACTION_NAMES, EXIT_STATE
from solvers.utils import dict_argmax

import random

class ValueIteration:
    def __init__(self, env: Grid, epsilon: float = EPSILON, value_initializer: str = 'zero'):
        self.env = env
        if value_initializer == 'random':
            self.state_values = {state: random.uniform(-1, 1) for state in self.env.states}
        else:
            self.state_values = {state: 0 for state in self.env.states}
        self.policy = {state: RIGHT for state in self.env.states}

        self.epsilon = epsilon

    def next_iteration(self) -> bool:
        new_state_values = dict()

        self.policy = dict()

        for state in self.env.states:
            action_values = dict()
            for action in self.env.actions:
                action_value = 0
                for stoch_action, probability in self.env.stoch_action(action).items():
                    next_state = self.env.attempt_move(state, stoch_action)
                    action_value += probability * (self.env.get_reward(state) + self.env.gamma * self.state_values[next_state])

                action_values[action] = action_value

            new_state_values[state] = max(action_values.values())
            self.policy[state] = dict_argmax(action_values)

        converged = self.check_convergence(new_state_values)
        self.state_values = new_state_values

        return converged

    def check_convergence(self, new_state_values: Dict[Tuple[int, int], float]) -> bool:
        differences = [abs(self.state_values[state] - new_state_values[state]) for state in self.env.states]
        return max(differences) < self.epsilon

    def print_values(self):
        for state, value in self.state_values.items():
            print(state, value)

    def print_values_and_policy(self):
        for state, value in self.state_values.items():
            print(state, ACTION_NAMES[self.policy[state]], value)

    def get_values_and_policy(self):
        data = []
        for state, value in self.state_values.items():
            if state == EXIT_STATE:
                continue
            data.append( (state[0], state[1], ACTION_NAMES[self.policy[state]], value) )

        return data


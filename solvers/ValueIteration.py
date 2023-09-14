from typing import Dict, Tuple

from envs.GridWorldWithKeys import GridWorldWithKeys, EPSILON, RIGHT
from solvers.utils import dict_argmax, VisualizerMixin

import random

class ValueIteration(VisualizerMixin):
    def __init__(self, env: GridWorldWithKeys, epsilon: float = EPSILON, gamma: float = 0.9, value_initializer: str = 'zero'):
        self.env = env
        if value_initializer == 'random':
            self.state_values = {state: random.uniform(-1, 1) for state in self.env.states}
        else:
            self.state_values = {state: 0 for state in self.env.states}
        self.policy = {state: RIGHT for state in self.env.states}

        self.epsilon = epsilon
        self.gamma = gamma


    def next_iteration(self) -> bool:
        new_state_values = dict()

        self.policy = dict()

        for state in self.env.states:
            action_values = dict()
            for action in self.env.actions:
                action_value = 0
                for probability, next_state, reward in self.env.get_transition_outcomes(state, action):
                    action_value += probability * (reward + self.gamma * self.state_values[next_state])

                action_values[action] = action_value

            new_state_values[state] = max(action_values.values())
            self.policy[state] = dict_argmax(action_values)

        converged = self.check_convergence(new_state_values)
        self.state_values = new_state_values

        return converged

    def check_convergence(self, new_state_values: Dict[Tuple[int, int], float]) -> bool:
        differences = [abs(self.state_values[state] - new_state_values[state]) for state in self.env.states]
        return max(differences) < self.epsilon






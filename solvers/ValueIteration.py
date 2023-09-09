from typing import Dict, Tuple

from envs.GridWorld import Grid, RIGHT, EPSILON, ACTION_NAMES, EXIT_STATE
from envs.GridWorldWithKeys import GridWorldWithKeys
from solvers.utils import dict_argmax

import random

class ValueIteration:
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
            desc = ''
            if state.position() in state.key_state:
                desc = '\nKey'
            if state.position() in self.env.hazards:
                desc = '\nHaz'
            if not state.key_state and state.position() in self.env.goal:
                desc = '\nGoal'

            data.append( (state.x, state.y, ACTION_NAMES[self.policy[state]], value, desc, state.key_state) )

        return data


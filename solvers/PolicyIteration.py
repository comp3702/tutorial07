from typing import Dict, Tuple

from envs.GridWorldWithKeys import GridWorldWithKeys, EPSILON, RIGHT
from solvers.utils import dict_argmax, VisualizerMixin
import numpy as np

class PolicyIteration(VisualizerMixin):
    def __init__(self, env: GridWorldWithKeys, epsilon: float = EPSILON, gamma: float = 0.9, policy_initializer: str = 'zero'):
        self.env = env
        self.state_values = {state: 0 for state in self.env.states}
        if policy_initializer == 'random':
            self.policy = {pi: np.random.choice(env.actions) for pi in self.env.states}
        else:
            self.policy = {pi: RIGHT for pi in self.env.states}
        self.epsilon = epsilon
        self.gamma = gamma

    def next_iteration(self) -> bool:
        self.policy_evaluation()
        return self.policy_improvement()

    def policy_evaluation(self):
        # use 'naive'/iterative policy evaluation
        value_converged = False
        while not value_converged:
            new_values = dict()
            for state in self.env.states:
                action_value = 0
                action = self.policy[state]
                for probability, next_state, reward in self.env.get_transition_outcomes(state, action):
                    action_value += probability * (reward + self.gamma * self.state_values[next_state])

                # for stoch_action, p in self.env.stoch_action(self.policy[s]).items():
                #     # Apply action
                #     s_next = self.env.attempt_move(s, stoch_action)
                #     action_value += p * (self.env.get_reward(s) + (self.env.gamma * self.values[s_next]))
                # Update state value with best action
                new_values[state] = action_value

            # Check convergence
            differences = [abs(self.state_values[s] - new_values[s]) for s in self.env.states]
            if max(differences) < self.epsilon:
                value_converged = True

            # Update values and policy
            self.state_values = new_values


    def policy_improvement(self) -> bool:
        new_policy = dict()

        for state in self.env.states:
            # Keep track of maximum value
            action_values = dict()
            for action in self.env.actions:
                action_value = 0
                for probability, next_state, reward in self.env.get_transition_outcomes(state, action):
                    action_value += probability * (reward + self.gamma * self.state_values[next_state])

                action_values[action] = action_value

            # Update policy
            new_policy[state] = dict_argmax(action_values)

        converged = self.convergence_check(new_policy)
        self.policy = new_policy

        return converged

    def convergence_check(self, new_policy: Dict[Tuple[int, int], int]) -> bool:
        return self.policy == new_policy



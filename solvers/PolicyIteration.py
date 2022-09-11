from typing import Dict, Tuple

from envs.GridWorld import RIGHT, Grid, ACTION_NAMES, EPSILON
from solvers.utils import dict_argmax


class PolicyIteration:
    def __init__(self, env: Grid, epsilon: float = EPSILON):
        self.env = env
        self.values = {state: 0 for state in self.env.states}
        self.policy = {pi: RIGHT for pi in self.env.states}
        self.epsilon = epsilon

    def next_iteration(self) -> bool:
        self.policy_evaluation()
        return self.policy_improvement()

    def policy_evaluation(self):
        # use 'naive'/iterative policy evaluation
        value_converged = False
        while not value_converged:
            new_values = dict()
            for s in self.env.states:
                total = 0
                for stoch_action, p in self.env.stoch_action(self.policy[s]).items():
                    # Apply action
                    s_next = self.env.attempt_move(s, stoch_action)
                    total += p * (self.env.get_reward(s) + (self.env.gamma * self.values[s_next]))
                # Update state value with best action
                new_values[s] = total

            # Check convergence
            differences = [abs(self.values[s] - new_values[s]) for s in self.env.states]
            if max(differences) < self.epsilon:
                value_converged = True

            # Update values and policy
            self.values = new_values


    def policy_improvement(self) -> bool:
        new_policy = dict()

        for s in self.env.states:
            # Keep track of maximum value
            action_values = dict()
            for a in self.env.actions:
                total = 0
                for stoch_action, p in self.env.stoch_action(a).items():
                    # Apply action
                    s_next = self.env.attempt_move(s, stoch_action)
                    total += p * (self.env.get_reward(s) + (self.env.gamma * self.values[s_next]))
                action_values[a] = total
            # Update policy
            new_policy[s] = dict_argmax(action_values)

        converged = self.convergence_check(new_policy)
        self.policy = new_policy

        return converged

    def convergence_check(self, new_policy: Dict[Tuple[int, int], int]) -> bool:
        return self.policy == new_policy

    def print_values_and_policy(self):
        for state, value in self.values.items():
            print(state, ACTION_NAMES[self.policy[state]], value)

    def print_values(self):
        for state, value in self.values.items():
            print(state, value)

    def print_policy(self):
        for state, policy in self.policy.items():
            print(state, policy)

from envs.GridWorld import EPSILON, Grid, EXIT_STATE, RIGHT
from solvers.PolicyIteration import PolicyIteration
import numpy as np

class PolicyIterationLinAlg(PolicyIteration):
    def __init__(self, env: Grid, epsilon: float = EPSILON):
        super().__init__(env, epsilon)
        
        # t model (lin alg)
        t_model = np.zeros([len(self.env.states), len(self.env.actions), len(self.env.states)])
        for state_index, state in enumerate(self.env.states):
            for action_index, action in enumerate(self.env.actions):
                # reward state always leads to exit
                if state in self.env.rewards.keys():
                    exit_state_index = self.env.states.index(EXIT_STATE)
                    t_model[state_index][action_index][exit_state_index] = 1.0
                elif state == EXIT_STATE:
                    t_model[state_index][action_index][self.env.states.index(EXIT_STATE)] = 1.0
                else:
                    for stoch_action, probability in self.env.stoch_action(action).items():
                        # Apply action
                        next_state = self.env.attempt_move(state, stoch_action)
                        next_state_index = self.env.states.index(next_state)
                        t_model[state_index][action_index][next_state_index] += probability
        self.t_model = t_model

        # r model (lin alg)
        r_model = np.zeros([len(self.env.states)])
        for state_index, state in enumerate(self.env.states):
            r_model[state_index] = self.env.get_reward(state)
        self.r_model = r_model

        # lin alg policy
        self.la_policy = np.zeros([len(self.env.states)], dtype=np.int64) + RIGHT

    def policy_evaluation(self):
        # use linear algebra for policy evaluation
        # V^pi = R + gamma T^pi V^pi
        # (I - gamma * T^pi) V^pi = R
        # Ax = b; A = (I - gamma * T^pi),  b = R
        state_numbers = np.array(range(len(self.env.states)))  # indices of every state
        t_pi = self.t_model[state_numbers, self.la_policy]
        values = np.linalg.solve(np.identity(len(self.env.states)) - (self.env.gamma * t_pi), self.r_model)
        self.values = {s: values[i] for i, s in enumerate(self.env.states)}
        # new_policy = {s: self.env.actions[self.la_policy[i]] for i, s in enumerate(self.env.states)}

    def policy_improvement(self) -> bool:
        converged = super().policy_improvement()

        for i, s in enumerate(self.env.states):
            self.la_policy[i] = self.policy[s]

        return converged
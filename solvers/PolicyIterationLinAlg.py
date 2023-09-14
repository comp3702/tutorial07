from envs.GridWorldWithKeys import GridWorldWithKeys, EXIT_STATE, EPSILON, RIGHT
from solvers.PolicyIteration import PolicyIteration
import numpy as np


class PolicyIterationLinAlg(PolicyIteration):
    def __init__(self, env: GridWorldWithKeys, epsilon: float = EPSILON, gamma: float = 0.9, policy_initializer: str = 'zero'):
        super().__init__(env, epsilon=epsilon, gamma=gamma, policy_initializer=policy_initializer)

        # r model (lin alg)
        r_model = np.zeros([len(self.env.states)])

        # t model (lin alg)
        t_model = np.zeros([len(self.env.states), len(self.env.actions), len(self.env.states)])
        for state_index, state in enumerate(self.env.states):
            for action_index, action in enumerate(self.env.actions):
                # hazard state always leads to exit
                if state.position() in self.env.hazards.keys():
                    exit_state_index = self.env.states.index(EXIT_STATE)
                    t_model[state_index][action_index][exit_state_index] = 1.0
                    r_model[state_index] = self.env.hazards[state.position()]
                elif not state.key_state and state.position() in self.env.goal.keys():
                    exit_state_index = self.env.states.index(EXIT_STATE)
                    t_model[state_index][action_index][exit_state_index] = 1.0
                    r_model[state_index] = self.env.goal[state.position()]
                elif state == EXIT_STATE:
                    t_model[state_index][action_index][self.env.states.index(EXIT_STATE)] = 1.0
                else:
                    for probability, next_state, reward in self.env.get_transition_outcomes(state, action):
                        next_state_index = self.env.states.index(next_state)
                        t_model[state_index][action_index][next_state_index] += probability

        self.t_model = t_model
        self.r_model = r_model

        # lin alg policy
        self.la_policy = np.array(list(self.policy.values()), dtype=np.int64)

    def policy_evaluation(self):
        # use linear algebra for policy evaluation
        # V^pi = R + gamma T^pi V^pi
        # (I - gamma * T^pi) V^pi = R
        # Ax = b; A = (I - gamma * T^pi),  b = R
        state_numbers = np.array(range(len(self.env.states)))  # indices of every state
        t_pi = self.t_model[state_numbers, self.la_policy]
        values = np.linalg.solve(np.identity(len(self.env.states)) - (self.gamma * t_pi), self.r_model)
        self.state_values = {s: values[i] for i, s in enumerate(self.env.states)}
        # new_policy = {s: self.env.actions[self.la_policy[i]] for i, s in enumerate(self.env.states)}

    def policy_improvement(self) -> bool:
        converged = super().policy_improvement()

        for i, s in enumerate(self.env.states):
            self.la_policy[i] = self.policy[s]

        return converged

from envs.GridWorld import EPSILON, Grid, EXIT_STATE, RIGHT
from solvers.PolicyIteration import PolicyIteration
import numpy as np

class PolicyIterationLinAlg(PolicyIteration):
    def __init__(self, env: Grid, epsilon: float = EPSILON):
        super().__init__(env, epsilon)
        
        # t model (lin alg)
        t_model = np.zeros([len(self.env.states), len(self.env.actions), len(self.env.states)])
        for i, s in enumerate(self.env.states):
            for j, a in enumerate(self.env.actions):
                if s in self.env.rewards.keys():
                    for k in range(len(self.env.states)):
                        if self.env.states[k] == (-1, -1):
                            t_model[i][j][k] = 1.0
                        else:
                            t_model[i][j][k] = 0.0
                elif s == EXIT_STATE:
                    t_model[i][j][self.env.states.index(EXIT_STATE)] = 1.0
                else:
                    for stoch_action, p in self.env.stoch_action(a).items():
                        # Apply action
                        s_next = self.env.attempt_move(s, stoch_action)
                        k = self.env.states.index(s_next)
                        t_model[i][j][k] += p
        self.t_model = t_model

        # r model (lin alg)
        r_model = np.zeros([len(self.env.states)])
        for i, s in enumerate(self.env.states):
            r_model[i] = self.env.get_reward(s)
        self.r_model = r_model

        # lin alg policy
        la_policy = np.zeros([len(self.env.states)], dtype=np.int64)
        for i, s in enumerate(self.env.states):
            la_policy[i] = RIGHT
            # la_policy[i] = random.randint(0, len(self.env.actions) - 1)
        self.la_policy = la_policy

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
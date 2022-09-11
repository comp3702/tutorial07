from envs.GridWorld import RIGHT


class PolicyIteration:
    def __init__(self, env):
        self.env = env
        self.values = {state: 0 for state in self.env.states}
        self.policy = {pi: RIGHT for pi in self.env.states}
        self.r = [0 for s in self.env.states]
        for idx, state in enumerate(self.env.states):
            if state in self.env.rewards.keys():
                self.r[idx] = self.env.rewards[state]
        print('r is ', self.r)

    def next_iteration(self):
        """
        TODO: Write code to orchestrate one iteration of PI here.
        """
        return

    def policy_evaluation(self):
        """
        TODO: Write code for the policy evaluation step of PI here. That is, update
        the current value estimates using the current policy estimate.
        """
        return

    def policy_improvement(self):
        """
        TODO: Write code to extract the best policy for a given value function here
        """
        return

    def convergence_check(self):
        """
        TODO: Write code to check if PI has converged here
        """
        return

    def print_values(self):
        for state, value in self.values.items():
            print(state, value)

    def print_policy(self):
        for state, policy in self.policy.items():
            print(state, policy)

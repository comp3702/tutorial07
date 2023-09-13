import unittest

from envs.GridWorld import Grid
from solvers.PolicyIteration import PolicyIteration


class TestPolicyIteration(unittest.TestCase):
    def setUp(self) -> None:
        self.env = Grid()

    def test_policy_evaluation(self):
        pi = PolicyIteration(self.env)

        pi.policy_evaluation()

        self.assertEqual(1, pi.state_values[(3, 2)])
        self.assertEqual(-100, pi.state_values[(3, 1)])

        self.assertAlmostEqual(-32.527, pi.state_values[(0, 0)], 3)
        self.assertAlmostEqual(-76.667, pi.state_values[(2, 1)], 3)

    def test_converge_after_5_iterations(self):
        pi = PolicyIteration(self.env)
        converged = False

        for _ in range(5):
            converged = pi.next_iteration()

        self.assertTrue(converged)
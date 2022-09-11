import unittest

from envs.GridWorld import Grid
from solvers.PolicyIteration import PolicyIteration


class TestPolicyIteration(unittest.TestCase):
    def setUp(self) -> None:
        self.env = Grid()

    def test_policy_evaluation(self):
        pi = PolicyIteration(self.env)

        pi.policy_evaluation()

        self.assertEqual(1, pi.values[(3, 2)])
        self.assertEqual(-100, pi.values[(3, 1)])

        self.assertAlmostEqual(-32.52716599187787, pi.values[(0, 0)])
        self.assertAlmostEqual(-76.66664787325631, pi.values[(2, 1)])

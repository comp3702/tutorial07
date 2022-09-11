import unittest

from envs.GridWorld import Grid, EPSILON, EXIT_STATE
from solvers.ValueIteration import ValueIteration


class TestValueIteration(unittest.TestCase):
    def setUp(self) -> None:
        self.env = Grid()

    def test_convergence(self):
        one_tile_env = Grid(1, 1, obstacles=())

        vi = ValueIteration(one_tile_env)

        self.assertFalse(vi.check_convergence({(0, 0): EPSILON, EXIT_STATE: 0}))
        self.assertTrue(vi.check_convergence({(0, 0): EPSILON/2, EXIT_STATE: 0}))

    def test_single_iteration(self):
        vi = ValueIteration(self.env)

        self.assertFalse(vi.next_iteration())
        # this only assigns the rewards
        self.assertEqual(1, vi.state_values[(3, 2)])
        self.assertEqual(-100, vi.state_values[(3, 1)])

    def test_iteration_2(self):
        vi = ValueIteration(self.env)

        vi.next_iteration()
        self.assertFalse(vi.next_iteration())
        self.assertAlmostEqual(0.72, vi.state_values[(2, 2)])

        self.assertEqual(1, vi.state_values[(3, 2)])
        self.assertEqual(-100, vi.state_values[(3, 1)])

    def test_iteration_3(self):
        vi = ValueIteration(self.env)

        vi.next_iteration()
        vi.next_iteration()
        self.assertFalse(vi.next_iteration())
        self.assertAlmostEqual(0.7848, vi.state_values[(2, 2)])
        self.assertAlmostEqual(0.0648, vi.state_values[(2, 1)])
        self.assertAlmostEqual(0.5184, vi.state_values[(1, 2)])

        self.assertEqual(1, vi.state_values[(3, 2)])
        self.assertEqual(-100, vi.state_values[(3, 1)])

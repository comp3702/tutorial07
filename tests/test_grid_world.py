import unittest

from envs.GridWorld import Grid, EXIT_STATE


class TestGridWorld(unittest.TestCase):
    def test_init(self):
        env = Grid()

        self.assertEqual(3, env.last_col)
        self.assertEqual(2, env.last_row)

        self.assertEqual(
            ((0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2), EXIT_STATE),
            env.states
        )
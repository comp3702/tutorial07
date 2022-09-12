import unittest

from envs.GridWorld import Grid, EXIT_STATE, RIGHT, LEFT, DOWN, UP


class TestGridWorld(unittest.TestCase):
    def setUp(self) -> None:
        self.env = Grid()

    def test_init(self):
        self.assertEqual(3, self.env.last_col)
        self.assertEqual(2, self.env.last_row)

        self.assertEqual(
            ((0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2), EXIT_STATE),
            self.env.states
        )

    def test_move_right(self):
        self.assertEqual((1, 0), self.env.attempt_move((0, 0), RIGHT))

    def test_move_right_to_wall(self):
        self.assertEqual((3, 0), self.env.attempt_move((3, 0), RIGHT))

    def test_move_left(self):
        self.assertEqual((0, 0), self.env.attempt_move((1, 0), LEFT))

    def test_move_left_to_wall(self):
        self.assertEqual((0, 0), self.env.attempt_move((0, 0), LEFT))

    def test_move_up(self):
        self.assertEqual((0, 1), self.env.attempt_move((0, 0), UP))

    def test_move_up_to_wall(self):
        self.assertEqual((0, 2), self.env.attempt_move((0, 2), UP))

    def test_move_down(self):
        self.assertEqual((0, 1), self.env.attempt_move((0, 2), DOWN))

    def test_move_down_to_wall(self):
        self.assertEqual((0, 0), self.env.attempt_move((0, 0), DOWN))

    def test_move_to_obstacle(self):
        self.assertEqual((0, 1), self.env.attempt_move((0, 1), RIGHT))

    def test_move_from_reward_state(self):
        self.assertEqual(EXIT_STATE, self.env.attempt_move((3, 2), RIGHT))

    def test_move_from_exit_state(self):
        self.assertEqual(EXIT_STATE, self.env.attempt_move(EXIT_STATE, RIGHT))

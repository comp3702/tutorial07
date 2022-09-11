import unittest

from solvers.utils import dict_argmax


class TestUtils(unittest.TestCase):
    def test_dict_argmax(self):
        d = {1: 2, 3: 4}
        self.assertEqual(3, dict_argmax(d))
from typing import Tuple


class GridWorldState():
    def __init__(self, x: int, y: int, key_state: Tuple[Tuple[int, int], ...]):
        self.x = x
        self.y = y
        self.key_state = key_state
    def __eq__(self, other):
        if not isinstance(other, GridWorldState):
            return False
        return self.x == other.x and self.y == other.y and self.key_state == other.key_state

    def __hash__(self):
        return hash((self.x, self.y, tuple(self.key_state)))

    def deepcopy(self):
        return GridWorldState(self.x, self.y, self.key_state)

    def __repr__(self):
        return f'({self.x}, {self.y}, {self.key_state})'
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)
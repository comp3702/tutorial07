class ValueIteration:
    def __init__(self, grid):
        self.grid = grid
        self.values = {state: 0 for state in self.grid.states}

    def next_iteration(self):
        new_values = dict()
        """
        TODO: Write code here to imlpement the VI value update
        Iterate over self.grid.states and self.grid.actions
        Use stoch_action(a) and attempt_move(s,a)
        """
        self.values = new_values

    def print_values(self):
        for state, value in self.values.items():
            print(state, value)

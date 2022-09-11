import time

from envs.GridWorld import Grid
from official.GridWorld_Solution import ValueIteration

MAX_ITER = 100

if __name__ == "__main__":
    grid = Grid()
    vi = ValueIteration(grid)

    start = time.time()
    print("Initial values:")
    vi.print_values()
    print()

    for i in range(MAX_ITER):
        vi.next_iteration()
        print("Values after iteration", i + 1)
        vi.print_values()
        print()
        if vi.converged:
            break

    end = time.time()
    print("Time to complete", i + 1, "VI iterations")
    print(end - start)

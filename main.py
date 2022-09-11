import time

from envs.GridWorld import Grid
from solvers.ValueIteration import ValueIteration

MAX_ITER = 100

if __name__ == "__main__":
    env = Grid()
    vi = ValueIteration(env)

    start = time.time()
    print("Initial values:")
    vi.print_values()
    print()

    for i in range(MAX_ITER):
        converged = vi.next_iteration()
        print("Values after iteration", i + 1)
        vi.print_values_and_policy()
        print()
        if converged:
            break

    end = time.time()
    print("Time to complete", i + 1, "VI iterations")
    print(end - start)

import argparse
import time
import sys

from envs.GridWorld import Grid
from solvers.PolicyIteration import PolicyIteration
from solvers.PolicyIterationLinAlg import PolicyIterationLinAlg
from solvers.ValueIteration import ValueIteration

MAX_ITER = 100

def run_value_iteration():
    env = Grid()
    vi = ValueIteration(env)

    print("Initial values:")
    vi.print_values()
    print()

    for i in range(MAX_ITER):
        converged = vi.next_iteration()
        print("Values after iteration", i + 1)
        vi.print_values_and_policy()
        print()
        if converged:
            return i + 1

def run_policy_iteration():
    env = Grid()
    pi = PolicyIteration(env)

    print("Initial policy and values:")
    pi.print_values_and_policy()
    print()

    for i in range(MAX_ITER):
        converged = pi.next_iteration()
        print("Policy and values after iteration", i + 1)
        pi.print_values_and_policy()
        print()
        if converged:
            return i + 1

def run_policy_iteration_lin_alg():
    env = Grid()
    pi = PolicyIterationLinAlg(env)

    print("Initial policy and values:")
    pi.print_values_and_policy()
    print()

    for i in range(MAX_ITER):
        converged = pi.next_iteration()
        print("Policy and values after iteration", i + 1)
        pi.print_values_and_policy()
        print()
        if converged:
            return i + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--solver', required=True, help='Solver to be used - one of value, policy, or lin_alg')
    args = parser.parse_args()

    start = time.time()

    if args.solver == 'policy':
        iter_count = run_policy_iteration()
    elif args.solver == 'lin_alg':
        iter_count = run_policy_iteration_lin_alg()
    elif args.solver == 'value':
        iter_count = run_value_iteration()
    else:
        raise Exception(f'Invalid policy: {args.solver}')

    end = time.time()
    print("Time to complete", iter_count, " iterations")
    print(end - start)

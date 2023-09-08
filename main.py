import argparse
import time
import seaborn as sns
import pandas as pd

from envs.GridWorld import Grid, GridWithKey, GridWithKeyAndCosts
from solvers.PolicyIteration import PolicyIteration
from solvers.PolicyIterationLinAlg import PolicyIterationLinAlg
from solvers.ValueIteration import ValueIteration
import matplotlib.pyplot as plt

MAX_ITER = 100

def run_value_iteration(env: Grid, value_initializer: str = 'zero'):
    vi = ValueIteration(env, value_initializer=value_initializer)

    print("Initial values:")
    vi.print_values()
    print()

    plt.ion()
    plt.show()


    data = vi.get_values_and_policy()
    heatmap(plt, data)
    plt.pause(20)

    for i in range(MAX_ITER):
        converged = vi.next_iteration()
        print("Values after iteration", i + 1)
        # vi.print_values_and_policy()
        data = vi.get_values_and_policy()
        heatmap(plt, data)
        print()
        if converged:
            plt.ioff()
            plt.show()
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

def heatmap(plt, data):
    plt.clf()
    data = pd.DataFrame(data, columns=['X', 'Y', 'A', 'R'])
    data['RA'] = data['R'].round(4).apply(str) + data['A']
    map_data = data[['X', 'Y', 'R']].pivot(index='Y', columns='X')
    # map_data.sort_index(level=0, ascending=False, inplace=True)
    map = sns.heatmap(map_data, vmin=-1, vmax=1, annot=data[['X', 'Y', 'RA']].pivot(index='Y', columns='X'), fmt='')

    # plt.show()
    # time.sleep(0.25)
    plt.pause(0.25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--solver', required=True, help='Solver to be used - one of value, policy, or lin_alg')
    parser.add_argument('-e', '--env', default='Grid', help='Environment to be used - one of Grid, GridWithKey or GridWithKeyAndCosts')
    parser.add_argument('-i', '--value_initializer', default='zero', help='Initial value strategy - one of zero, or random')
    parser.add_argument('-d', '--difficulty', type=int, default=0, help='Difficulty of the environment - number between 0 and 2')
    args = parser.parse_args()

    start = time.time()

    env = None
    if args.env == 'GridWithKey':
        print('Starting env GridWithKey...')
        if args.difficulty == 2:
            env = GridWithKey(x_size=10, y_size=6, keys=((2, 2), (9, 5)))
        elif args.difficulty == 1:
            env = GridWithKey(x_size=10, y_size=6)
        else:
            env = GridWithKey()
    elif args.env == 'GridWithKeyAndCost':
        print('Starting env GridWithKeyAndCost...')
        if args.difficulty == 2:
            env = GridWithKeyAndCosts(x_size=10, y_size=6, rewards={(8, 4): -1, (9, 5): 1}, keys=((2, 2), (9, 5)))
        elif args.difficulty == 1:
            env = GridWithKeyAndCosts(x_size=10, y_size=6)
        else:
            env = GridWithKeyAndCosts()
    else:
        print('Starting env Grid...')
        if args.difficulty == 2:
            env = Grid(x_size=10, y_size=6, obstacles=((1, 1), (6, 4), (6, 5)))
        elif args.difficulty == 1:
            env = Grid(x_size=10, y_size=6)
        else:
            env = Grid()


    if args.solver == 'policy':
        iter_count = run_policy_iteration()
    elif args.solver == 'lin_alg':
        iter_count = run_policy_iteration_lin_alg()
    elif args.solver == 'value':
        iter_count = run_value_iteration(env, value_initializer=args.value_initializer)
    else:
        raise Exception(f'Invalid policy: {args.solver}')

    end = time.time()
    print("Time to complete", iter_count, " iterations")
    print(end - start)

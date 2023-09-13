import argparse
import time
import seaborn as sns
import pandas as pd

from envs.GridWorld import Grid
from envs.GridWorldWithKeys import GridWorldWithKeys
from solvers.PolicyIteration import PolicyIteration
from solvers.PolicyIterationLinAlg import PolicyIterationLinAlg
from solvers.ValueIteration import ValueIteration
import matplotlib.pyplot as plt

MAX_ITER = 100

def run_value_iteration(env: GridWorldWithKeys, value_initializer: str = 'zero'):
    vi = ValueIteration(env, value_initializer=value_initializer)

    print("Initial values:")
    vi.print_values()
    print()

    plt.ion()
    plt.show()

    data = vi.get_values_and_policy()
    heatmap(plt, data)
    plt.pause(5)

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
            print("Value iteration has finished...close the visualization window")
            return i + 1


def run_policy_iteration(env: GridWorldWithKeys, policy_initializer: str = 'zero'):
    pi = PolicyIteration(env, policy_initializer=policy_initializer)

    print("Initial policy and values:")
    pi.print_values_and_policy()

    print()

    plt.ion()
    plt.show()

    data = pi.get_values_and_policy()
    heatmap(plt, data)
    plt.pause(5)


    for i in range(MAX_ITER):
        converged = pi.next_iteration()
        print("Policy and values after iteration", i + 1)
        # pi.print_values_and_policy()
        data = pi.get_values_and_policy()
        heatmap(plt, data)
        if converged:
            plt.ioff()
            plt.show()
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
    data = pd.DataFrame(data, columns=['X', 'Y', 'A', 'R', 'Desc', 'Keys'])
    # print(data)
    data['RA'] = data['R'].round(4).apply(str) + data['A'] + data['Desc']
    data['Y'] = data['Keys'].apply(str) + data['Y'].apply(str)
    map_data = data[['X', 'Y', 'R']].pivot(index='Y', columns=['X'])
    # map_data.sort_index(level=0, ascending=False, inplace=True)
    map = sns.heatmap(map_data, vmin=-1, vmax=1, annot=data[['X', 'Y', 'RA']].pivot(index='Y', columns=['X']), fmt='')

    # plt.show()
    # time.sleep(0.25)
    plt.pause(0.25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--solver', required=True, help='Solver to be used - one of value, policy, or lin_alg')
    # parser.add_argument('-e', '--env', default='Grid', help='Environment to be used - one of Grid, GridWithKey or GridWithKeyAndCosts')
    parser.add_argument('-i', '--initializer', default='zero', help='Initial value/policy strategy - one of zero, or random - for value iteration, this initializes state values and for policy iteration the default policy')
    parser.add_argument('-d', '--difficulty', type=int, default=0, help='Difficulty of the environment - number between 0 and 2')
    args = parser.parse_args()

    start = time.time()

    env = None
    print('Starting env GridWorldWithKeys...')
    if args.difficulty == 2:
        print('Bigger grid with multiple hazards')
        env = GridWorldWithKeys(x_size=10, y_size=6, keys=((9, 5),), hazards={(3, 1): -100, (6, 3): -100})
    elif args.difficulty == 1:
        print('Small grid with a key')
        env = GridWorldWithKeys()
    else:
        print('Small grid without keys')
        env = GridWorldWithKeys(keys=())

    if args.solver == 'policy':
        iter_count = run_policy_iteration(env, policy_initializer=args.initializer)
    elif args.solver == 'lin_alg':
        iter_count = run_policy_iteration_lin_alg()
    elif args.solver == 'value':
        iter_count = run_value_iteration(env, value_initializer=args.initializer)
    else:
        raise Exception(f'Invalid policy: {args.solver}')

    end = time.time()
    print("Time to complete", iter_count, " iterations")
    print(end - start)

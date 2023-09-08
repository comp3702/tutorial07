# Tutorial 07 - MDP with Value and Policy Iterations

Loosely based on the official solutions but adds tests and refactors each algo into a separate file.

The PI using linear algebra is done by subclassing the regular PolicyIteration class, 
rather than if conditions throughout the code. I will try to make it less cryptic if I have time later this week.

### Dependencies
Python 3.10 and only dependency is `numpy` and `seaborn`

    conda install numpy
    pip install seaborn

### Running
Use the `main.py` to just see the results with `-s solver` argument:

    python main.py -s value   // for value iteration

    # policy iteration will be disussed next week
    python main.py -s policy   // for policy iteration
    python main.py -s lin_alg   // for policy iteration using linear algebra

There are more commandline options:

    -e Grid|GridWithKey|GridWithKeyAndCosts - the environment to be used
    -i zero|random - initializer for the state values (unified random or zeros) 
    -d 0|1|2 - difficulty of the environment

To run the basic environment without keys:

    python main.py -s value

And you should see this:
![Without Key](resources/without_key.gif)

To run the most difficult environment with keys use:

    python main.py -s value -e GridWithKey -d 2

This should fail because the solver will fail to reach states with reward.

To solve it, use random initializer for the state values:

    python main.py -s value -e GridWithKey -d 2 -i random

Which should result in this (note that the starting state values are random and not 0):
![With key](resources/with_key.gif)

### Question 1
[q6-01.py](q6-01.py) has a solution to question 1 (markov chains).

    python q6-01.py

### More details
[Lecture by Andrew Ng](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=3198s)

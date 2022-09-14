# Tutorial 07 - MDP with Value and Policy Iterations

Loosely based on the official solutions but adds tests and refactors each algo into a separate file.

The PI using linear algebra is done by subclassing the regular PolicyIteration class, 
rather than if conditions throughout the code. I will try to make it less cryptic if I have time later this week.

### Dependencies
Python 3.10 and only dependency is `numpy`

    conda install numpy

### Running
Use the `main.py` to just see the results with `-s solver` argument:

    python main.py -s value   // for value iteration
    python main.py -s policy   // for policy iteration
    python main.py -s lin_alg   // for policy iteration using linear algebra

### More details
[Andrew's Ng Lesson](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=3198s)

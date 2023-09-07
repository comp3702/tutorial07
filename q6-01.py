import numpy as np

if __name__ == "__main__":
    P = np.array((
        (0.5, 0.5, 0, 0, 0, 0),
        (0.5, 0, 0.5, 0, 0, 0),
        (0, 0.5, 0, 0.5, 0, 0),
        (0, 0, 0.5, 0, 0.5, 0),
        (0, 0, 0, 0.5, 0, 0.5),
        (0, 0, 0, 0, 0.5, 0.5)
    ))

    x0 = np.array((0, 0, 1, 0, 0, 0))
    x1 = np.matmul(x0, P)
    print(x1)

    # x1 = np.array((0, 0.5, 0, 0.5, 0, 0))

    x = x1
    for i in range(100):
        x = np.matmul(x, P)
        print(f"x{i + 2}:\n{x}")


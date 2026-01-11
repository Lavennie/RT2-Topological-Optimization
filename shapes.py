import numpy as np


def random_points(N):
    """
    Generates random 2D point cloud inside the [-1, 1] square with N points.

    Parameters
    ----------
    N : int
        Number of points to generate.

    Returns
    -------
    ndarray of shape (N, 2)
        A randomly generate point cloud.

    """
    return np.random.rand(N, 2) * 2 - 1


def circle_noise(N):
    angles = np.random.rand(N, 1) * 2 * np.pi
    dists = np.random.rand(N, 1) * 0.3 + 0.65
    return np.clip(np.hstack((np.cos(angles), np.sin(angles))) * dists, -1, 1)


def infinity_noise(N):
    c1 = circle_noise(N // 2) * 0.55 - np.array([0.45, 0])
    c2 = circle_noise((N + 1) // 2) * 0.55 + np.array([0.45, 0])
    return np.clip(np.vstack((c1, c2)), -1, 1)


def square_noise(N):
    sides = np.random.randint(0, 4, size=N)
    t = np.random.uniform(-0.8, 0.8, size=N)
    d = np.random.normal(0, 0.05, size=N)  # offset

    x = np.zeros(N)
    y = np.zeros(N)

    # bottom edge (y = -1)
    mask = sides == 0
    x[mask] = t[mask]
    y[mask] = -0.8 + d[mask]

    # top edge (y = +1)
    mask = sides == 1
    x[mask] = t[mask]
    y[mask] = 0.8 + d[mask]

    # left edge (x = -1)
    mask = sides == 2
    x[mask] = -0.8 + d[mask]
    y[mask] = t[mask]

    # right edge (x = +1)
    mask = sides == 3
    x[mask] = 0.8 + d[mask]
    y[mask] = t[mask]

    return np.column_stack((x, y))

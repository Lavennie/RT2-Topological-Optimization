import numpy as np

def random_points(N):
    """
    Generates random 2D point cloud inside the unit square with N points.

    Parameters
    ----------
    N : int
        Number of points to generate.

    Returns
    -------
    ndarray of shape (N, 2)
        A randomly generate point cloud.

    """
    return np.random.rand(N, 2)

def circle_noise(N):
    angles = np.random.rand(N, 1) * 2 * np.pi
    dists = np.random.rand(N, 1) * 0.3 + 0.65
    return np.hstack((np.cos(angles), np.sin(angles))) * dists

def infinity_noise(N):
    c1 = circle_noise(N // 2) * 0.6 - np.array([0.5, 0])
    c2 = circle_noise((N + 1) // 2) * 0.6 + np.array([0.5, 0])
    return np.vstack((c1, c2))
    
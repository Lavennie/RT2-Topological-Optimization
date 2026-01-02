import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser


def get_gradient_vector(u, v, normalize=True):
    """
    Calculates the vector pointing from v to u.
    Used to push u away from v (or pull u towards v).
    """
    direction = u - v
    dist = np.linalg.norm(direction)

    # To avoid division by zero we return a zeros array
    if dist == 0:
        return np.zeros_like(u)

    if normalize:
        return direction / dist
    return direction


# The only domain we currently support is a square
def apply_domain_constraints(points, domain_type="square"):
    """
    Keeps points inside the domain [-1, 1].
    """
    return np.clip(points, -1.0, 1.0)


def compute_persistence(points):
    """
    Wrapper for ripser to ensure consistent settings.
    """
    return ripser(points, do_cocycles=True)


def plot_optimization(initial_points, current_points, iteration):
    """
    Helper to visualize progress.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(
        initial_points[:, 0], initial_points[:, 1], c="gray", alpha=0.5, label="Start"
    )
    plt.scatter(
        current_points[:, 0], current_points[:, 1], c="blue", label=f"Iter {iteration}"
    )
    plt.legend()
    plt.title("Point Cloud Deformation")
    plt.axis("equal")
    plt.show()

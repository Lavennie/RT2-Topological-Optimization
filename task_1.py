import numpy as np
from core import (
    get_gradient_vector,
    apply_domain_constraints,
    compute_persistence,
    plot_optimization,
)

LEARNING_RATE = 0.02
ITERATIONS = 150

SHOW_INTERMEDIATE_PLOTS = True


def get_critical_indices(diagrams, cocycles, target_idx, points):
    """
    Identifies the specific vertices responsible for the Birth and Death
    of the bar at index 'target_idx' in the 1D persistence diagram.
    """
    cycle_edges = cocycles[1][target_idx]

    birth_edge = None
    max_len = -1

    for edge in cycle_edges:
        u, v = int(edge[0]), int(edge[1])
        dist = np.linalg.norm(points[u] - points[v])
        if dist > max_len:
            max_len = dist
            birth_edge = (u, v)

    death_time = diagrams[1][target_idx][1]
    death_triplet = None

    epsilon = 1e-4

    num_points = len(points)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = np.linalg.norm(points[i] - points[j])

            # If this edge is approximately the death time
            if abs(dist - death_time) < epsilon:
                death_triplet = (i, j)
                break
        if death_triplet:
            break

    return birth_edge, death_triplet


# Main optimization loop

t = np.linspace(0, 2 * np.pi, 50)
noise = np.random.normal(0, 0.2, (50, 2))
points = np.column_stack((np.cos(t), np.sin(t))) + noise
initial_points = points.copy()

print("Starting Optimization...")
print("Goal: Prolong the longest 1D bar (make the loop clearer).")

for step in range(ITERATIONS):
    result = compute_persistence(points)
    dgm1 = result["dgms"][1]
    cocycles = result["cocycles"]

    if len(dgm1) == 0:
        print("No loops found. Adding noise to restart...")
        points += np.random.normal(0, 0.1, points.shape)
        continue

    # Find the Longest Bar
    lifetimes = dgm1[:, 1] - dgm1[:, 0]
    longest_idx = np.argmax(lifetimes)

    birth_pair, death_pair = get_critical_indices(
        result["dgms"], cocycles, longest_idx, points
    )

    if birth_pair:
        u, v = birth_pair
        grad = get_gradient_vector(points[u], points[v])  # Vector from v to u
        # Pull them together
        points[u] -= LEARNING_RATE * grad
        points[v] += LEARNING_RATE * grad

    if death_pair:
        u, v = death_pair
        grad = get_gradient_vector(points[u], points[v])
        # Push them apart
        points[u] += LEARNING_RATE * grad
        points[v] -= LEARNING_RATE * grad

    points = apply_domain_constraints(points)

    if step % 20 == 0 and SHOW_INTERMEDIATE_PLOTS:
        print(f"Iteration {step}: Max Lifetime = {lifetimes[longest_idx]:.4f}")
        # Only plot if running locally (disabled for speed if needed)
        plot_optimization(initial_points, points, step)

print("Optimization Complete.")
plot_optimization(initial_points, points, ITERATIONS)

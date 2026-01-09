import numpy as np
import gudhi
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class TopologicalOptimizer:
    def __init__(self, points, learning_rate=0.01, momentum=0.9):
        self.points = points.astype(np.float64)
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = np.zeros_like(points)

    def get_gradient_vector(self, p1, p2):
        """
        Computes the gradient of distance ||p1 - p2|| with respect to p1.
        Returns unit vector (p1 - p2) / ||p1 - p2||.
        """
        diff = p1 - p2
        dist = np.linalg.norm(diff)
        if dist < 1e-8:
            return np.zeros_like(diff), dist  # Handle coincident points safely
        return diff / dist, dist

    def optimize_rips_topology(self, points, homology_group_dim=1, epochs=50, lr=0.02):
        """
        Optimizes the point cloud to prolong the dominant H1 feature.
        Includes intermediate visualizations.
        """
        if homology_group_dim < 0:
            raise ValueError("Homology group dimension must be >= 0")
        print("Starting Optimization...")

        for epoch in range(epochs):
            rc = gudhi.RipsComplex(points=points)
            st = rc.create_simplex_tree(max_dimension=2)

            # Get persistence pairs for H_k
            persistence = st.persistence()
            hk_features = [
                (birth, death)
                for dim, (birth, death) in persistence
                if dim == homology_group_dim and death != float("inf")
            ]

            if len(hk_features) == 0:
                print(f"Epoch {epoch}: No H1 features found. Adding noise.")
                points += np.random.normal(0, 0.01, points.shape)
                continue

            # Find the most persistent feature
            persistences = [
                (death - birth, birth, death) for birth, death in hk_features
            ]
            max_pers, birth_time, death_time = max(persistences, key=lambda x: x[0])

            # Find edges at birth and death times
            # Get all edges (1-simplices) from the filtration
            simplices_1 = [
                simplex for simplex, filt in st.get_filtration() if len(simplex) == 2
            ]

            # Find birth edge (first edge to appear at birth_time)
            birth_edge = None
            for simplex in simplices_1:
                if abs(st.filtration(simplex) - birth_time) < 1e-9:
                    birth_edge = tuple(simplex)
                    break

            # Find death edge (edge that closes the loop at death_time)
            death_edge = None
            for simplex in simplices_1:
                if abs(st.filtration(simplex) - death_time) < 1e-9:
                    death_edge = tuple(simplex)
                    break

            if birth_edge is None or death_edge is None:
                print(f"Epoch {epoch}: Could not find critical edges. Skipping.")
                continue

            # Gradient descent
            bu, bv = birth_edge
            du, dv = death_edge

            grad_accum = np.zeros_like(points)

            # Minimize Birth (contract birth edge)
            dir_b, dist_b = self.get_gradient_vector(points[bu], points[bv])
            grad_accum[bu] -= dir_b  # Move towards bv
            grad_accum[bv] += dir_b  # Move towards bu

            # Maximize Death (expand death edge)
            dir_d, dist_d = self.get_gradient_vector(points[du], points[dv])
            grad_accum[du] += dir_d  # Move away from dv
            grad_accum[dv] -= dir_d  # Move away from du

            # Apply updates
            points += lr * grad_accum

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Max Persistence = {max_pers:.4f}")
                visualize_step(points, birth_edge, death_edge, epoch)

        return points


def visualize_step(points, birth_edge, death_edge, epoch, title="Optimization"):
    """
    Plots the point cloud and highlights the critical edges being optimized.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].set_title(f"Point Cloud Geometry (Epoch {epoch})")
    ax[0].scatter(points[:, 0], points[:, 1], s=10, c="gray", alpha=0.5)

    # Highlight Birth Edge (Green - Contract)
    if birth_edge is not None:
        u, v = birth_edge
        ax[0].plot(
            [points[u][0], points[v][0]],
            [points[u][1], points[v][1]],
            "b--",
            linewidth=3,
            label="Birth Edge (Contract)",
        )

    # Highlight Death Edge (Red - Expand)
    if death_edge is not None:
        u, v = death_edge
        ax[0].plot(
            [points[u][0], points[v][0]],
            [points[u][1], points[v][1]],
            "r--",
            linewidth=3,
            label="Death Edge (Expand)",
        )

    ax[0].legend(loc="upper right")
    ax[0].axis("equal")

    # We re-compute just to show the current state
    rc = gudhi.RipsComplex(points=points)
    st = rc.create_simplex_tree(max_dimension=2)
    st.compute_persistence()

    # Extract H1 points
    h1_points = [
        p[1] for p in st.persistence() if p[0] == 1 and p[1][1] != float("inf")
    ]
    if h1_points:
        h1_arr = np.array(h1_points)
        ax[1].scatter(h1_arr[:, 0], h1_arr[:, 1], c="blue", label="H1 Features")
        # Highlight the max persistence one
        pers = h1_arr[:, 1] - h1_arr[:, 0]
        max_idx = np.argmax(pers)
        ax[1].scatter(
            [h1_arr[max_idx, 0]],
            [h1_arr[max_idx, 1]],
            c="red",
            s=100,
            zorder=5,
            label="Target Feature",
        )

    ax[1].set_title("Persistence Diagram (H1)")
    ax[1].set_xlabel("Birth")
    ax[1].set_ylabel("Death")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    points = np.random.random((50, 2))

    opt = TopologicalOptimizer(points, learning_rate=0.05)

    print("Starting optimization...")
    target_info = opt.optimize_rips_topology(points=points, epochs=101)

    print("Done.")

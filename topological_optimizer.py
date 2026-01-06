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

    def compute_persistence(self):
        """Compute persistence using Gudhi AlphaComplex."""

        ac = gudhi.AlphaComplex(points=self.points)
        st = ac.create_simplex_tree()
        st.compute_persistence()
        return st

    def get_critical_features(self, simplex_tree, dim=1):
        """
        Identify critical vertices for birth and death.
        """
        pairs = simplex_tree.persistence_pairs()

        features = list()

        for birth_simplex, death_simplex in pairs:
            # Filter by dimension (birth simplex dim = feature dim)
            if len(birth_simplex) == dim + 1 and len(death_simplex) == dim + 2:

                b_val = simplex_tree.filtration(birth_simplex)
                d_val = simplex_tree.filtration(death_simplex)

                # Skip infinite features
                if np.isinf(d_val):
                    continue

                features.append(
                    {
                        "birth_verts": list(birth_simplex),
                        "death_verts": list(death_simplex),
                        "birth_val": b_val,
                        "death_val": d_val,
                        "persistence": d_val - b_val,
                    }
                )

        # Sort by persistence (descending)
        features.sort(key=lambda x: x["persistence"], reverse=True)
        return features

    def prolong_target_feature(self, dim=1, target_idx=0, noise_weight=0.5):
        """
        Prolong the target feature, kill others.
        Returns the critical vertices of the target for visualization.
        """
        st = self.compute_persistence()
        features = self.get_critical_features(st, dim=dim)

        if not features:
            self.points += np.random.normal(0, 0.01, self.points.shape)
            return None

        # Accumulate gradients
        grad_accum = np.zeros_like(self.points)

        # Optimizing target
        target_info = None
        if target_idx < len(features):
            target = features[target_idx]
            target_info = target  # For visualization return

            # Minimize birth
            # Note: To prolong (d - b), we minimize b.
            u, v = target["birth_verts"]
            direction = self.points[u] - self.points[v]
            dist = np.linalg.norm(direction) + 1e-8
            direction /= dist

            # Pull u towards v
            grad_accum[u] += direction
            grad_accum[v] -= direction

            # Maximize death
            d_verts = target["death_verts"]
            centroid = np.mean(self.points[d_verts], axis=0)
            for idx in d_verts:
                dir_c = self.points[idx] - centroid
                dir_c /= np.linalg.norm(dir_c) + 1e-8
                # Move AWAY from centroid (gradient ascent on radius)
                # Since we do p -= lr * grad, we need grad to be negative of direction
                grad_accum[idx] -= dir_c

        # Loop for optimizing noise
        for i, feat in enumerate(features):
            if i == target_idx:
                continue

            # Kill noise by pulling birth vertices together (collapse hole)
            u, v = feat["birth_verts"]
            direction = self.points[u] - self.points[v]
            dist = np.linalg.norm(direction) + 1e-8
            direction /= dist

            grad_accum[u] += direction * noise_weight
            grad_accum[v] -= direction * noise_weight

        # Apply Momentum & Update
        self.velocity = self.momentum * self.velocity + self.lr * grad_accum
        self.points -= self.velocity

        return target_info


def visualize_progress(optimizer, step_num, target_info):
    points = optimizer.points
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], c="black", s=15, label="Points")

    if target_info:
        # Highlight birth edge in red
        u, v = target_info["birth_verts"]
        p_u, p_v = points[u], points[v]
        plt.plot(
            [p_u[0], p_v[0]], [p_u[1], p_v[1]], "r-", linewidth=3, label="Birth Edge"
        )

        # Highlight death simplex in blue
        d_verts = target_info["death_verts"]
        d_points = points[d_verts]
        # Draw triangle
        t_draw = np.vstack([d_points, d_points])
        plt.plot(t_draw[:, 0], t_draw[:, 1], "b-", linewidth=2, label="Death Simplex")

    plt.title(f"Optimization Step {step_num}")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    points = np.random.random((50, 2))

    opt = TopologicalOptimizer(points, learning_rate=0.05)

    print("Starting optimization...")
    for i in range(100):
        # Prolong the main loop (dim=1, target_idx=0)
        target_info = opt.prolong_target_feature(dim=1, target_idx=0)

        # Visualize every 20 steps
        if i % 20 == 0:
            print(f"Visualizing step {i}...")
            visualize_progress(opt, i, target_info)

    print("Done.")

import numpy as np
import gudhi
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from shapes import *
from visualization import *


class TopologicalOptimizer:
    def __init__(
        self,
        points,
        learning_rate=0.02,
        momentum=0.9,
        extend=True,
        round_robin=False,
        target_cloud=None,
        homology_group_dim=-1. # negative homology will optimize both dimension (0 and 1)
    ):
        self.points = points.astype(np.float64)
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = np.zeros_like(points)
        self.extend = extend
        self.round_robin = round_robin
        self.feature_index = 0
        self.homology_group_dim = homology_group_dim
        self.lr = learning_rate
        if target_cloud is not None:
            diagrams = persistence_diagram(target_cloud)[1]
            diagrams = np.array(
                [(death - birth, birth, death) for birth, death in diagrams]
            )
            self.target_persistence = diagrams
        else:
            self.target_persistence = None

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

    def sort_persistences(self, persistences):
        return persistences[np.argsort(persistences[:, 0])[::-1]]

    def significant_features(self, persistences):
        """
        Features (their values) that live long enough sorted by their lifetime from longer to shorter.
        """
        return self.sort_persistences(persistences[persistences[:, 0] > 0.1])

    def largest_features(self, persistences, k):
        """
        The top k features that survive the longest.
        """
        if len(persistences) < k:
            return persistences
        return persistences[np.argpartition(persistences[:, 0], 0)[-k:]]

    def optimize_rips_iterator(self, points, birth_death, epoch):
        points, birth_edge, death_edge, max_pers = self.optimize_rips_step(points, epoch)
        return points, [birth_edge, death_edge]
    def optimize_rips_step(self, points, epoch):
        """
        Single step that optimizes the point cloud:
            - optimizes the most significant feature (default) or
            - optimizes significant featues in a round robin style (round_robin = True) or
            - optimizes all features toward a target diagram (persistence_diagram != None).
        """
        rc = gudhi.RipsComplex(points=points)
        st = rc.create_simplex_tree(max_dimension=2)

        # Get persistence pairs for H_k
        persistence = st.persistence()
        # choose or retrieve the homology group to optimize
        hom_grp_dim = self.homology_group_dim if self.homology_group_dim >= 0 else np.random.randint(0, 2)
        
        hk_features = [
            (birth, death)
            for dim, (birth, death) in persistence
            if dim == hom_grp_dim and death != float("inf")
        ]

        # no need to optimize if there aren't any features
        if len(hk_features) == 0:
            print(f"Epoch {epoch}: No H{hom_grp_dim} features found. Adding noise.")
            points += np.random.normal(0, 0.01, points.shape)
            return (points, None, None, 0)

        # Find the most persistent feature or pick the significant ones in
        persistences = np.array(
            [(death - birth, birth, death) for birth, death in hk_features]
        )
        # Optimize toward another persistence diagram, picks a random feature
        if self.target_persistence is not None:
            if len(self.target_persistence) == 0:
                raise Exception(
                    "Target persistence does not have any significant features."
                )
            # These will be transformed in to the target features in a round robin style
            persistences = self.sort_persistences(persistences)
            # largest = self.largest_features(persistences, len(self.target_persistence))
            # self.feature_index = self.feature_index % len(persistences) # Loop around back to the first feature
            self.feature_index = np.random.randint(0, len(persistences))
            max_pers, birth_time, death_time = persistences[self.feature_index, :]
            if self.feature_index < len(
                self.target_persistence
            ):  # feature to transform into one of the reference features
                # Choose whether to shorten or lengthen the feature
                self.extend = (
                    max_pers < self.target_persistence[self.feature_index, 0]
                )
            else:
                # Contract any other holes that exist
                self.extend = False

            # max_pers, birth_time, death_time = largest[np.argpartition(largest[:, 0], 0)[-self.feature_index - 1], :]
            self.feature_index += 1
        # Modify only significant features swapping among them in a round robin fashion
        elif self.round_robin:
            significant = self.significant_features(
                persistences
            )  # 0.1 is some arbitrary cutoff
            if len(significant) == 0:
                significant = np.array(
                    [persistences[0, :]]
                )  # Just take one diagram randomly and modify it
            self.feature_index = self.feature_index % len(
                significant
            )  # Loop around back to the first feature
            max_pers, birth_time, death_time = significant[
                np.argpartition(significant[:, 0], 0)[-self.feature_index - 1], :
            ]
            self.feature_index += 1
        # Optimize the most significant feature
        else:
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

        # it is normal for H0 to not have a birth edge (birth vertex, but that's not useful), don't end optimization
        if hom_grp_dim > 0 and birth_edge is None or death_edge is None:
            print(f"Epoch {epoch}, H{hom_grp_dim}: Could not find critical edges: birth={birth_edge}, death={death_edge}. Skipping.")
            return (points, None, None, max_pers)

        # Gradient descent
        if birth_edge is not None:
            bu, bv = birth_edge
        du, dv = death_edge

        grad_accum = np.zeros_like(points)

        if self.extend:
            # Target Feature: Linear force
            force_magnitude = 1.0
            direction = 1
        else:
            # Noise Feature: Square force (scale by persistence)
            current_persistence = death_time - birth_time
            force_magnitude = 2.0 * current_persistence
            direction = -1

        # Birth Edge Update
        if birth_edge is not None:
            dir_b, dist_b = self.get_gradient_vector(points[bu], points[bv])
            # If direction is 1 (target): contract birth (move bu -> bv)
            # If direction is -1 (noise): expand birth (move bu <- bv)
            grad_accum[bu] -= dir_b * direction * force_magnitude
            grad_accum[bv] += dir_b * direction * force_magnitude

        # Death Edge Update
        dir_d, dist_d = self.get_gradient_vector(points[du], points[dv])
        # If direction is 1 (target): expand death (move du <- dv)
        # If direction is -1 (noise): contract death (move du -> dv)
        grad_accum[du] += dir_d * direction * force_magnitude
        grad_accum[dv] -= dir_d * direction * force_magnitude

        # Apply updates
        points += self.lr * grad_accum

        # Clamp the point positions to [-1,1]
        points = np.clip(points, -1, 1)
        
        return (points, birth_edge, death_edge, max_pers)

    def optimize_rips_topology(self, points, epochs=50, lr=0.02):
        """
        Optimizes the point cloud to prolong the dominant feature.
        Includes intermediate visualizations.
        """
        
        for epoch in range(epochs):
            points, birth_edge, death_edge, max_pers = self.optimize_rips_step(points, epoch)
            
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
    ax[0].set_xlim(-1, 1)
    ax[0].set_ylim(-1, 1)

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

    ax[1].plot(
        [0, np.sqrt(3)], [0, np.sqrt(3)], "--", color="black", alpha=0.4, linewidth=1
    )
    ax[1].set_title("Persistence Diagram (H1)")
    ax[1].set_xlabel("Birth")
    ax[1].set_ylabel("Death")
    ax[1].set_xlim(0, np.sqrt(3))
    ax[1].set_ylim(0, np.sqrt(3))
    ax[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    # points = np.random.random((50, 2))
    points = circle_noise(100)
    reference = infinity_noise(100)
    
    plot_diagram(persistence_diagram(reference))

    opt = TopologicalOptimizer(
        points,
        learning_rate=0.02,
        extend=True,
        round_robin=False,
        target_cloud=reference,
        homology_group_dim=-1
    )
    point_cloud_persistence_anim(points, opt.optimize_rips_iterator, 50, "point_cloud.mp4", 20, extra_param=[], extra_display="birth death")

    print("Starting optimization...")
    #target_info = opt.optimize_rips_topology(
    #    points=points, epochs=500
    #)

    print("Done.")

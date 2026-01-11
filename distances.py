import numpy as np
from scipy.spatial.distance import directed_hausdorff
from persim import bottleneck
from visualization import *

def hausdorff_distance(cloud1, cloud2):
    return max(
        directed_hausdorff(cloud1, cloud2)[0],
        directed_hausdorff(cloud2, cloud1)[0]
    )

def bottleneck_distance(cloud1, cloud2):
    diag1 = persistence_diagram(cloud1)
    diag2 = persistence_diagram(cloud2)
    
    def finite(D):
        return [(b, d) for b, d in D if np.isfinite(d)]
    
    return max(
        bottleneck(finite(diag1[0]), finite(diag2[0])), 
        bottleneck(finite(diag1[1]), finite(diag2[1]))
    )


if __name__ == "__main__":
    prefix = "bubble_circle_"
    start_points = np.load(prefix + "start_points.npy")
    final_points = np.load(prefix + "optimized.npy")
    reference = np.load(prefix + "target_points.npy")
    
    
    mask = np.all((final_points >= -1) & (final_points <= 1), axis=1)
    final_points = final_points[mask]
    plot_points_and_diagram(reference)
    #plt.savefig(prefix + "target_points.png")
    plot_points_and_diagram(start_points)
    #plt.savefig(prefix + "start_points.png")
    plot_points_and_diagram(final_points)
    #plt.savefig(prefix + "optimized.png")
    print(hausdorff_distance(reference, final_points))
    print(bottleneck_distance(reference, final_points))
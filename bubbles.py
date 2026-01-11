import numpy as np
from shapes import *
from visualization import *
from distances import *

BLOAT_STEP = 0.01
SPREAD_STEP = 0.01
DRIFT_STEP = 0.1

def random_vector(length):
    angle = np.random.uniform(0, 2 * np.pi)
    return np.array([np.cos(angle), np.sin(angle)]) * length
def move_towards_vec(current, target, max_distance_delta):
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)

    delta = target - current
    dist = np.linalg.norm(delta)

    if dist <= max_distance_delta or dist == 0.0:
        return target.copy()
    
    return current + (delta / dist) * max_distance_delta
def move_towards(current, target, max_distance_delta):
    delta = target - current
    dist = np.linalg.norm(delta, axis=1)

    ended_mask = (dist == 0.0) | (dist <= max_distance_delta)
    res = np.empty_like(current)
    res[ended_mask] = target[ended_mask]
    res[~ended_mask] = current[~ended_mask] + delta[~ended_mask] / dist[~ended_mask][:, None] * max_distance_delta
    return res

def bloat(centroid, points):
    max_bloat = np.min([centroid[0] + 1, 1 - centroid[0], centroid[1] + 1, 1 - centroid[1]])
    from_center = points - centroid
    lengths = np.linalg.norm(from_center, axis=1)
    mask = lengths > 0
    targets = np.empty_like(points)
    targets[mask] = from_center / lengths[:, None] * max_bloat
    targets[~mask] = points[~mask]
    return centroid + move_towards(from_center, targets, BLOAT_STEP)
def closest_points(source, points, k):
    distances = np.linalg.norm(points - source, axis=1)
    nearest_indices = np.argpartition(distances, 2)[:k]
    return points[nearest_indices]
def signed_angle(a, b):
    """
    Returns the signed angle in degrees between 2D vectors a and b.
    Positive if b is counterclockwise from a, negative if clockwise.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    det = a[0]*b[1] - a[1]*b[0]   # a x b (2D cross)
    dot = np.dot(a, b)

    angle_rad = np.arctan2(det, dot)
    
    return angle_rad
def normalize(v):
    """
    Returns the normalized vector of v.
    If v is zero, returns zero vector.
    
    v: array-like [x, y, ...]
    """
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.zeros_like(v)
    return v / norm
def rotate_vector_2d(v, angle_deg):
    """
    Rotate a 2D vector v by angle_deg degrees counterclockwise.
    """
    angle_rad = np.radians(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    rotation_matrix = np.array([[c, -s],
                                [s,  c]])
    return rotation_matrix @ v
def spread(centroid, points):
    res = np.empty_like(points)
    for i in range(len(points)):
        closest = closest_points(points[i, :], np.delete(points, i, axis=0), 2)
        ca = closest[0, :] - centroid
        cb = closest[1, :] - centroid
        cp = points[i, :] - centroid
        
        a1 = signed_angle(cp, ca)
        a2 = signed_angle(cp, cb)
        
        # point is between closest ones - take the average
        if np.sign(a1) != np.sign(a2):
            target_pos = centroid + normalize((ca + cb) / 2) * (np.linalg.norm(ca) + np.linalg.norm(ca)) / 2
        # point is on one side of both closest points - push away from them
        else:
            target_pos = centroid + rotate_vector_2d(cp, -np.sign(a1))
        res[i, :] = move_towards_vec(points[i, :], target_pos, SPREAD_STEP)
    return res
def drift(centroids):
    if len(centroids) == 1:
        return np.array([move_towards_vec(centroids[0, :], np.array([0, 0]), DRIFT_STEP)])
    else:
        result = np.empty_like(centroids)
        for i in range(len(centroids)):
            c = centroids[i, :]
            if c[1] >= c[0] and c[1] >= -c[0]: # top wall
                wall = np.array([c[0], 1])
            elif c[1] < c[0] and c[1] < -c[0]: # bottom wall
                wall = np.array([c[0], -1])
            elif c[1] >= c[0] and c[1] < -c[0]: # left wall
                wall = np.array([-1, c[1]])
            else: # right wall
                wall = np.array([1, c[1]])
            other = np.vstack((wall, np.delete(centroids, i, axis=0)))
            from_other = c - other
            from_other[0, :] *= 2
            dists = np.linalg.norm(from_other, axis=1)
            t =  np.sin(2 * np.atan2(c[1], c[0])) ** 4 
            from_other[0, :] = (1 - t) * from_other[0, :] + t * normalize(-c) * dists[0] # combination of direction directly from wall and toward center
            #from_other[0, :] = normalize(-c) * dists[0] # wall direction toward center
            mask = dists > 0
            weight = (dists[mask] * dists[mask])[:, None]
            change = (from_other / weight).mean(axis=0)
            #change += random_vector(DRIFT_STEP / 10)
            result[i, :] = np.clip(c + change * DRIFT_STEP, -1, 1)
        return result
def get_centroids(points, N):
    rips = gd.RipsComplex(points=points, max_edge_length=MAX_RANGE)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    simplex_tree.compute_persistence()

    pairs = simplex_tree.persistence_pairs()

    # get all the hole birth-death simplices
    holes = []
    for (birth_sx, death_sx) in pairs:
        if (len(death_sx) == 3 and death_sx is not None):
            holes.append((birth_sx, death_sx))
    def distance(v1, v2):
        return np.linalg.norm(v1 - v2)
    def sort_key(vertex_indices):
        vertices = points[vertex_indices[1]]
        return max(distance(vertices[0, :], vertices[1, :]), distance(vertices[1, :], vertices[2, :]), distance(vertices[2, :], vertices[0, :]))

    # get the holes with the biggest death simplices
    holes.sort(key=sort_key, reverse=True)
    # take the first N holes and calculate their centroid
    centroids = []
    for i in range(N):
        centroids.append(points[holes[i][1]].mean(axis=0))
    return np.array(centroids)
def cluster(points, centroids):
    assignment = np.empty(len(points))
    for i in range(len(points)):
        dists = np.linalg.norm(centroids - points[i, :], axis=1)
        nearest_index = np.argmin(dists)
        assignment[i] = nearest_index
    return assignment
        
def effervescence(points, centroids, frame):
    global clustering
    centroids = drift(centroids)
    for i in range(len(centroids)):
        mask = clustering == i
        points[mask] = spread(centroids[i, :], bloat(centroids[i, :], points[mask]))
    return points, centroids

if __name__ == "__main__":
    np.random.seed(42)
    centroid_count = 1
    points = random_points(100)
    reference = circle_noise(100)
    centroids = get_centroids(points, centroid_count)
    clustering = cluster(points, centroids)
    plot_points_and_diagram(points)
    
    final_points = point_cloud_persistence_anim(points, effervescence, 100, "point_cloud.mp4", 5, extra_param=centroids, extra_display="centroids")
    
    np.save("start_points.npy", points)
    np.save("target_points.npy", reference)
    np.save("optimized.npy", final_points)
    print(hausdorff_distance(reference, final_points))
    print(bottleneck_distance(reference, final_points))
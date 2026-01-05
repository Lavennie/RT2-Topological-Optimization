import numpy as np
from shapes import *
from visualization import *

BLOAT_STEP = 0.01
SPREAD_STEP = 0.01
DRIFT_STEP = 0.01
points = circle_noise(100)
plot_points(points)

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
    targets = from_center / lengths[:, None] * max_bloat
    return centroid + move_towards(from_center, targets, BLOAT_STEP)
def closest_points(source, points):
    distances = np.linalg.norm(points - source, axis=1)
    nearest_indices = np.argpartition(distances, 2)[:2]
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
        closest = closest_points(points[i, :], np.delete(points, i, axis=0))
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
    if len(centroids == 1):
        return np.array([move_towards_vec(centroids[0, :], np.array([0, 0]), DRIFT_STEP)])
    else:
        return centroids
        
def effervescence(points, frame):
    global centroids
    centroids = drift(centroids)
    return spread(centroids[0, :], bloat(centroids[0, :], points))

centroids = np.array([[0.6, -0.3]])
plot_points(points)
func = lambda x, _: spread(centroids[0, :], bloat(centroids[0, :], x))

point_cloud_anim(points, effervescence, 100, "point_cloud.mp4", 5)
persistence_diagram_anim(points, effervescence, 100, "persistence_diagram.mp4", 10)
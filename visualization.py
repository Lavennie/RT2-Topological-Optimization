import numpy as np
from ripser import ripser
import gudhi as gd
from persim import plot_diagrams
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


MAX_RANGE = np.sqrt(3)
# randomly generate points
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
def plot_points(points, show=True):
    """
    Plot a 2D point cloud.

    Parameters
    ----------
    points : ndarray of shape (N, 2)
        Point positions.
    show : bool, optional
        If True (default), display the plot using plt.show().
        If False, the plot is created but not shown.

    Returns
    -------
    None
    """
    plt.plot(points[:, 0], points[:, 1], '.', color='black')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('equal')
    if show:
        plt.show()
# compute and plot the persitance diagram of the points
def persistence_diagram(points):
    """
    Calculate the persistance diagram of the given poin cloud via the Rips filtration.
    
    Parameters
    ----------
        points : ndarray of shape (N, 2)
            Starting point positions.
            
    Returns
    -------
    list of ndarray
        Persistence diagrams for each homology dimension.
        result[0] corresponds to H0, result[1] to H1.
    """
    result = ripser(points, maxdim=1)
    return result['dgms']
def plot_diagram(diagram, max_val=MAX_RANGE):
    """
    Plot a persistence diagram with fixed axis bounds.

    Parameters
    ----------
    diagram : list of ndarray
        Persistence diagrams as returned by ripser, one array per homology dimension.
    max_val : float, optional
        Maximum value for both birth and death axes. Defaults to MAX_RANGE.

    Returns
    -------
    None
    """
    plot_diagrams(diagram, xy_range=[0, max_val, 0, max_val])
    
def persistence_diagram_anim(points, iter_func, repeat_count, file, duration, max_val=MAX_RANGE, extra_param=None):
    """
    Animate the evolution of a persistence diagram under an iterative point update.

    Parameters
    ----------
    points : ndarray of shape (N, 2)
        Initial point cloud.
    iter_func : callable
        Function iter_func(points, frame) -> updated points.
    repeat_count : int
        Number of animation frames.
    file : str
        Output animation filename (e.g. "diagram.mp4").
    duration : float
        Total animation duration in seconds.
    max_val : float, optional
        Maximum birth/death value for axis limits.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")
    ax.set_title("Rips Filtration")
    
    ax.plot([0, max_val], [0, max_val], '--', color='black', alpha=0.4, linewidth=1)
    
    scat_h0 = ax.scatter([], [], c="blue", label="H0")
    scat_h1 = ax.scatter([], [], c="orange", label="H1")
    ax.legend(loc="lower right")
    
    points = points.copy()
    
    def update(frame):
        nonlocal points
        nonlocal extra_param
        if frame % 100 == 0:
            print(frame, '/', repeat_count)
        
        diagram = persistence_diagram(points)
        # iterater after drowing once, to still draw the starting frame
        if extra_param is None:
            points = iter_func(points, frame) 
        else:
            points, extra_param = iter_func(points, extra_param, frame)
        
        if len(diagram[0]) > 0:
            scat_h0.set_offsets(diagram[0])
        else:
            scat_h0.set_offsets(np.empty((0, 2)))
            
        if len(diagram) > 0 and len(diagram[1]) > 0:
            scat_h1.set_offsets(diagram[1])
        else:
            scat_h1.set_offsets(np.empty((0, 2)))
        
        ax.set_title(f"Persistence diagram ≤ {frame}")
        return scat_h0, scat_h1
    
    anim = FuncAnimation(fig, update, frames=repeat_count, interval=400)
    anim.save(file, writer=FFMpegWriter(fps=repeat_count / duration))
    
def point_cloud_anim(points, iter_func, repeat_count, file, duration, centroids=None):
    """
    Animate the evolution of a point cloud under an iterative point update.

    Parameters
    ----------
    points : ndarray of shape (N, 2)
        Initial point cloud.
    iter_func : callable
        Function iter_func(points, frame) -> updated points.
    repeat_count : int
        Number of animation frames.
    file : str
        Output animation filename (e.g. "point_cloud.mp4").
    duration : float
        Total animation duration in seconds.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_title("Rips Filtration")
    
    scat = ax.scatter([], [], c="blue", label="Points")
    if centroids is not None:
        scat_cent = ax.scatter([], [], c="orange", label="Centroids")
    ax.legend(loc="lower right")
    
    points = points.copy()
    
    def update(frame):
        nonlocal points
        nonlocal centroids
        if frame % 100 == 0:
            print(frame, '/', repeat_count)
        
        scat.set_offsets(points)
        ax.set_title(f"Point cloud {frame}")
        
        # iterater after drowing once, to still draw the starting frame
        if centroids is None:
            points = iter_func(points, frame) 
            return scat
        else:
            scat_cent.set_offsets(centroids)
            points, centroids = iter_func(points, centroids, frame)
            return scat, scat_cent
    
    anim = FuncAnimation(fig, update, frames=repeat_count, interval=400)
    anim.save(file, writer=FFMpegWriter(fps=repeat_count / duration))
def rips_anim_steps(points, file, duration):
    """
    Animate the Rips filtration by stepping through discrete filtration values.

    Edges and triangles appear at their filtration time, and balls grow in steps.

    Parameters
    ----------
    points : ndarray of shape (N, 2)
        Point cloud from which a simplicial complex is built.
    file : str
        Output animation filename (e.g. "rips_steps.mp4").
    duration : float
        Total animation duration in seconds.

    Returns
    -------
    None
    """
    rips = gd.RipsComplex(points=points, max_edge_length=MAX_RANGE)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    simplices = list(simplex_tree.get_filtration())
        
    simpl_by_f = dict()
    for s, f in simplices:
        if not f in simpl_by_f:
            simpl_by_f[f] = []
        simpl_by_f[f].append(s)
     
    filtration_values = sorted(simpl_by_f.keys())
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Rips Filtration")
    ax.scatter(points[:, 0], points[:, 1], color="black")
    
    
    edge_lines = []
    triangle_patches = []
    circle_patches = [plt.Circle(p, 0, fill=False, color='blue', alpha=0.3, linewidth=1.5) for p in points]
    for c in circle_patches:
        ax.add_patch(c)
    
    def update(frame):
        if frame % 100 == 0:
            print(frame, '/', len(filtration_values))
        f = filtration_values[frame]
        
        # udpdate circle radius
        for c in circle_patches:
            c.set_radius(f)
        
        # remove previous frame's edges/triangles
        for l in edge_lines:
            l.remove()
        edge_lines.clear()
        for t in triangle_patches:
            t.remove()
        triangle_patches.clear()
    
        # draw simplices up to current filtration value
        for fv in filtration_values[:frame + 1]:
            for s in simpl_by_f[fv]:
                if len(s) == 2:  # edge
                    i, j = s
                    line, = ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color="green", alpha=0.7, linewidth=2)
                    edge_lines.append(line)
                elif len(s) == 3:  # triangle
                    i, j, k = s
                    tri = plt.Polygon([points[i], points[j], points[k]], color="orange", alpha=0.25)
                    ax.add_patch(tri)
                    triangle_patches.append(tri)
    
        ax.set_title(f"Rips Filtration ≤ {f:.3f}")
        return edge_lines + triangle_patches + circle_patches
    
    anim = FuncAnimation(fig, update, frames=len(filtration_values), interval=400)
    anim.save(file, writer=FFMpegWriter(fps=len(filtration_values) / duration))
    
def rips_anim_smooth(points, file, duration):
    """
    Animate the Rips filtration with smooth ball growth.

    Balls grow continuously, and simplices appear when their filtration
    threshold is reached.

    Parameters
    ----------
    points : ndarray of shape (N, 2)
        Point cloud from which a simplicial complex is built.
    file : str
        Output animation filename (e.g. "rips_steps.mp4").
    duration : float
        Total animation duration in seconds.

    Returns
    -------
    None
    """
    rips = gd.RipsComplex(points=points, max_edge_length=MAX_RANGE)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    simplices = list(simplex_tree.get_filtration())
        
    simpl_by_f = dict()
    for s, f in simplices:
        if not f in simpl_by_f:
            simpl_by_f[f] = []
        simpl_by_f[f].append(s)
     
    filtration_values = sorted(simpl_by_f.keys())
    max_filtration = filtration_values[-1]
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Rips Filtration")
    ax.scatter(points[:, 0], points[:, 1], color="black")
    
    
    edge_lines = []
    triangle_patches = []
    circle_patches = [plt.Circle(p, 0, fill=False, color='blue', alpha=0.3, linewidth=1.5) for p in points]
    for c in circle_patches:
        ax.add_patch(c)
    
    def update(frame):
        if frame % 100 == 0:
            print(frame, '/', len(filtration_values))
        t = frame / len(filtration_values)
        radius = t * max_filtration
        
        # udpdate circle radius
        for c in circle_patches:
            c.set_radius(radius)
        
        # remove previous frame's edges/triangles
        for l in edge_lines:
            l.remove()
        edge_lines.clear()
        for t in triangle_patches:
            t.remove()
        triangle_patches.clear()
    
        # draw simplices up to current filtration value
        for fv in filtration_values:
            if fv > radius:
                break
            for s in simpl_by_f[fv]:
                if len(s) == 2:  # edge
                    i, j = s
                    line, = ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color="green", alpha=0.7, linewidth=2)
                    edge_lines.append(line)
                elif len(s) == 3:  # triangle
                    i, j, k = s
                    tri = plt.Polygon([points[i], points[j], points[k]], color="orange", alpha=0.25)
                    ax.add_patch(tri)
                    triangle_patches.append(tri)
    
        ax.set_title(f"Rips Filtration: ≤ {radius:.3f}")
        return edge_lines + triangle_patches + circle_patches
    
    anim = FuncAnimation(fig, update, frames=len(filtration_values), interval=400)
    anim.save(file, writer=FFMpegWriter(fps=len(filtration_values) / duration))
    

#persistence_diagram_anim(random_points(30), lambda x, _: x*0.9, 10, "persistance_diagram.mp4", 10)
#rips_anim_smooth(random_points(30), "rips_filtration_smooth.mp4", 10)
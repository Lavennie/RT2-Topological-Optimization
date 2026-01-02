import numpy as np
from ripser import ripser
import gudhi as gd
from persim import plot_diagrams
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


MAX_RANGE = np.sqrt(2)
# randomly generate points
def random_points(N):
    return np.random.rand(N, 2)
def plot_points(points, show=True):
    plt.plot(points[:, 0], points[:, 1], '.', color='black')
    if show:
        plt.show()
# compute and plot the persitance diagram of the points
def persistance_diagram(points):
    result = ripser(points, maxdim=1)
    return result['dgms']
def plot_diagram(diagram, max_val=MAX_RANGE):
    plot_diagrams(diagram, xy_range=[0, max_val, 0, max_val])

def rips_anim_steps(points, file, duration):
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
    
#rips_anim_smooth(random_points(30), "rips_filtration_smooth.mp4", 10)
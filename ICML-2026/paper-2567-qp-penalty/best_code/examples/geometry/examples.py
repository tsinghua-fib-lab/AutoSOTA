import numpy as np
import igl

import os, sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from matplotlib.tri import Triangulation
from matplotlib import pyplot as plt

# ant and cross models from the dataset compiled at https://github.com/duxingyi-charles/Locally-Injective-Mappings-Benchmark

def plot_domain(option,v,f):
    boundary_inds = igl.boundary_loop(f)
    b = v[boundary_inds,:]
    b_rep = np.vstack((b, b[0, :]))
    fig, ax = plt.subplots()
    plt.plot(b_rep[:, 0], b_rep[:, 1], 'k', linewidth=2)
    ax.triplot(Triangulation(v[:, 0], v[:, 1], triangles=f), color="k", linewidth=0.5,
                  alpha=0.5)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    out_dir = os.path.join("results", option)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'initial.pdf'))
    plt.savefig(os.path.join(out_dir, 'initial.png'), dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig)

def load_mesh(example):

    if example == "parameterization":
        path = "ant"
        v, f = igl.read_triangle_mesh(path + "/result.obj")
        v = v[:, 0:2]
        boundary_inds = igl.boundary_loop(f)
        b = v[boundary_inds, :]
    elif example == "cross":
        path = "cross"
        v, f = igl.read_triangle_mesh(path + "/result.obj")
        v = v[:,0:2]
        boundary_inds = igl.boundary_loop(f)
        b = v[boundary_inds,:]
        v, f = igl.read_triangle_mesh(path + "/input.obj")

    # ps.init()
    # ps.register_surface_mesh("mesh", v, f)
    # ps.show()

    plot_domain(example,v,f)

    return v,f,b
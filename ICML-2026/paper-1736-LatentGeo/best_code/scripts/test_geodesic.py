"""
This script exercises the installed package on a small ground-truth torus
"""
import numpy as np
import matplotlib.pyplot as plt
from latentgeodesics import GeodesicSolverNumpy, KExponentialNumpy


RADIUS_MAJOR = 4.0 / 5.0
RADIUS_MINOR = 0.2


def torus_point(v, u):
    """Return one point on the torus used in the ground-truth example."""
    x = (RADIUS_MAJOR + RADIUS_MINOR * np.cos(v)) * np.cos(u)
    y = (RADIUS_MAJOR + RADIUS_MINOR * np.cos(v)) * np.sin(u)
    z = RADIUS_MINOR * np.sin(v)
    return np.array([x, y, z], dtype=np.float64)

# just for plotting the torus
def torus_coordinates(resolution=100):
    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, 2*np.pi, resolution)
    u, v = np.meshgrid(u, v)

    x = (RADIUS_MAJOR + RADIUS_MINOR * np.cos(v)) * np.cos(u)
    y = (RADIUS_MAJOR + RADIUS_MINOR * np.cos(v)) * np.sin(u)
    z = RADIUS_MINOR * np.sin(v)

    return x ,y ,z

def phi(coord):
    rho = np.sqrt(coord[:, 0] ** 2 + coord[:, 1] ** 2)
    return ((rho - RADIUS_MAJOR) ** 2 + coord[:, 2] ** 2 - RADIUS_MINOR**2).reshape(1, -1)


def dphi(coord):
    rho = np.sqrt(coord[:, 0] ** 2 + coord[:, 1] ** 2)
    safe_rho = np.where(rho == 0, np.finfo(coord.dtype).eps, rho)
    grad = np.column_stack(
        (
            2 * coord[:, 0] * (rho - RADIUS_MAJOR) / safe_rho,
            2 * coord[:, 1] * (rho - RADIUS_MAJOR) / safe_rho,
            2 * coord[:, 2],
        )
    )
    return grad.reshape(1, -1, 3)


def main():
    x_a = torus_point(v=0.0, u=1.0)
    x_b = torus_point(v=1.0, u=0.0)

    solver = GeodesicSolverNumpy(3, phi, dphi)
    geodesic = solver.AugLagrangeMinimize(
        resolution=12,
        xA=x_a,
        xB=x_b,
        mu=100,
        alpha=100,
        disp=False,
    )
    geodesic_constraint = float(np.max(np.abs(phi(geodesic))))

    start = geodesic[0]
    direction = 13 * (geodesic[1] - geodesic[0])
    exponential = KExponentialNumpy(
        phi,
        dphi,
        start,
        direction,
        13,
        firstStep="linear",
        constrfact=8.0,
    )

    exp_constraint = float(np.max(np.abs(phi(exponential))))

    print("installation test passed")
    print(f"geodesic_shape={geodesic.shape} constraint_max={geodesic_constraint:.3e}")
    print(f"exponential_shape={exponential.shape} constraint_max={exp_constraint:.3e}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(*torus_coordinates(),  edgecolor='none', alpha=0.7)
    coord = torus_coordinates()
    ax.set_box_aspect([np.ptp(a) for a in coord])
    ax.view_init(elev=30, azim=45) 
    ax.plot(geodesic[:,0], geodesic[:,1], geodesic[:,2], color='red',marker="o", label='geodesic',zorder=3)
    ax.scatter(exponential[:,0], exponential[:,1], exponential[:,2], color='black', marker="x", label='exponential',zorder=4)
    ax.legend()
    plt.savefig("geodesic_test.png", dpi=300)
    plt.show()
    print("saved a figure as geodesic_test.png")
if __name__ == "__main__":
    main()
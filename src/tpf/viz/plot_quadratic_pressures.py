import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
import scipy.sparse as sps
from matplotlib import cm
from numba import njit, prange


def plot_quadratic_pressures(
    g: pp.Grid,
    bounding_box: dict[str, pp.number],
    coeffs: np.ndarray,
    title="Quadratic pressures",
    show=False,
    save_path=None,
    **kwargs,
):
    """
    Plot the quadratic pressures on a mesh.

    Parameters:
        g: The grid containing the nodes and elements.
        bounding_box: The bounding box of the grid.
        coeffs: The coeffs of the pressure to plot.
        title  The title of the plot. Default is "Quadratic pressures".
        show: Whether to show the plot. Default is False.
        save_path: The path to save the plot. Default is None.
        **kwargs: Additional keyword arguments for the plot.
            - log_scale: Whether to use a log scale for the z-axis. Default is False.
            - zmin: The minimum value for the z-axis. Default is None.
            - zmax: The maximum value for the z-axis. Default is None.
    Returns:
        None

    """
    x = np.linspace(bounding_box["xmin"], bounding_box["xmax"], 100)
    y = np.linspace(bounding_box["ymin"], bounding_box["ymax"], 100)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()

    cell_nodes_map = sps.find(g.cell_nodes().T)[1]
    nodes_of_cell = cell_nodes_map.reshape(g.num_cells, g.dim + 1)

    viz_mesh_to_grid = find_points_in_triangles(xv, yv, g.nodes[:2].T[nodes_of_cell])
    viz_mesh_coeffs = coeffs[viz_mesh_to_grid]

    pressure = (
        viz_mesh_coeffs[..., 0] * xv**2
        + viz_mesh_coeffs[..., 1] * xv * yv
        + viz_mesh_coeffs[..., 2] * xv
        + viz_mesh_coeffs[..., 3] * yv**2
        + viz_mesh_coeffs[..., 4] * yv
        + viz_mesh_coeffs[..., 5]
    )

    xv = xv.reshape((100, 100))
    yv = yv.reshape((100, 100))
    pressure = pressure.reshape((100, 100))

    # Create a 3D plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "3d"})
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Pressure")
    ax.set_title(title)

    zmin = kwargs.get("zmin", np.min(pressure))
    zmax = kwargs.get("zmax", np.max(pressure))
    use_log = kwargs.get("log_scale", False)

    # Handle log scale safely.
    if use_log:
        positive_mask = pressure > 0
        if not np.any(positive_mask):
            raise ValueError(
                "All pressure values are non-positive; cannot use log scale."
            )
        # Optionally clip or shift data slightly to avoid log(0)
        pressure_safe = pressure - np.min(pressure)
        zmin = max(zmin, np.nanmin(pressure_safe))
        ax.set_zscale("log")
    else:
        pressure_safe = pressure

    # Plot
    ax.plot_surface(
        xv, yv, pressure_safe, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    ax.view_init(elev=50, azim=30)
    ax.set_xlim(bounding_box["xmin"], bounding_box["xmax"])
    ax.set_ylim(bounding_box["ymin"], bounding_box["ymax"])
    ax.set_zlim(zmin, zmax)

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    plt.close(fig)


@njit(parallel=True)
def find_points_in_triangles(x, y, cells):
    """Find which cell each point belongs to using barycentric coordinates."""
    n_points = x.shape[0]
    n_cells = cells.shape[0]
    result = np.full(n_points, -1, dtype=np.int64)

    for i in prange(n_points):
        px, py = x[i], y[i]
        for j in range(n_cells):
            x1, y1 = cells[j, 0]
            x2, y2 = cells[j, 1]
            x3, y3 = cells[j, 2]

            det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            if det == 0.0:
                # Degenerate triangle, skip.
                continue

            w1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / det
            w2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / det
            w3 = 1.0 - w1 - w2

            if (w1 >= 0.0) and (w2 >= 0.0) and (w3 >= 0.0):
                result[i] = j
                # Stop once a containing triangle is found.
                break

    return result

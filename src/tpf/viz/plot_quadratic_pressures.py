import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from matplotlib import cm


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
    viz_mesh = np.vstack((xv, yv, np.zeros_like(xv)))  # shape (3, 10000)

    viz_mesh_to_grid = g.closest_cell(viz_mesh)
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
    # ax.set_box_aspect([1, 1, 1])  # aspect ratio is 1:1:1
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Pressure")
    ax.set_title(title)
    ax.plot_surface(xv, yv, pressure, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.view_init(elev=50, azim=30)
    ax.set_xlim(bounding_box["xmin"], bounding_box["xmax"])
    ax.set_ylim(bounding_box["ymin"], bounding_box["ymax"])
    ax.set_zlim(
        kwargs.get("zmin", np.min(pressure)),
        kwargs.get("zmax", np.max(pressure)),
    )
    ax.set_zscale("log" if kwargs.get("log_scale", False) else "linear")
    ax.set_title(title)

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    plt.close(fig)

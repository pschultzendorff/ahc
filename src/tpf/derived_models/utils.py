import numpy as np
import porepy as pp


def position_cell_id(
    g: pp.Grid, x: float, y: float, percentages: bool = True
) -> np.intp:
    """Identify the id of the cell corresponding to the given position.

    Parameters:
        g: Grid.
        x: x-coordinate.
        y: y-coordinate.
        percentages: Whether the coordinates are given in percentages of the
            width/height.

    Returns:
        id: Index of the cell closest to the given position.

    """
    # Ignore z-values of the grid.
    cell_centers = g.cell_centers[:2, :]

    # Transform to percentages if needed.
    if not percentages:
        height: float = np.max(g.nodes[1, :]) - np.min(g.nodes[1, :])
        width: float = np.max(g.nodes[0, :]) - np.min(g.nodes[0, :])
        x /= width
        y /= height

    min_x, min_y = np.min(cell_centers, axis=1)
    max_x, max_y = np.max(cell_centers, axis=1)
    center = np.argmin(
        np.sum(
            (cell_centers - np.array([[(max_x - min_x) * x], [(max_y - min_y) * y]]))
            ** 2,
            axis=0,
        )
    )
    return center


def center_cell_id(g: pp.Grid) -> np.intp:
    """Identify the center cell of the grid.

    Parameters:
        g: Grid.

    Returns:
        center: Index of the center cell.

    """
    return position_cell_id(g, 0.5, 0.5)


def corner_faces_id(g: pp.Grid, height: float, width: float, well_size) -> np.ndarray:
    """Identify the boundary faces in the corners of the grid corresponding to
    production wells.

    Parameters:
        g: Grid.

    Returns:
        corners: Indices of the boundary faces in the corners.

    """
    # Ignore z-values of the grid.
    boundary_faces: np.ndarray = g.get_boundary_faces()
    boundary_face_centers: np.ndarray = g.face_centers[:2, boundary_faces]
    # Find indices of faces in the corners.
    indices: list[np.ndarray] = []
    indices.append(
        np.argwhere(
            np.logical_and(
                boundary_face_centers[0] == 0,
                boundary_face_centers[1] <= well_size,
            ),
        )
    )
    indices.append(
        np.argwhere(
            np.logical_and(
                boundary_face_centers[0] == 0,
                boundary_face_centers[1] >= height - well_size,
            )
        )
    )
    indices.append(
        np.argwhere(
            np.logical_and(
                boundary_face_centers[0] == width,
                boundary_face_centers[1] <= well_size,
            )
        )
    )
    indices.append(
        np.argwhere(
            np.logical_and(
                boundary_face_centers[0] == width,
                boundary_face_centers[1] >= height - well_size,
            )
        )
    )
    indices.append(
        np.argwhere(
            np.logical_and(
                boundary_face_centers[0] <= well_size,
                boundary_face_centers[1] == 0,
            )
        )
    )
    indices.append(
        np.argwhere(
            np.logical_and(
                boundary_face_centers[0] >= width - well_size,
                boundary_face_centers[1] == 0,
            )
        )
    )
    indices.append(
        np.argwhere(
            np.logical_and(
                boundary_face_centers[0] <= well_size,
                boundary_face_centers[1] == height,
            )
        )
    )
    indices.append(
        np.argwhere(
            np.logical_and(
                boundary_face_centers[0] >= width - well_size,
                boundary_face_centers[1] == height,
            )
        )
    )
    return boundary_faces[np.concatenate(indices).flatten()]


def well_cell_id(
    g: pp.Grid,
    well_boundaries: np.ndarray,
) -> np.ndarray:
    """Identify the cells inside a well boundary.

    Assumes that a grid cell is either fully inside or outside the well. Thus it
    suffices, to check only cell centers to find whether a cell is inside a well.

    Parameters:
        g: Grid.
        well_boundaries ``shape=(2, nd)``: Lower left and upper right corner of the
            square well.

    Returns:
        corners: Indices of the cells inside the well.

    """
    # Ignore z-values of the grid.
    cell_centers: np.ndarray = g.cell_centers[:2, :]
    # Find indices of cells inside the well.
    indices = np.argwhere(
        np.logical_and(
            np.logical_and(
                cell_centers[0] >= well_boundaries[0, 0],
                cell_centers[0] <= well_boundaries[1, 0],
            ),
            np.logical_and(
                cell_centers[1] >= well_boundaries[0, 1],
                cell_centers[1] <= well_boundaries[1, 1],
            ),
        )
    )
    return indices

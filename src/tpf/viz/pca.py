import logging
import typing
from typing import Any

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def screeplot(pca: PCA, n: int | None = None) -> matplotlib.figure.Figure:
    """Create a scree plot for Principal Component Analysis (PCA) results.

    Parameters:
        pca: The PCA object containing the results.
        n: The number of principal components to plot.
            Defaults to None, which plots all components.

    Returns:
        fig: The resulting scree plot figure.

    """
    fig, ax = plt.subplots()
    if n is None:
        n = pca.n_components_
    elif n > pca.n_components_:
        logger.info(
            f"{n=} is larger than the number of components."
            + f" Using {pca.n_components_}"
        )
        n = pca.n_components_
    pcs: np.ndarray = np.arange(1, n + 1, 1)
    ax.plot(pcs, pca.explained_variance_ratio_[:n], "o-")
    ax.set_xticks(pcs)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained [%]")
    ax.set_title("Scree Plot")
    return fig


def biplot(
    score: np.ndarray,
    coeff: np.ndarray,
    labels: list[str] | None = None,
    colors: list[Any] | None = None,
    n: int | None = None,
    plot_graph: bool = False,
) -> matplotlib.figure.Figure:
    """Create a biplot for Principal Component Analysis (PCA) results.

    Parameters:
        score: The PCA scores, typically the transformed data.
        coeff ``shape=(n_features, n_components)``: The PCA coefficients, typically the
            principal components.
        labels: Labels for the variables. Defaults to None.
        colors: Colors for the points. Defaults to None.
        n: Number of variables to plot. Defaults to None, which plots all variables.
        plot_graph: If True, plots a line graph; otherwise, plots a scatter plot.
            Defaults to False.

    Returns:
        fig: The resulting biplot figure.

    """
    if n is None:
        n = coeff.shape[0]
        n = typing.cast(int, n)
    xs: np.ndarray = score[:, 0]
    ys: np.ndarray = score[:, 1]
    scalex: np.ndarray = 1.0 / (xs.max() - xs.min())
    scaley: np.ndarray = 1.0 / (ys.max() - ys.min())
    fig, ax = plt.subplots()
    if plot_graph:
        ax.plot(xs * scalex, ys * scaley, "o-")
    else:
        ax.scatter(xs * scalex, ys * scaley, color=colors)
    for i in range(n):  # type: ignore
        ax.arrow(0, 0, coeff[i, 0], coeff[i, 1], color="r", alpha=0.5)
        label: str = f"Var{str(i+1)}" if labels is None else labels[i]
        ax.text(
            coeff[i, 0] * 1.15,
            coeff[i, 1] * 1.15,
            label,
            color="g",
            ha="center",
            va="center",
        )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(f"PC{1}")
    ax.set_ylabel(f"PC{2}")
    ax.grid()
    return fig

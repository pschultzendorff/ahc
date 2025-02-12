import logging
import pathlib

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tpf_PorePy.src.tpf.viz.pca import biplot, screeplot

dirname: pathlib.Path = pathlib.Path(__file__).parent
(dirname / "pca_bl").mkdir(exist_ok=True)
logger = logging.getLogger(__name__)

n_cells: int = 100
n_iterations: int = 20

# Generate series of approximate solutions to Buckley-Leverett.
solutions: np.ndarray | list[np.ndarray] = [
    np.concat([np.ones((i,)), np.zeros((n_cells - i,))]) for i in range(n_iterations)
]
solutions = np.vstack(solutions)

# Standardize the data (important for PCA)
scaler = StandardScaler()
solutions_scaled = scaler.fit_transform(solutions)

# Perform PCA.
pca = PCA()
reduced_solutions = pca.fit_transform(solutions_scaled)

fig = screeplot(pca, n=10)
fig.savefig(dirname / "pca_bl" / "screeplot.png")

fig = biplot(
    reduced_solutions,
    pca.components_.T,
    labels=[""] * n_cells,
    n=n_cells,
    plot_graph=True,
)
fig.savefig(dirname / "pca_bl" / "biplot.png")

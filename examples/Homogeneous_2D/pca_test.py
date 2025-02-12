import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assume you have multiple solvers' solutions stored in a dictionary like this:
# Each solver's data is a 2D array (steps x dimensions of the solution vector)
# For example, solver_1 could be the series of solutions from solver 1
solver_1 = np.random.randn(100, 50)  # 100 steps, 50-dimensional solutions
solver_2 = np.random.randn(100, 50)  # 100 steps, 50-dimensional solutions
solver_3 = np.random.randn(100, 50)  # 100 steps, 50-dimensional solutions

# Stack all solvers' data together (each solver as a separate series of solutions)
solutions = [solver_1, solver_2, solver_3]

# Combine all solutions into one large matrix
all_solutions = np.vstack(solutions)

# Standardize the data (important for PCA)
scaler = StandardScaler()
all_solutions_scaled = scaler.fit_transform(all_solutions)

# Perform PCA (reduce to 2 dimensions for plotting)
pca = PCA(n_components=2)
reduced_solutions = pca.fit_transform(all_solutions_scaled)

# Plot the solution paths for each solver
plt.figure(figsize=(10, 8))

colors = ["r", "g", "b"]  # Colors for each solver's path
labels = ["Solver 1", "Solver 2", "Solver 3"]

start_idx = 0  # Start index for each solver

for i, solver in enumerate(solutions):
    # Get the reduced data for the current solver
    end_idx = start_idx + solver.shape[0]
    reduced_solver = reduced_solutions[start_idx:end_idx]

    # Plot the path for the current solver
    plt.plot(
        reduced_solver[:, 0],
        reduced_solver[:, 1],
        label=labels[i],
        color=colors[i],
        marker="o",
    )

    start_idx = end_idx  # Update the start index for the next solver

plt.title("Solution Path in PCA Reduced Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()

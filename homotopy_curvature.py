import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root


def f(x):
    return x**2 - 10 * x - 20


def f_prime(x):
    return 2 * x - 10


def f_double_prime(x):
    return 2


def g(x):
    return x**2 - 20


def g_prime(x):
    return 2 * x


def g_double_prime(x):
    return 2


def h(x, beta):
    return beta * g(x) + (1 - beta) * f(x)


def h_prime(x, beta):
    return beta * g_prime(x) + (1 - beta) * f_prime(x)


def h_double_prime(x, beta):
    return beta * g_double_prime(x) + (1 - beta) * f_double_prime(x)


# def homotopy_curvature(x, beta):
#     curvature_vector_x = (
#         h_double_prime(x, beta) / (h_prime(x, beta) ** 2) * (g(x) - f(x))
#     )


def homotopy_curvature(x, beta):
    h_double_prime(x, beta) / (1 + h_prime(x, beta) ** 2) ** (3 / 2)


def homotopy_tangent(x, beta):
    return (g(x) - f(x)) / h_prime(x, beta)


def homotopy_curvature_approx(xx, betas):
    h = betas[1] - betas[0]
    curvature = np.zeros_like(xx)
    # Central difference for interior points using slicing
    curvature[1:-1] = (xx[2:] - 2 * xx[1:-1] + xx[:-2]) / h**2
    return curvature


def solve_h_for_beta(beta, x0=0):
    if np.isscalar(beta):

        def h_for_fixed_beta(x):
            return h(x, beta)

        sol = root(h_for_fixed_beta, x0)
        return sol.x[0]
    else:
        # For array of betas, solve for each beta
        result = np.zeros_like(beta)
        for i, b in enumerate(beta):
            result[i] = solve_h_for_beta(b, x0)
        return result


def homotopy_curvature_plot():
    betas = np.linspace(0, 1, 100)
    xx = solve_h_for_beta(betas, x0=-3)
    plt.figure(figsize=(10, 6))
    plt.plot(betas, xx, label=r"$x$")
    tangent = homotopy_tangent(xx, betas)
    plt.plot(betas, tangent, label=r"$\dot{x}$")

    plt.quiver(
        betas[::10],
        xx[::10],
        np.full_like(betas[::10], -0.1),
        tangent[::10] / 10,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="green",
        alpha=0.7,
        width=0.005,
    )
    # plt.plot(betas, homotopy_curvature(xx, betas), label=r"$\ddot{x}$")
    plt.plot(betas, homotopy_curvature_approx(xx, betas), label=r"$\ddot{x}_{approx}$")
    plt.plot(betas, g(xx) - f(xx), label=r"$g(x) - f(x)$")
    plt.plot(betas, h_prime(xx, betas) ** 2, label=r"$\dot{h}^2$")
    plt.plot(betas, h_double_prime(xx, betas), label=r"$\ddot{h}$")

    plt.title("Homotopy Curvature")
    plt.xlabel(r"$\beta$")
    plt.xlim(1, 0)
    plt.ylim(-10, 10)
    plt.legend()
    plt.show()


homotopy_curvature_plot()

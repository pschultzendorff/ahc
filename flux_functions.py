import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

total_flow: float = 1.0  # []
permeability: float = 1.0  # [mD]

mu_w: float = 1.0  # Viscosity of wetting phase [mP]
mu_n: float = 1.0  # Viscosity of non-wetting phase [mP]

L: float = 1.0  # Characteristic length scale
P_c_bar: float = 1.0  # Characteristic capillary pressure

PECLET: float = total_flow * mu_n * L / (permeability * P_c_bar)


def k_rw(S: ArrayLike, model: str) -> np.ndarray:
    """Wetting relative permeability.

    Parameters:
        S: Saturation

    Returns:
        The value of the relative permeability.

    """
    S = np.array(S)

    # Compute relative permeability based on chosen model.
    if model in ["Corey", "power"]:
        rel_perm: np.ndarray = S**3
    elif model == "linear":
        rel_perm = S
    # Brooks-Corey-Mualem
    elif model.startswith("Brooks-Corey"):
        rel_perm = S ** (2.0 + 2.0 * 1.0)

    return rel_perm


def k_rn(S: ArrayLike, model: str) -> np.ndarray:
    """Nonwetting relative permeability.

    Parameters:
        S: Saturation

    Returns:
        The value of the relative permeability.

    """
    S = np.array(S)

    # Compute relative permeability based on chosen model.
    if model in ["Corey", "power"]:
        rel_perm: np.ndarray = (1 - S) ** 3
    elif model == "linear":
        rel_perm = 1 - S
    # Brooks-Corey-Mualem
    elif model.startswith("Brooks-Corey"):
        rel_perm = ((1 - S) ** 2.0) * ((1 - S**2.0) ** 1.0)

    return rel_perm


def p_c(S: ArrayLike, model: str, entry_pressure: float) -> np.ndarray:
    """Capillary pressure.

    Parameters:
        S: Saturation

    Returns:
        The value of the capillary pressure.

    """
    S = np.array(S)

    # Compute capillary pressure based on chosen model.
    if model == "linear":
        p_c = entry_pressure * (1 - S)
    # Brooks-Corey-Mualem
    elif model.startswith("Brooks-Corey"):
        p_c = entry_pressure * np.power(
            S,
            -1 / 2.0,
            out=np.zeros_like(S),
            where=S != 0,
        )

    return p_c


def lambda_w(S: ArrayLike, model: str) -> np.ndarray:
    """Linear wetting mobility.

    Parameters:
        S: Saturation

    Returns:
        The value of the wetting mobility.

    """
    return k_rw(S, model) / mu_w


def lambda_n(S: ArrayLike, model: str) -> np.ndarray:
    """Linear non-wetting mobility.

    Parameters:
        S: Saturation

    Returns:
        The value of the non-wetting mobility.

    """
    return k_rn(S, model) / mu_n


def viscous_flow(
    S: ArrayLike,
    rp_model: str,
) -> np.ndarray:
    """Viscous flow function.

    The equation implemented is
    .. math::
        λ_w/λ_T

    Parameters:
        S: Saturation

    Returns:
        The value of the viscous flow function.

    """
    lambda_total = lambda_w(S, rp_model) + lambda_n(S, rp_model)
    return lambda_w(S, rp_model) / lambda_total


# TODO Use permeability, nonwetting viscosity and total flow instead of Peclet number.
def capillary_flow(
    S: ArrayLike,
    rp_model: str,
    grad_pc: ArrayLike,
    peclet: ArrayLike,
) -> np.ndarray:
    r"""Capillary flux function.

    The equation implemented is
    .. math::
        (λ_w*k_rn/λ_T)*(\nabla P_c/P_e(P_c/L))

    Parameters:
        S: Saturation
        grad_pc: Gradient of capillary pressure, :math:`∇P_c`.
        peclet: Peclet number specifying the ratio :math:`P_e(P_c/L)`.

    Returns:
        The value of the capillary flux function.

    """
    grad_pc = np.array(grad_pc)
    peclet = np.array(peclet)

    lambda_total = lambda_w(S, rp_model) + lambda_n(S, rp_model)

    return (lambda_w(S, rp_model) * k_rn(S, rp_model) / lambda_total) * (
        grad_pc / (peclet * P_c_bar / L)
    )


def f_w(
    S: ArrayLike,
    rp_model: str,
    grad_pc: ArrayLike = 0,
    peclet: ArrayLike = 1,
) -> np.ndarray:
    r"""Wetting phase fractional flow function in the absence of gravity.

    The equation implemented is
    .. math::
        f_w = λ_w/λ_T + (λ_w*k_rn/λ_T)*(\nabla P_c/P_e(P_c/L))

    Parameters:
        S: Saturation
        grad_pc: Gradient of capillary pressure, :math:`\nabla P_c`. Default is 0.0.
        peclet: Peclet number specifying the ratio ratio :math:`P_e(P_c/L)`. Default is
            1.0.

    Returns:
        The value of the fractional flow function.

    """
    return viscous_flow(S, rp_model) + capillary_flow(S, rp_model, grad_pc, peclet)


def cap_press_gradient(
    S_U: ArrayLike, S_D: ArrayLike, cp_model: str, **kwargs
) -> np.ndarray:
    """Calculate the capillary pressure gradient.

    Parameters:
        S_U: Upstream saturation
        S_D: Downstream saturation
        cp_model: Capillary pressure model

    Returns:
        The value of the capillary pressure gradient.

    """
    # Calculate the capillary pressure for upstream and downstream saturations
    p_c_U = p_c(S_U, cp_model, entry_pressure=10.0)
    p_c_D = p_c(S_D, cp_model, entry_pressure=10.0)

    # Calculate the capillary pressure gradient
    return p_c_U - p_c_D


def find_zero_flux_points(rp_model: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Find the zero flux points of the fractional flow function.

    Parameters:
        S: Saturation
        rp_model: Relative permeability model
        **kwargs: Additional arguments for the f_w function. If ``grad_pc`` or
            ``peclet`` are provided, they have to be floats.

    Returns:
        Tuple of zero flux points and their corresponding values.

    """
    S: np.ndarray = np.linspace(0, 1, 100)
    f_w_values = f_w(S, rp_model, **kwargs)

    # Find zero flux points where the flux function changes sign.
    zero_flux_points = np.where(np.diff(np.sign(f_w_values)))[0]

    return S[zero_flux_points], f_w_values[zero_flux_points]


def calc_unit_flux_points(rp_model: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the unit flux points of the fractional flow function.

    Parameters:
        S: Saturation
        rp_model: Relative permeability model
        **kwargs: Additional arguments for the f_w function. If ``grad_pc`` or
            ``peclet`` are provided, they have to be floats.

    Returns:
        Tuple of unit flux points and their corresponding values.

    """
    S: np.ndarray = np.linspace(0, 1, 100)
    f_w_values = f_w(S, rp_model, **kwargs)

    # Find zero flux points where 1 - the flux function changes sign.
    unit_flux_points = np.where(np.diff(np.sign(1.0 - f_w_values)))[0]

    return S[unit_flux_points], f_w_values[unit_flux_points]


def F(
    S_U: ArrayLike,
    S_D: ArrayLike,
    rp_model: str,
    cp_model: str,
    upwinding: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Calculate the flux function F based on upstream and downstream saturations.

    Parameters:
        S_U: Upstream saturation.
        S_D: Downstream saturation.
        rp_model: Relative permeability model.
        cp_model: Capillary pressure model.
        upwinding: Boolean flag for upwinding. If False, uses averaged mobility. Default
            is True.

    Returns:
        The value of the flux function F(S_U, S_D)

    """
    S_U = np.array(S_U)
    S_D = np.array(S_D)

    # Calculate capillary pressure gradient.
    grad_pcs: np.ndarray = cap_press_gradient(S_U, S_D, cp_model)

    # Calculate flux function based on upstream saturations.
    f_w_SU: np.ndarray = f_w(S_U, rp_model, grad_pc=grad_pcs, **kwargs)

    # Precompute rel. perms. and mobility values.
    lambda_wSU = lambda_w(S_U, rp_model)
    lambda_nSU = lambda_n(S_U, rp_model)
    lambda_wSD = lambda_w(S_D, rp_model)
    lambda_nSD = lambda_n(S_D, rp_model)
    k_rn_SU = k_rn(S_U, rp_model)
    k_rn_SD = k_rn(S_D, rp_model)

    if upwinding:
        # Compute numerical fluxes for differing co-/counter-current cases.

        # Co-current, wetting forward, nonwetting forward.
        co_current: np.ndarray = (
            lambda_wSU
            * (1 + k_rn_SU * grad_pcs / (kwargs.get("peclet", 1.0) * P_c_bar / L))
            / (lambda_wSU + lambda_nSU)
        )
        # Counter-current, wetting backward, nonwetting forward.
        counter_current_1: np.ndarray = (
            lambda_wSD
            * (1 + k_rn_SU * grad_pcs / (kwargs.get("peclet", 1.0) * P_c_bar / L))
            / (
                lambda_wSD + lambda_nSU + 1e-10
            )  # Due to upwinding, we can have a division by zero.
        )
        # Counter-current, wetting forward, nonwetting backward.
        counter_current_2: np.ndarray = (
            lambda_wSU
            * (1 + k_rn_SD * grad_pcs / (kwargs.get("peclet", 1.0) * P_c_bar / L))
            / (
                lambda_wSU + lambda_nSD + 1e-10
            )  # Due to upwinding, we can have a division by zero.
        )

        # Upwinding depends on cap. press. gradient sign. For negative gradient, we have
        # a zero flux point. For positive gradient, we have a unit flux point.
        num_flux: np.ndarray = np.where(
            grad_pcs > 0,
            np.where(f_w_SU > 1, counter_current_1, co_current),
            np.where(f_w_SU < 0, counter_current_2, co_current),
        )

    else:
        # Average mobilities.
        lambda_wAVG: np.ndarray = (lambda_wSU + lambda_wSD) / 2
        lambda_nAVG: np.ndarray = (lambda_nSU + lambda_nSD) / 2
        k_rnAVG: np.ndarray = (k_rn_SU + k_rn_SD) / 2

        # Compute numerical flux for averaged mobility.
        num_flux = (
            lambda_wAVG
            * (1 + k_rnAVG * grad_pcs / (kwargs.get("peclet", 1.0) * P_c_bar / L))
            / (lambda_wAVG + lambda_nAVG)
        )

    return num_flux


def plot_f_w() -> None:
    """
    Plot the wetting phase fractional flow function for different Peclet numbers.
    Creates three plots showing how f_w varies with saturation for different
    capillary pressure effects.
    """
    # Create saturation values from 0 to 1
    S = np.linspace(0, 1, 100)

    # Different Peclet numbers
    pe_values = [0.1, 1.0, 10.0]  # Strong to weak capillary effects
    rp_models = ["linear", "Corey", "Brooks-Corey"]
    grad_pc = 1.0  # Fixed gradient of capillary pressure

    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 5))

    for i, pe in enumerate(pe_values):
        for j, rp_model in enumerate(rp_models):
            # Calculate f_w for the current Peclet number
            f_values = f_w(S, rp_model, grad_pc=grad_pc, peclet=pe)

            # Plot the curve
            axes[i, j].plot(
                S, f_values, "b-", linewidth=2, label="With capillary effects"
            )

            # Plot reference curve with no capillary effects
            viscous_only = f_w(S, rp_model, grad_pc=0)
            axes[i, j].plot(S, viscous_only, "r--", linewidth=1, label="Viscous only")

            # Configure plot
            axes[i, j].set_xlabel("Saturation (S)")
            axes[i, j].set_ylabel("Fractional Flow (f_w)")
            axes[i, j].set_title(f"Pe = {pe}, Model = {rp_model}")
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].legend()
            axes[i, j].set_xlim(0, 1)
            axes[i, j].set_ylim(0, max(1.1, max(f_values) * 1.1))

    plt.tight_layout()
    plt.show()


def plot_F() -> None:
    """Plot the flux function F as S_U and S_D vary between 0 and 1.

    Creates 3D surface plots for different rel. perm. and cap. press. models.

    """
    # Create a grid of S_U and S_D values
    S_values = np.linspace(0, 1, 50)
    S_U_grid, S_D_grid = np.meshgrid(S_values, S_values)

    pe_values = [1.0, 10.0]  # Strong to weak capillary effects
    rp_models = ["linear", "Corey", "Brooks-Corey"]
    cp_models = ["linear", "Brooks-Corey"]
    upwinding_flags = [True, False]

    for i, pe in enumerate(pe_values):
        for upwinding in upwinding_flags:
            # Create figure with subplots
            fig, axes = plt.subplots(
                len(rp_models),
                len(cp_models),
                figsize=(15, 10),  # Increased figure size
                subplot_kw={"projection": "3d"},
            )

            # Loop through relative permeability and capillary pressure models and plot
            # numerical flux function.
            for j, rp_model in enumerate(rp_models):
                for k, cp_model in enumerate(cp_models):
                    F_values = F(
                        S_U_grid,
                        S_D_grid,
                        rp_model=rp_model,
                        cp_model=cp_model,
                        upwinding=upwinding,
                        peclet=pe,
                    )

                    # Plot the surface
                    axes[j, k].plot_surface(
                        S_U_grid, S_D_grid, F_values, cmap="viridis"
                    )

                    # Set labels and title
                    axes[j, k].set_xlabel("S_U")
                    axes[j, k].set_ylabel("S_D")
                    axes[j, k].set_zlabel("F(S_U, S_D)")

                    # Split title into two lines for better readability
                    title_line1 = f"Pe = {pe}, RP Model = {rp_model}"
                    title_line2 = f"CP Model = {cp_model}, Upwinding = {upwinding}"
                    axes[j, k].set_title(f"{title_line1}\n{title_line2}")

                    # Turn to have the origin in the front.
                    axes[j, k].view_init(azim=-135)
                    # TODO Fix this!
                    axes[j, k].set_xlim(0.0, 1.0)
                    axes[j, k].set_zlim(0.0, 1.0)
                    axes[j, k].set_zlim(-0.2, 3.5)

            # Add more padding between subplots
            plt.tight_layout(pad=3.0)
            plt.show()


if __name__ == "__main__":
    plot_f_w()
    plot_F()

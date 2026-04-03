import pathlib

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from ahc.derived_models.fluid_values import oil, water
from ahc.derived_models.spe10 import HEIGHT
from ahc.derived_models.spe11 import case_B
from numpy.typing import ArrayLike

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

total_flow: float = 0.3  # []
permeability: float = case_B["PERMEABILITY"]["facies 3"]  # [m^2]

mu_w: float = water["viscosity"]  # Viscosity of wetting phase [mP??]
# mu_n: float = co2["viscosity"]  # Viscosity of non-wetting phase [mP??]
mu_n: float = oil["viscosity"]  # Viscosity of non-wetting phase [mP??]
rho_w: float = water["density"]  # Density of wetting phase [kg m^-3]
# rho_n: float = co2["density"]  # Density of non-wetting phase [kg m^-3]
rho_n: float = oil["density"]  # Density of non-wetting phase [kg m^-3]

G: float = 9.81 * HEIGHT / 100  # Gravity [m^2s^-2]
# G: float = 9.81 * case_A["HEIGHT"] / 100  # Gravity [m^2s^-2]

L: float = HEIGHT / 100  # Characteristic length scale [m]
# L: float = case_A["HEIGHT"] / 100  # Characteristic length scale [m]
P_c_bar: float = 100 * pp.PASCAL  # Characteristic capillary pressure [Pa]
# P_c_bar: float = case_A["ENTRY_PRESSURE"]["facies 3"]  # Entry pressure [Pa]
# P_c_bar: float = LeverettJfunction(  # type: ignore
#     case_B["PERMEABILITY"]["facies 3"], case_B["POROSITY"]["facies 3"]
# )  # Characteristic capillary pressure [Pa]


P_e: float = total_flow * mu_n * L / (permeability * P_c_bar)
N_g: float = permeability * (rho_w - rho_n) * G / (mu_n * total_flow)


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
        # Choose linear factor 4 * entry_pressure - entry_pressure as s^-0.5 (i.e.
        # Brooks-Corey with entry pressure 1) starts growing rapidly at p_c \approx 4
        p_c = entry_pressure + (4.0 * entry_pressure - entry_pressure) * (1 - S)
    # Brooks-Corey-Mualem
    elif model.startswith("Brooks-Corey"):
        p_c = np.maximum(
            entry_pressure
            * np.power(
                S,
                -1 / 2.0,
                out=np.zeros_like(S),
                where=S != 0,
            ),
            4.0 * entry_pressure,
        )
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


def buoyancy_flow(S: ArrayLike, rp_model: str, n_g: ArrayLike) -> np.ndarray:
    r"""Buoyancy flux function.

    The equation implemented is
    .. math::
        - (λ_w*k_rn/λ_T)*N_g

    Parameters:
        S: Saturation.
        rp_model: Relative permeability model.
        n_g: Gravity number, :math:`N_g`.

    Returns:
        The value of the buoyancy flux function.

    """
    n_g = np.array(n_g)

    lambda_total = lambda_w(S, rp_model) + lambda_n(S, rp_model)

    return -(lambda_w(S, rp_model) * k_rn(S, rp_model) / lambda_total) * n_g


def capillary_flow(
    S: ArrayLike,
    rp_model: str,
    grad_pc: ArrayLike,
    p_e: ArrayLike,
) -> np.ndarray:
    r"""Capillary flux function.

    The equation implemented is
    .. math::
        (λ_w*k_rn/λ_T)*(\nabla P_c/P_e(P_c/L))

    Parameters:
        S: Saturation
        grad_pc: Gradient of capillary pressure, :math:`∇P_c`.
        p_e: Peclet number specifying the ratio :math:`P_e(P_c/L)`.

    Returns:
        The value of the capillary flux function.

    """
    grad_pc = np.array(grad_pc)
    p_e = np.array(p_e)

    lambda_total = lambda_w(S, rp_model) + lambda_n(S, rp_model)

    return (lambda_w(S, rp_model) * k_rn(S, rp_model) / lambda_total) * (
        grad_pc / (p_e * P_c_bar / L)
    )


def f_w(
    S: ArrayLike,
    rp_model: str,
    n_g: ArrayLike = N_g,
    grad_pc: ArrayLike = 0.0,
    p_e: ArrayLike = P_e,
) -> np.ndarray:
    r"""Wetting phase fractional flow function in the absence of gravity.

    The equation implemented is
    .. math::
        f_w = λ_w/λ_T - (λ_w*k_rn/λ_T)*N_g + (λ_w*k_rn/λ_T)*(\nabla P_c/P_e(P_c/L))

    Parameters:
        S: Saturation
        rp_model: Relative permeability model.
        n_g: Gravity number, :math:`N_g`. Default is `N_g`.
        grad_pc: Gradient of capillary pressure, :math:`\nabla P_c`. Default is 0.0.
        p_e: Peclet number specifying the ratio ratio :math:`P_e(P_c/L)`. Default is
            `P_e`.

    Returns:
        The value of the fractional flow function.

    """
    return (
        viscous_flow(S, rp_model)
        + buoyancy_flow(S, rp_model, n_g)
        + capillary_flow(S, rp_model, grad_pc, p_e)
    )


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
    # Calculate the capillary pressure for upstream and downstream saturations.
    p_c_U = p_c(S_U, cp_model, entry_pressure=P_c_bar)
    p_c_D = p_c(S_D, cp_model, entry_pressure=P_c_bar)

    # Calculate the discretized capillary pressure gradient.
    return (p_c_D - p_c_U) / L


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


def F_with_capillary(
    S_U: ArrayLike,
    S_D: ArrayLike,
    rp_model: str,
    cp_model: str,
    upwinding: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Calculate the flux function F with viscous and capillary flow based on upstream and
    downstream saturations.

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

    # Calculate flux function based on upstream saturations. Set gravity to 0.
    f_w_SU: np.ndarray = f_w(S_U, rp_model, n_g=0.0, grad_pc=grad_pcs, **kwargs)

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
            * (1 + k_rn_SU * grad_pcs / (kwargs.get("p_e", 1.0) * P_c_bar / L))
            / (lambda_wSU + lambda_nSU)
        )
        # Counter-current, wetting forward, nonwetting backward.
        counter_current_1: np.ndarray = (
            lambda_wSU
            * (1 + k_rn_SD * grad_pcs / (kwargs.get("p_e", 1.0) * P_c_bar / L))
            / (
                lambda_wSD + lambda_nSD + 1e-20
            )  # Due to upwinding, we can have a division by zero.
        )
        # Counter-current, wetting backward, nonwetting forward.
        counter_current_2: np.ndarray = (
            lambda_wSD
            * (1 + k_rn_SU * grad_pcs / (kwargs.get("p_e", 1.0) * P_c_bar / L))
            / (
                lambda_wSD + lambda_nSU + 1e-20
            )  # Due to upwinding, we can have a division by zero.
        )

        # Upwinding depends on cap. press. gradient sign. For negative gradient, we have
        # a zero flux point. For positive gradient, we have a unit flux point.
        num_flux: np.ndarray = np.where(
            grad_pcs >= 0,
            np.where(f_w_SU >= 1, counter_current_1, co_current),
            np.where(f_w_SU <= 0, counter_current_2, co_current),
        )

    else:
        # Average mobilities.
        lambda_wAVG: np.ndarray = (lambda_wSU + lambda_wSD) / 2
        lambda_nAVG: np.ndarray = (lambda_nSU + lambda_nSD) / 2
        k_rnAVG: np.ndarray = (k_rn_SU + k_rn_SD) / 2

        # Compute numerical flux for averaged mobility.
        num_flux = (
            lambda_wAVG
            * (1 + k_rnAVG * grad_pcs / (kwargs.get("p_e", 1.0) * P_c_bar / L))
            / (lambda_wAVG + lambda_nAVG)
        )

    return num_flux


def F_with_gravity(
    S_U: ArrayLike,
    S_D: ArrayLike,
    rp_model: str,
    upwinding: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Calculate the flux function F with viscous and buoyancy flow based on upstream and
    downstream saturations.

    Parameters:
        S_U: Upstream saturation.
        S_D: Downstream saturation.
        rp_model: Relative permeability model.
        upwinding: Boolean flag for upwinding. If False, uses averaged mobility. Default
            is True.

    Returns:
        The value of the flux function F(S_U, S_D)

    """
    S_U = np.array(S_U)
    S_D = np.array(S_D)

    # Calculate capillary pressure gradient.

    # Calculate flux function based on upstream saturations. Set capillary gradient to
    # 0.
    f_w_SU: np.ndarray = f_w(S_U, rp_model, grad_pc=0.0, **kwargs)

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
            * (1 - k_rn_SU * kwargs.get("n_g", 1.0))
            / (lambda_wSU + lambda_nSU)
        )
        # Counter-current, wetting backward, nonwetting forward.
        counter_current_1: np.ndarray = (
            lambda_wSU
            * (1 - k_rn_SD * kwargs.get("n_g", 1.0))
            / (
                lambda_wSU + lambda_nSD + 1e-20
            )  # Due to upwinding, we can have a division by zero.
        )
        # Counter-current, wetting forward, nonwetting backward.
        counter_current_2: np.ndarray = (
            lambda_wSD
            * (1 - k_rn_SU * kwargs.get("n_g", 1.0))
            / (
                lambda_wSD + lambda_nSU + 1e-20
            )  # Due to upwinding, we can have a division by zero.
        )

        # Upwinding depends on the sign of the gravity number. For a negative gravity
        # number, we have a zero flux point. For a positive gravity number, we have a
        # unit flux point.
        num_flux: np.ndarray = np.where(
            kwargs.get("n_g", 1.0) <= 0,
            np.where(f_w_SU >= 1, counter_current_1, co_current),
            np.where(f_w_SU <= 0, counter_current_2, co_current),
        )

    else:
        # Average mobilities.
        lambda_wAVG: np.ndarray = (lambda_wSU + lambda_wSD) / 2
        lambda_nAVG: np.ndarray = (lambda_nSU + lambda_nSD) / 2
        k_rnAVG: np.ndarray = (k_rn_SU + k_rn_SD) / 2

        # Compute numerical flux for averaged mobility.
        num_flux = (
            lambda_wAVG
            * (1 - k_rnAVG * kwargs.get("n_g", 1.0))
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

    rp_models = ["linear", "Corey", "Brooks-Corey"]
    ng_values = [-10.0, 1.0, 10.0]  # Strong to weak gravity effects
    pe_values = [0.1, 1.0, 10.0]  # Strong to weak capillary effects
    grad_pc = 1.0  # Fixed gradient of capillary pressure

    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 5))

    for i, n_g in enumerate(ng_values):
        for j, rp_model in enumerate(rp_models):
            # Calculate f_w for the current gravity number and zero capillary flux.
            f_values = f_w(S, rp_model, n_g=n_g, grad_pc=0.0)

            # Plot the curve
            axes[i, j].plot(
                S, f_values, "b-", linewidth=2, label="With buoyancy effects"
            )

            # Plot reference curve with no buoyancy or capillary effects.
            viscous_only = f_w(S, rp_model, n_g=0.0, grad_pc=0.0)
            axes[i, j].plot(S, viscous_only, "r--", linewidth=1, label="Viscous only")

            # Configure plot
            axes[i, j].set_xlabel("Saturation (S)")
            axes[i, j].set_ylabel("Fractional Flow (f_w)")
            axes[i, j].set_title(rf"$N_g$ = {n_g}, Model = {rp_model}")
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].legend()
            axes[i, j].set_xlim(0, 1)
            axes[i, j].set_ylim(0, max(1.1, max(f_values) * 1.1))

    plt.tight_layout()
    fig.savefig(dirname / "f_w_buoyancy_plots.png", dpi=300, bbox_inches="tight")

    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 5))
    for i, p_e in enumerate(pe_values):
        for j, rp_model in enumerate(rp_models):
            # Calculate f_w for the current Peclet number and zero buoyancy flux.
            f_values = f_w(S, rp_model, n_g=0.0, grad_pc=grad_pc, p_e=p_e)

            # Plot the curve
            axes[i, j].plot(
                S, f_values, "b-", linewidth=2, label="With capillary effects"
            )

            # Plot reference curve with no buoyancy or capillary effects.
            viscous_only = f_w(S, rp_model, n_g=0.0, grad_pc=0.0)
            axes[i, j].plot(S, viscous_only, "r--", linewidth=1, label="Viscous only")

            # Configure plot
            axes[i, j].set_xlabel("Saturation (S)")
            axes[i, j].set_ylabel("Fractional Flow (f_w)")
            axes[i, j].set_title(rf"$P_e$ = {p_e}, Model = {rp_model}")
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].legend()
            axes[i, j].set_xlim(0, 1)
            axes[i, j].set_ylim(0, max(1.1, max(f_values) * 1.1))

    plt.tight_layout()
    fig.savefig(dirname / "f_w_capillary_plots.png", dpi=300, bbox_inches="tight")


def plot_F_capillary() -> None:
    """Plot the flux function F as S_U and S_D vary between 0 and 1.

    Creates 3D surface plots for different rel. perm. and cap. press. models.

    """
    # Create a grid of S_U and S_D values
    S_values = np.linspace(0, 1, 50)
    S_D_grid, S_U_grid = np.meshgrid(S_values, S_values)

    pe_values = [1.0, 3.0, 10.0]  # Strong to weak capillary effects
    rp_models = ["linear", "Corey", "Brooks-Corey"]
    cp_models = ["linear", "Brooks-Corey"]
    upwinding_flags = [True, False]

    for i, p_e in enumerate(pe_values):
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
                    F_values = F_with_capillary(
                        S_U_grid,
                        S_D_grid,
                        rp_model=rp_model,
                        cp_model=cp_model,
                        upwinding=upwinding,
                        p_e=p_e,
                    )

                    # Plot the surface
                    axes[j, k].plot_surface(
                        S_D_grid, S_U_grid, F_values, cmap="viridis"
                    )

                    # Set labels and title
                    axes[j, k].set_xlabel(r"$s_D$")
                    axes[j, k].set_ylabel(r"$s_U$")
                    axes[j, k].set_zlabel(r"$F(s_U, s_D)$")

                    # Split title into two lines for better readability
                    title_line1 = rf"$P_e$ = {p_e}, RP Model = {rp_model}"
                    title_line2 = f"CP Model = {cp_model}, Upwinding = {upwinding}"
                    axes[j, k].set_title(f"{title_line1}\n{title_line2}")

                    # Turn to have the origin in the front.
                    axes[j, k].view_init(azim=-135)
                    axes[j, k].set_xlim(0.0, 1.0)
                    axes[j, k].set_ylim(0.0, 1.0)
                    axes[j, k].set_zlim(-3.5, 3.5)

            # Add more padding between subplots
            plt.tight_layout(pad=3.0)
            fig.savefig(
                dirname / f"F_cap_plot_Pe_{p_e}_upwinding_{upwinding}.png",
                dpi=300,
                bbox_inches="tight",
            )


def plot_dSU_F_capillary() -> None:
    """Plot the derivative w.r.t. :math:``S_U`` of the flux function F as S_U and S_D
    vary between 0 and 1.

    Creates 3D surface plots for different rel. perm. and cap. press. models.

    """
    # Create a grid of S_U and S_D values. For the derivative, we need a finer grid.
    S_values = np.linspace(0, 1, 300)
    S_D_grid, S_U_grid = np.meshgrid(S_values, S_values)

    pe_values = [1.0, 3.0, 10.0]  # Strong to weak capillary effects
    rp_models = ["linear", "Corey", "Brooks-Corey"]
    cp_models = ["linear", "Brooks-Corey"]
    upwinding_flags = [True, False]

    for i, p_e in enumerate(pe_values):
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
                    F_values = F_with_capillary(
                        S_U_grid,
                        S_D_grid,
                        rp_model=rp_model,
                        cp_model=cp_model,
                        upwinding=upwinding,
                        p_e=p_e,
                    )
                    dSD_F_Values = np.gradient(F_values, 1 / 299, axis=1)

                    # Plot the surface
                    axes[j, k].plot_surface(
                        dSD_F_Values, S_U_grid, F_values, cmap="viridis"
                    )

                    # Set labels and title
                    axes[j, k].set_xlabel(r"$s_D$")
                    axes[j, k].set_ylabel(r"$s_U$")
                    axes[j, k].set_zlabel(r"$\partial_{s_D} F(s_U, s_D)$")

                    # Split title into two lines for better readability
                    title_line1 = rf"$P_e$ = {p_e}, RP Model = {rp_model}"
                    title_line2 = f"CP Model = {cp_model}, Upwinding = {upwinding}"
                    axes[j, k].set_title(f"{title_line1}\n{title_line2}")

                    # Turn to have the origin in the front.
                    axes[j, k].view_init(azim=-135)
                    axes[j, k].set_xlim(0.0, 1.0)
                    axes[j, k].set_ylim(0.0, 1.0)

            # Add more padding between subplots
            plt.tight_layout(pad=3.0)
            fig.savefig(
                dirname / f"dSD_F_cap_plot_Pe_{p_e}_upwinding_{upwinding}.png",
                dpi=300,
                bbox_inches="tight",
            )


def plot_F_gravity() -> None:
    """Plot the flux function F as S_U and S_D vary between 0 and 1.

    Creates 3D surface plots for different rel. perm. models and gravity numbers.

    """
    # Create a grid of S_U and S_D values
    S_values = np.linspace(0, 1, 50)
    S_D_grid, S_U_grid = np.meshgrid(S_values, S_values)

    ng_values = [-3.0, -1.0, 1.0, 3.0]  # Strong to weak capillary effects
    rp_models = ["linear", "Corey", "Brooks-Corey"]
    upwinding_flags = [True, False]

    for upwinding in upwinding_flags:
        # Create figure with subplots
        fig, axes = plt.subplots(
            len(rp_models),
            len(ng_values),
            figsize=(15, 10),  # Increased figure size
            subplot_kw={"projection": "3d"},
        )

        # Loop through relative permeability and capillary pressure models and plot
        # numerical flux function.
        for i, rp_model in enumerate(rp_models):
            for j, n_g in enumerate(ng_values):
                F_values = F_with_gravity(
                    S_U_grid,
                    S_D_grid,
                    rp_model=rp_model,
                    upwinding=upwinding,
                    n_g=n_g,
                )

                # Plot the surface
                axes[i, j].plot_surface(S_D_grid, S_U_grid, F_values, cmap="viridis")

                # Set labels and title
                axes[i, j].set_xlabel(r"$s_D$")
                axes[i, j].set_ylabel(r"$s_U$")
                axes[i, j].set_zlabel(r"$F(s_U, s_D)$")

                # Split title into two lines for better readability
                title_line1 = rf"$N_g$ = {n_g}, RP Model = {rp_model}"
                title_line2 = f"Upwinding = {upwinding}"
                axes[i, j].set_title(f"{title_line1}\n{title_line2}")

                # Turn to have the origin in the front.
                axes[i, j].view_init(azim=-135)
                axes[i, j].set_xlim(0.0, 1.0)
                axes[i, j].set_ylim(0.0, 1.0)
                axes[i, j].set_zlim(-3.5, 3.5)

        # Add more padding between subplots
        plt.tight_layout(pad=3.0)
        fig.savefig(
            dirname / f"F_grav_plot_upwinding_{upwinding}.png",
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    # plot_f_w()
    # plot_F_capillary()
    plot_dSU_F_capillary()
    # plot_F_gravity()

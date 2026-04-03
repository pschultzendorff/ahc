import pathlib

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from ahc.derived_models.fluid_values import oil, water
from ahc.derived_models.spe10 import HEIGHT
from numpy.typing import ArrayLike

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

RESOLUTION: int = 100  # Resolution for the 2D plots.

# Model parameters.
mu_w: float = water["viscosity"]  # Viscosity of wetting phase [mP??]
mu_n: float = oil["viscosity"]  # Viscosity of non-wetting phase [mP??]
rho_w: float = water["density"]  # Density of wetting phase [kg m^-3]
rho_n: float = oil["density"]  # Density of non-wetting phase [kg m^-3]

permeability: float = (
    (2.5e-12) ** -1 + (1.3e-13) ** -1
) ** -1  # [m^2] # Permeability of the porous medium. Harmonic mean of two cells in
# the high permeability region.
L: float = HEIGHT / (2 * 100)  # Characteristic length scale [m]
Pc_bar: float = 100 * pp.PASCAL  # Characteristic capillary pressure [Pa]

# Representative total flow rate from two cells in the high permeability region. It is
# important to have the SAME length scale and permeability that goes into the Peclet
# number to get the right ratio of viscous and capillary flow over the SAME interface.
# We multiply by the total mobility with linear rel. perms. at s=0.5.
total_mobility: float = 0.5 / mu_n + 0.5 / mu_w
print(total_mobility)
total_flow: float = (
    total_mobility * (5.36e7 * pp.PASCAL - 5.348e7 * pp.PASCAL) / L * permeability
)  # [m^3 s^-1]
total_flow = 0.5 * (5.3e7 * pp.PASCAL - 5.2e7 * pp.PASCAL) / L * permeability
# Gravity gradient between two cells. It is important to have the SAME length scale that
# goes into the gravity number to get the right ratio of viscous and buoyancy flow over
# the SAME interface.
grad_G: float = 9.81 * L  # Gravity [m^2s^-2]

peclet: float = total_flow * mu_n * L / (permeability * Pc_bar)
N_g: float = permeability * (rho_w - rho_n) * grad_G / (mu_n * total_flow)


def pc_bar_to_peclet(
    pc_bar: ArrayLike | None = None, peclet: ArrayLike | None = None, **kwargs
) -> np.ndarray:
    """Convert characteristic capillary pressure to Peclet number or vice versa.

    Parameters:
        pc_bar: Characteristic capillary pressure. Default is None.
        peclet: Peclet number specifying the ratio of viscous to capillary force.
            Default is None.
        **kwargs: Additional arguments for the conversion. If not provided, the global
            variables `L`, `mu_n`, `permeability`, and `total_flow` are used.
            - `L`: Characteristic length scale.
            - `mu_n`: Viscosity of the non-wetting phase.
            - `permeability`: Permeability of the porous medium.
            - `total_flow`: Representative total flow rate.

    Raises:
        ValueError: If both `entry_pressure` and `peclet` are provided or if neither
            is provided.


    Returns:
        The value of the Peclet number or entry pressure.

    """
    if pc_bar is None and peclet is None:
        raise ValueError("Either entry_pressure or peclet must be provided.")
    elif pc_bar is not None and peclet is not None:
        raise ValueError("Only one of entry_pressure or peclet can be provided.")
    L_local = kwargs.get("L", L)
    mu_n_local = kwargs.get("mu_n", mu_n)
    permeability_local = kwargs.get("permeability", permeability)
    total_flow_local = kwargs.get("total_flow", total_flow)

    if pc_bar is not None:
        # Convert entry pressure to Peclet number.
        pc_bar = np.asarray(pc_bar)
        return total_flow_local * mu_n_local * L_local / (permeability_local * pc_bar)
    else:
        # Convert Peclet number to entry pressure.
        peclet = np.asarray(peclet)
        return peclet * permeability_local / (total_flow_local * mu_n_local * L_local)


def k_rw(S: ArrayLike, model: str) -> np.ndarray:
    """Wetting relative permeability.

    Parameters:
        S: Saturation

    Returns:
        The value of the relative permeability.

    """
    S = np.array(S)

    # Compute relative permeability based on chosen model.
    if model == "Corey_2":
        rel_perm: np.ndarray = S**2
    elif model == "Corey_3":
        rel_perm = S**3
    elif model == "linear":
        rel_perm = S
    # Brooks-Corey-Mualem
    elif model == "Brooks-Corey":
        rel_perm = S ** (2.0 + 2.0 * 1.0)
    else:
        raise ValueError(f"Unknown relative permeability model: {model}")

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
    if model == "Corey_2":
        rel_perm: np.ndarray = (1 - S) ** 2
    elif model == "Corey_3":
        rel_perm = (1 - S) ** 3
    elif model == "linear":
        rel_perm = 1 - S
    # Brooks-Corey-Mualem
    elif model == "Brooks-Corey":
        rel_perm = ((1 - S) ** 2.0) * ((1 - S**2.0) ** 1.0)
    else:
        raise ValueError(f"Unknown relative permeability model: {model}")

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
        # Limit from below by entry pressure.
        p_c = np.clip(
            entry_pressure
            * np.power(
                S,
                -1 / 2.0,
                out=np.full_like(S, entry_pressure * 10.0),
                where=S != 0,
            ),
            entry_pressure,
            entry_pressure * 10.0,
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
    r"""Buoyancy flow function.

    The equation implemented is
    .. math::
        - (λ_w*k_rn/λ_T)*N_g

    Parameters:
        S: Saturation.
        rp_model: Relative permeability model.
        n_g: Gravity number, :math:`N_g`.

    Returns:
        The value of the buoyancy flow function.

    """
    n_g = np.array(n_g)

    lambda_total = lambda_w(S, rp_model) + lambda_n(S, rp_model)

    return -(lambda_w(S, rp_model) * k_rn(S, rp_model) / lambda_total) * n_g


def capillary_flow(
    S: ArrayLike,
    rp_model: str,
    grad_pc: ArrayLike,
    peclet: ArrayLike,
    **kwargs,
) -> np.ndarray:
    r"""Capillary flow function.

    The equation implemented is
    .. math::
        (λ_w*k_rn/λ_T)*(\nabla P_c/P_e(P_c/L))

    Parameters:
        S: Saturation
        grad_pc: Gradient of capillary pressure, :math:`∇P_c`.
        peclet: Peclet number specifying the ratio of viscous to capillary force.
        **kwargs: Additional arguments for the capillary flow function when computed in
            terms of the Peclet number.
            - `Pc_bar`: Characteristic capillary pressure. Default is the global
              `Pc_bar`.
            - `L`: Characteristic length scale. Default is the global `L`.

    Returns:
        The value of the capillary flow function.

    """
    peclet = np.array(peclet)
    grad_pc = np.array(grad_pc)
    lambda_total = lambda_w(S, rp_model) + lambda_n(S, rp_model)
    return (lambda_w(S, rp_model) * k_rn(S, rp_model) / lambda_total) * (
        grad_pc / (peclet * kwargs.get("Pc_bar", Pc_bar) / kwargs.get("L", L))
    )


def f_w(
    S: ArrayLike,
    rp_model: str,
    n_g: ArrayLike = N_g,
    grad_pc: ArrayLike = 0.0,
    peclet: ArrayLike = peclet,
    **kwargs,
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
        peclet: Peclet number specifying the ratio ratio :math:`P_e(P_c/L)`. Default is
        the global `P_e`.
        **kwargs: Additional arguments for the capillary flow function.

    Returns:
        The value of the fractional flow function.

    """
    return (
        viscous_flow(S, rp_model)
        + buoyancy_flow(S, rp_model, n_g)
        + capillary_flow(S, rp_model, grad_pc, peclet, **kwargs)
    )


def cap_press_gradient(
    s_U: ArrayLike, s_D: ArrayLike, cp_model: str, entry_pressure: float
) -> np.ndarray:
    """Calculate the capillary pressure gradient.

    Parameters:
        s_U: Upstream saturation
        s_D: Downstream saturation
        cp_model: Capillary pressure model
        entry_pressure: Entry pressure for the capillary pressure model

    Returns:
        The value of the capillary pressure gradient.

    """
    # Calculate the capillary pressure for upstream and downstream saturations.
    p_c_U = p_c(s_U, cp_model, entry_pressure)
    p_c_D = p_c(s_D, cp_model, entry_pressure)

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
    s_U: ArrayLike,
    s_D: ArrayLike,
    rp_model: str,
    cp_model: str,
    entry_pressure: float,
    upwinding: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Calculate the flow function F with viscous and capillary flow based on upstream and
    downstream saturations.

    Parameters:
        s_U: Upstream saturation.
        s_D: Downstream saturation.
        rp_model: Relative permeability model.
        cp_model: Capillary pressure model.
        entry_pressure: Entry pressure for the capillary pressure model.
        upwinding: Boolean flag for upwinding. If False, uses averaged mobility. Default
            is True.
        **kwargs: Additional arguments for the flow function :meth:`f_w` that get passed
         on to the capillary flow :meth:`capillary_flow`.
            - `peclet`: Peclet number specifying the ratio of viscous to capillary
                force. Default is the global `peclet`.
            - `Pc_bar`: Characteristic capillary pressure. Default is the global
              `Pc_bar`.

    Returns:
        The value of the numerical flow function F(s_U, s_D)

    """
    s_U = np.array(s_U)
    s_D = np.array(s_D)

    # Calculate capillary pressure gradient.
    grad_pcs: np.ndarray = cap_press_gradient(s_U, s_D, cp_model, entry_pressure)

    # Calculate fractional flow function based on upstream saturations. Set gravity to
    # 0.
    f_w_SU: np.ndarray = f_w(s_U, rp_model, n_g=0.0, grad_pc=grad_pcs, **kwargs)

    # Precompute rel. perms. and mobility values.
    lambda_wSU = lambda_w(s_U, rp_model)
    lambda_nSU = lambda_n(s_U, rp_model)
    lambda_wSD = lambda_w(s_D, rp_model)
    lambda_nSD = lambda_n(s_D, rp_model)
    k_rn_SU = k_rn(s_U, rp_model)
    k_rn_SD = k_rn(s_D, rp_model)

    peclet_local = kwargs.get("peclet", peclet)
    Pc_bar_local = kwargs.get("Pc_bar", Pc_bar)
    L_local = kwargs.get("L", L)

    if upwinding:
        # Compute numerical flow for differing co-/counter-current cases.

        # Co-current, wetting forward, nonwetting forward.
        co_current: np.ndarray = (
            lambda_wSU
            * (1 + k_rn_SU * grad_pcs / (peclet_local * Pc_bar_local / L_local))
            / (lambda_wSU + lambda_nSU)
        )
        # Counter-current, wetting forward, nonwetting backward.
        counter_current_1: np.ndarray = (
            lambda_wSU
            * (1 + k_rn_SD * grad_pcs / (peclet_local * Pc_bar_local / L_local))
            / (
                lambda_wSU + lambda_nSD + 1e-20
            )  # Due to upwinding, we can have a division by zero.
        )
        # Counter-current, wetting backward, nonwetting forward.
        counter_current_2: np.ndarray = (
            lambda_wSD
            * (1 + k_rn_SU * grad_pcs / (peclet_local * Pc_bar_local / L_local))
            / (
                lambda_wSD + lambda_nSU + 1e-20
            )  # Due to upwinding, we can have a division by zero.
        )

        # Upwinding depends on cap. press. gradient sign. For negative gradient, we have
        # a zero flux point. For positive gradient, we have a unit flux point.
        num_flow: np.ndarray = np.where(
            grad_pcs >= 0,
            np.where(f_w_SU >= 1, counter_current_1, co_current),
            np.where(f_w_SU <= 0, counter_current_2, co_current),
        )

    else:
        # Average mobilities.
        lambda_wAVG: np.ndarray = (lambda_wSU + lambda_wSD) / 2
        lambda_nAVG: np.ndarray = (lambda_nSU + lambda_nSD) / 2
        k_rnAVG: np.ndarray = (k_rn_SU + k_rn_SD) / 2

        # Compute numerical flow for averaged mobility.
        num_flow = (
            lambda_wAVG
            * (1 + k_rnAVG * grad_pcs / (peclet_local * Pc_bar_local / L_local))
            / (lambda_wAVG + lambda_nAVG)
        )

    return num_flow


def F_with_gravity(
    s_U: ArrayLike,
    s_D: ArrayLike,
    rp_model: str,
    upwinding: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Calculate the flow function F with viscous and buoyancy flow based on upstream and
    downstream saturations.

    Parameters:
        s_U: Upstream saturation.
        s_D: Downstream saturation.
        rp_model: Relative permeability model.
        upwinding: Boolean flag for upwinding. If False, uses averaged mobility. Default
            is True.

    Returns:
        The value of the flow function F(s_U, s_D)

    """
    s_U = np.array(s_U)
    s_D = np.array(s_D)

    # Calculate capillary pressure gradient.

    # Calculate flow function based on upstream saturations. Set capillary gradient to
    # 0.
    f_w_SU: np.ndarray = f_w(s_U, rp_model, grad_pc=0.0, **kwargs)

    # Precompute rel. perms. and mobility values.
    lambda_wSU = lambda_w(s_U, rp_model)
    lambda_nSU = lambda_n(s_U, rp_model)
    lambda_wSD = lambda_w(s_D, rp_model)
    lambda_nSD = lambda_n(s_D, rp_model)
    k_rn_SU = k_rn(s_U, rp_model)
    k_rn_SD = k_rn(s_D, rp_model)

    if upwinding:
        # Compute numerical flow for differing co-/counter-current cases.

        # Co-current, wetting forward, nonwetting forward.
        co_current: np.ndarray = (
            lambda_wSU
            * (1 - k_rn_SU * kwargs.get("n_g", N_g))
            / (lambda_wSU + lambda_nSU)
        )
        # Counter-current, wetting backward, nonwetting forward.
        counter_current_1: np.ndarray = (
            lambda_wSU
            * (1 - k_rn_SD * kwargs.get("n_g", N_g))
            / (
                lambda_wSU + lambda_nSD + 1e-20
            )  # Due to upwinding, we can have a division by zero.
        )
        # Counter-current, wetting forward, nonwetting backward.
        counter_current_2: np.ndarray = (
            lambda_wSD
            * (1 - k_rn_SU * kwargs.get("n_g", N_g))
            / (
                lambda_wSD + lambda_nSU + 1e-20
            )  # Due to upwinding, we can have a division by zero.
        )

        # Upwinding depends on the sign of the gravity number. For a negative gravity
        # number, we have a zero flux point. For a positive gravity number, we have a
        # unit flux point.
        num_flow: np.ndarray = np.where(
            kwargs.get("n_g", N_g) <= 0,
            np.where(f_w_SU >= 1, counter_current_1, co_current),
            np.where(f_w_SU <= 0, counter_current_2, co_current),
        )

    else:
        # Average mobilities.
        lambda_wAVG: np.ndarray = (lambda_wSU + lambda_wSD) / 2
        lambda_nAVG: np.ndarray = (lambda_nSU + lambda_nSD) / 2
        k_rnAVG: np.ndarray = (k_rn_SU + k_rn_SD) / 2

        # Compute numerical flow for averaged mobility.
        num_flow = (
            lambda_wAVG
            * (1 - k_rnAVG * kwargs.get("n_g", N_g))
            / (lambda_wAVG + lambda_nAVG)
        )

    return num_flow


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
    grad_pc_values = [
        30.0 * pp.PASCAL,
        100.0 * pp.PASCAL,
        500.0 * pp.PASCAL,
    ]  # Weak to strong capillary effects

    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 5))

    for i, n_g in enumerate(ng_values):
        for j, rp_model in enumerate(rp_models):
            # Calculate f_w for the current gravity number and zero capillary flow.
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
    for i, grad_pc in enumerate(grad_pc_values):
        for j, rp_model in enumerate(rp_models):
            # Calculate f_w for the current Peclet number and zero buoyancy flow.
            f_values = f_w(S, rp_model, n_g=0.0, grad_pc=grad_pc, peclet=peclet)

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
            axes[i, j].set_title(
                rf"$\nabla p_c$ = {grad_pc} Pa, Rel. perm. model = {rp_model}"
            )
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].legend()
            axes[i, j].set_xlim(0, 1)
            axes[i, j].set_ylim(0, max(1.1, max(f_values) * 1.1))

    plt.tight_layout()
    fig.savefig(dirname / "f_w_capillary_plots.png", dpi=300, bbox_inches="tight")


def plot_F_capillary() -> None:
    """Plot the flow function F as s_U and s_D vary between 0 and 1.

    Creates 3D surface plots for different rel. perm. and cap. press. models.

    """
    # Create a grid of s_U and s_D values
    S_values = np.linspace(0, 1, RESOLUTION)
    s_D_grid, s_U_grid = np.meshgrid(S_values, S_values)

    pe_values = [
        30.0 * pp.PASCAL,
        100.0 * pp.PASCAL,
        500.0 * pp.PASCAL,
        # 2000 * pp.PASCAL,
    ]  # Weak to strong capillary effects
    rp_models = ["Corey_2", "Corey_3", "Brooks-Corey"]
    upwinding_flags = [True]  # , False]

    for p_e in pe_values:
        for upwinding in upwinding_flags:
            # Create figure with subplots
            fig, axes = plt.subplots(
                3,
                2,
                figsize=(15, 10),  # Increased figure size
                subplot_kw={"projection": "3d"},
            )

            def plot(
                rp_model: str, cp_model: str, p_e: float, savepath: pathlib.Path | str
            ) -> None:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                F_values = F_with_capillary(
                    s_U_grid,
                    s_D_grid,
                    rp_model=rp_model,
                    cp_model=cp_model,
                    entry_pressure=p_e,
                    upwinding=upwinding,
                )

                # Plot the surface
                ax.plot_surface(s_D_grid, s_U_grid, F_values, cmap="viridis")  # type: ignore

                # Set labels and title
                ax.set_xlabel(r"$\hat{s}_{w,D}$")
                ax.set_ylabel(r"$\hat{s}_{w,U}$")
                ax.set_zlabel(r"$f_w(s_U, s_D)$")  # type: ignore

                # Split title into two lines for better readability
                # Peclet number only makes sense for nonzero capillary pressure.
                if p_e > 0.0:
                    peclet_local = pc_bar_to_peclet(p_e)
                    title_line1 = rf"$p_e = {p_e} \text{{Pa}}$, $P_e = {peclet_local:.1f} \text{{Pa}}$"
                else:
                    title_line1 = rf"$p_e = {p_e} \text{{Pa}}$"
                title_line2 = f"rel. perm.: {rp_model}, cap. press.: {cp_model}"
                ax.set_title(f"{title_line1}\n{title_line2}")

                # Turn to have the origin in the front.
                ax.view_init(azim=-135)  # type: ignore
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                ax.set_zlim(-3.5, 3.5)  # type: ignore
                # Add more padding between subplots
                plt.tight_layout(pad=3.0)
                fig.savefig(
                    savepath,
                    dpi=300,
                    bbox_inches="tight",
                )

            # Linear rel. perm., zero capillary pressure.
            plot(
                "linear",
                "linear",
                0.0,
                dirname / f"F_cap_p_e_{0.0}_upwinding_{upwinding}_0.png",
            )
            # Linear rel. perm., linear capillary pressure.
            plot(
                "linear",
                "linear",
                p_e,
                dirname / f"F_cap_p_e_{p_e}_upwinding_{upwinding}_1.png",
            )
            # Loop through relative permeability models, Brooks-Corey capillary
            # pressure.
            for i, rp_model in enumerate(rp_models):
                plot(
                    rp_model,
                    "Brooks-Corey",
                    p_e,
                    dirname / f"F_cap_p_e_{p_e}_upwinding_{upwinding}_{2 + i}.png",
                )


def plot_dSU_F_capillary() -> None:
    """Plot the derivative w.r.t. :math:``s_U`` of the flow function F as s_U and s_D
    vary between 0 and 1.

    Creates 3D surface plots for different rel. perm. and cap. press. models.

    """
    # Create a grid of s_U and s_D values. For the derivative, we need a finer grid.
    S_values = np.linspace(0, 1, RESOLUTION)
    s_D_grid, s_U_grid = np.meshgrid(S_values, S_values)

    pe_values = [
        # 30.0 * pp.PASCAL,
        # 100.0 * pp.PASCAL,
        500.0 * pp.PASCAL,
    ]  # Weak to strong capillary effects
    rp_models = ["Corey_2"]  # , "Corey_3", "Brooks-Corey"]
    upwinding_flags = [True]  # , False]

    for p_e in pe_values:
        for upwinding in upwinding_flags:
            # Create figure with subplots
            fig, axes = plt.subplots(
                3,
                2,
                figsize=(15, 10),  # Increased figure size
                subplot_kw={"projection": "3d"},
            )

            def plot(rp_model: str, cp_model: str, p_e: float, i: int, j: int) -> None:
                F_values = F_with_capillary(
                    s_U_grid,
                    s_D_grid,
                    rp_model=rp_model,
                    cp_model=cp_model,
                    entry_pressure=p_e,
                    upwinding=upwinding,
                )
                dSU_F_Values = np.gradient(F_values, 1 / 299, axis=0)

                # Remove the boundary values to avoid plotting artifacts.
                dSU_F_Values_reduced = dSU_F_Values[1:-1, 1:-1]
                s_D_grid_reduced = s_D_grid[1:-1, 1:-1]
                s_U_grid_reduced = s_U_grid[1:-1, 1:-1]

                # Plot the surface
                axes[i, j].plot_surface(
                    s_D_grid_reduced,
                    s_U_grid_reduced,
                    dSU_F_Values_reduced,
                    cmap="viridis",
                )

                # Set labels and title
                axes[i, j].set_xlabel(r"$s_{w,D}$")
                axes[i, j].set_ylabel(r"$s_U$")
                axes[i, j].set_zlabel(r"$\partial_{s_U} F(s_U, s_D)$")

                # Split title into two lines for better readability
                title_line1 = rf"$p_e$ = {p_e} Pa, RP Model = {rp_model}"
                title_line2 = f"CP Model = {cp_model}, Upwinding = {upwinding}"
                axes[i, j].set_title(f"{title_line1}\n{title_line2}")

                # Turn to have the origin in the front.
                axes[i, j].view_init(azim=-135)
                axes[i, j].set_xlim(0.0, 1.0)
                axes[i, j].set_ylim(0.0, 1.0)
                axes[i, j].set_zlim(-10.0, 10.0)

            # Linear rel. perm., zero capillary pressure.
            plot("linear", "linear", 0.0, 0, 0)
            # Linear rel. perm., linear capillary pressure.
            plot("linear", "linear", p_e, 0, 1)
            # Loop through relative permeability models, Brooks-Corey capillary
            # pressure.
            for i, rp_model in enumerate(rp_models):
                plot(rp_model, "Brooks-Corey", p_e, 1 + i // 2, i % 2)

            # Add more padding between subplots
            plt.tight_layout(pad=3.0)
            fig.savefig(
                dirname / f"dSD_F_cap_plot_p_e_{p_e}_upwinding_{upwinding}.png",
                dpi=300,
                bbox_inches="tight",
            )


def plot_F_gravity() -> None:
    """Plot the flow function F as s_U and s_D vary between 0 and 1.

    Creates 3D surface plots for different rel. perm. models and gravity numbers.

    """
    # Create a grid of s_U and s_D values
    S_values = np.linspace(0, 1, RESOLUTION)
    s_D_grid, s_U_grid = np.meshgrid(S_values, S_values)

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
        # numerical flow function.
        for i, rp_model in enumerate(rp_models):
            for j, n_g in enumerate(ng_values):
                F_values = F_with_gravity(
                    s_U_grid,
                    s_D_grid,
                    rp_model=rp_model,
                    upwinding=upwinding,
                    n_g=n_g,
                )

                # Plot the surface
                axes[i, j].plot_surface(s_D_grid, s_U_grid, F_values, cmap="viridis")

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
    plot_F_capillary()
    # plot_dSU_F_capillary()
    # plot_F_gravity()

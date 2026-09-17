import pathlib

import porepy as pp
from ahc.derived_models.fluid_values import co2_reservoir, water
from ahc.derived_models.spe10 import HEIGHT
from ahc.derived_models.spe11 import case_B
from flux_functions_spe10.flux_functions_spe10 import (
    plot_dSU_F_capillary,
    plot_F_capillary,
    plot_F_gravity,
)

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

total_flow: float = 0.3  # []
permeability: float = case_B["PERMEABILITY"]["facies 3"]  # [m^2]

mu_w: float = water["viscosity"]  # Viscosity of wetting phase [mP??]
mu_n: float = co2_reservoir["viscosity"]  # Viscosity of non-wetting phase [mP??]
rho_w: float = water["density"]  # Density of wetting phase [kg m^-3]
rho_n: float = co2_reservoir["density"]  # Density of non-wetting phase [kg m^-3]

G: float = 9.81  # Gravity [m^2s^-2]

L: float = HEIGHT / 100  # Characteristic length scale [m]
# L: float = case_A["HEIGHT"] / 100  # Characteristic length scale [m]
P_c_bar: float = 100 * pp.PASCAL  # Characteristic capillary pressure [Pa]
# P_c_bar: float = case_A["ENTRY_PRESSURE"]["facies 3"]  # Entry pressure [Pa]
# P_c_bar: float = LeverettJfunction(  # type: ignore
#     case_B["PERMEABILITY"]["facies 3"], case_B["POROSITY"]["facies 3"]
# )  # Characteristic capillary pressure [Pa]


P_e: float = total_flow * mu_n * L / (permeability * P_c_bar)
N_g: float = permeability * (rho_w - rho_n) * G / (mu_n * total_flow)


if __name__ == "__main__":
    # plot_f_w()
    plot_F_capillary()
    plot_dSU_F_capillary()
    plot_F_gravity()

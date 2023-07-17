import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from buckley_leverett import analytical_solution
import math

S_w = np.linspace(0, 1, 500)
dShatdS = 1 / 0.4

# linear f_w_1
f_w_1 = S_w
# concave f_w_1
f_w_1 = 1 - (1 - S_w) ** 2

f_w_2 = (S_w**3) / ((1 - S_w) ** 3 + S_w**3)

# Get the concave hull of f_w_2.
params = {
    # fluid and solid params
    "porosity": 1.0,
    "viscosity_w": 1.0,
    "viscosity_n": 1.0,
    "density_w": 1.0,
    "density_n": 1.0,
    "S_M": 1 - 0.0,
    "S_m": 0.0,
    "residual_saturation_w": 0.0,
    "residual_saturation_n": 0.0,
    # rel. perm model
    "rel_perm_model": "power",
    "rel_perm_linear_param_w": 1.0,
    "rel_perm_linear_param_n": 1.0,
    "limit_rel_perm": False,
    # Buckley-Leverett params
    "angle": math.pi / 2,
    "influx": 1.0,
    # Linear flow function. Necessary parameter for the analytical solver.
    "linear_flow": False,
}
analytical = analytical_solution.BuckleyLeverett(params)
concave_hull = analytical.concave_hull()[0](S_w)


# Define the homotopy continuation.
def f_w(k: float) -> np.ndarray:
    return k * f_w_1 + (1 - k) * f_w_2


def f_w_prime(k: float) -> np.ndarray:
    raise NotImplementedError


# Initial parameter.
init_k = 1.0


# Create the figure and the line to manipulate.
fig, ax = plt.subplots()
(line,) = ax.plot(S_w, f_w(init_k), lw=2, label="f_w")
# (v_line,) = ax.axvline(0.0, label="")
ax.plot(S_w[:-2], concave_hull[:-2], "g-", lw=2, label="concave hull")
ax.set_xlabel("S_w")
ax.set_ylabel("f_w")

# Adjust the main plot to make room for the sliders.
fig.subplots_adjust(left=0.25, bottom=0.25)

axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
param_slider = Slider(
    ax=axfreq,
    label="k",
    valmin=0.0,
    valmax=1.0,
    valinit=init_k,
)


# Define the update function and link to the slider.
def update(val):
    line.set_ydata(f_w(param_slider.val))
    fig.canvas.draw_idle()


param_slider.on_changed(update)

plt.show()

# Two-phase flow in PorePy.

This repo contains a working two-phase flow model in PorePy, as well as functionality to
include neural networks into PorePy models.


### Code
In ``./src`` the adaptions to PorePy are found; most notably a two-phase flow model
class and some functionality for that class and the neural networks components. The
neural networks are assumed to be implemented in ``PyTorch``

#### Progress bar
This repo adds progress bars, both for time steps and solver iterations to Porepy, for a
less cluttered interface. However, if run in Jupyter Notebooks, this displays
incorrectly and at each time step a blank line appears.

This is a well-known issue with `tqdm` and Jupyter:
https://github.com/jupyter-widgets/ipywidgets/issues/1845. To circumvent this, add the
content of `custom.css` to the same file in your Jupyter config (may be found under
`~/.jupyter/custom/custom.css`).

### Results
Under ``./results``, one find the results of all runs.
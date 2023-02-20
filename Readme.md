# Two-phase flow in PorePy.

This repo contains a working two-phase flow model in PorePy, functionality to
include neural networks into PorePy models, and some miscellaneous additions to PorePy
functionality.

Furthermore, some notebooks with setup tests and more advanced experiments are included.

As of now the code works with PorePy commit 278a2cf654224c6bd06b3c36b538bdb8fa03b8c7
(10th February 2023). Later or earlier commits migh work as well (as long as the new
model framework is implemented), but this was not tested.


### PorePy changes/additions
In ``./src`` the adaptions to PorePy are found.

#### Two-phase flow model
Two formulations of the model are implemented: Nonwetting pressure-wetting saturation
and (the more unusual) wetting pressure-wetting saturation.

#### Neural networks
Neural networks from `Pytorch` can be included as `ad.Function` into the PorePy `ad`
framework. This is done via the `ml.ml_ad.nn_wrapper` function, which transforms a
neural network into a function. This acts like the functions in `ad.functions`.

Some artificial data for rel. perm. and cap. pressure functions is provided, as well as
simple neural networks and a training function. 

#### Progress bar
This repo adds progress bars, both for time steps and solver iterations to Porepy, for a
less cluttered interface. However, if run in Jupyter Notebooks, this displays
incorrectly and at each time step a blank line appears.

This is a well-known issue with `tqdm` and Jupyter:
https://github.com/jupyter-widgets/ipywidgets/issues/1845. To circumvent this, add the
content of `custom.css` to the same file in your Jupyter config (may be found under
`~/.jupyter/custom/custom.css`).

#### Adaptive time stepping
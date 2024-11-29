# Two-phase flow in PorePy.

This repo contains a working two-phase flow model in PorePy, a posteriori error
estimators, an adaptive homotopy continuation algorithm, integration of neural networks
into PorePy models, and some miscellaneous additions to PorePy functionality.


As of now the code works with PorePy commit c5732db (19th October 2024). Some modules
are not yet updated to modern PorePy. Checkout the other branches for this.

### PorePy changes/additions
In ``./src`` the adaptions to PorePy are found.

#### Two-phase flow model
Implemented in the fractional flow formulation.

#### Neural networks
Neural networks from `Pytorch` can be included as `ad.Function` into the PorePy `ad`
framework. This is done via the `ml.ml_ad.nn_wrapper` function, which transforms a
neural network into a function. This acts like the functions in `ad.functions`.

Some artificial data for rel. perm. and cap. pressure functions is provided, as well as
simple neural networks and a training function. 

#### Homotopy Continuation

#### A posteriori error estimates
- Equilibrated flux and pressure reconstructions. Based on Jhabriel Valera's
  (unpublished) work.
- Error estimates.

# TODO
- We **always** work on only one subdomain. Remove all ``for`` loops through subdomains.
- Often we call ``pp.shift_solution_values`` even though this is not necessary!
- Rename every method and attribute that contains ``continuation`` to ``hc`` to make
  things more readable!
- Add keywords_only to all dataclasses!
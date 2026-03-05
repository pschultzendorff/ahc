# Two-phase flow in PorePy.
This repository extends the functionality of
[PorePy](https://github.com/pmgbergen/porepy) with the following:
- an incompressible & immiscible two-phase flow model in the fractional flow formulation
- a posteriori error estimators for the two-phase flow model
- an adaptive homotopy continuation algorithm based on the error estimators
- integration of neural networks into PorePy models
- some miscellaneous additions to PorePy functionality.

The code does not work with a current PorePy build, but requires [this
fork](https://github.com/pschultzendorff/porepy_hc), which adds some functionalities for
homotopy continuation to work properly.

#### Setup
The following should be done in either a docker container or a virtual environment. A
dockerfile & build will be provided at a later point.
1. Clone https://github.com/pschultzendorff/porepy_hc and follow the instructions to
   install PorePy
2. Clone this repository `clone ...`
3. Run


#### Two-phase flow model
Implemented in the fractional flow formulation.

#### A posteriori error estimates
The implementation closely follows [C. Cancès, I. Pop, and M. Vohralík, “An a posteriori
error estimate for vertex-centered finite volume discretizations of immiscible
incompressible two-phase flow,” Math. Comp., vol. 83, no. 285, pp. 153–188, Jan. 2014,
doi: 10.1090/S0025-5718-2013-02723-8.] and [M. Vohralík and M. F. Wheeler, “A posteriori
error estimates, stopping criteria, and adaptivity for two-phase flows,” Comput Geosci,
vol. 17, no. 5, pp. 789–812, Oct. 2013, doi: 10.1007/s10596-013-9356-0.]. The pressure
reconstruction reuses code of Jhabriel Valera [J. Varela, C. E. Schaerer, and E.
Keilegavlen, “A linear potential reconstruction technique based on Raviart-Thomas basis
functions for cell-centered finite volume approximations to the Darcy problem,”
Proceeding Series of the Brazilian Society of Computational and Applied Mathematics,
vol. 11, no. 1, Art. no. 1, Jan. 2025, doi: 10.5540/03.2025.011.01.0332.]

- Implements functions to calculate global and complementary pressure, equilibrate
  fluxes, and post-process and reconstruct pressures. 
- Guaranted error estimates
- Decomposition of the error into discretization, homotopy continuation, and
  linearization error

#### Adaptive Homotopy Continuation


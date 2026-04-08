# ahc - Adaptive Homotopy Continuation for Two-Phase Flow in PorePy.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19421277.svg)](https://doi.org/10.5281/zenodo.19421277)

This repository extends the functionality of
[PorePy](https://github.com/pmgbergen/porepy) with the following:
- an incompressible & immiscible two-phase flow model in the fractional flow formulation
- a posteriori error estimators for the two-phase flow model
- an adaptive homotopy continuation algorithm based on the error estimators

The code does not work with the current PorePy version, but requires [this
fork](https://github.com/pschultzendorff/porepy_hc), which adds some functionalities for
homotopy continuation to work properly.

If you use this code, please cite:

P. von Schultzendorff, J. W. Both, J. M. Nordbotten, T. H. Sandve, and M. Vohralík,
"Adaptive homotopy continuation solver for incompressible two-phase flow in porous
media"

## Installation and reproducing the examples

**Local setup (requires Python 3.12.11):**
```bash
git clone https://github.com/pschultzendorff/porepy_hc
pip install -e ./porepy_hc[development,testing]
git clone https://github.com/pschultzendorff/ahc
pip install -e ./ahc
sh ./ahc/run_all.sh
```

**Docker:**
```bash
git clone https://github.com/pschultzendorff/ahc
cd ahc
docker compose up
```

## Content

#### Two-phase flow model
Implemented in the fractional flow formulation.

#### A posteriori error estimates
The implementation closely follows [1] and [2]. The pressure reconstruction reuses code
of Jhabriel Varela [3].

- Implements functions to calculate global and complementary pressure, equilibrate
  fluxes, and post-process and reconstruct pressures.
- Guaranteed error estimates.
- Decomposition of the error into discretization, homotopy continuation, and
  linearization error.

#### Adaptive Homotopy Continuation
The adaptive homotopy continuation (AHC) algorithm gradually deforms a simple,
easy-to-solve problem (linear rel. perm., zero capillary pressure) into the target
nonlinear problem using adaptive stepping criteria:

- **Newton corrector loop**: The linearization error estimator in
  `ahc.models.error_estimate` provides a stopping criterion that allows early
  termination of the Newton corrector loop when the linearization error becomes
  sufficiently small relative to the HC error estimator.
- **Homotopy continuation loop**: After each Newton solve, the HC error estimator
  evaluates whether the homotopy step is small enough to proceed to the next step, or
  should be retried with a reduced step size. This adaptive step control, implemented in
  `ahc.models.homotopy_continuation`, prevents both over-refinement (excessive homotopy
  steps) and solver failure (oversized steps).

The combined effect is a globally adaptive algorithm that balances computational
efficiency with robustness by terminating iterations early when convergence criteria are
met.

---

## References
[1] C. Cancès, I. Pop, and M. Vohralík, "An a posteriori error estimate for
vertex-centered finite volume discretizations of immiscible incompressible two-phase
flow," *Math. Comp.*, vol. 83, no. 285, pp. 153–188, Jan. 2014,
doi: 10.1090/S0025-5718-2013-02723-8.

[2] M. Vohralík and M. F. Wheeler, "A posteriori error estimates, stopping criteria,
and adaptivity for two-phase flows," *Comput Geosci*, vol. 17, no. 5, pp. 789–812,
Oct. 2013, doi: 10.1007/s10596-013-9356-0.

[3] J. Varela, C. E. Schaerer, and E. Keilegavlen, "A linear potential reconstruction
technique based on Raviart-Thomas basis functions for cell-centered finite volume
approximations to the Darcy problem," *Proceeding Series of the Brazilian Society of
Computational and Applied Mathematics*, vol. 11, no. 1, Jan. 2025,
doi: 10.5540/03.2025.011.01.0332.

[4] M. A. Christie and M. J. Blunt, "Tenth SPE Comparative Solution Project: A
Comparison of Upscaling Techniques," *SPE Reservoir Evaluation & Engineering*, vol. 4,
no. 4, pp. 308–317, 2001, doi: 10.2118/72469-PA.

[5] J. M. Nordbotten, M. A. Ferno, B. Flemisch, A. R. Kovscek, and K.-A. Lie, "The 11th
Society of Petroleum Engineers Comparative Solution Project: Problem Definition," 
*SPE Journal*, vol. 29, no. 5, pp. 2507–2524, 2024, doi: 10.2118/218015-PA.

[6] X. Wang and H. A. Tchelepi, “Trust-region based solver for nonlinear transport in
heterogeneous porous media,” *Journal of Computational Physics*, vol. 253, pp. 114–137,
2013, doi: 10.1016/j.jcp.2013.06.041.

## AI disclosure

Generative AI (GitHub Copilot in VS Code, ChatGPT, and Microsoft Copilot) was used to
create scripted figures with `Matplotlib`.

## TODO 
Fix some of the tests
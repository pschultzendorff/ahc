# ``HCProtocol`` and ``TPFProtocol`` define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
class HybridUpwindingHC(HCProtocol, DarcyFluxes):
    r"""Mobility upwinded for viscous flux and averaged for capillary flux.

    F. P. Hamon, B. T. Mallison, and H. A. Tchelepi, “Implicit Hybrid Upwinding for
          two-phase flow in heterogeneous porous media with buoyancy and
          capillarity,” Computer Methods in Applied Mechanics and Engineering, vol.
          331, pp. 701–727, Apr. 2018, doi: 10.1016/j.cma.2017.10.008.

    """

    # def prepare_simulation(self) -> None:
    #     super().prepare_simulation()
    #     self.calc_capillary_diffusion_interpolants()

    # def calc_capillary_diffusion_interpolants(self) -> None:
    #     hybrid_upwind_constants: dict[str, Any] = self.params.get(
    #         "hc_hybrid_upwind_constants", {}
    #     )
    #     self.s_interpol_vals: np.ndarray = np.linspace(
    #         self.wetting.residual_saturation + self.wetting.saturation_epsilon,
    #         1
    #         - self.nonwetting.residual_saturation
    #         - self.nonwetting.saturation_epsilon,
    #         hybrid_upwind_constants.get("interpolation_degree", 100),
    #     )

    #     def capillary_diffusion(s_w: np.ndarray) -> np.ndarray:
    #         return (
    #             self.rel_perm_np(s_w, self.wetting, self._rel_perm_constants_1)
    #             * self.rel_perm_np(s_w, self.wetting, self._rel_perm_constants_1)
    #             / (
    #                 self.rel_perm_np(s_w, self.wetting, self._rel_perm_constants_1)
    #                 * self.wetting.viscosity
    #                 / self.nonwetting.viscosity
    #                 * self.rel_perm_np(s_w, self.wetting, self._rel_perm_constants_1)
    #             )
    #             * self.cap_press_deriv_np(s_w, self._cap_press_constants_1)
    #         )

    #     self.capillary_diffusion_vals: np.ndarray = capillary_diffusion(
    #         self.s_interpol_vals
    #     )

    # def capillary_flux(self, g: pp.Grid) -> pp.ad.Operator:
    #     tpfa = pp.ad.TpfaAd(self.flux_key, [g])
    #     diffusion_coeff = np.interp(
    #         self.s_interpol_vals, self.capillary_diffusion_vals
    #     )(self.wetting.s)
    #     cap_flux = (
    #         tpfa.flux() @ self.wetting.s
    #         + tpfa.bound_flux() @ self.bc_dirichlet_saturation_values(g, self.wetting)
    #     )

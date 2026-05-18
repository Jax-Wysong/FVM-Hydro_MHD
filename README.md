# Finite Volume Method for Hydro- and Magnetohydro-dynamics problems

A collection of finite-volume solvers for fluid and plasma physics, progressing from simple 1D advection through 2D compressible hydrodynamics and 1D ideal MHD. The chapter-based notebooks follow Zingale's *Introduction to Computational Astrophysical Hydrodynamics*, and the Euler equation solvers using the HLL and HLLC Riemann solvers were modeled after Toro's *Riemann Solvers and Numerical Methods for Fluid Dynamics*. The MHD code was modeled after Carolyn Wendeln's 1D MHD implementation, [Wendeln-FVM-github](https://github.com/wendelnc/1D_FVM_Euler_MHD.git).

---

## Structure

### Chapter 4 — Advection Basics
`Chapter-4-Advection-Basics/`

First-order finite-volume methods for the 1D linear advection equation. Covers the upwind scheme, FTCS instability, and implicit upwind with periodic and outflow boundary conditions. Tests on top-hat and Gaussian initial conditions.

### Chapter 5 — Second-Order Advection
`Chapter-5-Second-Order-Advection/`

Second-order FV solver for 1D linear advection with piecewise linear reconstruction and the minmod slope limiter. Includes a convergence study and a dimensionally-split 2D advection extension tested on top-hat and Gaussian profiles.

### Chapter 6 — Burgers' Equation
`Chapter-6-Burgers-Equation/`

Extends the advection framework to the inviscid Burgers equation. Implements unlimited, minmod, and MC slope limiters and a proper Riemann solver that handles both shocks and rarefactions. Tests rarefaction and sinusoidal initial conditions.

### Chapter 8 — Euler Equations
`Chapter-8-Eulers-Eqs/`

1D and 2D compressible hydrodynamics solvers for the Euler equations.

- **1D CGF solver** — piecewise constant and piecewise linear reconstruction with an exact/CGF Riemann solver; validated against the Sod shock tube exact solution.
- **1D HLL/HLLC solvers** — piecewise linear reconstruction with minmod limiting; HLL and HLLC Riemann solvers compared on the Sod shock tube. The HLL and HLLC solvers follow Toro's formulation.
- **2D HLL solver** — unsplit 2D Euler solver with HLL fluxes; tested on the Sedov blast wave.

Exact solution data and result visualizations (plots and animations) are included.

### MHD — 1D Ideal MHD
`MHD/`

1D ideal MHD solver for the full 8-variable system (density, three momentum components, total energy, and three magnetic field components). Uses an HLL Riemann solver adapted for MHD with fast magnetosonic wave speed estimates. Tested on two problems:

- **Brio-Wu shock tube** 
- **Dai-Woodward problem** 

The code structure and approach were modeled after Carolyn Wendeln's 1D MHD implementation.
[Wendeln-FVM-github](https://github.com/wendelnc/1D_FVM_Euler_MHD.git)

import numpy as np

def cons_to_prim(U, gamma):
    '''
    Definition: converts conserved variables to primitive variables

    Inputs:     U     : conserved state (rho, rho_ux, rho_uy, rho_uz, E, bx, by, bz)^T
                gamma : adiabatic index

    Outputs:    U_prim : primitive state (rho, ux, uy, uz, p, bx, by, bz)^T

    Dependencies: none

    Conversion:
        ux = (rho_ux)/rho,  uy = (rho_uy)/rho,  uz = (rho_uz)/rho
        KE = 0.5*rho*(ux^2 + uy^2 + uz^2)          (kinetic energy density)
        ME = 0.5*(bx^2 + by^2 + bz^2)              (magnetic energy density)
        p  = (gamma-1)*(E - KE - ME)               (pressure from gamma-law EOS)
        B components pass through unchanged
    '''
    # grab conservative variables
    rho_cons      = U[0]
    rho_ux_cons   = U[1]
    rho_uy_cons   = U[2]
    rho_uz_cons   = U[3]
    E_cons        = U[4]
    bx_cons       = U[5]
    by_cons       = U[6]
    bz_cons       = U[7]
    
    # new solution vector to return
    U_prim = np.zeros_like(U)
    
    # recover primitive variables
    rho_prim = rho_cons
    rho_prim = np.maximum(rho_prim, 1e-15)
    ux_prim  = rho_ux_cons/rho_prim   # x-velocity
    uy_prim  = rho_uy_cons/rho_prim   # y-velocity
    uz_prim  = rho_uz_cons/rho_prim   # z-velocity
    E_prim   = E_cons
    bx_prim  = bx_cons
    by_prim  = by_cons
    bz_prim  = bz_cons
    
    KE = 0.5*rho_prim*(ux_prim**2 + uy_prim**2 + uz_prim**2)   # kinetic energy density
    ME = 0.5*(bx_prim**2 + by_prim**2 + bz_prim**2)             # magnetic energy density
    
    p_prim = (gamma - 1.0) * (E_prim - KE - ME)   # pressure from gamma-law EOS
    p_prim = np.maximum(p_prim, 1e-15)
    
    # assign primitive variables to output vector
    U_prim[0] = rho_prim
    U_prim[1] = ux_prim
    U_prim[2] = uy_prim
    U_prim[3] = uz_prim
    U_prim[4] = p_prim
    U_prim[5] = bx_prim
    U_prim[6] = by_prim
    U_prim[7] = bz_prim
    
    return U_prim


def prim_to_cons(U, gamma):
    '''
    Definition: converts primitive variables to conserved variables

    Inputs:     U     : primitive state (rho, ux, uy, uz, p, bx, by, bz)^T
                gamma : adiabatic index

    Outputs:    U_cons : conserved state (rho, rho_ux, rho_uy, rho_uz, E, bx, by, bz)^T

    Dependencies: none

    Conversion:
        rho_ux = rho * ux,  rho_uy = rho * uy,  rho_uz = rho * uz  (momentum densities)
        KE = 0.5*rho*(ux^2 + uy^2 + uz^2)                           (kinetic energy density)
        ME = 0.5*(bx^2 + by^2 + bz^2)                               (magnetic energy density)
        E  = p/(gamma-1) + KE + ME                                  (total energy density)
        B components pass through unchanged
    '''
    # grab primitive variables
    rho_prim  = U[0]
    ux_prim   = U[1]
    uy_prim   = U[2]
    uz_prim   = U[3]
    p_prim    = U[4]
    bx_prim   = U[5]
    by_prim   = U[6]
    bz_prim   = U[7]    

    #--------- get energy from pressure ---------------
    KE = 0.5*rho_prim*(ux_prim**2 + uy_prim**2 + uz_prim**2)   # kinetic energy density
    ME = 0.5*(bx_prim**2 + by_prim**2 + bz_prim**2)             # magnetic energy density

    E_cons   = p_prim/(gamma-1) + KE + ME   # total energy density
    #--------------------------------------------------

    # new solution vector to return
    U_cons = np.zeros_like(U)
    
    # assemble conserved variables
    rho_cons      = rho_prim
    rho_ux_cons   = rho_prim*ux_prim   # x-momentum density
    rho_uy_cons   = rho_prim*uy_prim   # y-momentum density
    rho_uz_cons   = rho_prim*uz_prim   # z-momentum density
    bx_cons       = bx_prim
    by_cons       = by_prim
    bz_cons       = bz_prim  

    # assign conserved variables to output vector
    U_cons[0] = rho_cons
    U_cons[1] = rho_ux_cons
    U_cons[2] = rho_uy_cons
    U_cons[3] = rho_uz_cons
    U_cons[4] = E_cons
    U_cons[5] = bx_cons
    U_cons[6] = by_cons
    U_cons[7] = bz_cons
    
    return U_cons

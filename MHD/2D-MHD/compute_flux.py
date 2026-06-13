import numpy as np

# user defined libraries
import conversions

def compute_flux_x(q, gamma):
    '''
    Definition: evaluates the x-direction MHD flux vector F(U)_x

    Inputs:     q     : conserved state (rho, rho_ux, rho_uy, rho_uz, E, bx, by, bz)^T
                gamma : adiabatic index

    Outputs:    F : x-direction flux vector (8 components)

        F[0] = rho * ux                                        (mass flux)
        F[1] = rho*ux^2 + p + 0.5*||B||^2 - bx^2             (x-momentum: gas+magnetic pressure - tension)
        F[2] = rho*ux*uy - bx*by                              (y-momentum: Maxwell stress)
        F[3] = rho*ux*uz - bx*bz                              (z-momentum: Maxwell stress)
        F[4] = ux*(E + p + 0.5*||B||^2) - bx*(u.B)           (energy: work by total pressure - Poynting)
        F[5] = 0                                               (B_x constant)
        F[6] = ux*by - uy*bx                                  (induction: B_y evolution)
        F[7] = ux*bz - uz*bx                                  (induction: B_z evolution)

    Dependencies: cons_to_prim

    Reference: Gardiner & Stone (2005)
    '''
    q_prim = conversions.cons_to_prim(q, gamma)
    
    rho  = q_prim[0]
    ux   = q_prim[1]
    uy   = q_prim[2]
    uz   = q_prim[3]
    p    = q_prim[4]
    bx   = q_prim[5]
    by   = q_prim[6]
    bz   = q_prim[7]
    
    E = p/(gamma-1) + 0.5*rho*(ux**2 + uy**2 + uz**2) + 0.5*(bx**2 + by**2 + bz**2)
    
    F = np.zeros_like(q)
    F[0] = rho * ux
    F[1] = rho*(ux**2) + p + 0.5*(bx**2 + by**2 + bz**2) - bx**2   # magnetic pressure + tension
    F[2] = rho*ux*uy - bx*by                                          # Maxwell stress y
    F[3] = rho*ux*uz - bx*bz                                          # Maxwell stress z
    F[4] = ux*(E + p + 0.5*(bx**2 + by**2 + bz**2)) - bx*(ux*bx + uy*by + uz*bz)  # energy flux
    F[5] = 0                                                           # B_x has no x-flux (zero from ideal induction)
    F[6] = ux*by - uy*bx                                              # induction B_y
    F[7] = ux*bz - uz*bx                                              # induction B_z
    
    return F

def compute_flux_y(q, gamma):
    '''
    Definition: evaluates the y-direction MHD flux vector F(U)_y

    Inputs:     q     : conserved state (rho, rho_ux, rho_uy, rho_uz, E, bx, by, bz)^T
                gamma : adiabatic index

    Outputs:    F : y-direction flux vector (8 components)

        F[0] = rho * uy                                        (mass flux)
        F[1] = rho*ux*uy - bx*by                             (x-momentum: Maxwell stress)
        F[2] = rho*uy^2 + p + 0.5*||B||^2 - by^2             (y-momentum: gas+magnetic pressure - tension)
        F[3] = rho*uy*uz - by*bz                              (z-momentum: Maxwell stress)
        F[4] = uy*(E + p + 0.5*||B||^2) - by*(u.B)           (energy: work by total pressure - Poynting)
        F[5] = uy*bx - ux*by                                  (induction: B_y evolution)
        F[6] = 0                                              (B_y constant)
        F[7] = uy*bz - uz*by                                  (induction: B_z evolution)

    Dependencies: cons_to_prim

    Reference: Gardiner & Stone (2005)
    '''
    q_prim = conversions.cons_to_prim(q, gamma)
    
    rho  = q_prim[0]
    ux   = q_prim[1]
    uy   = q_prim[2]
    uz   = q_prim[3]
    p    = q_prim[4]
    bx   = q_prim[5]
    by   = q_prim[6]
    bz   = q_prim[7]
    
    E = p/(gamma-1) + 0.5*rho*(ux**2 + uy**2 + uz**2) + 0.5*(bx**2 + by**2 + bz**2)
    
    F = np.zeros_like(q)
    F[0] = rho * uy
    F[1] = rho*ux*uy - bx*by                                          # Maxwell stress x
    F[2] = rho*(uy**2) + p + 0.5*(bx**2 + by**2 + bz**2) - by**2      # magnetic pressure + tension
    F[3] = rho*uy*uz - by*bz                                          # Maxwell stress z
    F[4] = uy*(E + p + 0.5*(bx**2 + by**2 + bz**2)) - by*(ux*bx + uy*by + uz*bz)  # energy flux
    F[5] = uy*bx - ux*by                                              # induction B_x            
    F[6] = 0                                                          # B_y has no y-flux (zero from ideal induction)
    F[7] = uy*bz - uz*by                                              # induction B_z
    
    return F
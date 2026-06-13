import numpy as np

# user defined libraries
import conversions
import compute_flux

def HLL(U_L, U_R, gamma, space):
    '''
    Definition: computes the HLL approximate Riemann flux for the 2D ideal MHD equations

    Inputs:     U_L   : left  conserved state (rho, rho_ux, rho_uy, rho_uz, E, bx, by, bz)^T
                U_R   : right conserved state (same layout)
                gamma : adiabatic index
                space : 'x' or 'y' -- normal direction; selects normal velocity component
                        and flux function ('x' uses ux + compute_flux_x, 'y' uses uy + compute_flux_y)

    Outputs:    flux : HLL flux vector at the interface (same shape as U_L)

    Dependencies: cons_to_prim, compute_flux_x, compute_flux_y
    '''
    U_L_cons = U_L.copy()
    U_R_cons = U_R.copy()

    ################################################################################################
    # 1) Convert both states to primitive form; keep conserved copies for flux evaluation
    ################################################################################################

    U_L_prim = conversions.cons_to_prim(U_L, gamma)
    U_R_prim = conversions.cons_to_prim(U_R, gamma)
    
    rho_L  = U_L_prim[0]
    ux_L   = U_L_prim[1]
    uy_L   = U_L_prim[2]
    uz_L   = U_L_prim[3]
    p_L    = U_L_prim[4]
    bx_L   = U_L_prim[5]
    by_L   = U_L_prim[6]
    bz_L   = U_L_prim[7]
    
    rho_R  = U_R_prim[0]
    ux_R   = U_R_prim[1]
    uy_R   = U_R_prim[2]
    uz_R   = U_R_prim[3]
    p_R    = U_R_prim[4]
    bx_R   = U_R_prim[5]
    by_R   = U_R_prim[6]
    bz_R   = U_R_prim[7]

    ################################################################################################
    # 2) Compute squared sound speeds: c^2 = gamma*p/rho
    ################################################################################################

    c2_L = gamma*p_L/rho_L
    c2_R = gamma*p_R/rho_R

    ################################################################################################
    # 3) Compute fast magnetosonic wave speeds c_f using
    #    term1 = c^2 + ||B||^2/rho
    #    term2 = sqrt(term1^2 - 4*c^2*(bx^2/rho))
    #    c_f   = sqrt(0.5*(term1 + term2))
    ################################################################################################

    # ||B||^2 (magnetic field magnitude squared)
    b2_L = bx_L**2 + by_L**2 + bz_L**2
    b2_R = bx_R**2 + by_R**2 + bz_R**2
    
    # term 1 = c^2 + ||B||^2 / rho
    term1_L = c2_L + b2_L/rho_L
    term1_R = c2_R + b2_R/rho_R
    
    if space=='x':
        # term 2 = sqrt( term1^2 - 4* c^2 * (bx^2 / rho))
        term2_L = np.sqrt(term1_L**2 - 4*c2_L *(bx_L**2 / rho_L))
        term2_R = np.sqrt(term1_R**2 - 4*c2_R *(bx_R**2 / rho_R)) 
    elif space=='y':
        # term 2 = sqrt( term1^2 - 4* c^2 * (by^2 / rho))
        term2_L = np.sqrt(term1_L**2 - 4*c2_L *(by_L**2 / rho_L))
        term2_R = np.sqrt(term1_R**2 - 4*c2_R *(by_R**2 / rho_R)) 
    # c_f (fast magnetosonic wave speed)
    cf_L = np.sqrt(0.5*(term1_L + term2_L))
    cf_R = np.sqrt(0.5*(term1_R + term2_R))

    ################################################################################################
    # 4) Estimate bounding wave speeds from normal (x or y) velocity and fast magnetosonic speed
    # 5) Evaluate physical fluxes F_L and F_R
    ################################################################################################

    if space =='x':
        s_l = min(ux_L - cf_L, ux_R - cf_R)   # leftmost  wave speed
        s_r = max(ux_L + cf_L, ux_R + cf_R)   # rightmost wave speed
        F_L = compute_flux.compute_flux_x(U_L_cons, gamma)
        F_R = compute_flux.compute_flux_x(U_R_cons, gamma)
    elif space =='y':
        s_l = min(uy_L - cf_L, uy_R - cf_R)   # leftmost  wave speed
        s_r = max(uy_L + cf_L, uy_R + cf_R)   # rightmost wave speed
        F_L = compute_flux.compute_flux_y(U_L_cons, gamma)
        F_R = compute_flux.compute_flux_y(U_R_cons, gamma)
    else:
        print('error in Riemann flux HLL function with space vars')
    ################################################################################################
    # 6) Apply HLL upwinding based on wave speed signs
    ################################################################################################

    if s_l >= 0:
        flux = F_L
    elif s_r <= 0:
        flux = F_R
    else:
        flux = (s_r * F_L - s_l * F_R + s_l * s_r * (U_R - U_L)) / (s_r - s_l)
    
    return flux


# HLLD Riemann solver
# References: A multi-state HLL approximate Riemann solver for ideal magnetohydrodynamics
# Miyoshi and Kusano

def U_star_s(U_s, rho_s, ux_s, uy_s, uz_s, p_tot_s, E_s, bx_s, by_s, bz_s, cf_s, gamma, S_s, S_M, p_tot_star, space): # equations (43) - (48) 
    U_star_s = np.zeros_like(U_s)
    
    if space=='x':
        rho_s_star = rho_s * (S_s - ux_s)/(S_s - S_M)
        
        # compute star velocities first
        vx_star = S_M
        vy_star = uy_s - bx_s*by_s * (S_M - ux_s)/(rho_s*(S_s - ux_s)*(S_s - S_M) - bx_s**2)
        vz_star = uz_s - bx_s*bz_s * (S_M - ux_s)/(rho_s*(S_s - ux_s)*(S_s - S_M) - bx_s**2)

        
        if (np.abs(S_M - ux_s) < 1e-14 and (np.abs(S_s-(ux_s+cf_s))<1e-14 or np.abs(S_s-(ux_s-cf_s))<1e-14) and np.abs(by_s)<1e-14 and np.abs(bz_s)<1e-14 and np.abs(bx_s**2-gamma*p_tot_s)>1e-14):
            U_star_s[0] = rho_s * (S_s - ux_s)/(S_s - S_M)
            U_star_s[1] = rho_s_star * vx_star
            U_star_s[2] = rho_s_star * vy_star
            U_star_s[3] = rho_s_star * vz_star
            U_star_s[5] = bx_s
            U_star_s[6] = 0.0
            U_star_s[7] = 0.0
            vdotB_star = vx_star*bx_s + vy_star*U_star_s[6] + vz_star*U_star_s[7]
            U_star_s[4] = ((S_s - ux_s)*E_s - p_tot_s*ux_s + p_tot_star*S_M +
                        bx_s*(ux_s*bx_s + uy_s*by_s + uz_s*bz_s - vdotB_star)) / (S_s - S_M)

        else:
            U_star_s[0] = rho_s * (S_s - ux_s)/(S_s - S_M)
            U_star_s[1] = rho_s_star * vx_star
            U_star_s[2] = rho_s_star * vy_star
            U_star_s[3] = rho_s_star * vz_star
            U_star_s[5] = bx_s
            U_star_s[6] = by_s*(rho_s*(S_s-ux_s)**2 - bx_s**2) / (rho_s*(S_s-ux_s)*(S_s-S_M)-bx_s**2)
            U_star_s[7] = bz_s*(rho_s*(S_s-ux_s)**2 - bx_s**2) / (rho_s*(S_s-ux_s)*(S_s-S_M)-bx_s**2)
            vdotB_star = vx_star*bx_s + vy_star*U_star_s[6] + vz_star*U_star_s[7]
            U_star_s[4] = ((S_s - ux_s)*E_s - p_tot_s*ux_s + p_tot_star*S_M +
                        bx_s*(ux_s*bx_s + uy_s*by_s + uz_s*bz_s - vdotB_star)) / (S_s - S_M)
    elif space=='y':
        rho_s_star = rho_s * (S_s - uy_s)/(S_s - S_M)
        
        # compute star velocities first
        vx_star = ux_s - by_s*bx_s * (S_M - uy_s)/(rho_s*(S_s - uy_s)*(S_s - S_M) - by_s**2)
        vy_star = S_M
        vz_star = uz_s - by_s*bz_s * (S_M - uy_s)/(rho_s*(S_s - uy_s)*(S_s - S_M) - by_s**2)

        
        if (np.abs(S_M - uy_s) < 1e-14 and (np.abs(S_s-(uy_s+cf_s))<1e-14 or np.abs(S_s-(uy_s-cf_s))<1e-14) and np.abs(bx_s)<1e-14 and np.abs(bz_s)<1e-14 and np.abs(by_s**2-gamma*p_tot_s)>1e-14):
            U_star_s[0] = rho_s_star
            U_star_s[1] = rho_s_star * vx_star
            U_star_s[2] = rho_s_star * vy_star
            U_star_s[3] = rho_s_star * vz_star
            U_star_s[5] = 0.0
            U_star_s[6] = by_s
            U_star_s[7] = 0.0
            vdotB_star = vx_star*U_star_s[5] + vy_star*U_star_s[6] + vz_star*U_star_s[7]
            U_star_s[4] = ((S_s - uy_s)*E_s - p_tot_s*uy_s + p_tot_star*S_M +
                        by_s*(ux_s*bx_s + uy_s*by_s + uz_s*bz_s - vdotB_star)) / (S_s - S_M)

        else:
            U_star_s[0] = rho_s_star
            U_star_s[1] = rho_s_star * vx_star
            U_star_s[2] = rho_s_star * vy_star
            U_star_s[3] = rho_s_star * vz_star
            U_star_s[5] = bx_s*(rho_s*(S_s-uy_s)**2 - by_s**2) / (rho_s*(S_s-uy_s)*(S_s-S_M)-by_s**2)
            U_star_s[6] = by_s
            U_star_s[7] = bz_s*(rho_s*(S_s-uy_s)**2 - by_s**2) / (rho_s*(S_s-uy_s)*(S_s-S_M)-by_s**2)
            vdotB_star = vx_star*U_star_s[5] + vy_star*U_star_s[6] + vz_star*U_star_s[7]
            U_star_s[4] = ((S_s - uy_s)*E_s - p_tot_s*uy_s + p_tot_star*S_M +
                        by_s*(ux_s*bx_s + uy_s*by_s + uz_s*bz_s - vdotB_star)) / (S_s - S_M)


    
    return U_star_s # returns conservative state

def U_star_star(U_s, rho_L, ux_L, uy_L, uz_L, p_L, bx_L, by_L, bz_L, cf_L, S_L, \
                    rho_R, ux_R, uy_R, uz_R, p_R, bx_R, by_R, bz_R, cf_R, S_R, \
                    gamma, S_M, p_tot_star, space): # equations (59) - (63) 
    U_star_star_L = np.zeros_like(U_s)
    U_star_star_R = np.zeros_like(U_s)
    
    if space=='x':
        # compute shared v** and B** as scalars (same for L and R)
        vx_ss = S_M
        vy_ss = (np.sqrt(rho_L)*uy_L + np.sqrt(rho_R)*uy_R + (by_R-by_L)*np.sign(bx_L)) / (np.sqrt(rho_L) + np.sqrt(rho_R))
        vz_ss = (np.sqrt(rho_L)*uz_L + np.sqrt(rho_R)*uz_R + (bz_R-bz_L)*np.sign(bx_L)) / (np.sqrt(rho_L) + np.sqrt(rho_R))
        By_ss = (np.sqrt(rho_L)*by_R + np.sqrt(rho_R)*by_L + np.sqrt(rho_L*rho_R)*(uy_R-uy_L)*np.sign(bx_L)) / (np.sqrt(rho_L) + np.sqrt(rho_R))
        Bz_ss = (np.sqrt(rho_L)*bz_R + np.sqrt(rho_R)*bz_L + np.sqrt(rho_L*rho_R)*(uz_R-uz_L)*np.sign(bx_L)) / (np.sqrt(rho_L) + np.sqrt(rho_R))
        
        
        E_L = p_L/(gamma-1) + 0.5*(rho_L * (ux_L**2 + uy_L**2 + uz_L**2)) + 0.5*(bx_L**2 + by_L**2 + bz_L**2)
        E_R = p_R/(gamma-1) + 0.5*(rho_R * (ux_R**2 + uy_R**2 + uz_R**2)) + 0.5*(bx_R**2 + by_R**2 + bz_R**2)

        rho_star_star_L = rho_L * (S_L - ux_L)/(S_L - S_M)
        
        U_star_star_L[0] = rho_star_star_L
        U_star_star_L[1] = rho_star_star_L * vx_ss
        U_star_star_L[2] = rho_star_star_L * vy_ss
        U_star_star_L[3] = rho_star_star_L * vz_ss
        U_star_star_L[5], U_star_star_L[6], U_star_star_L[7] = bx_L, By_ss, Bz_ss
        
        rho_star_star_R = rho_R * (S_R - ux_R)/(S_R - S_M)
        
        U_star_star_R[0] = rho_star_star_R
        U_star_star_R[1] = rho_star_star_R * vx_ss
        U_star_star_R[2] = rho_star_star_R * vy_ss
        U_star_star_R[3] = rho_star_star_R * vz_ss
        U_star_star_R[5], U_star_star_R[6], U_star_star_R[7] = bx_R, By_ss, Bz_ss
        
        
        vdotB_ss = vx_ss*bx_L + vy_ss*By_ss + vz_ss*Bz_ss  # v**.B**
        # energy (eq 63)
        U_star_star_L[4] = E_L - np.sqrt(rho_L) * (ux_L*bx_L + uy_L*by_L + uz_L*bz_L - vdotB_ss) * np.sign(bx_L)
        U_star_star_R[4] = E_R + np.sqrt(rho_R) * (ux_R*bx_R + uy_R*by_R + uz_R*bz_R - vdotB_ss) * np.sign(bx_R)
    elif space=='y':
        # compute shared v** and B** as scalars (same for L and R)
        vx_ss = (np.sqrt(rho_L)*ux_L + np.sqrt(rho_R)*ux_R + (bx_R-bx_L)*np.sign(by_L)) / (np.sqrt(rho_L) + np.sqrt(rho_R))
        vy_ss = S_M
        vz_ss = (np.sqrt(rho_L)*uz_L + np.sqrt(rho_R)*uz_R + (bz_R-bz_L)*np.sign(by_L)) / (np.sqrt(rho_L) + np.sqrt(rho_R))
        Bx_ss = (np.sqrt(rho_L)*bx_R + np.sqrt(rho_R)*bx_L + np.sqrt(rho_L*rho_R)*(ux_R-ux_L)*np.sign(by_L)) / (np.sqrt(rho_L) + np.sqrt(rho_R))
        Bz_ss = (np.sqrt(rho_L)*bz_R + np.sqrt(rho_R)*bz_L + np.sqrt(rho_L*rho_R)*(uz_R-uz_L)*np.sign(by_L)) / (np.sqrt(rho_L) + np.sqrt(rho_R))
        
        
        E_L = p_L/(gamma-1) + 0.5*(rho_L * (ux_L**2 + uy_L**2 + uz_L**2)) + 0.5*(bx_L**2 + by_L**2 + bz_L**2)
        E_R = p_R/(gamma-1) + 0.5*(rho_R * (ux_R**2 + uy_R**2 + uz_R**2)) + 0.5*(bx_R**2 + by_R**2 + bz_R**2)

        rho_star_star_L = rho_L * (S_L - uy_L)/(S_L - S_M)
        
        U_star_star_L[0] = rho_star_star_L
        U_star_star_L[1] = rho_star_star_L * vx_ss
        U_star_star_L[2] = rho_star_star_L * vy_ss
        U_star_star_L[3] = rho_star_star_L * vz_ss
        U_star_star_L[5], U_star_star_L[6], U_star_star_L[7] = Bx_ss, by_L, Bz_ss
        
        rho_star_star_R = rho_R * (S_R - uy_R)/(S_R - S_M)
        
        U_star_star_R[0] = rho_star_star_R
        U_star_star_R[1] = rho_star_star_R * vx_ss
        U_star_star_R[2] = rho_star_star_R * vy_ss
        U_star_star_R[3] = rho_star_star_R * vz_ss
        U_star_star_R[5], U_star_star_R[6], U_star_star_R[7] = Bx_ss, by_R, Bz_ss
        
        
        vdotB_ss = vx_ss*Bx_ss + vy_ss*by_L + vz_ss*Bz_ss  # v**.B**
        # energy (eq 63)
        U_star_star_L[4] = E_L - np.sqrt(rho_L) * (ux_L*bx_L + uy_L*by_L + uz_L*bz_L - vdotB_ss) * np.sign(by_L)
        U_star_star_R[4] = E_R + np.sqrt(rho_R) * (ux_R*bx_R + uy_R*by_R + uz_R*bz_R - vdotB_ss) * np.sign(by_R)

    
    return U_star_star_L, U_star_star_R # returns conservative state

def HLLD(U_L, U_R, gamma, space):
    # get primitive form for easier computation
    U_L_prim = conversions.cons_to_prim(U_L, gamma)
    U_R_prim = conversions.cons_to_prim(U_R, gamma)
    
    rho_L  = U_L_prim[0]
    ux_L   = U_L_prim[1]
    uy_L   = U_L_prim[2]
    uz_L   = U_L_prim[3]
    p_L    = U_L_prim[4]
    bx_L   = U_L_prim[5]
    by_L   = U_L_prim[6]
    bz_L   = U_L_prim[7]
    
    rho_R  = U_R_prim[0]
    ux_R   = U_R_prim[1]
    uy_R   = U_R_prim[2]
    uz_R   = U_R_prim[3]
    p_R    = U_R_prim[4]
    bx_R   = U_R_prim[5]
    by_R   = U_R_prim[6]
    bz_R   = U_R_prim[7]
    
    #--------- compute fastest wave speed -----------------
    # c^2 (sound speed)
    c2_L = gamma*p_L/rho_L
    c2_R = gamma*p_R/rho_R
    
    # ||B||^2 (magnetic field magnitude)
    b2_L = bx_L**2 + by_L**2 + bz_L**2
    b2_R = bx_R**2 + by_R**2 + bz_R**2
    
    # term 1 = c^2 + ||B||^2 / rho
    term1_L = c2_L + b2_L/rho_L
    term1_R = c2_R + b2_R/rho_R
    
    if space=='x':
        # term 2 = sqrt( term1^2 - 4* c^2 * (bx^2 / rho))
        term2_L = np.sqrt(term1_L**2 - 4*c2_L *(bx_L**2 / rho_L))
        term2_R = np.sqrt(term1_R**2 - 4*c2_R *(bx_R**2 / rho_R)) 
    elif space=='y':
        # term 2 = sqrt( term1^2 - 4* c^2 * (bx^2 / rho))
        term2_L = np.sqrt(term1_L**2 - 4*c2_L *(by_L**2 / rho_L))
        term2_R = np.sqrt(term1_R**2 - 4*c2_R *(by_R**2 / rho_R)) 
    # c_f (fast magnetosonic wave speed)
    cf_L = np.sqrt(0.5*(term1_L + term2_L))
    cf_R = np.sqrt(0.5*(term1_R + term2_R))
    #--------------------------------------------------------
    
    if space =='x':
        # determine left and rightmost e.vals and S_M which is used to evaluate the average
        # normal velocity from the HLL average
        S_L    = min(ux_L - cf_L, ux_R - cf_R)
        S_R    = max(ux_L + cf_L, ux_R + cf_R)
        p_tot_L = p_L + 0.5*(bx_L**2 + by_L**2 + bz_L**2)
        p_tot_R = p_R + 0.5*(bx_R**2 + by_R**2 + bz_R**2)
        S_M    = ((S_R - ux_R) * rho_R * ux_R - (S_L - ux_L) * rho_L * ux_L - p_tot_R + p_tot_L) / ((S_R - ux_R)*rho_R - (S_L - ux_L)*rho_L)
        
        # get the conserved state and fluxes (U_L and U_R are already conservative form)
        F_L = compute_flux.compute_flux_x(U_L, gamma)
        F_R = compute_flux.compute_flux_x(U_R, gamma)
        
        # get E_L and E_R for U* computations
        # note! Miyoshi and Kusano use e, where I am using E
        # to represent that same quantity
        E_L = U_L[4]
        E_R = U_R[4]
        
        # comptue U* and F* for left and right
        p_tot_star = ((S_R - ux_R)*rho_R*p_tot_L - (S_L - ux_L)*rho_L*p_tot_R + rho_L*rho_R*(S_R - ux_R)*(S_L - ux_L)*(ux_R - ux_L)) / ((S_R - ux_R)*rho_R - (S_L - ux_L)*rho_L)
        
        U_star_L = U_star_s(U_L, rho_L, ux_L, uy_L, uz_L, p_tot_L, E_L, bx_L, by_L, bz_L, cf_L, gamma, S_L, S_M, p_tot_star, 'x')
        U_star_R = U_star_s(U_R, rho_R, ux_R, uy_R, uz_R, p_tot_R, E_R, bx_R, by_R, bz_R, cf_R, gamma, S_R, S_M, p_tot_star, 'x')
        
        S_L_star = S_M - np.abs(bx_L)/(np.sqrt(U_star_L[0])) # eqn (51)
        S_R_star = S_M + np.abs(bx_L)/(np.sqrt(U_star_R[0])) # eqn (51)
        
        U_star_L_prim = conversions.cons_to_prim(U_star_L, gamma)
        U_star_R_prim = conversions.cons_to_prim(U_star_R, gamma)

                                                            # rho*             ux*              uy*                uz*              p*                bx*                by*              bz*
        U_star_star_L, U_star_star_R = U_star_star(U_L, U_star_L_prim[0], U_star_L_prim[1], U_star_L_prim[2], U_star_L_prim[3], U_star_L_prim[4], U_star_L_prim[5], U_star_L_prim[6], U_star_L_prim[7], cf_L, S_L, \
                                                        U_star_R_prim[0], U_star_R_prim[1], U_star_R_prim[2], U_star_R_prim[3], U_star_R_prim[4], U_star_R_prim[5], U_star_R_prim[6], U_star_R_prim[7], cf_R, S_R, \
                                                        gamma, S_M, p_tot_star, 'x')
        
        F_star_L = F_L + S_L*(U_star_L - U_L) # eqn (64)
        F_star_R = F_R + S_R*(U_star_R - U_R) 
        
        F_star_star_L = F_L + S_L_star*U_star_star_L - (S_L_star - S_L)*U_star_L - S_L*U_L # eqn 65
        F_star_star_R = F_R + S_R_star*U_star_star_R - (S_R_star - S_R)*U_star_R - S_R*U_R 
    elif space =='y':
        # determine left and rightmost e.vals and S_M which is used to evaluate the average
        # normal velocity from the HLL average
        S_L    = min(uy_L - cf_L, uy_R - cf_R)
        S_R    = max(uy_L + cf_L, uy_R + cf_R)
        p_tot_L = p_L + 0.5*(bx_L**2 + by_L**2 + bz_L**2)
        p_tot_R = p_R + 0.5*(bx_R**2 + by_R**2 + bz_R**2)
        S_M    = ((S_R - uy_R) * rho_R * uy_R - (S_L - uy_L) * rho_L * uy_L - p_tot_R + p_tot_L) / ((S_R - uy_R)*rho_R - (S_L - uy_L)*rho_L)
        
        # get the conserved state and fluxes (U_L and U_R are already conservative form)
        F_L = compute_flux.compute_flux_y(U_L, gamma)
        F_R = compute_flux.compute_flux_y(U_R, gamma)
        
        # get E_L and E_R for U* computations
        # note! Miyoshi and Kusano use e, where I am using E
        # to represent that same quantity
        E_L = U_L[4]
        E_R = U_R[4]
        
        # comptue U* and F* for left and right
        p_tot_star = ((S_R - uy_R)*rho_R*p_tot_L - (S_L - uy_L)*rho_L*p_tot_R + rho_L*rho_R*(S_R - uy_R)*(S_L - uy_L)*(uy_R - uy_L)) / ((S_R - uy_R)*rho_R - (S_L - uy_L)*rho_L)
        
        U_star_L = U_star_s(U_L, rho_L, ux_L, uy_L, uz_L, p_tot_L, E_L, bx_L, by_L, bz_L, cf_L, gamma, S_L, S_M, p_tot_star, 'y')
        U_star_R = U_star_s(U_R, rho_R, ux_R, uy_R, uz_R, p_tot_R, E_R, bx_R, by_R, bz_R, cf_R, gamma, S_R, S_M, p_tot_star, 'y')
        
        S_L_star = S_M - np.abs(by_L)/(np.sqrt(U_star_L[0])) # eqn (51)
        S_R_star = S_M + np.abs(by_L)/(np.sqrt(U_star_R[0])) # eqn (51)
        
        U_star_L_prim = conversions.cons_to_prim(U_star_L, gamma)
        U_star_R_prim = conversions.cons_to_prim(U_star_R, gamma)

                                                            # rho*             ux*              uy*                uz*              p*                bx*                by*              bz*
        U_star_star_L, U_star_star_R = U_star_star(U_L, U_star_L_prim[0], U_star_L_prim[1], U_star_L_prim[2], U_star_L_prim[3], U_star_L_prim[4], U_star_L_prim[5], U_star_L_prim[6], U_star_L_prim[7], cf_L, S_L, \
                                                        U_star_R_prim[0], U_star_R_prim[1], U_star_R_prim[2], U_star_R_prim[3], U_star_R_prim[4], U_star_R_prim[5], U_star_R_prim[6], U_star_R_prim[7], cf_R, S_R, \
                                                        gamma, S_M, p_tot_star, 'y')
        
        F_star_L = F_L + S_L*(U_star_L - U_L) # eqn (64)
        F_star_R = F_R + S_R*(U_star_R - U_R) 
        
        F_star_star_L = F_L + S_L_star*U_star_star_L - (S_L_star - S_L)*U_star_L - S_L*U_L # eqn 65
        F_star_star_R = F_R + S_R_star*U_star_star_R - (S_R_star - S_R)*U_star_R - S_R*U_R 
    else:
        print('error in HLLD space vars')
        
    # determine HLLD Flux
    
    if S_L >0:
        flux = F_L
    elif (S_L<=0<=S_L_star):
        flux = F_star_L
    elif (S_L_star<=0<=S_M):
        flux = F_star_star_L
    elif (S_M<=0<=S_R_star):
        flux = F_star_star_R
    elif (S_R_star<=0<=S_R):
        flux = F_star_R
    elif (S_R<0):
        flux = F_R
    
    
    return flux

import numpy as np 

#user defined libraries
import conversions
from config import nghost


def Jacobian(U_prim, gamma):
    '''
    Definition: computes the 8x8 flux Jacobian matrix A = dF(W)/dW in primitive variable space

    Inputs:     U_prim : primitive state (rho, u, v, w, p, bx, by, bz)^T
                gamma  : adiabatic index

    Outputs:    A : 8x8 Jacobian matrix 

    Dependencies: none

    Reference: Wendeln's slide set 1

    A is used in compute_interface_states_PLM
    '''
    rho = U_prim[0]
    u  = U_prim[1]
    v  = U_prim[2]
    w  = U_prim[3]
    p   = U_prim[4]
    bx  = U_prim[5]
    by  = U_prim[6]
    bz  = U_prim[7]
    
    A_x = np.array([
        [u,   rho,   0,  0,   0,             0,                     0,       0     ],
        [0,    u,    0,  0, 1/rho,           0,                  by/rho,  bz/rho   ], # A_x[1,5] = 0 (was -bx/rho)
        [0,    0,    u,  0,   0,          -by/rho,               -bx/rho,    0     ],
        [0,    0,    0,  u,   0,          -bz/rho,                  0,    -bx/rho  ],
        [0, gamma*p, 0,  0,   u,             0,                     0,       0     ], #A_x[4,5] = 0  (was (gamma-1)*(u*bx+v*by+w*bz))
        [0,    0,    0,  0,   0,             0,                     0,       0     ],
        [0,    by,  -bx, 0,   0,            -v,                     u,       0     ],
        [0,    bz,   0, -bx,  0,            -w,                     0,       u     ]
        ], dtype=float)
    
    A_y = np.array([
        [v,   0,   rho,  0,   0,       0,               0,                     0   ],
        [0,   v,   0,    0,   0,    -by/rho,         -bx/rho,                  0   ], 
        [0,   0,   v,    0, 1/rho,   bx/rho,            0,                   bz/rho], #A_y[2,6] = 0  (was -by/rho)
        [0,   0,   0,    v,   0,       0,            -bz/rho,               -by/rho],
        [0,   0, gamma*p,0,   v,       0,               0,                     0   ], #A_y[4,6] = 0  (was (gamma-1)*(u*bx+v*by+w*bz))
        [0,  -by,  bx,   0,   0,       v,              -u,                     0   ],
        [0,   0,   0,    0,   0,       0,               0,                     0   ],  
        [0,   0,   bz,  -by,  0,       0,              -w,                     v   ]
    ], dtype=float)

    
    return A_x, A_y

def minmod(a, b):  # vectorized like Carolyn's so input can be vector valued
    '''
    Definition: minmod slope limiter -- returns the smaller-magnitude slope when both
                inputs agree in sign, and zero when they disagree (local extremum)

    Inputs:     a, b : forward and backward difference arrays (any broadcastable shape;
                    vectorized like Carolyn's so inputs can be vector-valued)

    Outputs:    slope : limited slope; zero where a and b have opposite signs,
                        the smaller-magnitude value otherwise

    Sign table:
        a > 0, b > 0  ->  min(a, b)          (both positive: pick shallower slope)
        a < 0, b < 0  ->  max(a, b)          (both negative: pick smaller magnitude)
        sign(a) != sign(b)  ->  0            (local extremum: flatten to avoid new extrema)
    '''
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


def PLM_x(q, Bx_face, dx, dt, gamma):  # takes conservative form
    '''
    Definition: PLM reconstruction along x -- Jacobian-based characteristic slope evolution

    Inputs:     q     : conserved array (DOFs_MHD_2D, ny+2*nghost, nx+2*nghost)
                dx    : cell size in x
                dt    : time step
                gamma : adiabatic index

    Outputs:    q_L, q_R : left/right interface states in conserved form

    Dependencies: cons_to_prim, prim_to_cons, minmod, Jacobian, nghost

    Note: this PLM applies the 8x8 Jacobian A to the MC-limited primitive slope:
        slope_L = (I - (dt/dx)*A) @ minmod(dl, dr)   (left-biased, characteristic-upwinded)
        slope_R = (I + (dt/dx)*A) @ minmod(dl, dr)   (right-biased)
    This bakes half-step time evolution into the interface states before the Riemann solve,
    unlike the primitive-space PLM in the 2D Euler code which has no Jacobian term.
    Reference: Wendeln's slide set 1 for Jacobian; Toro 8.2.2 for PLM with characteristic slope.
    '''
    ny_g = q.shape[1]   # ny + 2*nghost
    nx_g = q.shape[2]   # nx + 2*nghost
    w = conversions.cons_to_prim(q, gamma)   # work with primitive form for slope limiting
    w_L = np.zeros_like(q)
    w_R = np.zeros_like(q)

    for j in range(nghost - 1, ny_g - nghost + 1): # loops go left and right 1 of physical domain
        for i in range(nghost - 1, nx_g - nghost + 1):
            dc = 0.5 * (w[:, j, i+1] - w[:, j, i-1]) # centered difference
            dl = w[:, j, i+1] - w[:, j, i]    # forward  difference (delta+)
            dr = w[:, j, i]   - w[:, j, i-1]  # backward difference (delta-)

            # MC limiter (van Leer 1977, as in LeVeque 2002) -- modeled after Zingales Burgers code
            d1 = 2.0 * minmod(dl, dr)       
            ldeltau = minmod(dc, d1)   # limited slope in primitive space

            A_x, A_y = Jacobian(w[:, j, i], gamma)           # per-cell 8x8 Jacobian
            A_x[:, 5] = 0.0 # this seemed to help with the stripey behavior at tf=1
            slope_L = (np.eye(8) - (dt/dx) * A_x) @ ldeltau       # characteristic-upwinded left slope
            slope_R = (np.eye(8) + (dt/dx) * A_x) @ ldeltau       # characteristic-upwinded right slope
            
            # slope_L = (np.eye(8) - (dt/dx) * A_x)  @ dc     # use unlimited reconstruction for convergence test
            # slope_R = (np.eye(8) + (dt/dx) * A_x)  @ dc     
            

            # Gardiner & Stone (2005): transverse source term for By equation (x-sweep)
            #          | 0                |   rho
            #          | 0                |   ux
            #          | 0                |   uy
            # sigma =  | 0                |   uz
            #          | 0                |   p
            #          | 0                |   bx
            #          |uy*(partial_x bx) |   by
            #          | 0                |   bz
            uy    = w[2, j, i]
            #px_bx = (w[5, j, i+1] - w[5, j, i-1]) / (2*dx)
            px_bx = (Bx_face[j, i] - Bx_face[j, i-1]) / dx
            sigma = np.array([0, 0, 0, 0, 0, 0, uy*px_bx, 0])

            w_L[:, j, i] = w[:, j, i] + 0.5*slope_L + 0.5*dt*sigma
            w_R[:, j, i] = w[:, j, i] - 0.5*slope_R + 0.5*dt*sigma
            
            w_L[5, j, i] = Bx_face[j, i  ]
            w_R[5, j, i] = Bx_face[j, i-1]
            
            


    q_L = conversions.prim_to_cons(w_L, gamma)   # return interface states in conserved form
    q_R = conversions.prim_to_cons(w_R, gamma)
    return q_L, q_R

def PLM_y(q, By_face, dy, dt, gamma):  # takes conservative form
    '''
    Definition: PLM reconstruction along y -- Jacobian-based characteristic slope evolution

    Inputs:     q     : conserved array (DOFs_MHD_2D, ny+2*nghost, nx+2*nghost)
                dx    : cell size in x (used for sigma transverse correction)
                dy    : cell size in y
                dt    : time step
                gamma : adiabatic index

    Outputs:    q_L, q_R : left/right interface states in conserved form
                q_L[:,j,i] = state extrapolated to the top    face of cell (j,i) (from below)
                q_R[:,j,i] = state extrapolated to the bottom face of cell (j,i) (from above)

    Dependencies: cons_to_prim, prim_to_cons, minmod, Jacobian, nghost

    Note: this PLM applies the 8x8 Jacobian A to the MC-limited primitive slope:
        slope_L = (I - (dt/dy)*A) @ minmod(dl, dr)   (upward-biased, characteristic-upwinded)
        slope_R = (I + (dt/dy)*A) @ minmod(dl, dr)   (downward-biased)
    This bakes half-step time evolution into the interface states before the Riemann solve,
    unlike the primitive-space PLM in the 2D Euler code which has no Jacobian term.
    Reference: Wendeln's slide set 1 for Jacobian; Toro 8.2.2 for PLM with characteristic slope.
    '''
    ny_g = q.shape[1]   # ny + 2*nghost
    nx_g = q.shape[2]   # nx + 2*nghost
    w = conversions.cons_to_prim(q, gamma)   # work with primitive form for slope limiting
    w_L = np.zeros_like(q)
    w_R = np.zeros_like(q)

    for j in range(nghost - 1, ny_g - nghost + 1):
        for i in range(nghost - 1, nx_g - nghost + 1):
            dc = 0.5 * (w[:, j+1, i] - w[:, j-1, i]) # centered difference
            dl = w[:, j+1, i] - w[:, j, i]    # forward  difference (delta+, y-direction)
            dr = w[:, j, i]   - w[:, j-1, i]  # backward difference (delta-, y-direction)

            # MC limiter (van Leer 1977, as in LeVeque 2002) -- modeled after Zingales Burgers code
            d1 = 2.0 * minmod(dl, dr)       
            ldeltau = minmod(dc, d1)   # limited slope in primitive space

            A_x, A_y = Jacobian(w[:, j, i], gamma)           # per-cell 8x8 Jacobian
            A_y[:, 6] = 0.0
            slope_L = (np.eye(8) - (dt/dy) * A_y) @ ldeltau       # characteristic-upwinded upward slope
            slope_R = (np.eye(8) + (dt/dy) * A_y) @ ldeltau       # characteristic-upwinded downward slope
            
            # slope_L = (np.eye(8) - (dt/dy) * A_y)  @ dc     # use unlimited reconstruction for convergence test
            # slope_R = (np.eye(8) + (dt/dy) * A_y)  @ dc    

            # Gardiner & Stone (2005): transverse source term for Bx equation (y-sweep)
            #          | 0                |   rho
            #          | 0                |   ux
            #          | 0                |   uy
            # sigma =  | 0                |   uz
            #          | 0                |   p
            #          |ux*(partial_y by) |   bx
            #          | 0                |   by
            #          | 0                |   bz
            ux    = w[1, j, i]
            #py_by = (w[6, j+1, i] - w[6, j-1, i]) / (2*dy)
            py_by = (By_face[j, i] - By_face[j-1, i]) / dy
            sigma = np.array([0, 0, 0, 0, 0, ux*py_by, 0, 0])

            w_L[:, j, i] = w[:, j, i] + 0.5*slope_L + 0.5*dt*sigma
            w_R[:, j, i] = w[:, j, i] - 0.5*slope_R + 0.5*dt*sigma
            
            w_L[6, j, i] = By_face[j, i]
            w_R[6, j, i] = By_face[j-1, i]

    q_L = conversions.prim_to_cons(w_L, gamma)   # return interface states in conserved form
    q_R = conversions.prim_to_cons(w_R, gamma)
    return q_L, q_R

'''
# Updated PLM Reconstruction following Gardiner and Stone's 08 Athena paper

Step 1: Compute the eigenvalues and eigenvectors of the linearized equations 
in the primitive variables using the centered primitive variables in 1D. 
Expressions given in Appendix A
'''
def R_L_evecs_and_evals(w, gamma):
    rho = w[0]
    ux  = w[1]
    uy  = w[2]
    uz  = w[3]
    p   = w[4]
    bx  = w[5]
    by  = w[6]
    bz  = w[7]
    
    # compute sound speed
    a = np.sqrt(gamma*p/rho)
    
    # compute Alfven speed
    ca = np.sqrt((bx**2 + by**2 + bz**2) / rho)   # total Alfven speed
    cax = np.abs(np.sqrt(bx**2 / rho))             # x-component Alfven speed
    # compute fast magnetosonic wave speed
    cf = np.sqrt(0.5*np.abs(a**2 + ca**2 + np.sqrt((a**2 + ca**2)**2 - (4*a**2 * cax**2))))
    
    # compute slow magnetosonic wave speed
    cs = np.sqrt(0.5*np.abs(a**2 + ca**2 - np.sqrt((a**2 + ca**2)**2 - (4*a**2 * cax**2))))
    
    # compute sign of bx
    S = np.sign(bx)
    
    # compute alpha f and s (A16)
    
    if (np.abs(cf**2-cs**2) < 1e-12):
        alpha_f = 1.0
        alpha_s = 0.0
    else:
        alpha_f = np.sqrt(np.maximum(0.0, (a**2 - cs**2)/(cf**2 - cs**2)))
        alpha_s = np.sqrt(np.maximum(0.0, (cf**2 - a**2)/(cf**2 - cs**2)))

    # compute beta y and z (A17)
    if np.sqrt(by**2 + bz**2) < 1e-12:
        beta_y, beta_z = 1.0, 0.0
    else:
        beta_y = by / np.sqrt(by**2 + bz**2)
        beta_z = bz / np.sqrt(by**2 + bz**2)
    
    # compute terms (A13 - A15)
    Cff = cf*alpha_f
    Css = cs*alpha_s
    
    Qf = cf*alpha_f*S
    Qs = cs*alpha_s*S
    
    Af = a*alpha_f*np.sqrt(rho)
    As = a*alpha_s*np.sqrt(rho)
    
    # (19)
    Nf = Ns = 1/(2*a**2)
    
    # write off the vector of right-eigenvectors (columns of this matrix)
    #       rho                       ux                    uy          uz          p              bx            by                  bz
    R = np.array([
        [rho*alpha_f,                 0,              rho*alpha_s,      1,      rho*alpha_s,       0,            0,                rho*alpha_f     ],  #rho
        [-Cff,                        0,                 -Css,          0,          Css,           0,            0,                    Cff         ],  #ux
        [Qs*beta_y,               -beta_z,            -Qf*beta_y,       0,       Qf*beta_y,        0,         beta_z,               -Qs*beta_y     ],  #uy
        [Qs*beta_z,                beta_y,            -Qf*beta_z,       0,       Qf*beta_z,        0,        -beta_y,               -Qs*beta_z     ],  #uz
        [rho*a**2*alpha_f,           0,              rho*a**2*alpha_s,  0,     rho*a**2*alpha_s,   0,            0,              rho*a**2*alpha_f  ],  #p
        [       0,                   0,                   0,            0,            0,           0,            0,                     0          ],  #bx
        [As*beta_y,       -beta_z*S*np.sqrt(rho),    -Af*beta_y,        0,       -Af*beta_y,       0,   -beta_z*S*np.sqrt(rho),      As*beta_y     ],  #by
        [As*beta_z,        beta_y*S*np.sqrt(rho),    -Af*beta_z,        0,       -Af*beta_z,       0,    beta_y*S*np.sqrt(rho),      As*beta_z     ]   #bz
        ], dtype=float)
    

    # write off the vector of left-eigenvectors (rows of this matrix)
    #   rho     ux               uy              uz                  p              bx              by                                bz
    L = np.array([
        [0,  -Nf*Cff,      Nf*Qs*beta_y,     Nf*Qs*beta_z,     Nf*alpha_f/rho,       0,         Nf*As*beta_y/rho,               Nf*As*beta_z/rho     ],  #rho
        [0,     0,           -beta_z/2,       beta_y/2,              0,              0,    -beta_z*S/(2.0*np.sqrt(rho)),   beta_y*S/(2*np.sqrt(rho)) ],  #ux
        [0,  -Ns*Css,     -Ns*Qf*beta_y,    -Ns*Qf*beta_z,     Ns*alpha_s/rho,       0,        -Ns*Af*beta_y/rho,              -Ns*Af*beta_z/rho     ],  #uy
        [1,     0,              0,               0,              -1/(a**2),          0,               0,                              0              ],  #uz
        [0,   Ns*Css,     Ns*Qf*beta_y,     Ns*Qf*beta_z,      Ns*alpha_s/rho,       0,        -Ns*Af*beta_y/rho,              -Ns*Af*beta_z/rho     ],  #p
        [0,     0,              0,               0,                  0,              0,               0,                              0              ],  #bx
        [0,     0,          beta_z/2,        -beta_y/2,              0,              0,   -beta_z*S/(2*np.sqrt(rho)),      beta_y*S/(2*np.sqrt(rho)) ],  #by
        [0,   Nf*Cff,    -Nf*Qs*beta_y,    -Nf*Qs*beta_z,     Nf*alpha_f/rho,        0,         Nf*As*beta_y/rho,               Nf*As*beta_z/rho     ]   #bz
        ], dtype=float)
    
    # compute eigenvalues of A]
    #                 rho       ux        uy    uz     p   bx     by      bz
    lam = np.array([ux - cf, ux - cax, ux - cs, ux, ux,  ux+cs, ux + cax, ux + cf])
    return R, L, lam

'''
- Step 2: Compute the left-, right-, and centered-differences of the primitive variables (eq 36)
- Step 3: project the left, right, and centered differences on the characteristic variables (eq 37)
- Step 4: Apply monotonicity constraints to the differences in the characteristic variables, (eq 38)
- Step 5: Project the monotonized different in the characteristic variables back onto the primitive variables (eq 39)
- Step 6: Compute the left- and right-interface values using the monotonized difference in the primitive variables (eqs 40 and 41)
- Step 7: Perform the characteristic tracing, that is, subtract from the integral performed in step 6 that part of each wave family that does not reach the interface in dt/2 (eqs 42-43)
- Step 7.5: (necessary because we are using an HLL family Riemann solver) Add additional terms to step 8 eqs 42-43, (eqs 44-45)
- Step 8: Convert the left- and right-states in the primitive to the conserved variables, q_L and q_R
'''

def PLM_x_GS08(q, Bx_face, dx, dt, gamma):
    '''
    Definition: PLM reconstruction for 1D MHD, following GS08 Athena paper
    '''
    ny_g = q.shape[1]
    nx_g = q.shape[2]
    w= conversions.cons_to_prim(q, gamma)   # work with primitive form for slope limiting
    w_L = np.zeros_like(q)
    w_R = np.zeros_like(q)

    for j in range(nghost - 1, ny_g - nghost + 1): # loops go left and right 1 of physical domain
        for i in range(nghost - 1, nx_g - nghost + 1):
            ### step 2 ###
            dl = w[:,j,i+1] - w[:,j,i]   # forward  difference (delta+)
            dr = w[:,j,i] - w[:,j,i-1]   # backward difference (delta-)
            dc = 0.5*(w[:,j,i+1] - w[:,j,i-1]) # centered difference
            
            ### step 3 ###
            # get right and left eigenvectors
            R, L, lam = R_L_evecs_and_evals(w[:,j,i], gamma)
            dla = L @ dl
            dra = L @ dr
            dca = L @ dc

            ## step 4 ###
            # MC limiter (van Leer 1977, as in LeVeque 2002) -- modeled after Zingales Burgers code
            d1a = 2.0 * minmod(dla, dra)       
            daMC = minmod(dca, d1a)   # limited slope in primitive space

            ### step 5 ###
            dwMC = R @ daMC
            
            ### step 6 ###
            lam_M = np.max(lam)
            lam_0 = np.min(lam)
            w_hat_L = w[:,j,i] + (0.5 - np.maximum(lam_M,0)*dt/(2*dx)) * dwMC
            w_hat_R = w[:,j,i] - (0.5 - np.minimum(lam_0,0)*dt/(2*dx)) * dwMC

            ### step 7 ###
            sum_L = 0.0
            for alpha in range(len(lam)):
                if lam[alpha]>0:
                    sum_L += ((lam_M - lam[alpha])* np.dot(L[alpha,:], dwMC)) * R[:,alpha]
            sum_R = 0.0
            for alpha in range(len(lam)):
                if lam[alpha]<0:
                    sum_R += ((lam_0 - lam[alpha])* np.dot(L[alpha,:], dwMC)) * R[:,alpha]       
            
            # Gardiner & Stone (2005): transverse source term for By equation (x-sweep)
            #          | 0                |   rho
            #          | 0                |   ux
            #          | 0                |   uy
            # sigma =  | 0                |   uz
            #          | 0                |   p
            #          | 0                |   bx
            #          |uy*(partial_x bx) |   by
            #          | 0                |   bz
            uy    = w[2, j, i]
            px_bx = (Bx_face[j, i] - Bx_face[j, i-1]) / dx
            sigma = np.array([0, 0, 0, 0, 0, 0, uy*px_bx, 0])

            w_L[:,j,i] = w_hat_L + dt/(2*dx) * sum_L + 0.5*dt*sigma
            w_R[:,j,i] = w_hat_R + dt/(2*dx) * sum_R + 0.5*dt*sigma

            ### step 7.5 ###
            delta_w_L = 0.0
            delta_w_R = 0.0
            for alpha in range(len(lam)):
                if lam[alpha]<0:
                    delta_w_L -= dt/(2*dx) * ((lam[alpha] - lam_M)* np.dot(L[alpha,:], dwMC)) * R[:,alpha]
                elif lam[alpha]>0:
                    delta_w_R -= dt/(2*dx) * ((lam[alpha] - lam_0)* np.dot(L[alpha,:], dwMC)) * R[:,alpha]

            w_L[:,j,i] += delta_w_L
            w_R[:,j,i] += delta_w_R

            w_L[5, j, i] = Bx_face[j, i  ]
            w_R[5, j, i] = Bx_face[j, i-1]

        

    ### step 8 ###
    q_L = conversions.prim_to_cons(w_L, gamma)   # return interface states in conserved form
    q_R = conversions.prim_to_cons(w_R, gamma)
    return q_L, q_R

def PLM_y_GS08(q, By_face, dy, dt, gamma):
    '''
    Definition: PLM reconstruction for 1D MHD, following GS08 Athena paper
    '''
    # Permute the array of DOFs instead of remaking 'y' route
    ROT_Y = np.array([0, 2, 3, 1, 4, 6, 7, 5])   # (rho, vy, vz, vx, p, by, bz, bx)    
    ny_g = q.shape[1]
    nx_g = q.shape[2]
    w= conversions.cons_to_prim(q, gamma)   # work with primitive form for slope limiting
    w_L = np.zeros_like(q)
    w_R = np.zeros_like(q)

    for j in range(nghost - 1, ny_g - nghost + 1): # loops go left and right 1 of physical domain
        for i in range(nghost - 1, nx_g - nghost + 1):
            ### step 2 ###
            dl = w[ROT_Y,j+1,i] - w[ROT_Y,j,i]   # forward  difference (delta+)
            dr = w[ROT_Y,j,i] - w[ROT_Y,j-1,i]   # backward difference (delta-)
            dc = 0.5*(w[ROT_Y,j+1,i] - w[ROT_Y,j-1,i]) # centered difference
            
            ### step 3 ###
            # get right and left eigenvectors
            R, L, lam = R_L_evecs_and_evals(w[ROT_Y,j,i], gamma)
            dla = L @ dl
            dra = L @ dr
            dca = L @ dc

            ## step 4 ###
            # MC limiter (van Leer 1977, as in LeVeque 2002) -- modeled after Zingales Burgers code
            d1a = 2.0 * minmod(dla, dra)       
            daMC = minmod(dca, d1a)   # limited slope in primitive space

            ### step 5 ###
            dwMC = R @ daMC
            
            ### step 6 ###
            lam_M = np.max(lam)
            lam_0 = np.min(lam)
            w_hat_L = w[ROT_Y,j,i] + (0.5 - np.maximum(lam_M,0)*dt/(2*dy)) * dwMC
            w_hat_R = w[ROT_Y,j,i] - (0.5 - np.minimum(lam_0,0)*dt/(2*dy)) * dwMC

            ### step 7 ###
            sum_L = 0.0
            for alpha in range(len(lam)):
                if lam[alpha]>0:
                    sum_L += ((lam_M - lam[alpha])* np.dot(L[alpha,:], dwMC)) * R[:,alpha]
            sum_R = 0.0
            for alpha in range(len(lam)):
                if lam[alpha]<0:
                    sum_R += ((lam_0 - lam[alpha])* np.dot(L[alpha,:], dwMC)) * R[:,alpha]       
            
            # Gardiner & Stone (2005): transverse source term for Bx equation (y-sweep)
            #          | 0                |   rho
            #          | 0                |   ux
            #          | 0                |   uy
            # sigma =  | 0                |   uz
            #          | 0                |   p
            #          |ux*(partial_y by) |   bx
            #          | 0                |   by
            #          | 0                |   bz
            ux    = w[1, j, i]
            py_by = (By_face[j, i] - By_face[j-1, i]) / dy
            sigma = np.array([0, 0, 0, 0, 0, 0, 0, ux*py_by])

            w_L[ROT_Y,j,i] = w_hat_L + dt/(2*dy) * sum_L + 0.5*dt*sigma
            w_R[ROT_Y,j,i] = w_hat_R + dt/(2*dy) * sum_R + 0.5*dt*sigma
            
            ### step 7.5 ###
            delta_w_L = 0.0
            delta_w_R = 0.0
            for alpha in range(len(lam)):
                if lam[alpha]<0:
                    delta_w_L -= dt/(2*dy) * ((lam[alpha] - lam_M)* np.dot(L[alpha,:], dwMC)) * R[:,alpha]
                elif lam[alpha]>0:
                    delta_w_R -= dt/(2*dy) * ((lam[alpha] - lam_0)* np.dot(L[alpha,:], dwMC)) * R[:,alpha]
            
            w_L[ROT_Y,j,i] += delta_w_L
            w_R[ROT_Y,j,i] += delta_w_R
            
            
            w_L[6, j, i] = By_face[j, i  ]
            w_R[6, j, i] = By_face[j-1, i]
        

    ### step 8 ###
    q_L = conversions.prim_to_cons(w_L, gamma)   # return interface states in conserved form
    q_R = conversions.prim_to_cons(w_R, gamma)
    return q_L, q_R

def PPM_x_GS08(q, Bx_face, dx, dt, gamma):
    '''
    Definition: PPM reconstruction for 1D MHD, following GS08 Athena paper
    '''
    ny_g = q.shape[1]
    nx_g = q.shape[2]
    w= conversions.cons_to_prim(q, gamma)   # work with primitive form for slope limiting
    w_L = np.zeros_like(q)
    w_R = np.zeros_like(q)
    dwMC = np.zeros_like(q)

    for j in range(nghost - 1, ny_g - nghost + 1): # loops go left and right 1 of physical domain
        for i in range(nghost - 2, nx_g - nghost + 2):
            ### step 2 ###
            dl = w[:,j,i+1] - w[:,j,i]   # forward  difference (delta+)
            dr = w[:,j,i] - w[:,j,i-1]   # backward difference (delta-)
            dc = 0.5*(w[:,j,i+1] - w[:,j,i-1]) # centered difference
            
            ### step 3 ###
            # get right and left eigenvectors
            R, L, lam = R_L_evecs_and_evals(w[:,j,i], gamma)
            dla = L @ dl
            dra = L @ dr
            dca = L @ dc

            ## step 4 ###
            # MC limiter (van Leer 1977, as in LeVeque 2002) -- modeled after Zingales Burgers code
            d1a = 2.0 * minmod(dla, dra)       
            daMC = minmod(dca, d1a)   # limited slope 
            ### step 5 ###
            dwMC[:,j,i] = R @ daMC
    
    for j in range(nghost - 1, ny_g - nghost + 1): # loops go left and right 1 of physical domain
        for i in range(nghost - 1, nx_g - nghost + 1):            
            ### step 6 ###
            w_L_i = (w[:,j,i] + w[:,j,i-1])/2.0 - (dwMC[:,j,i] + dwMC[:,j,i-1])/6.0
            w_R_i = (w[:,j,i+1] + w[:,j,i])/2.0 - (dwMC[:,j,i+1] + dwMC[:,j,i])/6.0
            
            ### step 7 ###
            flat = (w_R_i - w[:,j,i])*(w[:,j,i] - w_L_i)<=0
            w_L_i = np.where(flat, w[:,j,i], w_L_i)
            w_R_i = np.where(flat, w[:,j,i], w_R_i)
            
            d = w_R_i - w_L_i
            w_6_i = 6.0*(w[:,j,i] - 0.5*(w_L_i + w_R_i))
            over_L = d*w_6_i > d*d
            w_L_i = np.where(over_L, 3.0*w[:,j,i]-2.0*w_R_i,w_L_i)
            
            d = w_R_i - w_L_i
            w_6_i = 6.0*(w[:,j,i]-0.5*(w_L_i+w_R_i))
            over_R = d*w_6_i < -d*d
            w_R_i = np.where(over_R, 3.0*w[:,j,i]-2.0*w_L_i,w_R_i)
            
            
            ### step 8 ###
            dwM = w_R_i - w_L_i
            w_6_i = 6*(w[:,j,i] - (w_L_i + w_R_i)/2.0)
            
            ### step 9 ###
            R, L, lam = R_L_evecs_and_evals(w[:,j,i], gamma)
            lam_M = np.max(lam)
            lam_max = max(lam_M,0.0)
            lam_0 = np.min(lam)
            lam_min = min(lam_0,0.0)
            w_hat_L = w_R_i - lam_max*(dt/(2*dx))*(dwM - (1 - lam_max * (2*dt/(3*dx)))*w_6_i)
            w_hat_R = w_L_i + lam_min*(dt/(2*dx))*(dwM + (1 - lam_min * (2*dt/(3*dx)))*w_6_i)
            
            ### step 10 ###
            sum_L = 0.0
            sum_R = 0.0
            for alpha in range(len(lam)):
                A = dt/(2*dx) *(lam_M - lam[alpha])
                B = 1/3 * (dt/dx)**2 * (lam_M**2 - lam[alpha]**2)
                C = dt/(2*dx) * (lam_0 - lam[alpha])
                D = 1/3 * (dt/dx)**2 * (lam_0**2 - lam[alpha]**2)
                if lam[alpha]>0:
                    sum_L += (np.dot(L[alpha,:], A*(dwM-w_6_i)+B*w_6_i)) * R[:,alpha]
                if lam[alpha]<0:
                    sum_R += (np.dot(L[alpha,:], C*(dwM+w_6_i)+D*w_6_i)) * R[:,alpha]       
            
            # Gardiner & Stone (2005): transverse source term for By equation (x-sweep)
            #          | 0                |   rho
            #          | 0                |   ux
            #          | 0                |   uy
            # sigma =  | 0                |   uz
            #          | 0                |   p
            #          | 0                |   bx
            #          |uy*(partial_x bx) |   by
            #          | 0                |   bz
            uy    = w[2, j, i]
            px_bx = (Bx_face[j, i] - Bx_face[j, i-1]) / dx
            sigma = np.array([0, 0, 0, 0, 0, 0, uy*px_bx, 0])

            w_L[:,j,i] = w_hat_L + sum_L + 0.5*dt*sigma
            w_R[:,j,i] = w_hat_R + sum_R + 0.5*dt*sigma

            ### step 10.5 ###
            delta_w_L = 0.0
            delta_w_R = 0.0
            for alpha in range(len(lam)):
                if lam[alpha]<0:
                    delta_w_L -= dt/(2*dx) * ((lam[alpha] - lam_M)* np.dot(L[alpha,:], dwM)) * R[:,alpha]
                elif lam[alpha]>0:
                    delta_w_R -= dt/(2*dx) * ((lam[alpha] - lam_0)* np.dot(L[alpha,:], dwM)) * R[:,alpha]

            w_L[:,j,i] += delta_w_L
            w_R[:,j,i] += delta_w_R

            w_L[5, j, i] = Bx_face[j, i  ]
            w_R[5, j, i] = Bx_face[j, i-1]

        

    ### step 8 ###
    q_L = conversions.prim_to_cons(w_L, gamma)   # return interface states in conserved form
    q_R = conversions.prim_to_cons(w_R, gamma)
    return q_L, q_R

def PPM_y_GS08(q, By_face, dy, dt, gamma):
    '''
    Definition: PPM reconstruction for 1D MHD, following GS08 Athena paper
    '''
    ROT_Y = np.array([0, 2, 3, 1, 4, 6, 7, 5])   # (rho, vy, vz, vx, p, by, bz, bx)    
    ny_g = q.shape[1]
    nx_g = q.shape[2]
    w= conversions.cons_to_prim(q, gamma)   # work with primitive form for slope limiting
    w_L = np.zeros_like(q)
    w_R = np.zeros_like(q)
    dwMC = np.zeros_like(q)

    for j in range(nghost - 2, ny_g - nghost + 2): # loops go left and right 1 of physical domain
        for i in range(nghost - 1, nx_g - nghost + 1):
            ### step 2 ###
            dl = w[ROT_Y,j+1,i] - w[ROT_Y,j,i]   # forward  difference (delta+)
            dr = w[ROT_Y,j,i] - w[ROT_Y,j-1,i]   # backward difference (delta-)
            dc = 0.5*(w[ROT_Y,j+1,i] - w[ROT_Y,j-1,i]) # centered difference
            
            ### step 3 ###
            # get right and left eigenvectors
            R, L, lam = R_L_evecs_and_evals(w[ROT_Y,j,i], gamma)
            dla = L @ dl
            dra = L @ dr
            dca = L @ dc

            ## step 4 ###
            # MC limiter (van Leer 1977, as in LeVeque 2002) -- modeled after Zingales Burgers code
            d1a = 2.0 * minmod(dla, dra)       
            daMC = minmod(dca, d1a)   # limited slope 
            ### step 5 ###
            dwMC[ROT_Y,j,i] = R @ daMC
    
    for j in range(nghost - 1, ny_g - nghost + 1): # loops go left and right 1 of physical domain
        for i in range(nghost - 1, nx_g - nghost + 1):            
            ### step 6 ###
            w_L_i = (w[ROT_Y,j,i] + w[ROT_Y,j-1,i])/2.0 - (dwMC[ROT_Y,j,i] + dwMC[ROT_Y,j-1,i])/6.0
            w_R_i = (w[ROT_Y,j+1,i] + w[ROT_Y,j,i])/2.0 - (dwMC[ROT_Y,j+1,i] + dwMC[ROT_Y,j,i])/6.0
            
            ### step 7 ###
            flat = (w_R_i - w[ROT_Y,j,i])*(w[ROT_Y,j,i] - w_L_i)<=0
            w_L_i = np.where(flat, w[ROT_Y,j,i], w_L_i)
            w_R_i = np.where(flat, w[ROT_Y,j,i], w_R_i)
            
            d = w_R_i - w_L_i
            w_6_i = 6.0*(w[ROT_Y,j,i] - 0.5*(w_L_i + w_R_i))
            over_L = d*w_6_i > d*d
            w_L_i = np.where(over_L, 3.0*w[ROT_Y,j,i]-2.0*w_R_i,w_L_i)
            
            d = w_R_i - w_L_i
            w_6_i = 6.0*(w[ROT_Y,j,i]-0.5*(w_L_i+w_R_i))
            over_R = d*w_6_i < -d*d
            w_R_i = np.where(over_R, 3.0*w[ROT_Y,j,i]-2.0*w_L_i,w_R_i)
            
            
            ### step 8 ###
            dwM = w_R_i - w_L_i
            w_6_i = 6*(w[ROT_Y,j,i] - (w_L_i + w_R_i)/2.0)
            
            ### step 9 ###
            R, L, lam = R_L_evecs_and_evals(w[ROT_Y,j,i], gamma)
            lam_M = np.max(lam)
            lam_max = max(lam_M,0.0)
            lam_0 = np.min(lam)
            lam_min = min(lam_0,0.0)
            w_hat_L = w_R_i - lam_max*(dt/(2*dy))*(dwM - (1 - lam_max * (2*dt/(3*dy)))*w_6_i)
            w_hat_R = w_L_i + lam_min*(dt/(2*dy))*(dwM + (1 - lam_min * (2*dt/(3*dy)))*w_6_i)
            
            ### step 10 ###
            sum_L = 0.0
            sum_R = 0.0
            for alpha in range(len(lam)):
                A = dt/(2*dy) *(lam_M - lam[alpha])
                B = 1/3 * (dt/dy)**2 * (lam_M**2 - lam[alpha]**2)
                C = dt/(2*dy) * (lam_0 - lam[alpha])
                D = 1/3 * (dt/dy)**2 * (lam_0**2 - lam[alpha]**2)
                if lam[alpha]>0:
                    sum_L += (np.dot(L[alpha,:], A*(dwM-w_6_i)+B*w_6_i)) * R[:,alpha]
                if lam[alpha]<0:
                    sum_R += (np.dot(L[alpha,:], C*(dwM+w_6_i)+D*w_6_i)) * R[:,alpha]       
            
            # Gardiner & Stone (2005): transverse source term for Bx equation (y-sweep)
            #          | 0                |   rho
            #          | 0                |   ux
            #          | 0                |   uy
            # sigma =  | 0                |   uz
            #          | 0                |   p
            #          |ux*(partial_y by) |   bx
            #          | 0                |   by
            #          | 0                |   bz
            ux    = w[1, j, i]
            py_by = (By_face[j, i] - By_face[j-1, i]) / dy
            sigma = np.array([0, 0, 0, 0, 0, 0, 0, ux*py_by])
            
            w_L[ROT_Y,j,i] = w_hat_L + sum_L + 0.5*dt*sigma
            w_R[ROT_Y,j,i] = w_hat_R + sum_R + 0.5*dt*sigma

            ### step 10.5 ###
            delta_w_L = 0.0
            delta_w_R = 0.0
            for alpha in range(len(lam)):
                if lam[alpha]<0:
                    delta_w_L -= dt/(2*dy) * ((lam[alpha] - lam_M)* np.dot(L[alpha,:], dwM)) * R[:,alpha]
                elif lam[alpha]>0:
                    delta_w_R -= dt/(2*dy) * ((lam[alpha] - lam_0)* np.dot(L[alpha,:], dwM)) * R[:,alpha]

            w_L[ROT_Y,j,i] += delta_w_L
            w_R[ROT_Y,j,i] += delta_w_R

            w_L[6, j, i] = By_face[j, i  ]
            w_R[6, j, i] = By_face[j-1, i]

        

    ### step 8 ###
    q_L = conversions.prim_to_cons(w_L, gamma)   # return interface states in conserved form
    q_R = conversions.prim_to_cons(w_R, gamma)
    return q_L, q_R

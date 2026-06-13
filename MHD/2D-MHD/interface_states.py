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
    q_L = np.zeros_like(q)
    q_R = np.zeros_like(q)

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
            px_bx = (w[5, j, i+1] - w[5, j, i-1]) / (2*dx)
            sigma = np.array([0, 0, 0, 0, 0, 0, uy*px_bx, 0])

            q_L[:, j, i] = w[:, j, i] + 0.5*slope_L + 0.5*dt*sigma
            q_R[:, j, i] = w[:, j, i] - 0.5*slope_R + 0.5*dt*sigma
            
            q_L[5, j, i] = Bx_face[j, i  ]
            q_R[5, j, i] = Bx_face[j, i-1]
            
            


    q_L = conversions.prim_to_cons(q_L, gamma)   # return interface states in conserved form
    q_R = conversions.prim_to_cons(q_R, gamma)
    return q_L, q_R

def PLM_y(q, By_face, dx, dy, dt, gamma):  # takes conservative form
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
    q_L = np.zeros_like(q)
    q_R = np.zeros_like(q)

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
            py_by = (w[6, j+1, i] - w[6, j-1, i]) / (2*dy)
            sigma = np.array([0, 0, 0, 0, 0, ux*py_by, 0, 0])

            q_L[:, j, i] = w[:, j, i] + 0.5*slope_L + 0.5*dt*sigma
            q_R[:, j, i] = w[:, j, i] - 0.5*slope_R + 0.5*dt*sigma
            
            q_L[6, j, i] = By_face[j, i]
            q_R[6, j, i] = By_face[j-1, i]

    q_L = conversions.prim_to_cons(q_L, gamma)   # return interface states in conserved form
    q_R = conversions.prim_to_cons(q_R, gamma)
    return q_L, q_R

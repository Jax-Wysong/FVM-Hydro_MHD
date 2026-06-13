'''
Initial condition constructors for 2D ideal MHD test problems.

Each test problem is split into two functions:
    - A pointwise function that returns the conserved variable vector at a
    single cell center (x, y).
    - A setup function that loops over the grid, fills the ghosted IC array,
    and initializes the staggered face-centered magnetic field arrays
    (Bx_face, By_face) required by the constrained transport (CT) update.

Conserved variable ordering used throughout this codebase:
    q = [rho, rho*vx, rho*vy, rho*vz, E, Bx, By, Bz]

Face-centered field conventions (Gardiner & Stone 2005, eqs. 16-17):
    Bx_face[j, i]   -- area-averaged Bx at the left/right face of cell (i,j)
                        shape (ny+2*nghost, nx+2*nghost)
    By_face[j, i]   -- area-averaged By at the bottom/top face of cell (i,j)
                        shape (ny+2*nghost, nx+2*nghost)

References:
    Gardiner & Stone (2005)
'''

import numpy as np

#user defined libs
import conversions
import boundary_conditions

def loop_advection(x, y, gamma, A0=1e-3,R=0.3):
    '''
    Definition: returns the conserved variable vector at a single cell center
                (x, y) for the magnetic field loop advection problem
                (GS05 Section 3.3.1 and 5.1).

                The flow carries a circular magnetic field loop.
                The loop is initialized from the
                z-component of the magnetic vector potential:
                    Az = A0*(R - r)   for r <= R
                    Az = 0            for r >  R
                giving:
                    Bx =  dAz/dy = -A0*(y/r)
                    By = -dAz/dx =  A0*(x/r)

    Inputs:     x     : x-coordinate of cell center
                y     : y-coordinate of cell center
                gamma : adiabatic index
                A0    : vector potential amplitude (default 1e-3)
                R     : loop radius (default 0.3)

    Outputs:    q_element : conserved variable vector at (x, y)
                            [rho, rho*vx, rho*vy, rho*vz, E, Bx, By, Bz]

    Dependencies: conversions.prim_to_cons
    '''
    v0     = np.sqrt(5)
    cos_t  = 2.0/v0
    sin_t  = 1.0/v0
    
    # v0     = 1.0
    # cos_t  = 0.0
    # sin_t  = 1.0
    
    rho = 1.0
    p   = 1.0
    ux  = v0*cos_t
    uy  = v0*sin_t
    uz  = 0.0 
    bz  = 0.0
    
    '''
    calculate bx and by from the vector potential,
    A = (0,0,Az)
    therefore, 
    Bx =   \partial_y Az   
    By = - \partial_x Az
    '''
    ## these will be calulated in other function
    ## placeholder for now
    bx = by = 0.0

    w_element = np.array([rho, ux, uy, uz, p, bx, by, bz])
    q_element = conversions.prim_to_cons(w_element,gamma)
    return q_element

def advection_setup(xL, xR, yL, yR, nx, ny, dx, dy, X, Y, gamma, nghost, DOFs):
    '''
    Definition: builds the ghosted initial condition array and staggered
                face-centered magnetic field arrays for the field loop
                advection test (GS05 Section 3.3.1 and 5.1).

                Calls loop_advection() at every interior cell center to fill
                the conserved variable array. The staggered Bx_face and
                By_face arrays are computed by finite-differencing the vector
                potential Az at cell corners, guaranteeing that the initial
                magnetic field is exactly divergence-free on the discrete grid.
                The cell-centered Bx and By stored in the ghosted array are
                then derived as averages of the surrounding face values
                (GS05 eqs. 19-20).

    Inputs:     xL, xR : left and right domain boundaries in x
                yL, yR : left and right domain boundaries in y
                nx, ny : number of interior cells in x and y
                dx, dy : cell widths in x and y
                X      : 1-D array of interior cell-center x-coordinates, shape (nx,)
                Y      : 1-D array of interior cell-center y-coordinates, shape (ny,)
                gamma  : adiabatic index
                nghost : number of ghost cells on each side
                DOFs   : number of conserved variables (8)

    Outputs:    q_ghost_IC_advection :  ghosted conserved variable array,
                                        shape (DOFs, ny+2*nghost, nx+2*nghost)
                Bx_face              : face-centered Bx, shape (ny+2*nghost, nx+2*nghost)
                By_face              : face-centered By, shape (ny+2*nghost, nx+2*nghost)

    Dependencies: loop_advection
    '''
    ####################################################################################################
    # Build Ghosted Initial Condition Array
    ####################################################################################################
    #need vector valued solution vector U = (rho, rho u, rho v, E, bx, by, bz)
    q_IC_advection = np.array([[loop_advection(x, y, gamma) for x in X] for y in Y]) # shape: (ny, nx, DOF)
    q_IC_advection = np.moveaxis(q_IC_advection, -1, 0) # shape: (DOF, ny, nx)
    q_ghost_IC_advection = np.zeros((DOFs, ny + 2*nghost, nx + 2*nghost))
    q_ghost_IC_advection[:, nghost:nghost+ny, nghost:nghost+nx] = q_IC_advection
    # --------------------------------------------------------------------------


    ### need to make and initialize face-centered Bx and By arrays
    A0_loop = 1e-3
    R_loop  = 0.3

    # A_z evaluated at cell corners (xL + i*dx, yL + j*dy)
    x_corn = xL + np.arange(-nghost, nx+nghost+1) * dx   # nx+2*nghost+1 points
    y_corn = yL + np.arange(-nghost, ny+nghost+1) * dy   # ny+2*nghost+1 points
    XX, YY = np.meshgrid(x_corn, y_corn)      # (ny+2*nghost+1, nx+2*nghost+1)
    R_c = np.sqrt(XX**2 + YY**2)
    Az  = np.where(R_c <= R_loop, A0_loop*(R_loop - R_c), 0.0)

    # Bx = dAz/dy at each x-face
    # Bx_face made so that Bx_face[j,i  ] is at the right face of the cell
    #                      Bx_face[j,i-1] is at the left face of the cell
    Bx_face = np.zeros((ny+2*nghost, nx+2*nghost))
    Bx_face[nghost:nghost+ny, nghost:nghost+nx] = (Az[nghost+1:nghost+ny+1, nghost+1:nghost+nx+1] - Az[nghost:nghost+ny, nghost+1:nghost+nx+1]) / dy
    
    #periodic boundary
    Bx_face[:,  :nghost]   = Bx_face[:, -2*nghost:-nghost]   
    Bx_face[:, -nghost:]   = Bx_face[:,   nghost:2*nghost]
    Bx_face[:nghost, : ]   = Bx_face[-2*nghost:-nghost, :]   
    Bx_face[-nghost:,: ]   = Bx_face[nghost:2*nghost,   :]
        
    # By = -dAz/dx at each y-face
    # By_face made so that By_face[j,i  ] is at the top face of the cell
    #                      By_face[j-1,i] is at the bottom face of the cell
    By_face = np.zeros((ny+2*nghost, nx+2*nghost))
    By_face[nghost:nghost+ny, nghost:nghost+nx] = -(Az[nghost+1:nghost+ny+1, nghost+1:nghost+nx+1] - Az[nghost+1:nghost+ny+1, nghost:nghost+nx]) / dx
    
    # periodic boundary
    By_face[:,  :nghost]   = By_face[:, -2*nghost:-nghost]   
    By_face[:, -nghost:]   = By_face[:, nghost:2*nghost]
    By_face[:nghost, :]    = By_face[-2*nghost:-nghost, :]   
    By_face[-nghost:,:]    = By_face[nghost:2*nghost, :]

    # Derive cell-centered B from staggered averages (eqs 19-20)
    q_ghost_IC_advection[5, nghost:nghost+ny, nghost:nghost+nx] = 0.5*(Bx_face[nghost:nghost+ny, nghost-1:nghost+nx-1] + Bx_face[nghost:nghost+ny, nghost:nghost+nx])
    q_ghost_IC_advection[6, nghost:nghost+ny, nghost:nghost+nx] = 0.5*(By_face[nghost-1:nghost+ny-1, nghost:nghost+nx] + By_face[nghost:nghost+ny, nghost:nghost+nx])

    return q_ghost_IC_advection, Bx_face, By_face


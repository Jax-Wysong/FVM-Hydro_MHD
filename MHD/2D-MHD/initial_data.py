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
import CT_update_RK
import Riemann
import plotting
import update_solution

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
    
    # x advection
    # v0     = 2.0
    # cos_t  = 1.0
    # sin_t  = 0.0
    
    # y advection
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
    return w_element

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
    w_fill_hydro = np.array([[loop_advection(x, y, gamma) for x in X] for y in Y]) # shape: (ny, nx, DOF)
    w_fill_hydro = np.moveaxis(w_fill_hydro, -1, 0) # shape: (DOF, ny, nx)
    
    Ny = ny + 2*nghost
    Nx = nx + 2*nghost
    
    w_ghost_IC_advection = np.zeros((DOFs, Ny, Nx))
    w_ghost_IC_advection[:, nghost:nghost+ny, nghost:nghost+nx] = w_fill_hydro
    # --------------------------------------------------------------------------
    

    ### need to make and initialize face-centered Bx and By arrays
    A0_loop = 1e-3
    R_loop  = 0.3

    # A_z evaluated at cell corners (xL + i*dx, yL + j*dy)
    x_corn = xL + (np.arange(Nx + 1) - nghost) * dx   
    y_corn = yL + (np.arange(Ny + 1) - nghost) * dy
    XX, YY = np.meshgrid(x_corn, y_corn)      
    R_c = np.sqrt(XX**2 + YY**2)
    Az  = np.where(R_c <= R_loop, A0_loop*(R_loop - R_c), 0.0)
    
    # Bx = dAz/dy at each x-face
    # Bx_face made so that Bx_face[j,i  ] is at the right face of the cell
    #                      Bx_face[j,i-1] is at the left face of the cell
    Bx_face = np.zeros((Ny, Nx+1))
    Bx_face[nghost:nghost+ny, nghost:nghost+nx+1] = (Az[nghost+1:nghost+ny+1, nghost:nghost+nx+1] - Az[nghost:nghost+ny, nghost:nghost+nx+1]) / dy
        
    # By = -dAz/dx at each y-face
    # By_face made so that By_face[j,i  ] is at the top face of the cell
    #                      By_face[j-1,i] is at the bottom face of the cell
    By_face = np.zeros((Ny+1, Nx))
    By_face[nghost:nghost+ny+1, nghost:nghost+nx] = -(Az[nghost:nghost+ny+1, nghost+1:nghost+nx+1] - Az[nghost:nghost+ny+1, nghost:nghost+nx]) / dx
    
    # Fill periodic ghost zones for staggered fields
    boundary_conditions.periodic_bc(Bx_face, ny, nx, 'Bx-cellFace')
    boundary_conditions.periodic_bc(By_face, ny, nx, 'By-cellFace')
    
    # Derive cell-centered B from staggered averages (eqs 19-20)
    w_ghost_IC_advection[5, :, :] = 0.5 * (Bx_face[:, :-1] + Bx_face[:, 1:])
    w_ghost_IC_advection[6, :, :] = 0.5 * (By_face[:-1, :] + By_face[1:, :])
    
    boundary_conditions.periodic_bc(w_ghost_IC_advection, ny, nx, 'q-cellCenter')

    # converting to conservative form will apply a calculation
    # that creates E. Since we filled w with the derived center mag fields
    # this will mean that E inherently is made properly
    
    q_ghost_IC_advection = conversions.prim_to_cons(w_ghost_IC_advection, gamma)
        

    print(f"Step: 0, Time: 0.0")
    divB = (
        (Bx_face[:, 1:] - Bx_face[:, :-1]) / dx
        +
        (By_face[1:, :] - By_face[:-1, :]) / dy
    )
    divB_phys = divB[nghost:nghost+ny, nghost:nghost+nx]

    print(f"  max physical face |div B| = {np.abs(divB_phys).max():.3e}")
       
    return q_ghost_IC_advection, Bx_face, By_face


def smooth_mhd_vortex(x, y, gamma=5/3, t=0.0, kappa=1.0, xi=1.0):
    # background state
    rho = np.ones_like(x)
    vx0, vy0, vz0 = 1.0, 1.0, 0.0
    p0 = 1.0

    # vortex coordinates
    X = x - vx0*t
    Y = y - vy0*t
    r2 = X**2 + Y**2

    g = np.exp(0.5*(1.0 - r2))

    # velocity perturbation
    dvx = -(kappa/(2.0*np.pi)) * Y * g
    dvy =  (kappa/(2.0*np.pi)) * X * g

    vx = vx0 + dvx
    vy = vy0 + dvy
    vz = np.zeros_like(x)

    # placeholders
    Bx = np.zeros_like(x)
    By = np.zeros_like(x)
    # defined this way
    Bz = np.zeros_like(x)

    # pressure perturbation
    dp = ((xi**2*(1.0 - r2) - kappa**2)
          / (8.0*np.pi**2)) * np.exp(1.0 - r2)

    # need this later
    p = p0 + dp

    w_element = np.array([
        rho,
        vx,
        vy,
        vz,
        p,
        Bx,
        By,
        Bz
    ])

    return w_element


def smooth_mhd_vortex_setup(xL, xR, yL, yR, nx, ny, dx, dy, X, Y, gamma, nghost, DOFs, xi=1.0):
    ####################################################################################################
    # Build Ghosted Initial Condition Array
    ####################################################################################################
    #need vector valued solution vector U = (rho, rho u, rho v, E, bx, by, bz)
    w_fill_hydro = np.array([[smooth_mhd_vortex(x, y, gamma) for x in X] for y in Y]) # shape: (ny, nx, DOF)
    w_fill_hydro = np.moveaxis(w_fill_hydro, -1, 0) # shape: (DOF, ny, nx)
    
    Ny = ny + 2*nghost
    Nx = nx + 2*nghost
    
    w_ghost_IC_smooth_vortex = np.zeros((DOFs, Ny, Nx))
    w_ghost_IC_smooth_vortex[:, nghost:nghost+ny, nghost:nghost+nx] = w_fill_hydro
    # --------------------------------------------------------------------------
    

    ### need to make and initialize face-centered Bx and By arrays
    A0_amp = xi/(2*np.pi)

    # A_z evaluated at cell corners (xL + i*dx, yL + j*dy)
    x_corn = xL + (np.arange(Nx + 1) - nghost) * dx   
    y_corn = yL + (np.arange(Ny + 1) - nghost) * dy
    XX, YY = np.meshgrid(x_corn, y_corn)      
    R2_c = XX**2 + YY**2
    Az  = A0_amp * np.exp(0.5*(1-R2_c))
    
    # Bx = dAz/dy at each x-face
    # Bx_face made so that Bx_face[j,i  ] is at (j, i+1/2)
    #                      Bx_face[j,i-1] is at (j, i-1/2)
    Bx_face = np.zeros((Ny, Nx+1))
    Bx_face[nghost:nghost+ny, nghost:nghost+nx+1] = (Az[nghost+1:nghost+ny+1, nghost:nghost+nx+1] - Az[nghost:nghost+ny, nghost:nghost+nx+1]) / dy
        
    # By = -dAz/dx at each y-face
    # By_face made so that By_face[j,i  ] is at the top face of the cell
    #                      By_face[j-1,i] is at the bottom face of the cell
    By_face = np.zeros((Ny+1, Nx))
    By_face[nghost:nghost+ny+1, nghost:nghost+nx] = -(Az[nghost:nghost+ny+1, nghost+1:nghost+nx+1] - Az[nghost:nghost+ny+1, nghost:nghost+nx]) / dx
    
    # Fill periodic ghost zones for staggered fields
    boundary_conditions.periodic_bc(Bx_face, ny, nx, 'Bx-cellFace')
    boundary_conditions.periodic_bc(By_face, ny, nx, 'By-cellFace')
    
    # Derive cell-centered B from staggered averages (eqs 19-20)
    w_ghost_IC_smooth_vortex[5, :, :] = 0.5 * (Bx_face[:, :-1] + Bx_face[:, 1:])
    w_ghost_IC_smooth_vortex[6, :, :] = 0.5 * (By_face[:-1, :] + By_face[1:, :])

    boundary_conditions.periodic_bc(w_ghost_IC_smooth_vortex, ny, nx, 'q-cellCenter')

    # converting to conservative form will apply a calculation
    # that created E. Since we filled w with the derived center mag fields
    # this will mean that E inherently is made properly
    
    q_ghost_IC_smooth_vortex = conversions.prim_to_cons(w_ghost_IC_smooth_vortex, gamma)
        
    print(f"Step: 0, Time: 0.0")
    divB = (
        (Bx_face[:, 1:] - Bx_face[:, :-1]) / dx
        +
        (By_face[1:, :] - By_face[:-1, :]) / dy
    )
    divB_phys = divB[nghost:nghost+ny, nghost:nghost+nx]

    print(f"  max physical face |div B| = {np.abs(divB_phys).max():.3e}")
        
    return q_ghost_IC_smooth_vortex, Bx_face, By_face


def smooth_vortex_convergence_test(nghost, DOFs, CFL, tf, gamma):
    
    xL      = -5.0;  xR  = 5.0
    yL      = -5.0;  yR  = 5.0

    N_list = [16,32,64,128]
    
    # Columns: rho, ux, Bx, p
    L2 = np.zeros((len(N_list), 4))
        
    for i, Ni in enumerate(N_list):
        nx      = Ni
        ny      = Ni

        dx = (xR - xL) / nx
        dy = (yR - yL) / ny

        X  = xL + (np.arange(nx) + 0.5) * dx
        Y  = yL + (np.arange(ny) + 0.5) * dy
        
        q_ghost_IC_smooth_vortex, \
                Bx_face, By_face    = smooth_mhd_vortex_setup(xL, xR, yL, yR, nx, ny, dx, dy, X, Y, gamma, nghost, DOFs)

        q_init = q_ghost_IC_smooth_vortex.copy()
        Bx_init = Bx_face.copy()
        By_init = By_face.copy()

        q_sol_smooth_vortex, q0_smooth_vortex, all_solns_smooth_vortex, all_t_smooth_vortex = update_solution.evolve(
                q_init, nghost, DOFs, nx, ny, dx, dy, CFL, tf, gamma, Bx_init, By_init, Riemann.HLLD, integrator='RK2', limiter='none')
                    
        w_sol = conversions.cons_to_prim(q_sol_smooth_vortex, gamma)
        w0_sol = conversions.cons_to_prim(q0_smooth_vortex, gamma)
        

        rho_err = w_sol[0] - w0_sol[0]
        ux_err  = w_sol[1] - w0_sol[1]
        Bx_err  = w_sol[5] - w0_sol[5]
        p_err   = w_sol[4] - w0_sol[4]

        # discrete L2 norm
        L2[i, 0] = np.sqrt(dx * dy * np.sum(rho_err**2))
        L2[i, 1] = np.sqrt(dx * dy * np.sum(ux_err**2))
        L2[i, 2] = np.sqrt(dx * dy * np.sum(Bx_err**2))
        L2[i, 3] = np.sqrt(dx * dy * np.sum(p_err**2))

        print(f"\n\nN = {Ni} x {Ni}")
        print(f"  rho L2 = {L2[i,0]:.6e}")
        print(f"  ux  L2 = {L2[i,1]:.6e}")
        print(f"  Bx  L2 = {L2[i,2]:.6e}")
        print(f"  p   L2 = {L2[i,3]:.6e}\n\n")

        if i > 0:
            order_rho = np.log2(L2[i-1, 0] / L2[i, 0])
            order_ux  = np.log2(L2[i-1, 1] / L2[i, 1])
            order_Bx  = np.log2(L2[i-1, 2] / L2[i, 2])
            order_p   = np.log2(L2[i-1, 3] / L2[i, 3])

            print(f"\nOrder from {N_list[i-1]} to {Ni}:")
            print(f"  rho order: {order_rho:.3f}")
            print(f"  ux  order: {order_ux:.3f}")
            print(f"  Bx  order: {order_Bx:.3f}")
            print(f"  p   order: {order_p:.3f}")

    return L2

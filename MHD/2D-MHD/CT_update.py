import numpy as np

# user defined libraries
from config import nghost, DOFs
import conversions
import boundary_conditions


# F^{n+1/2}_{x, i+1/2, j}, F^{n+1/2}_{x, i-1/2, j}
# F^{n+1/2}_{y, i, j+1/2}, F^{n+1/2}_{y, i, j-1/2}
# All in conserved form
def GS_step_3(Fx_p, Fx_m, Fy_p, Fy_m, nx, ny, dx, dy, dt, q, gamma): 
    '''
    -------------------------- ** Step (3) of G&S (2005)** -----------------------------------
    Using any of the three EMF CT algorithms, integrate the face centered fluxes to the
    grid cell corner as described in Section 4.1.1. We are using the EMFz^c algorithm 
    (eqs (41) and (50))
    ------------------------------------------------------------------------------------------
    Before we can apply eqs 41 and 50, we need to get our face centered EMF from our face centered fluxes.
    Recall that EMF = - v x B (mentioned at the beginning of 2.2). Thus, we have the relations:
    
    F^x_{By} = vx*By - vy*Bx = -EMF_z
    F^y_{Bx} = vy*Bx - vx*By = +EMF_z
    
    where the flux terms are just by definition those terms.
    
    This implies that
    
    e_{Bx}  F^{n+1/2}_{y,i,j±1/2} = +EMF^{n+1/2}_{z,i,j±1/2}
                            Fy_p[5]=  EMF^{n+1/2}_{z,i,j+1/2}
                            Fy_m[5]=  EMF^{n+1/2}_{z,i,j-1/2}
    e_{By}  F^{n+1/2}_{x,i±1/2,j} = -EMF^{n+1/2}_{z,i±1/2,j}
                            Fx_p[6]= -EMF^{n+1/2}_{z,i+1/2,j}
                            Fx_m[6]= -EMF^{n+1/2}_{z,i-1/2,j}
    
    so to get the face centered EMFs, we just need to access the face-centered fluxes at the appropriate places
    '''
    EMFz_x_p = -Fx_p[6] #EMF^{n+1/2}_{z,i+1/2,j}
    EMFz_x_m = -Fx_m[6] #EMF^{n+1/2}_{z,i-1/2,j}
    
    EMFz_y_p =  Fy_p[5] #EMF^{n+1/2}_{z,i,j+1/2}
    EMFz_y_m =  Fy_m[5] #EMF^{n+1/2}_{z,i,j-1/2}
        
    '''
    Now, we need to use equation (41) to find the cell-corner EMFs, EMF^{n+1/2}_{z,i±1/2,j±1/2}
    But before we can do that, we need to use equation (50) to find the upwinded partial derivatives,
    where the partial derivatives at 1/4 and 3/4 indexes are found with equation (45).
    However, in order to do that, we will need the EMFs at cell centers found by using the 
    current solution values pertaining to velocity and magnetic field.
    '''
    # We use the primative form for this calculation as all its values are from cell centers
    w = conversions.cons_to_prim(q, gamma)
    ux = w[1]
    uy = w[2]
    Bx = w[5]
    By = w[6]
    # EMFz = vy*Bx - vx*By
    EMFz_cc = uy*Bx - ux*By
    
    #### initializing arrays for corner EMF (think nodes not elements)
    EMFz_corner = np.zeros((ny+2*nghost+1,nx+2*nghost+1)) 
    
    for j in range(nghost, nghost+ny+1):
        for i in range(nghost, nghost+nx+1):
            # Compute physical array corners
            # Ghost boundary corners are filled by periodic copy below, matching
            # the same pattern used by periodic_bc for q and by the ghost face
            # assignments for Bx_face/By_face.

            # equation (50) gives us the piecewise relations used in the following
            ########################################
            # finding
            # (partial_y EMFz)_{i+1/2,j+1/4}
            ########################################
            ux_p1 = Fx_p[0,j,i] # = rho * ux_{x,i+1/2,j} rho (should be) > 0
            if  np.sign(ux_p1) > 0:
                dEz_dy_xp_quarter = 2.0*(EMFz_y_p[j,i] - EMFz_cc[j,i])/dy # Equation (45)
            elif np.sign(ux_p1) < 0:
                dEz_dy_xp_quarter = 2.0*(EMFz_y_p[j,i+1] - EMFz_cc[j,i+1])/dy
            else:
                dEz_dy_xp_quarter = 0.5*(2.0*(EMFz_y_p[j,i] - EMFz_cc[j,i])/dy + 2.0*(EMFz_y_p[j,i+1] - EMFz_cc[j,i+1])/dy)
            
            ######################################## 
            # finding 
            # (partial_y EMFz)_{i+1/2,j+3/4}
            ########################################
            ux_p2 = Fx_p[0,j+1,i] # rho * ux_{x,i+1/2,j+1}
            if np.sign(ux_p2) > 0:
                dEz_dy_xp_three_quarter = 2.0*(EMFz_cc[j+1,i] - EMFz_y_p[j,i])/dy
            elif np.sign(ux_p2) < 0:
                dEz_dy_xp_three_quarter = 2.0*(EMFz_cc[j+1,i+1] - EMFz_y_p[j,i+1])/dy
            else:
                dEz_dy_xp_three_quarter = 0.5*(2.0*(EMFz_cc[j+1,i] - EMFz_y_p[j,i])/dy + 2.0*(EMFz_cc[j+1,i+1] - EMFz_y_p[j,i+1])/dy)
            
            ######################################## 
            # finding 
            # (partial_x EMFz)_{i+1/4,j+1/2}
            ########################################
            uy_p1 = Fy_p[0,j,i] # rho * uy_{y,i,j+1/2} used for upwind calculation, rho (should be) > 0
            if  np.sign(uy_p1) > 0:
                dEz_dx_yp_quarter = 2.0*(EMFz_x_p[j,i] - EMFz_cc[j,i])/dx # Equation (45)
            elif np.sign(uy_p1) < 0:
                dEz_dx_yp_quarter = 2.0*(EMFz_x_p[j+1,i] - EMFz_cc[j+1,i])/dx
            else:
                dEz_dx_yp_quarter = 0.5*(2.0*(EMFz_x_p[j,i] - EMFz_cc[j,i])/dx + 2.0*(EMFz_x_p[j+1,i] - EMFz_cc[j+1,i])/dx)
            
            ######################################## 
            # finding 
            # (partial_x EMFz)_{i+3/4,j+1/2}
            ########################################
            uy_p2 = Fy_p[0,j,i+1] # rho * uy_{y,i+1,j+1/2}
            if np.sign(uy_p2) > 0:
                dEz_dx_yp_three_quarter = 2.0*(EMFz_cc[j,i+1] - EMFz_x_p[j,i])/dx
            elif np.sign(uy_p2) < 0:
                dEz_dx_yp_three_quarter = 2.0*(EMFz_cc[j+1,i+1] - EMFz_x_p[j+1,i])/dx
            else:
                dEz_dx_yp_three_quarter = 0.5*(2.0*(EMFz_cc[j,i+1] - EMFz_x_p[j,i])/dx + 2.0*(EMFz_cc[j+1,i+1] - EMFz_x_p[j+1,i])/dx)
            
            
            EMFz_corner[j,i] = (
                0.25*(EMFz_x_p[j,i] + EMFz_x_p[j+1,i] + EMFz_y_p[j,i] + EMFz_y_p[j,i+1]) \
                + dy/8*(dEz_dy_xp_quarter - dEz_dy_xp_three_quarter) \
                + dx/8*(dEz_dx_yp_quarter - dEz_dx_yp_three_quarter)
            ) # Equation (41)

    # Periodic boundaries
    EMFz_corner[:,  :nghost]   = EMFz_corner[:, -2*nghost-1:-nghost-1]   
    EMFz_corner[:, -nghost:]   = EMFz_corner[:,   nghost+1:2*nghost+1]
    EMFz_corner[:nghost, : ]   = EMFz_corner[-2*nghost-1:-nghost-1, :]   
    EMFz_corner[-nghost:,: ]   = EMFz_corner[nghost+1:2*nghost+1,   :]
        
    
    '''
    Now we can finally get the cell cornered Fluxes from equations (64) - (67)
    which will become the input to step 4!
    '''
    # initializing the flux corner arrays
    Fy_L_corner_pp = np.zeros((DOFs, ny+2*nghost+1, nx+2*nghost+1)) # F^{L}_{y,i+1/2,j+1/2}
    Fy_R_corner_pp = np.zeros_like(Fy_L_corner_pp)                  # F^{R}_{y,i+1/2,j+1/2}
    Fx_L_corner_pp = np.zeros_like(Fy_L_corner_pp)                  # F^{L}_{x,i+1/2,j+1/2}
    Fx_R_corner_pp = np.zeros_like(Fy_L_corner_pp)                  # F^{R}_{x,i+1/2,j+1/2}
    
    Fy_L_corner_pm = np.zeros_like(Fy_L_corner_pp)                  # F^{L}_{y,i+1/2,j-1/2}
    Fx_L_corner_mp = np.zeros_like(Fy_L_corner_pp)                  # F^{L}_{x,i-1/2,j+1/2}
    Fy_R_corner_pm = np.zeros_like(Fy_L_corner_pp)                  # F^{R}_{y,i+1/2,j-1/2}
    Fx_R_corner_mp = np.zeros_like(Fy_L_corner_pp)                  # F^{R}_{y,i-1/2,j+1/2}
    for j in range(nghost,nghost+ny+1):
        for i in range(nghost,nghost+nx+1):
            
            Fy_L_corner_pp[:,j,i] = Fy_p[:,j,i]
            Fy_L_corner_pp[5,j,i] += (EMFz_corner[j,i] - EMFz_y_p[j,i])
            
            Fy_R_corner_pp[:,j,i] = Fy_p[:,j,i+1]
            Fy_R_corner_pp[5,j,i] += (EMFz_corner[j,i] - EMFz_y_p[j,i+1])
            
            Fx_L_corner_pp[:,j,i] = Fx_p[:,j,i]
            Fx_L_corner_pp[6,j,i] -= (EMFz_corner[j,i] - EMFz_x_p[j,i])
            
            Fx_R_corner_pp[:,j,i] = Fx_p[:,j+1,i]
            Fx_R_corner_pp[6,j,i] -= (EMFz_corner[j,i] - EMFz_x_p[j+1,i])
            # ------------------------------------------------------------------- #
            Fy_L_corner_pm[:,j,i] = Fy_m[:,j,i]
            Fy_L_corner_pm[5,j,i] += (EMFz_corner[j-1,i] - EMFz_y_m[j,i])
            
            Fx_L_corner_mp[:,j,i] = Fx_m[:,j,i]
            Fx_L_corner_mp[6,j,i] -= (EMFz_corner[j,i-1] - EMFz_x_m[j,i])
            
            Fy_R_corner_pm[:,j,i] = Fy_m[:,j,i+1]
            Fy_R_corner_pm[5,j,i] += (EMFz_corner[j-1,i] - EMFz_y_m[j,i+1])
            
            Fx_R_corner_mp[:,j,i] = Fx_m[:,j+1,i]
            Fx_R_corner_mp[6,j,i] -= (EMFz_corner[j,i-1] - EMFz_x_m[j+1,i])
    # Periodic boundaries
    Fy_L_corner_pp[:,:,  :nghost]   = Fy_L_corner_pp[:,:, -2*nghost-1:-nghost-1]   
    Fy_L_corner_pp[:,:, -nghost:]   = Fy_L_corner_pp[:,:,   nghost+1:2*nghost+1]
    Fy_L_corner_pp[:,:nghost, : ]   = Fy_L_corner_pp[:,-2*nghost-1:-nghost-1, :]   
    Fy_L_corner_pp[:,-nghost:,: ]   = Fy_L_corner_pp[:,nghost+1:2*nghost+1,   :]

    Fy_R_corner_pp[:,:,  :nghost]   = Fy_R_corner_pp[:,:, -2*nghost-1:-nghost-1]   
    Fy_R_corner_pp[:,:, -nghost:]   = Fy_R_corner_pp[:,:,   nghost+1:2*nghost+1]
    Fy_R_corner_pp[:,:nghost, : ]   = Fy_R_corner_pp[:,-2*nghost-1:-nghost-1, :]   
    Fy_R_corner_pp[:,-nghost:,: ]   = Fy_R_corner_pp[:,nghost+1:2*nghost+1,   :]
    
    Fx_L_corner_pp[:,:,  :nghost]   = Fx_L_corner_pp[:,:, -2*nghost-1:-nghost-1]   
    Fx_L_corner_pp[:,:, -nghost:]   = Fx_L_corner_pp[:,:,   nghost+1:2*nghost+1]
    Fx_L_corner_pp[:,:nghost, : ]   = Fx_L_corner_pp[:,-2*nghost-1:-nghost-1, :]   
    Fx_L_corner_pp[:,-nghost:,: ]   = Fx_L_corner_pp[:,nghost+1:2*nghost+1,   :]
    
    Fx_R_corner_pp[:,:,  :nghost]   = Fx_R_corner_pp[:,:, -2*nghost-1:-nghost-1]   
    Fx_R_corner_pp[:,:, -nghost:]   = Fx_R_corner_pp[:,:,   nghost+1:2*nghost+1]
    Fx_R_corner_pp[:,:nghost, : ]   = Fx_R_corner_pp[:,-2*nghost-1:-nghost-1, :]   
    Fx_R_corner_pp[:,-nghost:,: ]   = Fx_R_corner_pp[:,nghost+1:2*nghost+1,   :]
    
    Fy_L_corner_pm[:,:,  :nghost]   = Fy_L_corner_pm[:,:, -2*nghost-1:-nghost-1]   
    Fy_L_corner_pm[:,:, -nghost:]   = Fy_L_corner_pm[:,:,   nghost+1:2*nghost+1]
    Fy_L_corner_pm[:,:nghost, : ]   = Fy_L_corner_pm[:,-2*nghost-1:-nghost-1, :]   
    Fy_L_corner_pm[:,-nghost:,: ]   = Fy_L_corner_pm[:,nghost+1:2*nghost+1,   :]

    Fx_L_corner_mp[:,:,  :nghost]   = Fx_L_corner_mp[:,:, -2*nghost-1:-nghost-1]   
    Fx_L_corner_mp[:,:, -nghost:]   = Fx_L_corner_mp[:,:,   nghost+1:2*nghost+1]
    Fx_L_corner_mp[:,:nghost, : ]   = Fx_L_corner_mp[:,-2*nghost-1:-nghost-1, :]   
    Fx_L_corner_mp[:,-nghost:,: ]   = Fx_L_corner_mp[:,nghost+1:2*nghost+1,   :] 
    
    Fy_R_corner_pm[:,:,  :nghost]   = Fy_R_corner_pm[:,:, -2*nghost-1:-nghost-1]   
    Fy_R_corner_pm[:,:, -nghost:]   = Fy_R_corner_pm[:,:,   nghost+1:2*nghost+1]
    Fy_R_corner_pm[:,:nghost, : ]   = Fy_R_corner_pm[:,-2*nghost-1:-nghost-1, :]   
    Fy_R_corner_pm[:,-nghost:,: ]   = Fy_R_corner_pm[:,nghost+1:2*nghost+1,   :] 
    
    Fx_R_corner_mp[:,:,  :nghost]   = Fx_R_corner_mp[:,:, -2*nghost-1:-nghost-1]   
    Fx_R_corner_mp[:,:, -nghost:]   = Fx_R_corner_mp[:,:,   nghost+1:2*nghost+1]
    Fx_R_corner_mp[:,:nghost, : ]   = Fx_R_corner_mp[:,-2*nghost-1:-nghost-1, :]   
    Fx_R_corner_mp[:,-nghost:,: ]   = Fx_R_corner_mp[:,nghost+1:2*nghost+1,   :] 
    
    return EMFz_corner, Fy_L_corner_pp, Fy_R_corner_pp, Fx_L_corner_pp, Fx_R_corner_pp, Fy_L_corner_pm, Fx_L_corner_mp, Fy_R_corner_pm, Fx_R_corner_mp

def GS_step_4(Fy_L_corner_pp, Fy_R_corner_pp,\
            Fx_L_corner_pp, Fx_R_corner_pp,\
            Fy_L_corner_pm, Fx_L_corner_mp,\
            Fy_R_corner_pm, Fx_R_corner_mp,\
            q_x_L_hat, q_x_R_hat,\
            q_y_L_hat, q_y_R_hat,\
            nx,ny,dx,dy,dt,q,gamma,\
            Bx_face, By_face): 
    ############################################################################################
    # -------------------------- ** Step (4) of G&S (2005)** -----------------------------------
    # Compute the four updated interface states via Eqs. (60-63) with the source 
    # terms detailed in Section 4.1.2
    ############################################################################################
    
    q_x_L = q_x_L_hat.copy()
    q_x_R = q_x_R_hat.copy()
    q_y_L = q_y_L_hat.copy()
    q_y_R = q_y_R_hat.copy()
    # need cell centered variables at time n for Sx/y
    w = conversions.cons_to_prim(q, gamma)
    uz = w[3]
    bx = w[5]
    by = w[6]
    bz = w[7]
    
    
    
    for j in range(nghost, nghost+ny):
        for i in range(nghost, nghost+nx):
            
            # source term Sx at (i,j) 
            dBx_dx = (Bx_face[j,i] - Bx_face[j,i-1])/dx
            Sx = np.array([0,
                        bx[j,i], by[j,i], bz[j,i],
                        bz[j,i]*uz[j,i],
                        0, 0,
                        uz[j,i]])*dBx_dx
            
            # source term Sx at (i+1,j)
            dBx_dx_ip1j = (Bx_face[j,i+1] - Bx_face[j,i])/dx
            Sx_ip1j = np.array([0,
                                bx[j,i+1], by[j,i+1], bz[j,i+1],
                                bz[j,i+1]*uz[j,i+1],
                                0,0,
                                uz[j,i+1]])*dBx_dx_ip1j
            
            # source term Sy at (i,j)
            dBy_dy = (By_face[j,i] - By_face[j-1,i])/dy
            Sy  = np.array([0,
                            bx[j, i], by[j, i], bz[j, i],
                            bz[j, i]*uz[j, i],
                            0, 0,
                            uz[j, i]]) * dBy_dy
            
            # Source term Sy at cell (i, j+1)  for eq (63)
            dBy_dy_ijp1 = (By_face[j+1, i] - By_face[j, i]) / dy
            Sy_ijp1 = np.array([0,
                                bx[j+1, i], by[j+1, i], bz[j+1, i],
                                bz[j+1, i]*uz[j+1, i],
                                0, 0,
                                uz[j+1, i]]) * dBy_dy_ijp1
            
            
            #eqs (60) - (63)
            q_x_L[:,j,i  ] = q_x_L_hat[:,j,i  ] - 0.5*(dt/dy)*(Fy_L_corner_pp[:,j,i] - Fy_L_corner_pm[:,j,i]) + dt/2 * Sx
            q_x_R[:,j,i+1] = q_x_R_hat[:,j,i+1] - 0.5*(dt/dy)*(Fy_R_corner_pp[:,j,i] - Fy_R_corner_pm[:,j,i]) + dt/2 * Sx_ip1j
            q_y_L[:,j,i  ] = q_y_L_hat[:,j,i  ] - 0.5*(dt/dx)*(Fx_L_corner_pp[:,j,i] - Fx_L_corner_mp[:,j,i]) + dt/2 * Sy
            q_y_R[:,j+1,i] = q_y_R_hat[:,j+1,i] - 0.5*(dt/dx)*(Fx_R_corner_pp[:,j,i] - Fx_R_corner_mp[:,j,i]) + dt/2 * Sy_ijp1
    
    # apply the boundary conditions
    boundary_conditions.periodic_bc(q_x_L, nghost)
    boundary_conditions.periodic_bc(q_x_R, nghost)
    boundary_conditions.periodic_bc(q_y_L, nghost)
    boundary_conditions.periodic_bc(q_y_R, nghost)
    
    return q_x_L, q_x_R, q_y_L, q_y_R

def GS_step_5(q_x_L, q_x_R, q_y_L, q_y_R, Bx_face, By_face, nx, ny, gamma, Riemann):
    ############################################################################################
    # -------------------------- ** Step (5) of G&S (2005)** -----------------------------------
    # Compute the x and y interface fluxes associated with the interface states
    # found from step (4) via a Riemann solver
    ############################################################################################
    

    #################################
    # Get F^{n+1/2}_{y, i, j + 1/2}
    #################################
    F_y_half = np.zeros_like(q_x_L)
    for j in range(nghost-1, nghost+ny):    
        for i in range(nghost-1, nghost+nx+1):
            # For y-fluxes - inject By_face[j,i] (top face of cell j,i) into both states
            q_y_L[6, j, i  ] = By_face[j, i]
            q_y_R[6, j+1, i] = By_face[j, i]
            # F^{n+1/2}_{y,i,j+1/2}
            F_y_half[:,j,i] = Riemann(q_y_L[:,j,i], q_y_R[:,j+1,i  ],gamma, 'y')
    

    
    #################################
    # Get F^{n+1/2}_{x, i + 1/2, j}
    #################################
    F_x_half = np.zeros_like(q_x_L)
    for i in range(nghost-1, nghost+nx):  
        for j in range(nghost-1, nghost+ny+1):
            # For x-fluxes - inject Bx_face[j,i] (right face of cell j,i) into both states
            q_x_L[5, j, i  ] = Bx_face[j, i]
            q_x_R[5, j, i+1] = Bx_face[j, i]
            # F^{n+1/2}_{x,i+1/2,j}
            F_x_half[:,j,i] = Riemann(q_x_L[:,j,i  ], q_x_R[:,j,i+1],gamma, 'x') 
 

    
    # apply the boundary conditions
    boundary_conditions.periodic_bc(F_x_half, nghost)
    boundary_conditions.periodic_bc(F_y_half, nghost)
    
    return F_x_half, F_y_half

def GS_step_6(Fx, Fy, ux, uy, Bx_face, By_face, nx, ny, dx, dy, dt, gamma): 
    ############################################################################################
    # -------------------------- ** Step (6) of G&S (2005)** -----------------------------------
    # Compute the grid cell corner centered EMF using any of the three EMF CT algorithms
    # advanced to time t^{n+1/2} as described in Section 4.2
    ############################################################################################
    '''
    -------------------------- ** Step (3) of G&S (2005)** -----------------------------------
    Using any of the three EMF CT algorithms, integrate the face centered fluxes to the
    grid cell corner as described in Section 4.1.1.
    ------------------------------------------------------------------------------------------
    Before we can apply eqs 41 and 50, we need to get our face centered EMF from our face centered fluxes.
    Recall that EMF = - v x B (mentioned at the beginning of 2.2). Thus, we have the relations:
    
    F^x_{By} = vx*By - vy*Bx = -EMF_z
    F^y_{Bx} = vy*Bx - vx*By = +EMF_z
    
    Which implies that
    
    e_{Bx} · F^{n+1/2}_{y,i,j±1/2} = +EMF^{n+1/2}_{z,i,j±1/2}
                            Fy_p[5]=  EMF^{n+1/2}_{z,i,j+1/2}
                            Fy_m[5]=  EMF^{n+1/2}_{z,i,j-1/2}
    e_{By} · F^{n+1/2}_{x,i±1/2,j} = -EMF^{n+1/2}_{z,i±1/2,j}
                            Fx_p[6]= -EMF^{n+1/2}_{z,i+1/2,j}
                            Fx_m[6]= -EMF^{n+1/2}_{z,i-1/2,j}
    
    so to get the face centered EMFs, we just need to access the face-centered fluxes at the appropriate places
    '''
    EMFz_x_p = -Fx[6] #EMF^{n+1/2}_{z,i+1/2,j}
    
    EMFz_y_p =  Fy[5] #EMF^{n+1/2}_{z,i,j+1/2}
        
    '''
    Now, we need to use equation (41) to find the cell-corner EMFs, EMF^{n+1/2}_{z,i±1/2,j±1/2}
    But before we can do that, we need to use equation (50) to find the upwinded partial derivatives,
    where the partial derivatives at 1/4 and 3/4 indexes are found with equation (45).
    However, in order to do that, we will need the EMFs at cell centers found by using the 
    current solution values pertaining to velocity and magnetic field.
    '''
    

    
    #### initializing arrays
    Bx_half_cc = np.zeros_like(Bx_face) # ghosted
    By_half_cc = np.zeros_like(Bx_face) # ghosted
    EMFz_cc = np.zeros((ny+2*nghost,nx+2*nghost))
    
    EMFz_corner = np.zeros((ny+2*nghost+1,nx+2*nghost+1)) 
    
    for j in range(nghost, nghost+ny):
        for i in range(nghost, nghost+nx):
            # calculate Bx and By from (75) - (76) with face centered averages
            Bx_half_cc[j,i] = 0.5*(Bx_face[j,i] + Bx_face[j,i-1])  
            By_half_cc[j,i] = 0.5*(By_face[j,i] + By_face[j-1,i])
            # EMFz = vy*Bx - vx*By
            EMFz_cc[j,i] = uy[j,i]*Bx_half_cc[j,i] - ux[j,i]*By_half_cc[j,i]
    
    #periodic boundary
    EMFz_cc[:,  :nghost]   = EMFz_cc[:, -2*nghost:-nghost]   
    EMFz_cc[:, -nghost:]   = EMFz_cc[:,   nghost:2*nghost]
    EMFz_cc[:nghost, : ]   = EMFz_cc[-2*nghost:-nghost, :]   
    EMFz_cc[-nghost:,: ]   = EMFz_cc[nghost:2*nghost,   :]
              
    
    for j in range(nghost, nghost+ny+1):
        for i in range(nghost, nghost+nx+1):
            
            # equation (50) gives us the piecewise relations used in the following
            ######################################## 
            # finding 
            # (partial_y EMFz)_{i+1/2,j+1/4}
            ########################################
            ux_p1 = Fx[0,j,i] # rho * ux_{x,i+1/2,j} rho (should be) > 0
            if  np.sign(ux_p1) > 0:
                dEz_dy_xp_quarter = 2.0*(EMFz_y_p[j,i] - EMFz_cc[j,i])/dy # Equation (45)
            elif np.sign(ux_p1) < 0:
                dEz_dy_xp_quarter = 2.0*(EMFz_y_p[j,i+1] - EMFz_cc[j,i+1])/dy
            else:
                dEz_dy_xp_quarter = 0.5*(2.0*(EMFz_y_p[j,i] - EMFz_cc[j,i])/dy + 2.0*(EMFz_y_p[j,i+1] - EMFz_cc[j,i+1])/dy)
            
            ######################################## 
            # finding 
            # (partial_y EMFz)_{i+1/2,j+3/4}
            ########################################
            ux_p2 = Fx[0,j+1,i] # rho * ux_{x,i+1/2,j+1}
            if np.sign(ux_p2) > 0:
                dEz_dy_xp_three_quarter = 2.0*(EMFz_cc[j+1,i] - EMFz_y_p[j,i])/dy
            elif np.sign(ux_p2) < 0:
                dEz_dy_xp_three_quarter = 2.0*(EMFz_cc[j+1,i+1] - EMFz_y_p[j,i+1])/dy
            else:
                dEz_dy_xp_three_quarter = 0.5*(2.0*(EMFz_cc[j+1,i] - EMFz_y_p[j,i])/dy + 2.0*(EMFz_cc[j+1,i+1] - EMFz_y_p[j,i+1])/dy)
            
            ######################################## 
            # finding 
            # (partial_x EMFz)_{i+1/4,j+1/2}
            ########################################
            uy_p1 = Fy[0,j,i] # rho * uy_{y,i,j+1/2} used for upwind calculation, rho (should be) > 0
            if  np.sign(uy_p1) > 0:
                dEz_dx_yp_quarter = 2.0*(EMFz_x_p[j,i] - EMFz_cc[j,i])/dx # Equation (45)
            elif np.sign(uy_p1) < 0:
                dEz_dx_yp_quarter = 2.0*(EMFz_x_p[j+1,i] - EMFz_cc[j+1,i])/dx
            else:
                dEz_dx_yp_quarter = 0.5*(2.0*(EMFz_x_p[j,i] - EMFz_cc[j,i])/dx + 2.0*(EMFz_x_p[j+1,i] - EMFz_cc[j+1,i])/dx)
            
            ######################################## 
            # finding 
            # (partial_x EMFz)_{i+3/4,j+1/2}
            ########################################
            uy_p2 = Fy[0,j,i+1] # rho * uy_{y,i+1,j+1/2}
            if np.sign(uy_p2) > 0:
                dEz_dx_yp_three_quarter = 2.0*(EMFz_cc[j,i+1] - EMFz_x_p[j,i])/dx
            elif np.sign(uy_p2) < 0:
                dEz_dx_yp_three_quarter = 2.0*(EMFz_cc[j+1,i+1] - EMFz_x_p[j+1,i])/dx
            else:
                dEz_dx_yp_three_quarter = 0.5*(2.0*(EMFz_cc[j,i+1] - EMFz_x_p[j,i])/dx + 2.0*(EMFz_cc[j+1,i+1] - EMFz_x_p[j+1,i])/dx)
            
            
            EMFz_corner[j,i] = (
                0.25*(EMFz_x_p[j,i] + EMFz_x_p[j+1,i] + EMFz_y_p[j,i] + EMFz_y_p[j,i+1]) \
                + dy/8*(dEz_dy_xp_quarter - dEz_dy_xp_three_quarter) \
                + dx/8*(dEz_dx_yp_quarter - dEz_dx_yp_three_quarter)
            ) # Equation (41)

    # Periodic boundaries
    EMFz_corner[:,  :nghost]   = EMFz_corner[:, -2*nghost-1:-nghost-1]   
    EMFz_corner[:, -nghost:]   = EMFz_corner[:,   nghost+1:2*nghost+1]
    EMFz_corner[:nghost, : ]   = EMFz_corner[-2*nghost-1:-nghost-1, :]   
    EMFz_corner[-nghost:,: ]   = EMFz_corner[nghost+1:2*nghost+1,   :]
        
    return EMFz_corner

def GS_step_7(EMFz_corner, Bx_face, By_face, nx, ny, dx, dy, dt):
    ############################################################################################
    # -------------------------- ** Step (7) of G&S (2005)** -----------------------------------
    #  Advance the surface averaged normal components of the magnetic field from time
    # t^{n} to time t^{n+1} using Eqs. (14) and (15)
    ############################################################################################
    
    new_Bx_face = np.zeros_like(Bx_face)
    new_By_face = np.zeros_like(By_face)
    for j in range(nghost, nghost+ny):
        for i in range(nghost, nghost+nx+1):  # nx+1 x-faces per row
            # eq (14)
            new_Bx_face[j, i] = Bx_face[j, i] - (dt/dy)*(EMFz_corner[j, i] - EMFz_corner[j-1, i])
    for j in range(nghost, nghost+ny+1):  # ny+1 y-faces per column
        for i in range(nghost, nghost+nx):
            # eq (15)
            new_By_face[j, i] = By_face[j, i] + (dt/dx)*(EMFz_corner[j, i] - EMFz_corner[j, i-1])
    
    #periodic boundary
    new_Bx_face[:,  :nghost]   = new_Bx_face[:, -2*nghost:-nghost]   
    new_Bx_face[:, -nghost:]   = new_Bx_face[:,   nghost:2*nghost]
    new_Bx_face[:nghost, : ]   = new_Bx_face[-2*nghost:-nghost, :]   
    new_Bx_face[-nghost:,: ]   = new_Bx_face[nghost:2*nghost,   :]
    
    new_By_face[:,  :nghost]   = new_By_face[:, -2*nghost:-nghost]   
    new_By_face[:, -nghost:]   = new_By_face[:,   nghost:2*nghost]
    new_By_face[:nghost, : ]   = new_By_face[-2*nghost:-nghost, :]   
    new_By_face[-nghost:,: ]   = new_By_face[nghost:2*nghost,   :]

    return new_Bx_face, new_By_face

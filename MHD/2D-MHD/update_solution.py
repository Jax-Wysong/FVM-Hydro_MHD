import numpy as np 

# user defined libraries
import plotting
import boundary_conditions as BC
import eignvalues as evals
import conversions
import CT_update as CT
import interface_states

def evolve(q_ghosted, nghost, DOFs, nx, ny, dx, dy, C, t_final, gamma, Bx_face, By_face, Riemann,
        snapshot_callback=None, snapshot_every=0):
    '''
    Definition: advances the 2D ideal MHD equations in time using an unsplit CTU finite-volume scheme
                with CT to handle the divergence free term, GS05

    Inputs:     U_ghosted    : initial ghosted conserved array (DOFs_MHD_2D, ny+2*nghost, nx+2*nghost)
                nghost       : number of ghost cells for BCs
                DOFs         : number of degrees of freedom per cell, 8 for ideal MHD [rho, u, v, w, P, Bx, By, Bz]
                nx           : number of interior x-cells
                ny           : number of interior y-cells
                dx           : x-direction spatial mesh size
                dy           : y-direction spatial mesh size
                C            : CFL number (stability requires C < 1)
                t_final      : final simulation time
                gamma        : adiabatic index
                Riemann      : callable -- Riemann solver (Riemann_HLL or Riemann_HLLD)

    Outputs:    U_interior : final interior conserved state â€” shape (DOFs_MHD_2D, ny, nx)
                all_solns  : list of interior snapshots at each time step (for animation)
                all_t      : list of corresponding simulation times

    Dependencies: boundary_conditions, conversions, eigenvalues, interface_states,
                CT_update
    '''
    q = q_ghosted.copy() # q is reserved for the conserved variables
    q_0 = q.copy() # for plotting initial time solution
    
    all_solns = [] # movie  stuff
    all_t     = [] # for movie purposes
    t = 0.0
    nt = 0  # time step counter
    
    all_solns.append(q[:, nghost:nghost+ny, nghost:nghost+nx].copy())
    all_t.append(t)
    while t < t_final:

        ############################################################################################
        # 1) Apply periodic BCs on both ghost-cell boundaries
        ############################################################################################

        BC.periodic_bc(q, nghost)

        ############################################################################################
        # 2) CFL-limited time step: dt = C*dx / max(|eigenvalues|); clip to not overshoot t_final
        ############################################################################################

        w = conversions.cons_to_prim(q[:, nghost:nghost+ny, nghost:nghost+nx], gamma)  # w(rho, u, p, B)(w is for primative vars)
        alpha_x = evals.eigenvalues(w, gamma, 'x')                      # all 8 MHD wave speed magnitudes
        alpha_y = evals.eigenvalues(w, gamma, 'y')                      # all 8 MHD wave speed magnitudes
        a_x_max = max(alpha_x)
        a_y_max = max(alpha_y)
        dt = C * min(dx/a_x_max, dy/a_y_max)   # CFL condition: dt set by fastest wave in the domain
        dt = min(dt, t_final - t)  # adjust final time step to hit t_final exactly

        ############################################################################################
        # 3) Reconstruct left/right interface states with a piecewise linear method PLM (8.71 Zingale)
        # ----------------------- ** Step (1) of G&S (2005)** -------------------------------------
        # Calculate x and y interface states using the multi-D source terms (eqns. 30-32)
        ############################################################################################

        # PLM: characteristic-upwinded interface states
        q_L_x_hat, q_R_x_hat = interface_states.PLM_x(q, Bx_face, dx, dt, gamma)     # returns q^{n+1/2}_{i+1/2,j,L/R}
        q_L_y_hat, q_R_y_hat = interface_states.PLM_y(q, By_face, dy, dt, gamma) # returns q^{n+1/2}_{i,j+1/2,L/R}
        
        BC.periodic_bc(q_L_x_hat, nghost)
        BC.periodic_bc(q_R_x_hat, nghost)
        BC.periodic_bc(q_L_y_hat, nghost)
        BC.periodic_bc(q_R_y_hat, nghost)


        ############################################################################################
        # 4) Transverse-flux-corrected x/y-interface states (Zingale 8.6):
        #    Compute y/x-direction transverse fluxes at each interface     (eqs. 8.72, 8.73)
        ############################################################################################
        # ----------------------- ** Step (2) of G&S (2005)** --------------------------------------
        # --------------------Calculate the x- and y- interface fluxes, ----------------------------
        # --------------------------F^{n+1/2}_{y, i, j \pm 1/2} and --------------------------------
        # ------------------------- F^{n+1/2}_{x, i\pm 1/2, j} -------------------------------------
        ############################################################################################

        ##################################
        # Get F^{n+1/2}_{y, i, j \pm 1/2}
        ##################################
        F_T_y_plus = np.zeros_like(q)
        F_T_y_minus = np.zeros_like(q)
        for j in range(nghost-1, nghost+ny):    
            for i in range(nghost-1, nghost+nx+1):
                # F^{n+1/2}_{y,i,j+1/2}   
                F_T_y_plus[:,j,i] = Riemann(q_L_y_hat[:,j,i], q_R_y_hat[:,j+1,i],gamma, 'y') # 8.72 returns cons flux
                
                # F^{n+1/2}_{y,i,j-1/2}
                F_T_y_minus[:,j,i]= Riemann(q_L_y_hat[:,j-1,i], q_R_y_hat[:,j,i],gamma, 'y') # 8.73 returns cons flux 
        
        #################################
        # Get F^{n+1/2}_{x, i \pm 1/2, j}
        #################################
        F_T_x_plus = np.zeros_like(q)
        F_T_x_minus = np.zeros_like(q)
        for i in range(nghost-1, nghost+nx):  
            for j in range(nghost-1, nghost+ny+1):
                # F^{n+1/2}_{x,i+1/2,j}
                F_T_x_plus[:,j,i] = Riemann(q_L_x_hat[:,j,i  ], q_R_x_hat[:,j,i+1],gamma, 'x') # 8.72 returns cons flux
                
                # F^{n+1/2}_{x,i-1/2,j}
                F_T_x_minus[:,j,i]= Riemann(q_L_x_hat[:,j,i-1], q_R_x_hat[:,j,i  ],gamma, 'x') # 8.73 returns cons flux   

        # filling periodic BCs
        BC.periodic_bc(F_T_x_plus,  nghost)
        BC.periodic_bc(F_T_x_minus, nghost)
        BC.periodic_bc(F_T_y_plus,  nghost)
        BC.periodic_bc(F_T_y_minus, nghost)

        #################################
        # Feed these fluxes into step (3)
        # of G&S to calculate EMFs and
        # and corner fluxes
        #################################
        EMFz_corner,\
        Fy_L_corner_pp, Fy_R_corner_pp,\
        Fx_L_corner_pp, Fx_R_corner_pp,\
        Fy_L_corner_pm, Fx_L_corner_mp,\
        Fy_R_corner_pm, Fx_R_corner_mp = CT.GS_step_3(F_T_x_plus, F_T_x_minus,\
                                                    F_T_y_plus, F_T_y_minus,\
                                                        nx, ny, dx, dy, dt, q, gamma)
        # returns 
        # Fx/y_corner_p/m/m/p[8, ny+2*nghost+1, nx+2*nghost+1] 
        
        # eqs 68-69 for half step update of the interface B vectors
        # B^{n+1/2} at the cell interfaces
        Bx_face_half = Bx_face.copy()
        By_face_half = By_face.copy()
        for j in range(nghost, nghost+ny):
            for i in range(nghost, nghost+nx+1):
                Bx_face_half[j,i] = Bx_face[j,i] + 0.5*(dt/dy)*(EMFz_corner[j-1,i] - EMFz_corner[j,i])
        for j in range(nghost, nghost+ny+1):
            for i in range(nghost, nghost+nx):
                By_face_half[j,i] = By_face[j,i] - 0.5*(dt/dx)*(EMFz_corner[j,i-1] - EMFz_corner[j,i])

        # periodic boundaries
        Bx_face_half[:,  :nghost]   = Bx_face_half[:, -2*nghost:-nghost]   
        Bx_face_half[:, -nghost:]   = Bx_face_half[:,   nghost:2*nghost]
        Bx_face_half[:nghost, : ]   = Bx_face_half[-2*nghost:-nghost, :]   
        Bx_face_half[-nghost:,: ]   = Bx_face_half[nghost:2*nghost,   :]
            
        By_face_half[:,  :nghost]   = By_face_half[:, -2*nghost:-nghost]   
        By_face_half[:, -nghost:]   = By_face_half[:,   nghost:2*nghost]
        By_face_half[:nghost, : ]   = By_face_half[-2*nghost:-nghost, :]   
        By_face_half[-nghost:,: ]   = By_face_half[nghost:2*nghost,   :]
            
        
        ##########################################
        # Feed the new corner fluxes into step (4)
        # of G&S to calculate the updated
        # interface states (60) - (63)
        ##########################################
        q_x_L, q_x_R,\
        q_y_L, q_y_R = CT.GS_step_4(Fy_L_corner_pp, Fy_R_corner_pp,\
                                Fx_L_corner_pp, Fx_R_corner_pp,\
                                Fy_L_corner_pm, Fx_L_corner_mp,\
                                Fy_R_corner_pm, Fx_R_corner_mp,\
                                q_L_x_hat, q_R_x_hat,\
                                q_L_y_hat, q_R_y_hat,\
                                nx,ny,dx,dy,dt,q, gamma,\
                                Bx_face,By_face)
        # returns BC ghost filled vectors
        
        ##########################################
        # step (5)
        # compute x- and y- interface fluxes
        # associated with the interface states
        # from step (4) giving us the half step
        # x- and y- fluxes
        ##########################################
        F_x_half,\
        F_y_half = CT.GS_step_5(q_x_L, q_x_R,\
                            q_y_L, q_y_R,\
                            Bx_face_half, By_face_half,\
                            nx,ny, gamma, Riemann)
        
        # returns ghost BC filled fluxes
        
        ##########################################
        # step (6)
        # use these new fluxes from step (5) to 
        # compute the cell-centered EMFs
        # advanced to time t^{n+1/2}
        ##########################################
        # To do this step, we need the half-step
        # ux and uy values so we can update the
        # cell-centered EMF values (eq 74)
        # We get these updated values by doing
        # eq (77) using our already half-step
        # fluxes F_T_x_plus/minus and F_T_y_plus/minus
        ############################################
        q_temp = q.copy()
        for j in range(nghost, nghost+ny):
            for i in range(nghost, nghost+nx):
                q_temp[:,j, i] = q_temp[:,j, i] - 0.5*(dt/dx) * (F_T_x_plus[:,j, i] - F_T_x_minus[:, j, i]) - 0.5*(dt/dy) * (F_T_y_plus[:,j,i] - F_T_y_minus[:,j,i])
        w_temp = conversions.cons_to_prim(q_temp, gamma)
        ux = w_temp[1]
        uy = w_temp[2]
        
        EMFz_corner = CT.GS_step_6(F_x_half, F_y_half, ux, uy, Bx_face_half, By_face_half, nx, ny, dx, dy, dt, gamma)
        
        # returns full vector with periodic BC applied
        
        ##########################################
        # step (7)
        # Advance the surface averaged normal comps
        # of the mag field from t^n to t^{n+1}
        # via equations (14) and (15)
        ##########################################
        
        Bx_face, By_face = CT.GS_step_7(EMFz_corner, Bx_face, By_face, nx, ny, dx, dy, dt)

        # returns full vector with periodic BC applied
        
        Bx_cc = np.zeros_like(Bx_face)
        By_cc = np.zeros_like(By_face)
        # eqs (19)-(20): cell-centered B = average of left and right staggered faces
        for j in range(nghost, nghost+ny):
            for i in range(nghost, nghost+nx):
                # calculate Bx and By from (75) - (76) with face centered averages
                Bx_cc[j,i] = 0.5*(Bx_face[j,i] + Bx_face[j,i-1])  
                By_cc[j,i] = 0.5*(By_face[j,i] + By_face[j-1,i])

        # periodic boundary application
        Bx_cc[:,  :nghost]   = Bx_cc[:, -2*nghost:-nghost]   
        Bx_cc[:, -nghost:]   = Bx_cc[:,   nghost:2*nghost]
        Bx_cc[:nghost, : ]   = Bx_cc[-2*nghost:-nghost, :]   
        Bx_cc[-nghost:,: ]   = Bx_cc[nghost:2*nghost,   :]        

        By_cc[:,  :nghost]   = By_cc[:, -2*nghost:-nghost]   
        By_cc[:, -nghost:]   = By_cc[:,   nghost:2*nghost]
        By_cc[:nghost, : ]   = By_cc[-2*nghost:-nghost, :]   
        By_cc[-nghost:,: ]   = By_cc[nghost:2*nghost,   :]       

        
        ############################################################################################
        # 6) Conservative update: U^(n+1) = U^n - (dt/dx)(F_x_i - F_x_{i-1})
        #                                        - (dt/dy)(F_y_j - F_y_{j-1}) (Zingale 8.78)
        # -------------------------- ** Step (8) of G&S (2005)** -----------------------------------
        # Advance the remaining volume averaged conserved quantities from time t^{n} to 
        # time t^{n+1}
        ############################################################################################
        # ----------------------- ** Step (8) of G&S (2005)** --------------------------------------
        # --------------------Advance the remaining cell-centered conserved quantities -------------
        ############################################################################################

        
        for j in range(nghost, nghost+ny):
            for i in range(nghost, nghost+nx):
                q[:,j, i] = q[:,j, i] - (dt/dx) * (F_x_half[:,j, i] - F_x_half[:, j, i-1]) - (dt/dy) * (F_y_half[:,j,i] - F_y_half[:,j-1,i])

        #replace the Bx and By in U with the cell-centered B's found from the staggered average
        q[5, nghost:nghost+ny, nghost:nghost+nx] = Bx_cc[nghost:nghost+ny, nghost:nghost+nx]
        q[6, nghost:nghost+ny, nghost:nghost+nx] = By_cc[nghost:nghost+ny, nghost:nghost+nx]

        
        ############################################################################################
        # 7) Advance time, print progress every 100 steps, store interior snapshot
        ############################################################################################

        t += dt
        nt += 1
        if nt % 10 == 0:
            print(f"Step: {nt}, Time: {t:.10f}")
            divB = (Bx_face[nghost:nghost+ny, nghost:nghost+nx] - Bx_face[nghost:nghost+ny, nghost-1:nghost+nx-1]) / dx \
                + (By_face[nghost:nghost+ny, nghost:nghost+nx] - By_face[nghost-1:nghost+ny-1, nghost:nghost+nx]) / dy
            print(f"  max face |div B| = {np.abs(divB).max():.3e}")
            

        if snapshot_callback is not None and snapshot_every > 0 and nt % snapshot_every == 0:
            snapshot_callback(q[:, nghost:nghost+ny, nghost:nghost+nx], t, nt)

        all_solns.append(q[:, nghost:nghost+ny, nghost:nghost+nx].copy())
        all_t.append(t)

    print(f"Finished at Step: {nt}, Time: {t:.3f}")
    return q[:, nghost:nghost+ny, nghost:nghost+nx], q_0[:, nghost:nghost+ny, nghost:nghost+nx], all_solns, all_t


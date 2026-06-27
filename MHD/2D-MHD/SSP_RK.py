import numpy as np

# user defined libraries
from config import nghost, DOFs
import conversions
import interface_states
import boundary_conditions
import CT_update_RK



def f(q, Bx_face, By_face, nx, ny, dx, dy, gamma, Riemann):
    ## note to self:
    ## q, Bx_face, and By_face should all have BCs
    ## applied before coming in here!
    ## I plan on doing this at the beginning of the main loop
    
    ###### Ghost values for SHAPING new vectors #######
    ## for reference, q has shape (DOFs, Ny, Nx) ##
    Nx = nx + 2*nghost
    Ny = ny + 2*nghost
    
    ############################################################
    # step 1
    ############################################################
    # reconstruct q at the x- and y-interfaces 
    ############################################################
    q_L_x, q_R_x = interface_states.PLM_x(q, Bx_face, dx, gamma, dt=0.0, integrator='RK2') 
    q_L_y, q_R_y = interface_states.PLM_y(q, By_face, dy, gamma, dt=0.0, integrator='RK2') 
    
    ############################################################
    # step 2
    ############################################################
    # calculate x- and y-fluxes
    # F_x[:j,i] = flux at face (j,i+1/2)
    # F_y[:j,i] = flux at face (j+1/2,i)
    ############################################################
    F_x = np.zeros((DOFs, Ny, Nx+1))
    F_y = np.zeros((DOFs, Ny+1, Nx))
    for j in range(nghost-1, nghost+ny+1):  
        for i in range(nghost, nghost+nx+1):
            #   ________ ________   
            #  |        |        |
            #  |        |        |
            #  |   i-1  |    i   |
            #  |        |        |
            #  |________|________|
            #  
            #           (right face of cell i-1, left face of cell i)
            F_x[:,j,i] = Riemann(q_L_x[:,j,i-1], q_R_x[:,j,i],gamma, 'x') # 8.72 returns cons flux_{i+1/2}
            
    for j in range(nghost, nghost+ny+1):    
        for i in range(nghost-1, nghost+nx+1):
            #           (top face of cell j-1, bottom face of cell j)
            F_y[:,j,i] = Riemann(q_L_y[:,j-1,i], q_R_y[:,j,i],gamma, 'y') # 8.72 returns cons flux_{j+1/2}
    
    ############################################################
    # step 3
    ############################################################
    # calculate face-centered electric fields
    # EMFx_face[j,i] = EMF at face (j,i+1/2) 
    #                  shape (Ny,Nx+1)
    # EMFy_face[j,i] = EMF at face (j+1/2,i)
    #                  shape (Ny+1,Nx)
    ############################################################
    EMFx_face = -F_x[6,:,:] # shape (Ny, Nx+1)
    EMFy_face =  F_y[5,:,:] # shape (Ny+1,Nx)
    
    ############################################################
    # step 4
    ############################################################
    # produce cell-centered EMF necessary for the
    # EMF_corner update. This comes from the cell-centered
    # solution vector (q)
    # EMF_cc.shape = (Ny, Nx)
    # EMF_cc[j,i] = EMF at cell-center in cell (j,i)
    ############################################################
    w = conversions.cons_to_prim(q, gamma)
    ux = w[1]
    uy = w[2]
    Bx = w[5]
    By = w[6]
    EMF_cc = uy*Bx - ux*By # shape (Ny, Nx)
    boundary_conditions.periodic_bc(EMF_cc, ny, nx, 'EMF-cellCenter')
        
    ############################################################
    # step 5
    ############################################################
    # interpolate EMF_face to the cell corners
    # EMF_corner[j,i] = EMF in corner (j+1/2, i+1/2)
    #                  shape (Ny+1,Nx+1)
    ############################################################
    # needed for the sign of velocities at cell faces
    # to calculate corner EMFs for upwind calculations
    rho_ux_face = F_x[0, :, :]   # shape (Ny, Nx+1)
    rho_uy_face = F_y[0, :, :]   # shape (Ny+1, Nx)
    
    EMF_corner = CT_update_RK.calculate_EMF_corner(EMFx_face, EMFy_face, \
                                                    EMF_cc, \
                                                    rho_ux_face, \
                                                    rho_uy_face, \
                                                    nx, ny, Nx, Ny, dx, dy)

    ############################################################
    # step 6
    ############################################################
    # build the RHS (spatial derivative only) update of q for RK
    ############################################################
    kq = np.zeros_like(q)
    for j in range(nghost, nghost+ny):
        for i in range(nghost, nghost+nx):
            kq[:,j,i] = -(F_x[:,j,i+1] - F_x[:,j,i])/dx \
                        -(F_y[:,j+1,i] - F_y[:,j,i])/dy
    
    ############################################################
    # step 7
    ############################################################
    # build the RHS (spatial derivative only) update for
    # Bx and By face vectors using the corner EMFs
    ############################################################
    kBx_face = np.zeros_like(Bx_face) # shape (Ny, Nx+1)
    kBy_face = np.zeros_like(By_face) # shape (Ny+1, Nx)
    
    # x face loop (needs a plus 1 in x-dir)
    for j in range(nghost, nghost+ny):
        for i in range(nghost, nghost+nx+1):
            kBx_face[j,i] = -(EMF_corner[j+1,i] - EMF_corner[j,i])/dy
    # y face loop (needs a plus 1 in y-dir)
    for j in range(nghost, nghost+ny+1):
        for i in range(nghost, nghost+nx):
            kBy_face[j,i] = (EMF_corner[j,i+1] - EMF_corner[j,i])/dx
    
    # align solution with cell-centered derived fields
    ## probably redundant but conceptually cleaner
    kq[5, :, :] = 0.5 * (kBx_face[:, :-1] + kBx_face[:, 1:])
    kq[6, :, :] = 0.5 * (kBy_face[:-1, :] + kBy_face[1:, :])

    return kq, kBx_face, kBy_face

def RK2_step(Riemann, q, Bx_face, By_face, nx, ny, dx, dy, dt, gamma):
    ####################
    # stage 1
    ####################
    # make sure q uses face average mag fields for B before starting
    q[5, :, :] = 0.5 * (Bx_face[:, :-1] + Bx_face[:, 1:])
    q[6, :, :] = 0.5 * (By_face[:-1, :] + By_face[1:, :])
    
    k1q, k1Bx_face, k1By_face = f(q, Bx_face, By_face, nx, ny, dx, dy, gamma, Riemann) 
    q_star = q + dt * k1q
    Bx_face_star = Bx_face + dt * k1Bx_face
    By_face_star = By_face + dt * k1By_face
    
    # apply boundary conditions to new vectors
    boundary_conditions.periodic_bc(q_star, ny, nx, 'q-cellCenter') #need this before reconstructing a again in k2
    boundary_conditions.periodic_bc(Bx_face_star, ny, nx, 'Bx-cellFace')
    boundary_conditions.periodic_bc(By_face_star, ny, nx, 'By-cellFace')
    
    # realign q with cell-center derived Bx and By
    q_star[5, :, :] = 0.5 * (Bx_face_star[:, :-1] + Bx_face_star[:, 1:])
    q_star[6, :, :] = 0.5 * (By_face_star[:-1, :] + By_face_star[1:, :])
    
    ####################
    # stage 2
    ####################
    k2q, k2Bx_face, k2By_face = f(q_star, Bx_face_star, By_face_star, nx, ny, dx, dy, gamma, Riemann)               
    q_next = q + 0.5 * dt * (k1q + k2q)
    Bx_face_next = Bx_face + 0.5 * dt * (k1Bx_face + k2Bx_face)
    By_face_next = By_face + 0.5 * dt * (k1By_face + k2By_face)
    
    # realign q with cell-center derived Bx and By
    q_next[5, :, :] = 0.5 * (Bx_face_next[:, :-1] + Bx_face_next[:, 1:])
    q_next[6, :, :] = 0.5 * (By_face_next[:-1, :] + By_face_next[1:, :])
    
    boundary_conditions.periodic_bc(q_next, ny, nx, 'q-cellCenter')
    boundary_conditions.periodic_bc(Bx_face_next, ny, nx, 'Bx-cellFace')
    boundary_conditions.periodic_bc(By_face_next, ny, nx, 'By-cellFace')
    return q_next, Bx_face_next, By_face_next

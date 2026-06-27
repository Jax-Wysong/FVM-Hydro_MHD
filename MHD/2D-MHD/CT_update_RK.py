import numpy as np

# user defined libraries
from config import nghost, DOFs
import conversions
import boundary_conditions


def calculate_EMF_corner(EMFx_face, EMFy_face, \
                        EMF_cc, \
                        rho_ux_face, \
                        rho_uy_face, \
                        nx, ny, Nx, Ny, dx, dy): 
    '''
    -------------------------- ** finding corner EMFs G&S (2005)** -----------------------------------
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
    
    This is was done previously and input into this function as
    EMFx_face
    EMFy_face 
    EMFx_face[j,i] = EMF at face (j,i+1/2) 
                    shape (Ny,Nx+1)
    EMFy_face[j,i] = EMF at face (j+1/2,i)
                    shape (Ny+1,Nx)
    
    Now, we need to use equation (41) to find the cell-corner EMFs, EMF^{n+1/2}_{z,i±1/2,j±1/2}
    But before we can do that, we need to use equation (50) to find the upwinded partial derivatives,
    where the partial derivatives at 1/4 and 3/4 indexes are found with equation (45).
    However, in order to do that, we will need the EMFs at cell centers found by using the 
    current solution values pertaining to velocity and magnetic field. This is inpute as 
    
    EMF_cc
    '''

    
    #### initializing array for corner EMF
    EMF_corner = np.zeros((Ny+1,Nx+1)) 
    
    for j in range(nghost, nghost+ny+1):
        for i in range(nghost, nghost+nx+1):
            # equation (50) gives us the piecewise relations used in the following
            ########################################
            # finding
            # (partial_y EMFz)_{i+1/2,j+1/4}
            ########################################
            if  np.sign(rho_ux_face[j-1,i]) > 0:
                dEz_dy_xp_quarter = 2.0*(EMFy_face[j,i-1] - EMF_cc[j-1,i-1])/dy # Equation (45)
            elif np.sign(rho_ux_face[j-1,i]) < 0:
                dEz_dy_xp_quarter = 2.0*(EMFy_face[j,i] - EMF_cc[j-1,i])/dy
            else:
                dEz_dy_xp_quarter = 0.5*(2.0*(EMFy_face[j,i-1] - EMF_cc[j-1,i-1])/dy + 2.0*(EMFy_face[j,i] - EMF_cc[j-1,i])/dy)
            
            ######################################## 
            # finding 
            # (partial_y EMFz)_{i+1/2,j+3/4}
            ########################################
            if np.sign(rho_ux_face[j,i]) > 0:
                dEz_dy_xp_three_quarter = 2.0*(EMF_cc[j,i-1] - EMFy_face[j,i-1])/dy
            elif np.sign(rho_ux_face[j,i]) < 0:
                dEz_dy_xp_three_quarter = 2.0*(EMF_cc[j,i] - EMFy_face[j,i])/dy
            else:
                dEz_dy_xp_three_quarter = 0.5*(2.0*(EMF_cc[j,i-1] - EMFy_face[j,i-1])/dy + 2.0*(EMF_cc[j,i] - EMFy_face[j,i])/dy)
            
            ######################################## 
            # finding 
            # (partial_x EMFz)_{i+1/4,j+1/2}
            ########################################
            if  np.sign(rho_uy_face[j,i-1]) > 0:
                dEz_dx_yp_quarter = 2.0*(EMFx_face[j-1,i] - EMF_cc[j-1,i-1])/dx # Equation (45)
            elif np.sign(rho_uy_face[j,i-1]) < 0:
                dEz_dx_yp_quarter = 2.0*(EMFx_face[j,i] - EMF_cc[j,i-1])/dx
            else:
                dEz_dx_yp_quarter = 0.5*(2.0*(EMFx_face[j-1,i] - EMF_cc[j-1,i-1])/dx + 2.0*(EMFx_face[j,i] - EMF_cc[j,i-1])/dx)
            
            ######################################## 
            # finding 
            # (partial_x EMFz)_{i+3/4,j+1/2}
            ########################################
            if np.sign(rho_uy_face[j,i]) > 0:
                dEz_dx_yp_three_quarter = 2.0*(EMF_cc[j-1,i] - EMFx_face[j-1,i])/dx
            elif np.sign(rho_uy_face[j,i]) < 0:
                dEz_dx_yp_three_quarter = 2.0*(EMF_cc[j,i] - EMFx_face[j,i])/dx
            else:
                dEz_dx_yp_three_quarter = 0.5*(2.0*(EMF_cc[j-1,i] - EMFx_face[j-1,i])/dx + 2.0*(EMF_cc[j,i] - EMFx_face[j,i])/dx)
            
            
            EMF_corner[j,i] = (
                0.25*(EMFx_face[j-1,i] + EMFx_face[j,i] + EMFy_face[j,i-1] + EMFy_face[j,i]) \
                + dy/8*(dEz_dy_xp_quarter - dEz_dy_xp_three_quarter) \
                + dx/8*(dEz_dx_yp_quarter - dEz_dx_yp_three_quarter)
            ) # Equation (41)
            
    boundary_conditions.periodic_bc(EMF_corner, ny, nx, 'EMF-cellCorner')
    
    
    return EMF_corner

def update_energy(q, gamma):
    w = conversions.cons_to_prim(q.copy(), gamma)
    q = conversions.prim_to_cons(w, gamma)
    return q

'''
####################################################################################################
2D Magnetohydrodynamic Equations
Jax Wysong
06/13/2026
####################################################################################################

The 2D Magnetohydrodynamic Equations are given by

∂(ρ)/∂t     + ∇·(ρu)                                   = 0
∂(ρu)/∂t    + ∇·(ρu⊗u + (p + 0.5*∥B∥^2 )I - B⊗B )     = 0
∂(B)/∂t     + ∇·( u⊗B - B⊗u )                        = 0
∂(E)/∂t     + ∇·( u(E + p + 0.5*∥B∥^2 ) - B(u·B))      = 0

with the additional requirement that ∇·B = 0

ρ is the density
u is the velocity vector (u_x,u_y,u_z)
B is the magnetic field vector (b_x,b_y,b_z)
E is the total energy
p is the  pressure
I is the identity matrix
E is the total energy = (p / (γ-1)) + 0.5ρ*∥u∥^2 + 0.5*∥B∥^2
γ is the adiabatic index = 5/3 
∥·∥ is the Euclidean vector norm

Written slightly different we have

∂q/∂t + ∂F(q)/∂x + ∂G(q)/∂y = 0

where the conserved variables q are

        | ρ    |
        | ρu_x |
        | ρu_y |
q =     | ρu_z |
        | E    |
        | B_x  |
        | B_y  |
        | B_z  |

        |              ρu_x                |
        | ρu_xu_x + p + 0.5*∥B∥^2 - B_xB_x  |
        |        ρu_xu_y - B_xB_y          |
F(q) =  |        ρu_xu_z - B_xB_z          |
        | u_x(E + p + 0.5*∥B∥^2) - B_x(u·B) |
        |               0                  |
        |        u_xB_y - u_yB_x           |
        |        u_xB_z - u_zB_x           |

        |              ρu_y                |
        |        ρu_yu_x - B_yB_x          |
        | ρu_yu_y + p + 0.5*∥B∥^2 - B_yB_y  |
G(q) =  |        ρu_yu_z - B_yB_z          |
        | u_y(E + p + 0.5*∥B∥^2) - B_y(u·B) |
        |        u_yB_x - u_xB_y           |
        |               0                  |
        |        u_yB_z - u_zB_y           |

For convenience, we also introduce the primative variables w

        | ρ   |
        | u_x |
        | u_y |
w =     | u_z |
        | p   |
        | B_x |
        | B_y |
        | B_z |

I am solving this system of equations using a Finite Volume Unsplit, Corner Transport Upwind (CTU) method 
with Constrained Transport (CT).
The CTU procedure follows that described by Zingale and Gardiner and Stone (2005), GS05
and the CT is described by GS05.
'''


import numpy as np

# user defined libraries
import Riemann
import update_solution
import initial_data
import plotting
from config import nghost, DOFs

#==============================================================================================================
####################################
# Field loop advection start
####################################
#==============================================================================================================
gamma   = 5/3

t0      = 0.0
tf      = 2.0

xL      = -1.0;  xR  = 1.0
yL      = -0.5;  yR  = 0.5

N       = 16
nx      = 2*N
ny      = N

dx = (xR - xL) / nx  
dy = (yR - yL) / ny   

X  = xL + (np.arange(nx) + 0.5) * dx   
Y  = yL + (np.arange(ny) + 0.5) * dy


BC      ='periodic' #this doesn't get passed in as of yet
CFL     = 0.4


q_ghost_IC_advection, \
        Bx_face, By_face    = initial_data.advection_setup(xL, xR, yL, yR, nx, ny, dx, dy, X, Y, gamma, nghost, DOFs)
# returns 
# q_ghost_IC_advection[8, ny+2*nghost, nx+2*nghost]
# Bx_face[ny+2*nghost, nx+2*nghost] 
# By_face[ny+2*nghost, nx+2*nghost] 

####################################################################################################
# Evolve and Plot for Loop Advection Tests
####################################################################################################
# Runs HLL or HLLD solver with PLM reconstruction 
# Is capable is saving solution snapshots every
# N number of time steps. Currently set to 20

snapshot = lambda q, t, nt: plotting.save_snapshot_advection(X, Y, q, t, nt, N, gamma, 'Field Loop Advection', 'HLLD')

q_sol_advection, q0_advection,all_solns_advection, all_t_advection = update_solution.evolve(
        q_ghost_IC_advection, nghost, DOFs, nx, ny, dx, dy, CFL, tf, gamma, Bx_face, By_face, Riemann.HLLD, integrator='RK2',
        snapshot_callback=snapshot, snapshot_every=20)


plotting.plot_solution_advection(X, Y, q_sol_advection, q0_advection, tf, gamma, N, 'Field Loop Advection', 'HLLD-RK2',True)
# ==============================================================================================================
####################################
# Field loop advection finished
####################################
# ==============================================================================================================
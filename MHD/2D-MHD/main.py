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


import argparse
import os
import numpy as np

# user defined libraries
import Riemann
import update_solution
import initial_data
import plotting
import conversions
from config import nghost, DOFs

# #==============================================================================================================
# ####################################
# # Field loop advection start
# ####################################
# #==============================================================================================================
# parser = argparse.ArgumentParser()
# parser.add_argument('--outdir', default='.', help='Directory to save output figures')
# parser.add_argument('--N',      type=int,   default=64,  help='Base grid resolution (nx=2N, ny=N)')
# parser.add_argument('--tf',     type=float, default=2.0, help='Final simulation time')
# parser.add_argument('--CFL',    type=float, default=0.4, help='CFL number')
# args = parser.parse_args()
# os.makedirs(args.outdir, exist_ok=True)

# gamma   = 5/3

# t0      = 0.0
# tf      = args.tf

# xL      = -1.0;  xR  = 1.0
# yL      = -0.5;  yR  = 0.5

# N       = args.N
# nx      = 2*N
# ny      = N

# dx = (xR - xL) / nx
# dy = (yR - yL) / ny

# X  = xL + (np.arange(nx) + 0.5) * dx
# Y  = yL + (np.arange(ny) + 0.5) * dy


# BC      ='periodic' #this doesn't get passed in as of yet
# CFL     = args.CFL


# q_ghost_IC_advection, \
#         Bx_face, By_face    = initial_data.advection_setup(xL, xR, yL, yR, nx, ny, dx, dy, X, Y, gamma, nghost, DOFs)
# # returns 
# # q_ghost_IC_advection[8, ny+2*nghost, nx+2*nghost]
# # Bx_face[ny+2*nghost, nx+2*nghost] (Periodic BC already implemented)
# # By_face[ny+2*nghost, nx+2*nghost] (Periodic BC already implemented)

# ####################################################################################################
# # Evolve and Plot for Loop Advection Tests
# ####################################################################################################
# # Runs HLL or HLLD solver with PLM reconstruction 
# # Is capable is saving solution snapshots every
# # N number of time steps. Currently set to 20

# snapshot = lambda q, t, nt: plotting.save_snapshot_advection(X, Y, q, t, nt, N, gamma, 'Field Loop Advection', 'HLLD-RK2', outdir=args.outdir)

# q_sol_advection, q0_advection,all_solns_advection, all_t_advection = update_solution.evolve(
#         q_ghost_IC_advection, nghost, DOFs, nx, ny, dx, dy, CFL, tf, gamma, Bx_face, By_face, Riemann.HLLD, integrator='RK2',
#         snapshot_callback=snapshot, snapshot_every=20)


# plotting.plot_solution_advection(X, Y, q_sol_advection, q0_advection, tf, gamma, N, 'Field Loop Advection', 'HLLD-RK2', save=True, outdir=args.outdir)

# plotting.movie_maker(X, Y, all_solns_advection, all_t_advection, gamma, 'Field Loop Advection HLLD-PLM-RK2-CT', outdir=args.outdir)
# # ==============================================================================================================
# ####################################
# # Field loop advection finished
# ####################################
# # ==============================================================================================================

#==============================================================================================================
####################################
# Smooth Vortex start
####################################
#==============================================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='.', help='Directory to save output figures')
parser.add_argument('--N',      type=int,   default=64,  help='Base grid resolution (nx=2N, ny=N)')
parser.add_argument('--tf',     type=float, default=2.0, help='Final simulation time')
parser.add_argument('--CFL',    type=float, default=0.4, help='CFL number')
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

gamma   = 5/3

t0      = 0.0
tf      = args.tf

xL      = -5.0;  xR  = 5.0
yL      = -5.0;  yR  = 5.0

N       = args.N
nx      = N
ny      = N

dx = (xR - xL) / nx
dy = (yR - yL) / ny

X  = xL + (np.arange(nx) + 0.5) * dx
Y  = yL + (np.arange(ny) + 0.5) * dy

BC      ='periodic' #this doesn't get passed in as of yet
CFL     = args.CFL

# L2 = initial_data.smooth_vortex_convergence_test(nghost, DOFs, CFL, tf, gamma)

# print(f'L2\n\n\n')

q_ghost_IC_smooth_vortex, \
        Bx_face, By_face    = initial_data.smooth_mhd_vortex_setup(xL, xR, yL, yR, nx, ny, dx, dy, X, Y, gamma, nghost, DOFs)
# returns 
# q_ghost_IC[8, ny+2*nghost, nx+2*nghost]
# Bx_face[ny+2*nghost, nx+2*nghost] (Periodic BC already implemented)
# By_face[ny+2*nghost, nx+2*nghost] (Periodic BC already implemented)
# BCs implemented for both

####################################################################################################
# Evolve and Plot for Convergence Tests
####################################################################################################
# Runs HLL or HLLD solver with PLM reconstruction 
# Is capable is saving solution snapshots every
# N number of time steps. Currently set to 20

snapshot = lambda q, t, nt: plotting.save_snapshot_advection(X, Y, q, t, nt, N, gamma, 'Smooth MHD Vortex', 'HLLD-RK2', outdir=args.outdir)

q_sol_smooth_vortex, q0_smooth_vortex, all_solns_smooth_vortex, all_t_smooth_vortex = update_solution.evolve(
        q_ghost_IC_smooth_vortex, nghost, DOFs, nx, ny, dx, dy, CFL, tf, gamma, Bx_face, By_face, Riemann.HLLD, integrator='RK2', limiter='MC',
        snapshot_callback=snapshot, snapshot_every=50)

w_sol = conversions.cons_to_prim(q_sol_smooth_vortex, gamma)
w0_sol = conversions.cons_to_prim(q0_smooth_vortex, gamma)


rho_err = w_sol[0] - w0_sol[0]
ux_err  = w_sol[1] - w0_sol[1]
Bx_err  = w_sol[5] - w0_sol[5]
p_err   = w_sol[4] - w0_sol[4]

L2 = np.zeros((4))
# discrete L2 norm
L2[0] = np.sqrt(dx * dy * np.sum(rho_err**2))
L2[1] = np.sqrt(dx * dy * np.sum(ux_err**2))
L2[2] = np.sqrt(dx * dy * np.sum(Bx_err**2))
L2[3] = np.sqrt(dx * dy * np.sum(p_err**2))

print(f"\n\nN = {N} x {N}")
print(f"  rho L2 = {L2[0]:.6e}")
print(f"  ux  L2 = {L2[1]:.6e}")
print(f"  Bx  L2 = {L2[2]:.6e}")
print(f"  p   L2 = {L2[3]:.6e}\n\n")

plotting.plot_solution_advection(X, Y, q_sol_smooth_vortex, q0_smooth_vortex, tf, gamma, N, 'Smooth MHD Vortex', 'HLLD-RK2', save=True, outdir=args.outdir)

#plotting.movie_maker(X, Y, all_solns_smooth_vortex, all_t_smooth_vortex, gamma, 'Smooth MHD Vortex HLLD-PLM-RK2-CT', outdir=args.outdir)
# ==============================================================================================================
####################################
# Smooth Vortex finished
####################################
# ==============================================================================================================
'''
2D Ideal MHD solver based off of Gardiner and Stone's 2005 paper
which describes a constrained transport (CT) method for handling the
divergence free constraint and the corner transport upwind method
(CTU) for 2nd order accuracy.
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
tf      = 1.0

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
# Bx_face[ny+2*nghost, nx+2*nghost] (Periodic BC already implemented)
# By_face[ny+2*nghost, nx+2*nghost] (Periodic BC already implemented)

####################################################################################################
# Evolve and Plot for Loop Advection Tests
####################################################################################################
# Runs HLL or HLLD solver with PLM reconstruction 
# Is capable is saving solution snapshots every
# N number of time steps. Currently set to 20

snapshot = lambda q, t, nt: plotting.save_snapshot_advection(X, Y, q, t, nt, N, gamma, 'Field Loop Advection', 'HLLD')

q_sol_advection, q0_advection,all_solns_advection, all_t_advection = update_solution.evolve(
    q_ghost_IC_advection, nghost, DOFs, nx, ny, dx, dy, CFL, tf, gamma, Bx_face, By_face, Riemann.HLLD,
    snapshot_callback=snapshot, snapshot_every=20)


plotting.plot_solution_advection(X, Y, q_sol_advection, q0_advection, tf, gamma, N, 'Field Loop Advection', 'HLLD',True)
# ==============================================================================================================
####################################
# Field loop advection finished
####################################
# ==============================================================================================================
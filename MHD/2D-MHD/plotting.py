import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

## user defined libraries
import conversions

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24
MSU_GREEN = "#18453B"  

matplotlib.rc('font', size=SMALL_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
matplotlib.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plot_solution_advection(X, Y, q_sol, q0, t, gamma, N, case='Advection', Riemann='HLL', save=True):
    '''
    Definition: plots numerical MHD solution against the exact solution at a given time

    Inputs:     X, Y         : 1D coordinate arrays for x and y axes (length nx, ny respectively)
                q_sol        : conserved variable array (DOFs, ny, nx) -- interior cells only
                t            : simulation time (used in the plot title)
                gamma        : adiabatic index
                case         : problem name for the plot title (default 'Brio-Wu')
                Godunov_type : reconstruction label for the title (default 'Piecewise Linear')
                Riemann      : solver label for the title (default 'HLL')

    Outputs:    matplotlib 2x3 figure (displayed via plt.show()):
                [0,0] density rho          [0,1] x-velocity v_x      [0,2] B_y
                [1,0] pressure p           [1,1] y-velocity v_y      [1,2] total energy E
                numerical solution only for this
                
    Dependencies: cons_to_prim, numpy, matplotlib
    '''
    
    w0   = conversions.cons_to_prim(q0,    gamma)
    w_sol = conversions.cons_to_prim(q_sol, gamma)

    B0  = w0[5]**2  + w0[6]**2
    B   = w_sol[5]**2 + w_sol[6]**2

    vmin = 0.0
    vmax = B0.max()   # pin scale to initial solution so outliers at t>0 don't compress the loop
    extent = [X[0], X[-1], Y[0], Y[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=100)
    fig.suptitle(f'{case} — {Riemann}')

    im0 = axes[0].imshow(B0, origin='lower', extent=extent, aspect='equal',
                        cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title('Bx^2 + By^2   t = 0.00')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    im1 = axes[1].imshow(B, origin='lower', extent=extent, aspect='equal',
                        cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title(f'Bx^2 + By^2   t = {t:.2f}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    plt.tight_layout()
    if save:
        fname = os.path.join(_HERE, f"{case.replace(' ', '_')}_N{N}_{Riemann}_t{t:.2f}.png")
        fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()


def save_snapshot_advection(X, Y, q, t, nt, N, gamma, case='Field Loop', Riemann='HLL'):
    '''
    Save a single B^2 image during a running simulation without blocking execution.
    Closes the figure after saving so figures don't accumulate in memory.
    Filename includes the step number so snapshots sort chronologically.
    '''
    w  = conversions.cons_to_prim(q, gamma)
    B2 = w[5]**2 + w[6]**2
    extent = [X[0], X[-1], Y[0], Y[-1]]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=80)
    im = ax.imshow(B2, origin='lower', extent=extent, aspect='equal', cmap='viridis')
    fig.colorbar(im, ax=ax)
    ax.set_title(f'Bx^2 + By^2   t = {t:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    fname = os.path.join(_HERE, f"{case.replace(' ', '_')}_N{N}_{Riemann}_step{nt:05d}.png")
    fig.savefig(fname, dpi=80, bbox_inches='tight')
    plt.close(fig)   # must close — plt.show() is not called, so this is the only cleanup

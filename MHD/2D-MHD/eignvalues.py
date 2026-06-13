import numpy as np


def eigenvalues(U, gamma, space):  # takes primitive U
    '''
    Definition: computes all 8 eigenvalue magnitudes of the 2D ideal MHD system for CFL stepping

    Inputs:     U     : primitive state (rho, ux, uy, uz, p, bx, by, bz)^T -- interior cells
                gamma : adiabatic index

    Outputs:    alpha : array of 8 maximum eigenvalue magnitudes, one per wave family

    Dependencies: none

    The 8 MHD wave families and their characteristic speeds (in s-direction):
        lambda_1 = |us - cf|    fast magnetosonic (left-going)
        lambda_2 = |us - cas|   Alfven (left-going)
        lambda_3 = |us - cs|    slow magnetosonic (left-going)
        lambda_4 = |us|         entropy wave
        lambda_5 = |us|         divergence wave 
        lambda_6 = |us + cs|    slow magnetosonic (right-going)
        lambda_7 = |us + cas|   Alfven (right-going)
        lambda_8 = |us + cf|    fast magnetosonic (right-going)

    Wave speed definitions:
        c   = sqrt(gamma*p/rho)                  (sound speed)
        ca  = sqrt(||B||^2/rho)                  (total Alfven speed)
        cax = |sqrt(bx^2/rho)|                   (x-component Alfven speed)
        cf  = sqrt(0.5*(c^2+ca^2 + sqrt((c^2+ca^2)^2 - 4*c^2*cax^2)))   (fast magnetosonic)
        cs  = sqrt(0.5*(c^2+ca^2 - sqrt((c^2+ca^2)^2 - 4*c^2*cax^2)))   (slow magnetosonic)

    A small floor (1e-15) prevents zero eigenvalues from stalling the time step.
    '''
    rho  = U[0]
    ux   = U[1]
    uy   = U[2]
    uz   = U[3]
    p    = U[4]
    bx   = U[5]
    by   = U[6]
    bz   = U[7]
    
    # compute sound speed
    c = np.sqrt(gamma*p/rho)
    
    if space =='x':
        # compute Alfven speed
        ca = np.sqrt((bx**2 + by**2 + bz**2) / rho)   # total Alfven speed
        cax = np.abs(np.sqrt(bx**2 / rho))             # x-component Alfven speed
        
        # compute fast magnetosonic wave speed
        cf = np.sqrt(0.5*np.abs(c**2 + ca**2 + np.sqrt((c**2 + ca**2)**2 - (4*c**2 * cax**2))))
        
        # compute slow magnetosonic wave speed
        cs = np.sqrt(0.5*np.abs(c**2 + ca**2 - np.sqrt((c**2 + ca**2)**2 - (4*c**2 * cax**2))))
        
        # floor to prevent zero eigenvalues from stalling the time step
        em = 1.0e-15
        
        max_eigen_1 = max(em, np.max(np.abs(ux - cf )))   # fast magnetosonic (left)
        max_eigen_2 = max(em, np.max(np.abs(ux - cax)))   # Alfven (left)
        max_eigen_3 = max(em, np.max(np.abs(ux - cs )))   # slow magnetosonic (left)
        max_eigen_4 = max(em, np.max(np.abs(ux      )))   # entropy wave
        max_eigen_5 = max(em, np.max(np.abs(ux      )))   # divergence wave 
        max_eigen_6 = max(em, np.max(np.abs(ux + cs )))   # slow magnetosonic (right)
        max_eigen_7 = max(em, np.max(np.abs(ux + cax)))   # Alfven (right)
        max_eigen_8 = max(em, np.max(np.abs(ux + cf )))   # fast magnetosonic (right)
    elif space =='y':
        # compute Alfven speed
        ca = np.sqrt((bx**2 + by**2 + bz**2) / rho)   # total Alfven speed
        cay = np.abs(np.sqrt(by**2 / rho))             # x-component Alfven speed
        
        # compute fast magnetosonic wave speed
        cf = np.sqrt(0.5*np.abs(c**2 + ca**2 + np.sqrt((c**2 + ca**2)**2 - (4*c**2 * cay**2))))
        
        # compute slow magnetosonic wave speed
        cs = np.sqrt(0.5*np.abs(c**2 + ca**2 - np.sqrt((c**2 + ca**2)**2 - (4*c**2 * cay**2))))
        
        # floor to prevent zero eigenvalues from stalling the time step
        em = 1.0e-15
        
        max_eigen_1 = max(em, np.max(np.abs(uy - cf )))   # fast magnetosonic (left)
        max_eigen_2 = max(em, np.max(np.abs(uy - cay)))   # Alfven (left)
        max_eigen_3 = max(em, np.max(np.abs(uy - cs )))   # slow magnetosonic (left)
        max_eigen_4 = max(em, np.max(np.abs(uy      )))   # entropy wave
        max_eigen_5 = max(em, np.max(np.abs(uy      )))   # divergence wave 
        max_eigen_6 = max(em, np.max(np.abs(uy + cs )))   # slow magnetosonic (right)
        max_eigen_7 = max(em, np.max(np.abs(uy + cay)))   # Alfven (right)
        max_eigen_8 = max(em, np.max(np.abs(uy + cf )))   # fast magnetosonic (right)
    else:
        print('error in eigenvalues() function')

    alpha = np.array([max_eigen_1, max_eigen_2, max_eigen_3, max_eigen_4,
                    max_eigen_5, max_eigen_6, max_eigen_7, max_eigen_8])
    
    return alpha
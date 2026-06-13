def outflow_bc(q, ng, nx, ny):
    '''
    Definition: fills ghost cells on all four domain boundaries with outflow (zero-gradient) BCs

    Inputs:     q  : solution array of shape (DOFs, ny+2*nghost, nx+2*nghost)
                ny : number of interior cells in y
                nx : number of interior cells in x

    Outputs:    U : same array with ghost cells set equal to the nearest interior edge value

    Dependencies: nghost 

    Outflow BC sets each ghost cell equal to the adjacent interior edge value (zero gradient),
    which lets waves exit the domain without spurious reflections.
    Interior cells run from index nghost to nghost+n-1 (inclusive) in each direction.

        left  ghosts : U[:, :, 0:nghost]        = U[:, :, nghost:nghost+1]        (nearest left  column)
        right ghosts : U[:, :, nghost+nx:]      = U[:, :, nghost+nx-1:nghost+nx]  (nearest right column)
        bottom ghosts: U[:, 0:nghost, :]        = U[:, nghost:nghost+1, :]         (nearest bottom row)
        top    ghosts: U[:, nghost+ny:, :]      = U[:, nghost+ny-1:nghost+ny, :]   (nearest top    row)
    '''
    # Left boundary: copy leftmost interior column into left ghost columns
    q[:, :, 0:ng] = q[:, :, ng:ng+1]

    # Right boundary: copy rightmost interior column into right ghost columns
    q[:, :, ng+nx:] = q[:, :,ng+nx-1:ng+nx]

    # Bottom boundary: copy bottommost interior row into bottom ghost rows
    q[:, 0:ng, :] = q[:, ng:ng+1, :]

    # Top boundary: copy topmost interior row into top ghost rows
    q[:, ng+ny:, :] = q[:, ng+ny-1:ng+ny, :]

    return q

def periodic_bc(q, ng):
    '''
    Definition: fills ghost cells on all four domain boundaries with periodic BCs

    Inputs:     U  : solution array of shape (DOFs, ny+2*nghost, nx+2*nghost)

    Outputs:    U : same array with ghost cells set equal wrap around the domain

    Dependencies: nghost 

    If we have the domain of grid cells denoted by |-----|, where |~~~~~| signifies ghost cells

    [0]   [1]   [2]   [3]         [.]         [N]  [N+1] [N+2] [N+3]
    |~~~~~|~~~~~|-----|-----|-----|-----|-----|-----|-----|~~~~~|~~~~~|

    Our swap for periodic boundary conditions is as follows:
    q_sys[0] = q_sys[N] 
    q_sys[1] = q_sys[N+1]
    q_sys[N+2] = q_sys[2]
    q_sys[N+3] = q_sys[3]    
    
    '''
    
    q[:,:, :ng] = q[:,:, -2*ng:-ng]
    q[:,:, -ng:] = q[:,:, ng:2*ng]
    
    q[:,:ng, :] = q[:,-2*ng:-ng, :]
    q[:,-ng:, :] = q[:,ng:2*ng, :] 


    return q

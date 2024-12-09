import numpy as np
import matplotlib.pyplot as plt


# Ψ^(n+1) = (I - (iτ / hbar) * H) * Ψ^n (9.32)
# Ψ^(n+1) = (I + iτ/2ħ H)^(-1) (I - iτ/2ħ H) Ψ^n                      (9.40)


def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10, 0, 0.5]):
    """
    Solves the 1D time-dependent Schrödinger equation using FTCS or Crank-Nicholson scheme.

    Parameters:
        nspace (int) => Number of spatial grid points.
        ntime (int) => Number of time steps.
        tau (float) => Time step.
        method (str) => Method to use ('ftcs' or 'crank').
        length (float) => Length of spatial grid: [-L/2, L/2]. Default to 200.
        potential (list) => Spatial index values at which the potential V(x) = 1. Default to empty.
        wparam (list): Parameters for initial wave packet [sigma0, x0, k0]. Default [10, 0, 0.5].

    Returns:
        psi_grid (2D array): Wavefunction ψ(x, t) at all grid points and times.
        x_grid (1D array): Spatial grid points.
        t_grid (1D array): Time steps.
        prob_array (1D array): Total probability at each time step.
    """

    # constants
    hbar = 1  # Planck's constant
    L = length  # spatial grid length #system extends from -L/2 to L/2
    dx = L / nspace  # spatial step size
    x_grid = np.linspace(-L / 2, L / 2, nspace, endpoint=False)  # spatial grid
    t_grid = np.linspace(0, ntime * tau, ntime)  # time grid

    #init. param.

    # Plan:
    # Setup spatial from -L/2 to L/2? and temporal grids   V
    # initial wave packet Psi(x, 0)                         X
    # Hamiltonian H (tridiagonal matrix)                    V
    # H = -(hbar^2 / 2m)*(∂^2 / ∂x^2) + V(x) (9.27) then if m = 1/2 and nbar=1 : H=− ∂^2/ ∂x^2x +V(x)
    # then let's approximate second dwerivative:
    V = np.zeros(nspace)  # default potential
    # if potential == []:
    #     V = None #to avoid error
    for index in potential:
        V[index] = 1
    diagonal = -2 / dx**2 + V #diag. terms
    off_diagonal = np.ones(nspace - 1) / dx**2
    H = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    # periodic
    H[0, -1] = H[-1, 0] = 1 / dx**2

    print(H) #test


    # if method == 'ftcs':
    #     # Constructing matrix A = I - iτ/ħ H for FTCS
    #     # stability check
    #     # Evolving Psi n using FTCS
    # elif method == 'crank':
    #     # Construct matrices A and B (the parts of the formula in ()) for Crank-Nicholson
    #     # Solving A Psi^(n+1) = B Psi^n
    # else:
    #     raise ValueError("Invalid method. Please choose 'ftcs' or 'crank'.")

    # # BC
    # # checking normalization using probability density (9.44?)

    # return psi_grid, x_grid, t_grid, prob_array

sch_eqn(nspace = 400, ntime = 1000, tau = 0.1, length = 200)

# to test H
# Output: - doesn't look corect
# [[-8.  4.  0. ...  0.  0.  4.]
#  [ 4. -8.  4. ...  0.  0.  0.]
#  [ 0.  4. -8. ...  0.  0.  0.]
#  ...
#  [ 0.  0.  0. ... -8.  4.  0.]
#  [ 0.  0.  0. ...  4. -8.  4.]
#  [ 4.  0.  0. ...  0.  4. -8.]]

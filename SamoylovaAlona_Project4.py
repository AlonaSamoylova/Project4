import numpy as np
import matplotlib.pyplot as plt


# Ψ^(n+1) = (I - (iτ / hbar) * H) * Ψ^n (9.32)
# Ψ^(n+1) = (I + iτ/2ħ H)^(-1) (I - iτ/2ħ H) Ψ^n                      (9.40)


def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=None, wparam=[10, 0, 0.5]):
    """
    Solves the 1D time-dependent Schrödinger equation using FTCS or Crank-Nicholson scheme.
    """
    # Plan:
    # Setup spatial from -L/2 to L/2? and temporal grids
    # initial wave packet Psi(x, 0)
    # Hamiltonian H (tridiagonal matrix) 


    if method == 'ftcs':
        # Constructing matrix A = I - iτ/ħ H for FTCS
        # stability check
        # Evolving Psi n using FTCS
    elif method == 'crank':
        # Construct matrices A and B (the parts of the formula in ()) for Crank-Nicholson
        # Solving A Psi^(n+1) = B Psi^n
    else:
        raise ValueError("Invalid method. Please choose 'ftcs' or 'crank'.")

    # BC
    # checking normalization using probability density (9.44?)

    return psi_grid, x_grid, t_grid, prob_array
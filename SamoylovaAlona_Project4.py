import numpy as np
import matplotlib.pyplot as plt


# Ψ^(n+1) = (I - (iτ / hbar) * H) * Ψ^n (9.32)
# Ψ^(n+1) = (I + iτ/2ħ H)^(-1) (I - iτ/2ħ H) Ψ^n                      (9.40)


def hamiltonian(nspace, potential=None, dx=1): #new function added
    """
    Constructs the Hamiltonian matrix for a free particle with the given potential.
    
    Parameters:
        nspace (int) => Number of spatial grid points.
        potential (1D array or None) => potential values at each grid point. If None, a zero potential is assumed.
        dx (float) => Spatial step size. Default is 1 (normalized).
    
    Returns:
        H (2D array)=> Hamiltonian matrix for the system.
    """
    if potential is None:
        potential = np.zeros(nspace)  # default to zero potential if none provided
    else:
        # to ensure potential array is the same length as nspace by padding with zeros if needed
        if len(potential) < nspace:
            potential = np.pad(potential, (0, nspace - len(potential)), mode='constant')
        elif len(potential) > nspace:
            raise ValueError("Potential array is longer than the number of grid points")
    
    H = np.zeros((nspace, nspace), dtype=float)  # Initializing matrix , firstly i wanted to set type to complex but we need only real values?

    # in the considered case  H=− ∂^2/ ∂x^2x +V(x) if m = 1/2 and nbar=1 by (9.27)

    # Kinetic energy operator (second derivative approximation with periodic boundary)
    for i in range(nspace):
        H[i, i] = -2  # center term (discretized second derivative) 
        H[i, (i + 1) % nspace] = 1  # Right neighbor (BC)   #% is needed for rounding, without it 4 -2 will look like 3.980025  -1.99 in the matrix (for ex.)
        H[i, (i - 1) % nspace] = 1  # Left neighbor (BC)

    # Apply the potential V(x) to the diagonal terms of the Hamiltonian
    for i in range(nspace):
        H[i, i] += potential[i]  # adds potential to the diagonal

    # scaling by -ħ² / 2m (assuming hbar = 1 and m = 1/2 so -h² / 2m = -1)
    H *= -0.5 / dx**2
    
    return H




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
        wparam (list) => Parameters for initial wave packet [sigma0, x0, k0]. Default [10, 0, 0.5].

    Returns:
        psi_grid (2D array) => Wavefunction ψ(x, t) at all grid points and times.
        x_grid (1D array) => Spatial grid points.
        t_grid (1D array) => Time steps.
        prob_array (1D array) => Total probability at each time step.
    """

    # Setup:

    # constants
    hbar = 1  # Planck's constant
    mass = 0.5 ## Mass of the particle as given in the instructions
    L = length  # spatial grid length #system extends from -L/2 to L/2
    dx = L / nspace  # spatial step size
    x_grid = np.linspace(-L / 2, L / 2, nspace, endpoint=False)  # spatial grid
    t_grid = np.linspace(0, ntime * tau, ntime)  # time grid

    # initial wave packet Psi(x, 0)
    # According to the textbook : Gaussian wave packet; the initial wave function is ψ(x, t = 0) = (1 / √(σ₀√π)) * exp[i*k₀x - (x - x₀)² / 2σ₀²]         (9.42)

    # wave packet parameters
    x0 = 0.0           # Initial position of the wave packet center
    velocity = 0.5     # Average velocity of the wave packet
    k0 = mass * velocity / hbar  # Average wave number
    sigma0 = L / 10.0  # Standard deviation of the wave packet
    Norm = 1 / (np.sqrt(sigma0 * np.sqrt(np.pi)))  # Normalization constant

    # Initialize wave function (Gaussian wave packet) - Eq. (9.42)
    psi = Norm * np.exp(-(x_grid - x0)**2 / (2 * sigma0**2)) * np.exp(1j * k0 * x_grid)


    # Hamiltonian H (tridiagonal matrix)                    
    # H = -(hbar^2 / 2m)*(∂^2 / ∂x^2) + V(x) (9.27) then if m = 1/2 and nbar=1 : H=− ∂^2/ ∂x^2x +V(x)
    H = hamiltonian(nspace, potential, dx)


    if method == 'ftcs':
        # Ψ^(n+1) = (I - (iτ / hbar) * H) * Ψ^n (9.32)

        # stability check : "The disadvantage of the FTCS scheme is that it is numerically unstable if the time step is too large."
        # Matrix coefficients; Discretization parameter is given by: r = tau / (dx**2) * hbar / (2 * m)  but nbar/2m =1 so
        r = tau / (dx**2)
        # at the same time h_coeff = ħ² / (2m) so max allowable time step is tau_max = dx**2 / (2 * h_coeff / hbar) but in our case when nbar=1 and m=0.5: #from wiki
        tau_max = dx**2 / 2 # max allowable time step
        if r > 0.5 or tau > tau_max:
            raise ValueError(f"Unstable: Time step τ={tau:.4e} exceeds stability limit τ_max={tau_max:.4e}. Reduce τ.")
        

        # constructing matrix A = I - iτ/ħ H for FTCS
        I = np.identity(nspace, dtype=complex)
        A = I - 1j * tau / hbar * H

        # initializing the wave function grid for all times
        psi_grid = np.zeros((ntime, nspace), dtype=complex)
        psi_grid[0, :] = psi  # IC

        prob_array = np.zeros(ntime)  # to track normalization (total probability)
        prob_array[0] = np.sum(np.abs(psi)**2) * dx  # initial normalization

        # time evolution loop; Evolving Psi n using FTCS
        for n in range(1, ntime):
            psi = A @ psi  # FTCS update: Ψ^(n+1) = A Ψ^n
            psi_grid[n, :] = psi  # to store wavefunction for this time step
            prob_array[n] = np.sum(np.abs(psi)**2) * dx  # computes total probability

    elif method == 'crank':
        # Ψ^(n+1) = (I + iτ/2ħ H)^(-1) (I - iτ/2ħ H) Ψ^n                      (9.40)
        # Construct matrices A and B (the parts of the formula in ()) for Crank-Nicholson
        # Solving A Psi^(n+1) = B Psi^n
    
        # A = I + (iτ / 2ħ) H, B = I - (iτ / 2ħ) H
        A = np.identity(nspace) + 1j * tau / (2 * hbar) * H
        B = np.identity(nspace) - 1j * tau / (2 * hbar) * H

        #to precompute the Crank-Nicholson matrix: dCN = A⁻¹ * B as in example
        dCN = np.dot(np.linalg.inv(A), B)

        # initializing storage for Ψ at all time steps as in 'ftcs' method
        psi_grid = np.zeros((ntime, nspace), dtype=complex)
        prob_array = np.zeros(ntime)  # total probability at each time step
        psi_grid[0, :] = psi  # IC

        # time evolution
        for n in range(1, ntime):
            psi = np.dot(dCN, psi)  # CN update
            # checking normalization using probability density (9.44?)
            # applying normalization  #another way: psi /= np.sqrt(np.sum(np.abs(psi)**2))  # Normalize
            psi /= np.linalg.norm(psi)
            psi_grid[n, :] = psi
            prob_array[n] = np.sum(np.abs(psi)**2) * dx  # Probability   

    else:
        raise ValueError("Invalid method. Please choose 'ftcs' or 'crank'.")

    return psi_grid, x_grid, t_grid, prob_array



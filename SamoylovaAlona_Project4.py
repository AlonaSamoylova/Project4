import numpy as np
import matplotlib.pyplot as plt

# a function to get user input with a default value, needed as we need many (8) possible parameters, as test cases are not given I assumed that all parametrs may be changed

def get_input(prompt, default=None):
    """
    Prompts the user for input and returns the entered value.
    If no input is given, it returns the default value instead.
    Supports both single values (int/float) and lists.
    """
    if default is not None:
        print(f"Suggested value: {default}")  # suggest the default value
    
    user_input = input(f"{prompt}: ")

    # if the user presses Enter without typing, use the default (if available).
    if user_input.strip() == "" and default is not None:
        return default
    
    # try to convert the input to a float for single numbers or lists.
    try:
        return float(user_input)
    except ValueError:
        try:
            # if input cannot be converted to a float, attempt to parse it as a list (for wave packet parameters, etc.).
            return [float(i) for i in user_input.split(',')]
        except ValueError:
            print("Invalid input format. Using default value.")
            return default  # return the default if parsing fails.



def get_input(prompt, default):
    """
    Prompts the user for input and returns the entered value.
    If no input is given, it returns the default value insted.
    """
    user_input = input(f"{prompt}. The default value is: {default}. : ") #"The input() function allows user input". The default value is also shown to th euser
    return float(user_input) if user_input.strip() else default

# prompts for the function
print("Please provide the input values for the simulation (or press Enter to use default test case values):")

# gets number of spatial grid points (suggesting a default value of 400)
nspace = get_input("Please enter number of spatial grid points.The is no hardcoded default value but '400' is suggested", 400)         #the default value was not given in instructions and 
# as I should leave the function def. line unchanged, I cannot hardcode default one but at least can suggest, '400' for ex.

# number of time steps (suggesting a default value of 1000)
ntime = get_input("Please enter the number of time steps. There is no default value but '1000' is suggested", 1000)    #1000 is suggested

# time step (suggesting a default value of 0.1)
tau = get_input("Please enter time step. There is no default value but '0.1' is suggested", 0.1)  #0.1 is suggested

method = get_input("Please enter method", 'ftcs')        #Method to use ('ftcs' or 'crank').

# Length of spatial grid
length = get_input("Please enter the length of spatial grid", 200)        # Default value is 200

# tspatial index values at which the potential V(x) = 1
potential = get_input("Please enter spatial index values at which the potential V(x) = 1", []) #Default to empty.

#parameters for initial wave packet
wparam = get_input("Please enter parameters for initial wave packet [sigma0, x0, k0]", [10, 0, 0.5]) #Default [10, 0, 0.5].


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

def sch_plot(plot_type='psi', t_index=None, save_to_file=False, filename="sch_plot.png"):
    """
    Plots the wave function ψ(x,t) or the probability density |ψ(x,t)|² at a specific time.
    
    Parameters:
        plot_type (str) => 'psi' for the real part of the wave function, 'prob' for the probability density.
        t_index (int) => Time index at which to plot (if None, the last time step is used).
        save_to_file (bool) => If True, saves the plot to a file.
        filename (str) => The filename to save the plot (only if save_to_file=True).
    
    Returns:
        None (Displays the plot).
    """
    # Calling sch_eqn to get its results and use as extra parameters.

    # sch_eqn (function) => Solves the 1D time-dependent Schrödinger equation using FTCS or Crank-Nicholson scheme. It returns the following parameters:
    #     psi_grid (2D array) => Wavefunction ψ(x, t) at all grid points and times.
    #     x_grid (1D array) => Spatial grid points.
    #     t_grid (1D array) => Time steps.
    #     prob_array (1D array) => Total probability at each time step.

    psi_grid, x_grid, t_grid, prob_array = sch_eqn(nspace=nspace, ntime=ntime, tau=tau, method=method, length=length, potential=potential, wparam=wparam)

    # Determines time index for plotting
    if t_index is None:
        t_index = len(t_grid) - 1  # Default to the last time step

    # extracting the time step and corresponding wave function or probability density
    time = t_grid[t_index]
    if plot_type == 'psi':
        # plots the real part of the wave function at the selected time step
        plt.figure(figsize=(10, 6))
        plt.plot(x_grid, np.real(psi_grid[t_index, :]), label=f'ψ(x,t) at t={time:.2f}')
        plt.title(f"Wave Function ψ(x,t) at t = {time:.2f}", fontsize=16)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('ψ(x,t)', fontsize=12)


     
sch_plot()
# # Created by Alona S

# the code is published in the following repository https://github.com/AlonaSamoylova/Project4

# Project 4

import numpy as np
import matplotlib.pyplot as plt

# a function to get user input with a default value, needed as we need many (8) possible parameters, and as test cases were not given I assumed that all parametrs may be changed

import ast 
# needed as list separation by coma in the user prompt wasn't efficient

def get_input(prompt, default=None):
    """
    Prompts the user for input and returns the entered value.
    If no input is given, it returns the default value instead.
    Supports single values (int/float), lists, and strings.
    """
    user_input = input(f"{prompt}. The default value is: {default}. : ")  # Show the prompt with the default value

    # If the user presses Enter without typing, use the default (if available).
    if user_input.strip() == "" and default is not None:
        return default
    
    # converts string input to lowercase before processing
    user_input = user_input.strip().lower()


    # Try to convert the input to a int for single numbers.
    try:
        return int(user_input)
    except ValueError:
        # if conversion fails, check if it's a valid list input
        try:
            # try to interpret the input as a Python-style list using ast.literal_eval
            return ast.literal_eval(user_input)
        except (ValueError, SyntaxError):
            # if it's neither a number nor a list, treat it as a string.
            return user_input.strip()


# prompts for the function
print("Please provide the input values for the simulation (or press Enter to use default test case values):")

# gets number of spatial grid points (suggesting a default value of 300)
nspace = get_input("Please enter number of spatial grid points.The is no hardcoded default value but '300' is suggested", 300)         #the default value was not given in instructions and 
# as I should leave the function def. line unchanged, I cannot hardcode default one but at least can suggest, '300' for ex., 

# number of time steps (suggesting a default value of 100)
ntime = get_input("Please enter the number of time steps. There is no default value but '100' is suggested", 100) 

# time step (suggesting a default value of 0.1)
tau = get_input("Please enter time step. There is no default value but '0.001' is suggested based on a test", 0.001)  #0.001 is suggested 

method = get_input("Please enter method. Type ftcs or crank", 'ftcs')        #Method to use ('ftcs' or 'crank').

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

    # in the considered case  H=− ∂^2/ ∂x^2 +V(x) if m = 1/2 and nbar=1 by (9.27)

    # Kinetic energy operator (second derivative approximation with periodic boundary)
    for i in range(nspace):
        H[i, i] = -2  # center term (discretized second derivative) 
        H[i, (i + 1) % nspace] = 1  # Right neighbor (BC)   #% is needed for rounding, without it 4 -2 will look like 3.980025  -1.99 in the matrix (for ex.)
        H[i, (i - 1) % nspace] = 1  # Left neighbor (BC)

    # applying the potential V(x) to the diagonal terms of the Hamiltonian
    for i in range(nspace):
        H[i, i] += potential[i]  # adds potential to the diagonal

    # scaling by -ħ² / 2m (assuming hbar = 1 and m = 1/2 so -h² / 2m = -1)
    H *= -0.5 / dx**2
    
    return H

# function to get spectral radius for ftcs stability check

# contains the code taken from NairMalavika_SamoylovaAlona_Lab10.py


def spectral_radius(A):
    """
    Calculates the maxium absolute eigenvalue of a 2-D array A
        
    Args:
        A : 2D array from part 1

    Returns:
        maximum absolute value of eigenvalues
    """    
    
    #determine the eigenvalues, only get that value from the tuple
    eigenvalues, _ = np.linalg.eig(A)
    
    #determine the maxiumum absolute value
    max_value = max(abs(eigenvalues))
    
    return max_value

# the main function

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
    # mass = 0.5 ## Mass of the particle as given in the instructions
    L = length  # spatial grid length #system extends from -L/2 to L/2
    dx = L / nspace  # spatial step size
    x_grid = np.linspace(-L / 2, L / 2, nspace, endpoint=False)  # spatial grid
    t_grid = np.linspace(0, ntime * tau, ntime)  # time grid

    # initial wave packet Psi(x, 0)
    # According to the textbook : Gaussian wave packet; the initial wave function is ψ(x, t = 0) = (1 / √(σ₀√π)) * exp[i*k₀x - (x - x₀)² / 2σ₀²]         (9.42)

    # wave packet parameters
    # Let's unpack the initial position of the wave packet center, average wave number and the standard deviation of the wave packet
    sigma0, x0, k0 = wparam

    Norm = 1 / (np.sqrt(sigma0 * np.sqrt(np.pi)))  # Normalization constant


  
    # # initializing wave function (Gaussian wave packet) - Eq. (9.42)
    psi_init = Norm * np.exp(-(x_grid - x0)**2 / (2 * sigma0**2)) * np.exp(1j * k0 * x_grid)

    # initializing the wave function grid for all times
    psi_grid = np.zeros((nspace, ntime), dtype=complex)
    psi_grid[:, 0] = psi_init  # Initial condition
    psi = psi_init.copy()
    

    # initializing Hamiltonian using the following formula:              
    # H = -(hbar^2 / 2m)*(∂^2 / ∂x^2) + V(x) (9.27) then if m = 1/2 and nbar=1 : H=− ∂^2/ ∂x^2x +V(x)
    H = hamiltonian(nspace, potential, dx)

    # Normalize H
    eigenvalues_H, _ = np.linalg.eig(H)
    H_normalized = H / max(abs(eigenvalues_H))

    eigenvalues_H, _ = np.linalg.eig(H)

    #probability storage
    prob_array = np.zeros(ntime)
    prob_array[0] = np.sum(np.abs(psi)**2)

    # let's start solving the equation for each time step
    for n in range(1, ntime):
        # For FTCS method:

        if method == 'ftcs':
            # Ψ^(n+1) = (I - (iτ / hbar) * H) * Ψ^n (9.32)

            # spectral radius stability check
            
            # # constructing matrix A = I - iτ/ħ H is not neccessary for the update (this is set this explicitely) but it can help in checking stability
            A = np.eye(H.shape[0]) - (1j*tau/hbar)*H_normalized
            eigenval = spectral_radius(A)
            #looking for radius
            rho = eigenval -1 #i don't have to take abs of eigenval again as it was taken in 
            if rho> 1e-6:
                #value error to terminate integration if the soln. is unstable
                raise ValueError(f"Unstable: Time step τ={tau:.4e} exceeds stability limit. Reduce τ.")

       
            # updates the wavefunction using FTCS method
            psi = psi + (-1j * tau) * np.dot(H, psi)  # Update ψ^(n+1)
            
        elif method == 'crank':
            # Ψ^(n+1) = (I + iτ/2ħ H)^(-1) (I - iτ/2ħ H) Ψ^n
            # let's construct matrices A and B (the parts of the formula in ()) for Crank-Nicholson scheme
            # and then solve A Psi^(n+1) = B Psi^n
            
            # A = I + (iτ / 2ħ) H, B = I - (iτ / 2ħ) H
            A = np.identity(nspace) + 1j * tau / (2 * hbar) * H
            B = np.identity(nspace) - 1j * tau / (2 * hbar) * H

            # to precompute the Crank-Nicholson matrix: dCN = A⁻¹ * B as in example
            dCN = np.dot(np.linalg.inv(A), B)

            # time evolution
            psi = np.dot(dCN, psi)  # updates psi using precomputed matrix dCN
      
        else:
            raise ValueError("Invalid method. Please choose 'ftcs' or 'crank'.")
        
        #same for all methods
        psi_grid[:, n] = psi
        prob_array[n] = np.sum(np.abs(psi)**2)

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

    # Calling sch_eqn to get the required outputs for plotting
    psi_grid, x_grid, t_grid, prob_array = sch_eqn(nspace=nspace, ntime=ntime, tau=tau, method=method, length=length, potential=potential, wparam=wparam)
    
    # validating t_index (it must be within the bounds of t_grid)
    if t_index is None:
        t_index = len(t_grid) - 1  # Defaults to the last time step
    if t_index < 0 or t_index >= len(t_grid):
        raise ValueError(f"Invalid t_index {t_index}. Must be between 0 and {len(t_grid) - 1}.")
    
    # extracting time and the corresponding data
    time = t_grid[t_index]

    # Plotting the data based on plot_type
    if plot_type == 'psi':
        # plots the real and imaginary parts of the wave function
        plt.figure(figsize=(10, 6))
        plt.plot(x_grid, np.real(psi_grid[:, t_index]), label='Real part of ψ(x,t)') # we need only real part
        plt.title(f"Wave Function ψ(x,t) at t = {time:.2f}", fontsize=16)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('ψ(x,t)', fontsize=12)

    # Explanation:

    # From instructions: prob_array is "1-D array that gives the total probability computed for each timestep (which should be conserved)" 
    # The prob_array matches the instructions perfectly. It ensures the conservation of total probability over time and helps validate the numerical accuracy of chosen method. 
    
    # However, I can't directly use prob_array for plotting is that it represents the total probability over all spatial grid points at each time step, rather than the local probability density at each point in space.
    # so instead of integral of the abs [psi^2] given by prob_array, I need only abs [psi^2]
    # or in other words for spatial plots or local probability densities, i need to compute ∣ψ(x,t)∣ ^2 directly from the wavefunction

    elif plot_type == 'prob':
        # plots the probability density |ψ(x,t)|²
        plt.figure(figsize=(10, 6))
        plt.plot(x_grid, np.abs(psi_grid[:, t_index])**2, label='$|ψ(x,t)|^2$')
        plt.title(f"Probability Density |ψ(x,t)|² at t = {time:.2f}", fontsize=16)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('|ψ(x,t)|²', fontsize=12)

    else:
        # to raise an error if an invalid plot_type is provided
        raise ValueError("Invalid plot_type. Use 'psi' for wave function or 'prob' for probability density.")

    # adds grid and legend to the plot
    plt.grid(True)
    plt.legend()

    # saves the plot to a file if requested
    if save_to_file:
        plt.savefig(filename)
        print(f"Plot saved to {filename}")

    # shows the plot
    plt.show()




print("\n") #adding space #to separate it from previous user input


#asking the user if they want to plot the results and call sch_plot()
plot = input("Do you want to plot the results? Please enter 'yes' to plot: ").strip().lower()

if plot == 'yes':
    # New sch_plot parameter prompts
    plot_type = input("Please enter plot type ('psi' for wavefunction, 'prob' for probability density): ").strip().lower()
    
    # checking plot_type input
    if plot_type not in ['psi', 'prob']:
        print("Invalid input for the plot type. Plotting the default 'psi' plot.")
        plot_type = 'psi'
        
    t_index_input = input("Please enter time index (leave blank for the last time step, enter 0 for t=0): ").strip()
    if t_index_input == '':
        t_index = None  # Default to last time step
    elif t_index_input == '0':
        t_index = 0  # Set to t=0
    else:
        t_index = int(t_index_input)  # Convert input to integer

    t_index = int(t_index_input) if t_index_input else None

    save_to_file_input = input("Do you want to save the plot to a file? (yes or no): ").strip().lower()
    if save_to_file_input in ['yes', 'y']:
        save_to_file = True
        filename = input("Enter filename for saving the plot (default is 'sch_plot.png'): ").strip()
        if not filename:
            filename = "sch_plot.png"
    else:
        save_to_file = False
        filename = "sch_plot.png"  # Exists for the function call but will not be used if save_to_file is False.

    # calling sch_plot()
    sch_plot(plot_type=plot_type, t_index=t_index, save_to_file=save_to_file, filename=filename)
else:
    print("No plot will be generated.")

#           Thank you for reading this!
#               Hope you enjoyed it :)
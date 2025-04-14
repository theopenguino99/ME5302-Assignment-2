# Importing dependencies
import numpy as np
from matplotlib import pyplot as plt
# Defining parameters
N = 121 # 61 or 121 # Number of points along x axis
dx = 1.0 / (N - 1) # Grid spacing in x direction
M = 121 # 61 or 121 # Number of points along y axis
dy = 1.0 / (M - 1) # Grid spacing in y direction
Ra = 3.5e3 # 2.5e4 # Rayleigh number
Pr = 0.7 # Prandtl number
CFL_crit = 2 * np.sqrt(2) # Critical CFL number for stability
margin_h = 0.1 # Margin for the time step
h = margin_h * CFL_crit / (1/dx + 1/dy)/200 # Time step
beta = 1.5 # relaxation factor for iterative methods to solve algebraic equations
'''
For the modified RK4 method, the CFL number has to be smaller than 2*sqrt(2)
The CFL number is defined as:
    CFL = u*dt/dx + v*dt/dy
The maximum velocity is taken to be 1.0
Hence the time step is given by:
    dt = CFL*dx/u = 

'''

# Initialisation at t=0 with boundary conditions
u = np.zeros((N, M)) # x-velocity
v = np.zeros((N, M)) # y-velocity
T = np.zeros((N, M)) # Temperature
rT = np.zeros((N, M)) # Residual temperature
vor = np.zeros((N, M)) # Vorticity
rvor = np.zeros((N, M)) # Residual vorticity
p = np.zeros((N, M)) # Stream function initialised to be 0
rp = np.zeros((N, M)) # Residual stream function
for i in range(N):
    T[i, 0] = 0.5 * np.cos(np.pi * i / (N-1))+1 # Bottom boundary condition of T = 0.5cos(pi*x)+1


def resvor(vor):
    global T, u, v
    rvor = np.zeros_like(vor)
    for i in range(1, N - 1):
        for j in range(1, M - 1):
            dvorx2 = (vor[i+1, j] - 2*vor[i, j] + vor[i-1, j]) / dx**2
            dvory2 = (vor[i, j+1] - 2*vor[i, j] + vor[i, j-1]) / dy**2
            dvorx1 = u[i, j] * (vor[i+1, j] - vor[i-1, j]) / (2*dx)
            dvory1 = v[i, j] * (vor[i, j+1] - vor[i, j-1]) / (2*dy)
            dTx = (T[i+1, j] - T[i-1, j]) / (2*dx)

            rvor[i, j] = (dvorx2 + dvory2) * Pr + Pr * Ra * dTx - dvorx1 - dvory1
            
    return rvor

def restemp(T):
    global u, v
    rtemp = np.zeros_like(T)
    for i in range(1, N - 1):
        for j in range(1, M - 1):
            dTx2 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2
            dTy2 = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2
            dTx1 = u[i, j] * (T[i+1, j] - T[i-1, j]) / (2*dx)
            dTy1 = v[i, j] * (T[i, j+1] - T[i, j-1]) / (2*dy)

            rtemp[i, j] = dTx2 + dTy2 - dTx1 - dTy1
            
    return rtemp

def step(var, method="rk4_modified", ResFunction=resvor):

    # Define 3 different methods for solving the Poisson equation
    # 1. Euler explicit
    # 2. RK4 modified explicit method proposed by James
    # 3. RK4 explicit method (requires more memory to store intermediate variables)
    if method == "euler":
        rvar = ResFunction(var)
        var[1:N-1, 1:M-1] += h * rvar[1:N-1, 1:M-1]
    
    elif method == "rk4_modified":
        vori = np.copy(var)

        # 1st stage
        rvar = ResFunction(var)
        vori[1:N-1, 1:M-1] = var[1:N-1, 1:M-1] + 0.25 * h * rvar[1:N-1, 1:M-1]
        # 2nd stage
        rvar = ResFunction(vori)
        vori[1:N-1, 1:M-1] = var[1:N-1, 1:M-1] + (h / 3.0) * rvar[1:N-1, 1:M-1]
        # 3rd stage
        rvar = ResFunction(vori)
        vori[1:N-1, 1:M-1] = var[1:N-1, 1:M-1] + 0.5 * h * rvar[1:N-1, 1:M-1]
        # 4th stage
        rvar = ResFunction(vori)
        var[1:N-1, 1:M-1] += h * rvar[1:N-1, 1:M-1]
    else:
        raise ValueError("Unknown method: {}".format(method))
    return var


# Calculation of residual of the Poisson equation
def resp(vor, p, conditions=2):
    rp = np.zeros_like(p)
    if conditions == 1:
        range_x = range(1, N-1)
        range_y = range(1, M-1)
    elif conditions == 2:
        range_x = range(2, N-2)
        range_y = range(2, M-2)
    else:
        raise ValueError("Invalid number of conditions. Use 1 or 2.")

    for i in range_x:
        for j in range_y:
            rp[i, j] = vor[i, j] - (
                (p[i+1, j] - 2*p[i, j] + p[i-1, j]) / dx**2 +
                (p[i, j+1] - 2*p[i, j] + p[i, j-1]) / dy**2
            )
    
    return rp

# Calculate \psi_{i,j}^{n+1} (TO BE DONE AFTER UPDATING VORTICITY!)
def solp(vor, p, beta, conditions=2, method = 'point Jacobi relaxation'):
    # Calculate residual of the Poisson equation
    rp = resp(vor, p, conditions)
    
    # Coefficients for iterative methods
    b_W = 1 / dx**2
    b_E = b_W
    b_S = 1 / dy**2
    b_N = b_S
    b_P = 2 * (b_W + b_S)

    # Defining grid to update based on boundary conditions of \psi
    if conditions == 1:
        range_x = range(1, N-1)
        range_y = range(1, M-1)
    elif conditions == 2:
        range_x = range(2, N-2)
        range_y = range(2, M-2)
    else:
        raise ValueError("Invalid number of conditions. Use 1 or 2.")
    
    if method == 'point Jacobi relaxation':
        for i in range_x:
            for j in range_y:
                p[i, j] += beta * rp[i, j] / b_P
    
    return p


def BCp(p):
    # Update p along the vertical boundaries
    for j in range(M):  
        p[1, j] = 0.25 * p[2, j]  
        p[N - 2, j] = 0.25 * p[N - 3, j]  

    # Update p along the horizontal boundaries
    for i in range(N):  
        p[i, 1] = 0.25 * p[i, 2] 
        p[i, M - 2] = 0.25 * p[i, M - 3]

    return p

def BCvor(vor):
    # Update vorticity at the boundaries using 2nd order approximation
    for j in range(M):
        vor[0, j] = 3.0 * p[1, j] / dx**2 - 0.5 * vor[1, j]
        vor[N-1, j] = 3.0 * p[N-2, j] / dx**2 - 0.5 * vor[N-2, j]
    
    # Update along the horizontal boundaries (i-loop)
    for i in range(1, N-1):
        vor[i, 0] = 3.0 * p[i, 1] / dy**2 - 0.5 * vor[i, 1]
        vor[i, M-1] = 3.0 * p[i, M-2] / dy**2 - 0.5 * vor[i, M-2]
    
    return vor  

def BCT(T):
    # Update temperature at the left boundary
    for j in range(M):
        T[0, j] = (4/3) * T[1, j] - (1/3) * T[2, j]
    
    # Update temperature at the right boundary
    for j in range(M):
        T[N-1, j] = (4/3) * T[N-2, j] - (1/3) * T[N-3, j]
    
    return T

def caluv(u,v,p):
    # Apply physical boundary conditions of 0 velocity
    for j in range(M):
        u[0, j] = 0
        u[N-1, j] = 0
        v[0, j] = 0
        v[N-1, j] = 0

    for i in range(N):
        u[i, 0] = 0
        v[i, 0] = 0
        u[i, M-1] = 0
        v[i, M-1] = 0

    # Update velocity components based on stream function
    for i in range(1, N-1):
        for j in range(1, M-1):
            u[i, j] = 0.5 * (p[i, j+1] - p[i, j-1]) / dy
            v[i, j] = -0.5 * (p[i-1, j] - p[i+1, j]) / dx

# Initialise errvor and errp for convergence check to be infinity
errp = np.inf
errvor = np.inf
iter = 23
# Function to update the heatmap for each frame
# while errp > 1e-4 and errvor > 1e-4:
for _ in range(iter):

    # Use global variables to update them in the loop


    # Perform one time step of the simulation:
    # Compute residual vorticity in place and update vorticity
    vor = step(vor, method="euler", ResFunction=resvor)

    # Compute residual Poisson equation in place and update stream function
    p = solp(vor, p, beta, conditions=2, method='point Jacobi relaxation')

    # Update boundary conditions for stream function
    p = BCp(p)

    # Update boundary conditions for vorticity
    vor = BCvor(vor)

    # Update velocity components in place based on stream function
    caluv(u, v, p)

    # Compute residual temperature in place and update temperature
    T = step(T, method="euler", ResFunction=restemp)

    # Update Temprature field
    T = BCT(T)

    # Update errvor and errp which resemble L2 norm using the sum of squares
    errvor = np.sqrt(np.sum(vor[1:N-1, 1:M-1]**2)/(N-2)/(M-2))
    errp = np.sqrt(np.sum(p[1:N-1, 1:M-1]**2)/(N-2)/(M-2))
    print("errp: ", errp, "errvor: ", errvor)

print('My time step is:', h)
print('Total time elapsed is:', h*iter)
print('My errp is:', errp)  
print('My errvor is:', errvor)

# Plot Temperature
plt.figure(figsize=(8, 6))
plt.imshow(T.T, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', aspect='auto')
plt.title("Temperature Field")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="Temperature")
plt.show()
# Plot velocity field in a different figure
plt.figure(figsize=(8, 6))
plt.imshow(u.T, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', aspect='auto')
plt.title("Velocity Field (u-component)")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="Velocity (u)")
plt.show()


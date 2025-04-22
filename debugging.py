# Importing dependencies
import numpy as np
from matplotlib import pyplot as plt
import time
# Defining parameters
N = 61 # 121 # Number of points along x axis
dx = 1.0 / (N - 1) # Grid spacing in x direction
M = 61 # 121 # Number of points along y axis
dy = 1.0 / (M - 1) # Grid spacing in y direction
Ra = 3.5e3 # 2.5e4 # Rayleigh number
Pr = 0.7 # Prandtl number
h = 0.00001 #Time step
beta = 0.4 # relaxation factor for iterative methods to solve algebraic equations
iter = 1500
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
    rvor = np.zeros_like(vor)
    for i in range(1, N - 1):
        for j in range(1, M - 1):
            dvorx2 = (vor[i+1, j] - 2*vor[i, j] + vor[i-1, j]) / (dx**2)
            dvory2 = (vor[i, j+1] - 2*vor[i, j] + vor[i, j-1]) / (dy**2)
            dvorx1 = u[i, j] * (vor[i+1, j] - vor[i-1, j]) / (2*dx)
            dvory1 = v[i, j] * (vor[i, j+1] - vor[i, j-1]) / (2*dy)
            dTx = (T[i+1, j] - T[i-1, j]) / (2*dx)

            rvor[i, j] = (dvorx2 + dvory2) * Pr + Pr * Ra * dTx - dvorx1 - dvory1
            
    return rvor

def restemp(T):
    rtemp = np.zeros_like(T)
    for i in range(1, N - 1):
        for j in range(1, M - 1):
            dTx2 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / (dx**2)
            dTy2 = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / (dy**2)
            dTx1 = u[i, j] * (T[i+1, j] - T[i-1, j]) / (2*dx)
            dTy1 = v[i, j] * (T[i, j+1] - T[i, j-1]) / (2*dy)

            rtemp[i, j] = dTx2 + dTy2 - dTx1 - dTy1
            
    return rtemp

def solT(rT, method="euler"):
    
    if method == "euler":
        T[1:N-1, 1:M-1] += h * rT[1:N-1, 1:M-1]
    
    elif method == "rk4_modified":
        Ti = np.copy(T)
        # 1st stage
        Ti[1:N-1, 1:M-1] = T[1:N-1, 1:M-1] + 0.25 * h * rT[1:N-1, 1:M-1]
        # 2nd stage
        rT = restemp(Ti)
        Ti[1:N-1, 1:M-1] = T[1:N-1, 1:M-1] + (h / 3.0) * rT[1:N-1, 1:M-1]
        # 3rd stage
        rT = restemp(Ti)
        Ti[1:N-1, 1:M-1] = T[1:N-1, 1:M-1] + 0.5 * h * rT[1:N-1, 1:M-1]
        # 4th stage
        rT = restemp(Ti)
        T[1:N-1, 1:M-1] += h * rT[1:N-1, 1:M-1]

    elif method == 'Point Jacobi':
        a_p_T = (2/dx**2 + 2/dy**2) + 1/h
        for i in range(1, N - 1):
            for j in range(1, M - 1):
                T[i, j] += beta * rT[i, j] / a_p_T

    return T

def solvor(rvor, method="euler"):

    if method == "euler":
        vor[1:N-1, 1:M-1] += h * rvor[1:N-1, 1:M-1]
    
    elif method == "rk4_modified":
        vori = np.copy(vor)

        # 1st stage
        vori[1:N-1, 1:M-1] = vor[1:N-1, 1:M-1] + 0.25 * h * rvor[1:N-1, 1:M-1]
        # 2nd stage
        rvor = resvor(vori)
        vori[1:N-1, 1:M-1] = vor[1:N-1, 1:M-1] + (h / 3.0) * rvor[1:N-1, 1:M-1]
        # 3rd stage
        rvor = resvor(vori)
        vori[1:N-1, 1:M-1] = vor[1:N-1, 1:M-1] + 0.5 * h * rvor[1:N-1, 1:M-1]
        # 4th stage
        rvor = resvor(vori)
        vor[1:N-1, 1:M-1] += h * rvor[1:N-1, 1:M-1]
    
    elif method == 'Point Jacobi':
        a_p_vor = Pr * (2/dx**2 + 2/dy**2) + 1/h
        for i in range(1, N - 1):
            for j in range(1, M - 1):
                vor[i, j] += beta * rvor[i, j] / a_p_vor
        
    return vor

# Calculation of residual of the Poisson equation
def resp(p):
    rp = np.zeros_like(p)
    
    for i in range(1, N-2):
        for j in range(1, M-2):
            rp[i, j] = vor[i, j] - (
                (p[i+1, j] - 2*p[i, j] + p[i-1, j]) / (dx**2) +
                (p[i, j+1] - 2*p[i, j] + p[i, j-1]) / (dy**2)
            )
    
    return rp

def solp(rp, beta):
    # Coefficients for iterative methods
    b_W = 1 / dx**2
    b_S = 1 / dy**2
    b_P = -2 * (b_W + b_S)
    
    for i in range(2, N-2):
        for j in range(1, M-1):
            p[i, j] += beta * rp[i, j] / b_P

    # # Update p at the boundaries
    # p[0, :] = 0 #Left
    # p[N - 1, :] = 0 #Right
    # p[:, 0] = 0 #Bottom
    # p[:, M - 1] = 0 #Top
            
    return p

def BCp(p):
    # Update p along the vertical boundaries
    for j in range(1, M-1):  
        p[1, j] = 0.25 * p[2, j]  #Left
        p[N - 2, j] = 0.25 * p[N - 3, j]  #Right

    # Update p along the horizontal boundaries
    for i in range(1, N-1):  
        p[i, 1] = 0.25 * p[i, 2] #Bottom
        p[i, M - 2] = 0.25 * p[i, M - 3] #Top

    # Update p at the boundaries
    p[0, :] = 0 #Left
    p[N - 1, :] = 0 #Right
    p[:, 0] = 0 #Bottom
    p[:, M - 1] = 0 #Top

    return p

def BCvor(vor):
    # Update vorticity at the boundaries using 2nd order approximation
    for j in range(M):
        vor[0, j] = 3.0 * p[1, j] / (dx**2) - 0.5 * vor[1, j]
        vor[N-1, j] = 3.0 * p[N-2, j] / (dx**2) - 0.5 * vor[N-2, j]
    
    # Update along the horizontal boundaries (i-loop)
    for i in range(1, N-1):
        vor[i, 0] = 3.0 * p[i, 1] / (dy**2) - 0.5 * vor[i, 1]
        vor[i, M-1] = 3.0 * p[i, M-2] / (dy**2) - 0.5 * vor[i, M-2]
    
    return vor  

def BCT(T):
    # Update temperature at the left boundary
    for j in range(M):
        T[0, j] = (4/3) * T[1, j] - (1/3) * T[2, j]
    
    # Update temperature at the right boundary
    for j in range(M):
        T[N-1, j] = (4/3) * T[N-2, j] - (1/3) * T[N-3, j]

    # Update temperature at the top boundary (added: isothermal condition T=0)
    for i in range(N):
        T[i, N-1] = 0.0
    
    return T

def caluv(u,v,p):
    # Apply physical boundary conditions of 0 velocity
    for j in range(M):
        u[0, j] = 0
        u[N-1, j] = 0
        v[0, j] = 0
        v[N-1, j] = 0

    for i in range(1, N-1):
        u[i, 0] = 0
        v[i, 0] = 0
        u[i, M-1] = 0
        v[i, M-1] = 0

    # Update velocity components based on stream function
    for i in range(1, N-1):
        for j in range(1, M-1):
            u[i, j] = 0.5 * (p[i, j+1] - p[i, j-1]) / dy
            v[i, j] = 0.5 * (p[i-1, j] - p[i+1, j]) / dx

# Initialise errvor and errp for convergence check to be infinity
errp = np.inf
errvor = np.inf
errT = np.inf
iter_no = 0
# Function to update the heatmap for each frame
# while errp > 1e-4 and errvor > 1e-4:

start_time = time.time()

for _ in range(iter):

    # Compute residual vorticity in place and update vorticity
    rvor = resvor(vor)
    vor = solvor(rvor, method="euler")

    # Compute residual Poisson equation in place and update stream function
    rp = resp(p)
    p = solp(rp, beta)

    # Update boundary conditions for stream function
    p = BCp(p)

    # Update boundary conditions for vorticity
    vor = BCvor(vor)

    # Update velocity components in place based on stream function
    caluv(u, v, p)

    # Compute residual temperature in place and update temperature
    rT = restemp(T)
    T = solT(rT, method="euler")

    # Update Temprature field
    T = BCT(T)

    # Update iteration number
    iter_no += 1

    errvor = np.sqrt(np.sum(rvor**2))
    errp = np.sqrt(np.sum(rp**2))
    errT = np.sqrt(np.sum(rT**2))

    if iter_no % 100 == 0:
        print('Iteration number', iter_no, "errp: ", errp, "errvor: ", errvor, "errT: ", errT)
    
end_time = time.time()
total_time = end_time - start_time

print('My time step is:', h)
print('Total time elapsed is:', h*iter)
print('Total time taken for the simulation:', total_time, 'seconds')

# Create 2x2 grid and plot contours for T, p, u, and v
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# Plot Temperature contours 
contour1 = axes[0, 0].contourf(T.T, levels=20, cmap='viridis')
axes[0, 0].set_title("Temperature Contours")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
fig.colorbar(contour1, ax=axes[0, 0], label="Temperature")
# Plot Stream function contours
contour2 = axes[0, 1].contourf(p.T, levels=20, cmap='viridis')
axes[0, 1].set_title("Stream function Contours")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")
fig.colorbar(contour2, ax=axes[0, 1], label="Stream function")
# Plot X-velocity contours
contour3 = axes[1, 0].contourf(u.T, levels=20, cmap='viridis')
axes[1, 0].set_title("X-velocity Contours")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
fig.colorbar(contour3, ax=axes[1, 0], label="X-velocity")
# Plot Y-velocity contours
contour4 = axes[1, 1].contourf(v.T, levels=20, cmap='viridis')
axes[1, 1].set_title("Y-velocity Contours")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")
fig.colorbar(contour4, ax=axes[1, 1], label="Y-velocity")
# Importing dependencies
import numpy as np
from matplotlib import pyplot as plt
import time
# Defining parameters
N = 61 # 61 or 121 # Number of points along x axis
dx = 1.0 / (N - 1) # Grid spacing in x direction
M = 61 # 61 or 121 # Number of points along y axis
dy = 1.0 / (M - 1) # Grid spacing in y direction
Ra = 3.5e3 # 2.5e4 # Rayleigh number
Pr = 0.7 # Prandtl number
CFL_crit = 2 * np.sqrt(2) # Critical CFL number for stability
margin_h = 0.1 # Margin for the time step
h = 0.00001 #margin_h * CFL_crit / (1/dx + 1/dy)/ 60 # Time step
beta = 0.4 # relaxation factor for iterative methods to solve algebraic equations
iter = 6000

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
dTdx1 = np.zeros((M, N))
dTdy1 = np.zeros((M, N))

# Initialise errvor and errp for convergence check to be infinity
errp = np.inf
errvor = np.inf
iter_no = 0

start_time = time.time()

for _ in range(iter):

    # Compute residual vorticity in place and update vorticity
    rvor = np.zeros_like(vor)
    for i in range(1, N - 1):
        for j in range(1, M - 1):
            dvorx2 = (vor[i+1, j] - 2*vor[i, j] + vor[i-1, j]) / dx**2
            dvory2 = (vor[i, j+1] - 2*vor[i, j] + vor[i, j-1]) / dy**2
            dvorx1 = u[i, j] * (vor[i+1, j] - vor[i-1, j]) / (2*dx)
            dvory1 = v[i, j] * (vor[i, j+1] - vor[i, j-1]) / (2*dy)
            dTx = (T[i+1, j] - T[i-1, j]) / (2*dx)

            rvor[i, j] = (dvorx2 + dvory2) * Pr + Pr * Ra * dTx - dvorx1 - dvory1
    
    vor[1:N-1, 1:M-1] += h * rvor[1:N-1, 1:M-1]
    # a_p = Pr * (2/dx**2 + 2/dy**2) + 1/h
    # for i in range(1, N - 1):
    #     for j in range(1, M - 1):
    #         vor[i, j] += beta * rvor[i, j] / a_p
    
    # Compute residual Poisson equation in place and update stream function
    rp = np.zeros_like(p)
    for i in range(2, N-2):
        for j in range(2, M-2):
            rp[i, j] = vor[i, j] - (
                (p[i+1, j] - 2*p[i, j] + p[i-1, j]) / dx**2 +
                (p[i, j+1] - 2*p[i, j] + p[i, j-1]) / dy**2
            )
    
    b_W = 1 / dx**2
    b_E = b_W
    b_S = 1 / dy**2
    b_N = b_S
    b_P = -2 * (b_W + b_S)

    p[2:N-2, 2:M-2] += beta * rp[2:N-2, 2:M-2] / b_P

    # Update boundary conditions for stream function
    for j in range(1, M-1):  
        p[1, j] = 0.25 * p[2, j]  
        p[N - 2, j] = 0.25 * p[N - 3, j]  

    for i in range(1, N-1):  
        p[i, 1] = 0.25 * p[i, 2] 
        p[i, M - 2] = 0.25 * p[i, M - 3]

    # Update boundary conditions for vorticity
    for j in range(M):
        vor[0, j] = (3.0 * p[1, j]) / dx**2 - 0.5 * vor[1, j]
        vor[N-1, j] = (3.0 * p[N-2, j]) / dx**2 - 0.5 * vor[N-2, j]
    
    for i in range(1, N-1):
        vor[i, 0] = (3.0 * p[i, 1]) / dy**2 - 0.5 * vor[i, 1]
        vor[i, M-1] = (3.0 * p[i, M-2]) / dy**2 - 0.5 * vor[i, M-2]

    # Update velocity components in place based on stream function
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
        dTdy1[i, 0] = 0
        dTdy1[i, M-1] = 0

    for i in range(1, N-1):
        for j in range(1, M-1):
            u[i, j] = 0.5 * (p[i, j+1] - p[i, j-1]) / dy
            v[i, j] = 0.5 * (p[i-1, j] - p[i+1, j]) / dx
            dTdx1[i,j] = (T[i+1,j]-T[i-1,j]) / (2*dx)
            dTdy1[i,j] = (T[i,j+1]-T[i,j-1]) / (2*dy)

    # Compute residual temperature in place and update temperature
    Ti = np.copy(T)
    
    # 1st stage
    rtemp = np.zeros_like(T)
    for i in range(1, N - 1):
        for j in range(1, M - 1):
            dTx2 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2
            dTy2 = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2
            dTx1 = u[i, j] * (T[i+1, j] - T[i-1, j]) / (2*dx)
            dTy1 = v[i, j] * (T[i, j+1] - T[i, j-1]) / (2*dy)

            rtemp[i, j] = dTx2 + dTy2 - dTx1 - dTy1


    # a_p_T = (2/dx**2 + 2/dy**2) + 1/h
    # for i in range(1, N - 1):
    #     for j in range(1, M - 1):
    #         T[i, j] += beta * rtemp[i, j] / a_p_T

    T[1:N-1, 1:M-1] += h * rtemp[1:N-1, 1:M-1]

    # Ti[1:N-1, 1:M-1] = T[1:N-1, 1:M-1] + 0.25 * h * rtemp[1:N-1, 1:M-1]
    # # 2nd stage
    # rtempi = np.zeros_like(Ti)
    # for i in range(1, N - 1):
    #     for j in range(1, M - 1):
    #         dTx2 = (Ti[i+1, j] - 2*Ti[i, j] + Ti[i-1, j]) / dx**2
    #         dTy2 = (Ti[i, j+1] - 2*Ti[i, j] + Ti[i, j-1]) / dy**2
    #         dTx1 = u[i, j] * (Ti[i+1, j] - Ti[i-1, j]) / (2*dx)
    #         dTy1 = v[i, j] * (Ti[i, j+1] - Ti[i, j-1]) / (2*dy)

    #         rtempi[i, j] = dTx2 + dTy2 - dTx1 - dTy1
    
    # Ti[1:N-1, 1:M-1] = T[1:N-1, 1:M-1] + (h / 3.0) * rtempi[1:N-1, 1:M-1]
    # # 3rd stage
    # rtempii = np.zeros_like(T)
    # for i in range(1, N - 1):
    #     for j in range(1, M - 1):
    #         dTx2 = (Ti[i+1, j] - 2*Ti[i, j] + Ti[i-1, j]) / dx**2
    #         dTy2 = (Ti[i, j+1] - 2*Ti[i, j] + Ti[i, j-1]) / dy**2
    #         dTx1 = u[i, j] * (Ti[i+1, j] - Ti[i-1, j]) / (2*dx)
    #         dTy1 = v[i, j] * (Ti[i, j+1] - Ti[i, j-1]) / (2*dy)

    #         rtempii[i, j] = dTx2 + dTy2 - dTx1 - dTy1
    # Ti[1:N-1, 1:M-1] = T[1:N-1, 1:M-1] + 0.5 * h * rtempii[1:N-1, 1:M-1]
    # # 4th stage
    # rtempiii = np.zeros_like(T)
    # for i in range(1, N - 1):
    #     for j in range(1, M - 1):
    #         dTx2 = (Ti[i+1, j] - 2*Ti[i, j] + Ti[i-1, j]) / dx**2
    #         dTy2 = (Ti[i, j+1] - 2*Ti[i, j] + Ti[i, j-1]) / dy**2
    #         dTx1 = u[i, j] * (Ti[i+1, j] - Ti[i-1, j]) / (2*dx)
    #         dTy1 = v[i, j] * (Ti[i, j+1] - Ti[i, j-1]) / (2*dy)

    #         rtempiii[i, j] = dTx2 + dTy2 - dTx1 - dTy1
    # T[1:N-1, 1:M-1] += h * rtempiii[1:N-1, 1:M-1]

    # Update Temprature field
    for j in range(M):
        T[0, j] = (4/3) * T[1, j] - (1/3) * T[2, j]
    
    for j in range(M):
        T[N-1, j] = (4/3) * T[N-2, j] - (1/3) * T[N-3, j]

    # Update iteration number
    iter_no += 1

    # Update errvor and errp which resemble L2 norm using the sum of squares
    errvor = np.max(np.abs(rvor[1:M-1,1:N-1])) #np.sqrt(np.sum(vor[1:N-1, 1:M-1]**2)/(N-2)/(M-2))
    errp = np.max(np.abs(rp[2:M-2,2:N-2])) #np.sqrt(np.sum(p[1:N-1, 1:M-1]**2)/(N-2)/(M-2))
    print('Iteration number', iter_no, "errp: ", errp, "errvor: ", errvor)

print('My time step is:', h)
print('Total time elapsed is:', h*iter)
print('My errp is:', errp)  
print('My errvor is:', errvor)

# Create a 2x2 grid for plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot Temperature
im1 = axes[0, 0].imshow(T.T, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', aspect='auto')
axes[0, 0].set_title("Temperature Field")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
fig.colorbar(im1, ax=axes[0, 0], label="Temperature")

# Plot Vorticity
im2 = axes[0, 1].imshow(vor.T, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', aspect='auto')
axes[0, 1].set_title("Vorticity Field")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")
fig.colorbar(im2, ax=axes[0, 1], label="Vorticity")

# Plot X-velocity
im3 = axes[1, 0].imshow(u.T, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', aspect='auto')
axes[1, 0].set_title("X-velocity Field")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
fig.colorbar(im3, ax=axes[1, 0], label="X-velocity")

# Plot Y-velocity
im4 = axes[1, 1].imshow(v.T, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', aspect='auto')
axes[1, 1].set_title("Y-velocity Field")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")
fig.colorbar(im4, ax=axes[1, 1], label="Y-velocity")

# Adjust layout
plt.tight_layout()
plt.show()
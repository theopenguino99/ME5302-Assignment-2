import numpy as np
import matplotlib.pyplot as plt


# ================================
# Define all functions
# ================================

# 1) Setting iteration and convergence criteria
itr = 1
eplon1 = 1e-3
eplon2 = 1e-4
eplon3 = 1e-3
dt = 0.00001  # time step size

Pr = 0.7    # Prandtl number
Ra = 3.5e3  # Rayleigh number

# 2) Mesh generation
N = 61
M = 61

dx = 1 / (N-1)
dy = 1 / (M-1)
beta = 0.4

# 3) Set up initial flow field
u = np.zeros((N, M))
v = np.zeros((N, M))
T = np.zeros((N, M))
p = np.zeros((N, M))
vor = np.zeros((N, M))
dTdx = np.zeros((N, M))
dTdy = np.zeros((N, M))
rvor = np.zeros((N, M))
rp = np.zeros((N, M))
rT = np.zeros((N, M))

def resvor(vor):
    rvor = np.zeros_like(vor)
    for i in range(1, N-1):
        for j in range(1, M-1):
            dvordx2 = (vor[i+1,j]-2*vor[i,j]+vor[i-1,j]) / dx**2
            dvordy2 = (vor[i,j+1]-2*vor[i,j]+vor[i,j-1]) / dy**2
            dTdx = (T[i+1,j]-T[i-1,j]) / (2*dx)
            dvordx1 = (vor[i+1,j]-vor[i-1,j]) / (2*dx)
            dvordy1 = (vor[i,j+1]-vor[i,j-1]) / (2*dy)
            rvor[i,j] = Pr * (dvordx2 + dvordy2) + Pr * Ra * dTdx - u[i,j] * dvordx1 - v[i,j] * dvordy1
    return rvor

def solvor(rvor):
    vor[1:M-1,1:N-1] += rvor[1:M-1,1:N-1] * dt
    return vor

def solve_rT(T):
    rT = np.zeros_like(T)
    for i in range(1, N-1):
        for j in range(1, M-1):
            dTdx2 = (T[i+1,j]-2*T[i,j]+T[i-1,j]) / dx**2
            dTdy2 = (T[i,j+1]-2*T[i,j]+T[i,j-1]) / dy**2
            dTdx = (T[i+1,j]-T[i-1,j]) / (2*dx)
            dTdy = (T[i,j+1]-T[i,j-1]) / (2*dy)
            rT[i,j] = dTdx2 + dTdy2 - (u[i,j]*dTdx + v[i,j]*dTdy)
    return rT

def solve_T(rT):
    T[1:N-1,1:M-1] += rT[1:N-1,1:M-1] * dt
    return T

def solve_rp(p):
    rp = np.zeros_like(p)
    for i in range(2, M-2):
        for j in range(2, N-2):
            dpdx2 = (p[i+1,j]-2*p[i,j]+p[i-1,j]) / dx**2
            dpdy2 = (p[i,j+1]-2*p[i,j]+p[i,j-1]) / dy**2
            rp[i,j] = vor[i,j] - (dpdx2 + dpdy2)
    return rp

def solve_p(rp, beta):
    
    b_W = 1/dx**2
    b_S = 1/dy**2
    b_P = -2 * (b_W + b_S)
    p[2:M-2,2:N-2] += rp[2:M-2,2:N-2] * beta / b_P
    return p

def BC_p(p):
    p[1:N-1,1] = 0.25 * p[1:N-1,2]
    p[1:N-1,M-2] = 0.25 * p[1:N-1,M-3]
    p[1,1:M-1] = 0.25 * p[2,1:M-1]
    p[N-2,1:M-1] = 0.25 * p[N-3,1:M-1]
    return p

def BC_T(T):
    T[0,:] = (4*T[1,:] - T[2,:]) / 3
    T[N-1,:] = (4*T[N-2,:] - T[N-3,:]) / 3
    return T

def BC_vor(vor):
    vor[:,0] = (3/dx**2)*p[:,1] - 0.5*vor[:,1]
    vor[:,M-1] = (3/dx**2)*p[:,M-2] - 0.5*vor[:,M-2]
    vor[0,:] = (3/dy**2)*p[1,:] - 0.5*vor[1,:]
    vor[N-1,:] = (3/dy**2)*p[N-2,:] - 0.5*vor[N-2,:]
    return vor

def compute_velocity(u,v,p):
    u[:,[0,-1]] = 0
    v[:,[0,-1]] = 0
    u[[0,-1],:] = 0
    v[[0,-1],:] = 0
    # dTdy[[0,-1],:] = 0
    for j in range(1,M-1):
        for i in range(1,N-1):
            u[j,i] = (p[j+1,i] - p[j-1,i]) / (2*dy)
            v[j,i] = (p[j,i-1] - p[j,i+1]) / (2*dx)
            # dTdx[j,i] = (T[j,i+1]-T[j,i-1]) / (2*dx)
            # dTdy[j,i] = (T[j+1,i]-T[j-1,i]) / (2*dy)
    # return u, v, dTdx, dTdy

def compute_err(rvor, rp, rT, M, N):
    
    errvor = np.max(np.abs(rvor[1:M-1,1:N-1]))
    errp = np.max(np.abs(rp[2:M-2,2:N-2]))
    errT = np.max(np.abs(rT[1:M-1,1:N-1]))
    return errvor, errp, errT

# def compute_Nu_vel(M, N, u, v, dTdy):
#     u_max = np.max(u[:,N//2])
#     u_idx = np.argmax(u[:,N//2])
#     v_max = np.max(v[M//2,:])
#     v_idx = np.argmax(v[M//2,:])
#     Nu_0 = np.mean(np.abs(dTdy[1,1:N-1]))
#     Nu_half = np.mean(np.abs(dTdy[N//2,1:N-1]))
#     Nu_max = np.max(np.abs(dTdy[1,1:N-1]))
#     Numax_idx = np.argmax(np.abs(dTdy[1,1:N-1]))
#     return u_max, v_max, Nu_0, Nu_half, Nu_max, u_idx, v_idx, Numax_idx



# 4) Set temperature boundary condition
for i in range(N):
    T[M-1, i] = 0.5 * np.cos(np.pi * i * dx) + 1  # Lower boundary

# 5) Enter loop to solve
iter = 1000
for _ in range(iter):
    
    rvor = resvor(vor)
    vor = solvor(rvor)

    rp = solve_rp(p)
    p = solve_p(p, beta)

    p = BC_p(p)

    vor = BC_vor(vor)

    compute_velocity(u, v, p)

    rT = solve_rT(T)
    T = solve_T(rT)

    T = BC_T(T)

    errvor, errp, errT = compute_err(rvor, rp, rT, M, N)
    

    # if itr % 100 == 0:
    print(f"Iteration: {itr}, Errorvor: {errvor}, ErrorT: {errT}, Errorp: {errp}")

    # if (errvor <= eplon1) and (errp <= eplon2) and (errT <= eplon3):
    #     break

    itr += 1

# 7) Plot contours
# Create a 2x2 grid for plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot Temperature
im1 = axes[0, 0].imshow(T, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', aspect='auto')
axes[0, 0].set_title("Temperature Field")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
fig.colorbar(im1, ax=axes[0, 0], label="Temperature")

# Plot vorticity
im2 = axes[0, 1].imshow(vor, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', aspect='auto')
axes[0, 1].set_title("vorticity Field")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")
fig.colorbar(im2, ax=axes[0, 1], label="vorticity")

# Plot X-velocity
im3 = axes[1, 0].imshow(u, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', aspect='auto')
axes[1, 0].set_title("X-velocity Field")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
fig.colorbar(im3, ax=axes[1, 0], label="X-velocity")

# Plot Y-velocity
im4 = axes[1, 1].imshow(v, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', aspect='auto')
axes[1, 1].set_title("Y-velocity Field")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")
fig.colorbar(im4, ax=axes[1, 1], label="Y-velocity")

# Adjust layout
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
import time

def chebyshev_gauss_lobatto_points(n):
    """Generate Chebyshev-Gauss-Lobatto points in [0,1]"""
    i = np.arange(n)
    x = 0.5 * (1 - np.cos(i * np.pi / (n - 1)))
    return x

def compute_gdq_weights(x, n):
    """Compute the GDQ weighting coefficients for first and second derivatives"""
    # First derivative weighting coefficients
    w1 = np.zeros((n, n))
    # Second derivative weighting coefficients
    w2 = np.zeros((n, n))
    
    # Compute first derivative weights
    for i in range(n):
        for j in range(n):
            if i != j:
                w1[i, j] = 1.0
                for k in range(n):
                    if k != i and k != j:
                        w1[i, j] *= (x[i] - x[k]) / (x[j] - x[k])
                w1[i, j] /= (x[i] - x[j])
    
    # Compute row sum rule for diagonal elements
    for i in range(n):
        w1[i, i] = -np.sum(w1[i, :])
    
    # Compute second derivative weights
    for i in range(n):
        for j in range(n):
            if i != j:
                w2[i, j] = 2 * w1[i, j] * w1[i, i] - 2 * w1[i, j] / (x[i] - x[j])
            else:
                w2[i, i] = 0
                for k in range(n):
                    if k != i:
                        w2[i, i] -= 2 * w1[i, k] * w1[i, k]
    
    return w1, w2

# Define simulation parameters
N = 21  # Number of points along x axis (keep moderate for GDQ)
M = 21  # Number of points along y axis
Ra = 3.5e3  # Rayleigh number
Pr = 0.7  # Prandtl number
max_iter = 300  # Maximum number of iterations
tol = 1e-6  # Convergence tolerance
beta = 0.1  # Relaxation parameter

# Generate non-uniform grid using Chebyshev-Gauss-Lobatto points
x = chebyshev_gauss_lobatto_points(N)
y = chebyshev_gauss_lobatto_points(M)

# Compute GDQ weights for derivatives
wx1, wx2 = compute_gdq_weights(x, N)
wy1, wy2 = compute_gdq_weights(y, M)

# Initialize fields
T = np.zeros((N, M))  # Temperature
psi = np.zeros((N, M))  # Stream function
vor = np.zeros((N, M))  # Vorticity
u = np.zeros((N, M))  # x-velocity
v = np.zeros((N, M))  # y-velocity

# Apply temperature boundary conditions
for i in range(N):
    T[i, 0] = 0.5 * np.cos(np.pi * x[i]) + 1  # Bottom boundary condition

# Create meshgrid for plotting
X, Y = np.meshgrid(x, y, indexing='ij')

start_time = time.time()

# Main iteration loop
for iter_no in range(max_iter):
    # Store previous values for convergence check
    psi_old = psi.copy()
    T_old = T.copy()
    vor_old = vor.copy()
    
    # Step 1: Solve stream function equation (Poisson equation)
    # ∇²ψ = -ω
    for i in range(1, N-1):
        for j in range(1, M-1):
            rhs = -vor[i, j]
            lap_psi = 0
            # Second derivatives in x and y
            for k in range(N):
                lap_psi += wx2[i, k] * psi[k, j]
            for k in range(M):
                lap_psi += wy2[j, k] * psi[i, k]
            
            # Update stream function using relaxation
            psi[i, j] = psi[i, j] + beta * (rhs - lap_psi)
    
    # Apply boundary conditions for stream function (zero at all boundaries)
    psi[0, :] = 0
    psi[N-1, :] = 0
    psi[:, 0] = 0
    psi[:, M-1] = 0
    
    # Step 2: Compute velocity components from stream function
    for i in range(1, N-1):
        for j in range(1, M-1):
            u[i, j] = 0
            v[i, j] = 0
            for k in range(M):
                u[i, j] += wy1[j, k] * psi[i, k]
            for k in range(N):
                v[i, j] -= wx1[i, k] * psi[k, j]
    
    # Apply no-slip boundary conditions
    u[0, :] = 0
    u[N-1, :] = 0
    u[:, 0] = 0
    u[:, M-1] = 0
    v[0, :] = 0
    v[N-1, :] = 0
    v[:, 0] = 0
    v[:, M-1] = 0
    
    # Step 3: Update vorticity field
    for i in range(1, N-1):
        for j in range(1, M-1):
            # Compute Laplacian of vorticity
            lap_vor = 0
            for k in range(N):
                lap_vor += wx2[i, k] * vor[k, j]
            for k in range(M):
                lap_vor += wy2[j, k] * vor[i, k]
            
            # Compute advection terms
            adv_x = 0
            adv_y = 0
            for k in range(N):
                adv_x += u[i, j] * wx1[i, k] * vor[k, j]
            for k in range(M):
                adv_y += v[i, j] * wy1[j, k] * vor[i, k]
            
            # Compute temperature gradient in x
            dTx = 0
            for k in range(N):
                dTx += wx1[i, k] * T[k, j]
            
            # Update vorticity using relaxation
            rhs = Pr * lap_vor + Pr * Ra * dTx - adv_x - adv_y
            vor[i, j] = vor[i, j] + beta * rhs
    
    # Apply vorticity boundary conditions
    for j in range(1, M-1):
        # Left boundary: Second-order approximation
        vor[0, j] = 0
        for k in range(1, 3):  # Use first two interior points
            vor[0, j] += (2 * k * psi[k, j]) / (x[k] - x[0])**2
        vor[0, j] *= -1
        
        # Right boundary: Second-order approximation
        vor[N-1, j] = 0
        for k in range(N-3, N-1):  # Use last two interior points
            vor[N-1, j] += (2 * k * psi[k, j]) / (x[k] - x[N-1])**2
        vor[N-1, j] *= -1
    
    for i in range(1, N-1):
        # Bottom boundary: Second-order approximation
        vor[i, 0] = 0
        for k in range(1, 3):  # Use first two interior points
            vor[i, 0] += (2 * k * psi[i, k]) / (y[k] - y[0])**2
        vor[i, 0] *= -1
        
        # Top boundary: Second-order approximation
        vor[i, M-1] = 0
        for k in range(M-3, M-1):  # Use last two interior points
            vor[i, M-1] += (2 * k * psi[i, k]) / (y[k] - y[M-1])**2
        vor[i, M-1] *= -1
    
    # Step 4: Update temperature field
    for i in range(1, N-1):
        for j in range(1, M-1):
            # Compute Laplacian of temperature
            lap_T = 0
            for k in range(N):
                lap_T += wx2[i, k] * T[k, j]
            for k in range(M):
                lap_T += wy2[j, k] * T[i, k]
            
            # Compute advection terms
            adv_x = 0
            adv_y = 0
            for k in range(N):
                adv_x += u[i, j] * wx1[i, k] * T[k, j]
            for k in range(M):
                adv_y += v[i, j] * wy1[j, k] * T[i, k]
            
            # Update temperature using relaxation
            rhs = lap_T - adv_x - adv_y
            T[i, j] = T[i, j] + beta * rhs
    
    # Apply temperature boundary conditions
    # Bottom is already set
    # Adiabatic (zero gradient) conditions for left and right walls
    for j in range(1, M-1):
        # Left wall: adiabatic, set gradient to zero
        sum_left = 0
        for k in range(1, N):
            sum_left += wx1[0, k] * T[k, j]
        T[0, j] = T[1, j] - sum_left / wx1[0, 0]
        
        # Right wall: adiabatic, set gradient to zero
        sum_right = 0
        for k in range(N-1):
            sum_right += wx1[N-1, k] * T[k, j]
        T[N-1, j] = T[N-2, j] - sum_right / wx1[N-1, N-1]
    
    # Top wall: enforce temperature
    T[:, M-1] = 0  # Cold top wall
    
    # Calculate errors for convergence check
    err_psi = np.max(np.abs(psi - psi_old))
    err_T = np.max(np.abs(T - T_old))
    err_vor = np.max(np.abs(vor - vor_old))
    
    # Print progress
    if iter_no % 10 == 0 or iter_no == max_iter - 1:
        print(f"Iteration {iter_no}: err_psi = {err_psi:.6e}, err_T = {err_T:.6e}, err_vor = {err_vor:.6e}")
    
    # Check for convergence
    if max(err_psi, err_T, err_vor) < tol:
        print(f"Converged after {iter_no+1} iterations")
        break

end_time = time.time()
print(f"Simulation completed in {end_time - start_time:.2f} seconds")

# Create a 2x2 grid for plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot Temperature
im1 = axes[0, 0].contourf(X, Y, T, 20, cmap='viridis')
axes[0, 0].set_title("Temperature Field")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
fig.colorbar(im1, ax=axes[0, 0], label="Temperature")

# Plot Vorticity
im2 = axes[0, 1].contourf(X, Y, vor, 20, cmap='coolwarm')
axes[0, 1].set_title("Vorticity Field")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")
fig.colorbar(im2, ax=axes[0, 1], label="Vorticity")

# Plot Stream Function
im3 = axes[1, 0].contourf(X, Y, psi, 20, cmap='RdBu_r')
axes[1, 0].set_title("Stream Function")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
fig.colorbar(im3, ax=axes[1, 0], label="Stream Function")

# Plot Velocity Magnitude
vel_mag = np.sqrt(u**2 + v**2)
im4 = axes[1, 1].contourf(X, Y, vel_mag, 20, cmap='viridis')
axes[1, 1].set_title("Velocity Magnitude")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")
fig.colorbar(im4, ax=axes[1, 1], label="Velocity Magnitude")

# Add velocity vector plot
skip = 2  # Show fewer vectors for clarity
axes[1, 1].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                 u[::skip, ::skip], v[::skip, ::skip], 
                 scale=50, color='k', alpha=0.7)

# Adjust layout
plt.tight_layout()
plt.show()
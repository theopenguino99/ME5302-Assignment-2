clc; close all; clear all;

% Defining parameters
N = 61; % Number of points along x axis
dx = 1.0 / (N - 1); % Grid spacing in x direction
M = 61; % Number of points along y axis
dy = 1.0 / (M - 1); % Grid spacing in y direction
Ra = 3.5e3; % Rayleigh number
Pr = 0.7; % Prandtl number
h = 0.00001; % Time step
beta = 0.4; % relaxation factor for iterative methods to solve algebraic equations

% Initialisation at t=0 with boundary conditions
u = zeros(N, M); % x-velocity
v = zeros(N, M); % y-velocity
T = zeros(N, M); % Temperature
rT = zeros(N, M); % Residual temperature
vor = zeros(N, M); % Vorticity
rvor = zeros(N, M); % Residual vorticity
p = zeros(N, M); % Stream function initialised to be 0
rp = zeros(N, M); % Residual stream function

% Boundary condition for temperature
for i = 1:N
    T(i, 1) = 0.5 * cos(pi * (i-1) / (N-1)) + 1; % Bottom boundary condition of T = 0.5cos(pi*x)+1
end

% Function to calculate vorticity residual
function rvor = resvor(vor, u, v, T, dx, dy, Pr, Ra)
    [N, M] = size(vor);
    rvor = zeros(size(vor));
    for i = 2:(N-1)
        for j = 2:(M-1)
            dvorx2 = (vor(i+1, j) - 2*vor(i, j) + vor(i-1, j)) / (dx^2);
            dvory2 = (vor(i, j+1) - 2*vor(i, j) + vor(i, j-1)) / (dy^2);
            dvorx1 = u(i, j) * (vor(i+1, j) - vor(i-1, j)) / (2*dx);
            dvory1 = v(i, j) * (vor(i, j+1) - vor(i, j-1)) / (2*dy);
            dTx = (T(i+1, j) - T(i-1, j)) / (2*dx);

            rvor(i, j) = (dvorx2 + dvory2) * Pr - Pr * a * dTx - dvorx1 - dvory1;
        end
    end
end

% Function to calculate temperature residual
function rtemp = restemp(T, u, v, dx, dy)
    [N, M] = size(T);
    rtemp = zeros(size(T));
    for i = 2:(N-1)
        for j = 2:(M-1)
            dTx2 = (T(i+1, j) - 2*T(i, j) + T(i-1, j)) / (dx^2);
            dTy2 = (T(i, j+1) - 2*T(i, j) + T(i, j-1)) / (dy^2);
            dTx1 = u(i, j) * (T(i+1, j) - T(i-1, j)) / (2*dx);
            dTy1 = v(i, j) * (T(i, j+1) - T(i, j-1)) / (2*dy);

            rtemp(i, j) = dTx2 + dTy2 - dTx1 - dTy1;
        end
    end
end

% Function to solve temperature field
function T = solT(T, rT, h, method)
    [N, M] = size(T);
    
    if strcmp(method, 'euler')
        T(2:N-1, 2:M-1) = T(2:N-1, 2:M-1) + h * rT(2:N-1, 2:M-1); 
    end
end

% Function to solve vorticity field
function vor = solvor(beta, vor, rvor, h, dx, dy, Pr, method)
    [N, M] = size(vor);

    if strcmp(method, 'euler')
        vor(2:N-1, 2:M-1) = vor(2:N-1, 2:M-1) + h * rvor(2:N-1, 2:M-1);
    
    elseif strcmp(method, 'rk4_modified')
        vori = vor;

        % 1st stage
        vori(2:N-1, 2:M-1) = vor(2:N-1, 2:M-1) + 0.25 * h * rvor(2:N-1, 2:M-1);
        % 2nd stage
        rvor = resvor(vori, u, v, T, dx, dy, Pr, Ra);
        vori(2:N-1, 2:M-1) = vor(2:N-1, 2:M-1) + (h / 3.0) * rvor(2:N-1, 2:M-1);
        % 3rd stage
        rvor = resvor(vori, u, v, T, dx, dy, Pr, Ra);
        vori(2:N-1, 2:M-1) = vor(2:N-1, 2:M-1) + 0.5 * h * rvor(2:N-1, 2:M-1);
        % 4th stage
        rvor = resvor(vori, u, v, T, dx, dy, Pr, Ra);
        vor(2:N-1, 2:M-1) = vor(2:N-1, 2:M-1) + h * rvor(2:N-1, 2:M-1);
    
    elseif strcmp(method, 'Point Jacobi')
        a_p_vor = Pr * (2/dx^2 + 2/dy^2) + 1/h;
        for i = 2:(N-1)
            for j = 2:(M-1)
                vor(i, j) = vor(i, j) + beta * rvor(i, j) / a_p_vor;
            end
        end
    end
end

% Calculation of residual of the Poisson equation
function rp = resp(p, vor, dx, dy)
    [N, M] = size(p);
    rp = zeros(size(p));
    
    for i = 3:(N-2)
        for j = 3:(M-2)
            rp(i, j) = vor(i, j) ...
                - (p(i+1, j) - 2*p(i, j) + p(i-1, j)) / (dx^2) ...
                - (p(i, j+1) - 2*p(i, j) + p(i, j-1)) / (dy^2);
        end
    end
end

% Function to solve stream function
function p = solp(p, rp, dx, dy, beta)
    [N, M] = size(p);
    % Coefficients for iterative methods
    b_W = 1 / dx^2;
    b_S = 1 / dy^2;
    b_P = -2 * (b_W + b_S);
    
    for i = 3:(N-2)
        for j = 3:(M-2)
            p(i, j) = p(i, j) + beta * rp(i, j) / b_P;
        end
    end
end

% Function to apply boundary conditions to stream function
function p = BCp(p)
    [N, M] = size(p);
    % Update p along the vertical boundaries
    for j = 2:(M-1)  
        p(2, j) = 0.25 * p(3, j);  % Left
        p(N-1, j) = 0.25 * p(N-2, j);  % Right
    end

    % Update p along the horizontal boundaries
    for i = 2:(N-1)  
        p(i, 2) = 0.25 * p(i, 3); % Bottom
        p(i, M-1) = 0.25 * p(i, M-2); % Top
    end

    % % Update p at the boundaries
    % p(1, :) = 0; % Left
    % p(N, :) = 0; % Right
    % p(:, 1) = 0; % Bottom
    % p(:, M) = 0; % Top
end

% Function to apply boundary conditions to vorticity
function vor = BCvor(vor, p, dx, dy)
    [N, M] = size(vor);
    % Update vorticity at the boundaries using 2nd order approximation
    % for j = 1:M
    %     vor(1, j) = 3.0 * p(2, j) / (dx^2) - 0.5 * vor(2, j);
    %     vor(N, j) = 3.0 * p(N-1, j) / (dx^2) - 0.5 * vor(N-1, j);
    % end
    % 
    % % Update along the horizontal boundaries (i-loop)
    % for i = 2:(N-1)
    %     vor(i, 1) = 3.0 * p(i, 2) / (dy^2) - 0.5 * vor(i, 2);
    %     vor(i, M) = 3.0 * p(i, M-1) / (dy^2) - 0.5 * vor(i, M-1);
    % end
    for i = 2:N-1
        vor(i,1) = 2*p(i,2)/(dy^2);
        vor(i,M) = 2*p(i,M-1)/(dy^2);
    end
    for j = 1:M
        vor(1,j) = 2*p(2,j)/(dx^2);
        vor(N,j) = 2*p(N-1,j)/(dx^2);
    end

end  

% Function to apply boundary conditions to temperature
function T = BCT(T)
    [N, M] = size(T);
    % Update temperature at the left boundary
    for j = 1:M
        T(1, j) = (4/3) * T(2, j) - (1/3) * T(3, j);
    end
    
    % Update temperature at the right boundary
    for j = 1:M
        T(N, j) = (4/3) * T(N-1, j) - (1/3) * T(N-2, j);
    end

    % % Update temperature at the top boundary (added: isothermal condition T=0)
    % for i = 1:N
    %     T(i, N) = 0.0;
    % end
    % for i = 2:N-1
    %     T(i, 1) = 0.5 * cos(pi * (i-1) / (N-1)) + 1;
    % end
end

% Function to calculate velocity components from stream function
function [u, v, T] = caluv(T, u, v, p, dx, dy)
    [N, M] = size(u);
    % Apply physical boundary conditions of 0 velocity
    for j = 1:M
        u(1, j) = 0;
        u(N, j) = 0;
        v(1, j) = 0;
        v(N, j) = 0;
    end

    for i = 2:(N-1)
        u(i, 1) = 0;
        v(i, 1) = 0;
        u(i, M) = 0;
        v(i, M) = 0;
    end
    for i = 1:N
        T(i, N) = 0.0;
    end
    for i = 2:N-1
        T(i, 1) = 0.5 * cos(pi * (i-1) / (N-1)) + 1;
    end

    % Update velocity components based on stream function
    for i = 2:(N-1)
        for j = 2:(M-1)
            u(i, j) = 0.5 * (p(i, j+1) - p(i, j-1)) / dy;
            v(i, j) = 0.5 * (p(i-1, j) - p(i+1, j)) / dx;
        end
    end
end

% Initialise errors for convergence check
iter_no = 0;

% Start timer
tic;

% Main simulation loop
while true
    % Compute residual vorticity and update vorticity
    rvor = resvor(vor, u, v, T, dx, dy, Pr, Ra);
    vor = solvor(beta, vor, rvor, h, dx, dy, Pr, 'euler');

    % Compute residual Poisson equation and update stream function
    rp = resp(p, vor, dx, dy);
    p = solp(p, rp, dx, dy, beta);

    % Update boundary conditions for stream function
    p = BCp(p);

    % Update boundary conditions for vorticity
    vor = BCvor(vor, p, dx, dy);

    % Update velocity components based on stream function
    [u, v, T] = caluv(T, u, v, p, dx, dy);

    % Compute residual temperature and update temperature
    rT = restemp(T, u, v, dx, dy);
    T = solT(T, rT, h, 'euler');

    % Update Temperature field
    T = BCT(T);

    % Update iteration number
    iter_no = iter_no + 1;

    % Calculate errors
    errvor = sqrt(sum(sum(rvor.^2)));
    errp = sqrt(sum(sum(rp.^2)));
    errT = sqrt(sum(sum(rT.^2)));

    if mod(iter_no, 100) == 0
        fprintf('Iteration number %d, errp: %f, errvor: %f, errT: %f\n', iter_no, errp, errvor, errT);
    end

        % Check convergence
    if errp < 1e-3 && errvor < 1e-3 && errT < 1e-3
        converged = true;
        fprintf('Converged at iteration %d: errp = %f, errvor = %f, errT = %f\n', iter_no, errp, errvor, errT);
        break;
    end
end

% End timer and calculate elapsed time
elapsed_time = toc;

fprintf('My time step is: %f\n', h);
fprintf('Total time elapsed is: %f\n', h*iter_no);
fprintf('Total time taken for the simulation: %f seconds\n', elapsed_time);

% Create figure with 2x2 subplots
figure('Position', [100, 100, 900, 750]);

% Plot Temperature contours 
subplot(2, 2, 1);
[~, contour1] = contourf(T', 20);
colorbar;
title('Temperature Contours');
xlabel('x');
ylabel('y');

% Plot Stream function contours
subplot(2, 2, 2);
[~, contour2] = contourf(p', 20);
colorbar;
title('Stream function Contours');
xlabel('x');
ylabel('y');

% Plot X-velocity contours
subplot(2, 2, 3);
[~, contour3] = contourf(u', 20);
colorbar;
title('X-velocity Contours');
xlabel('x');
ylabel('y');

% Plot Y-velocity contours
subplot(2, 2, 4);
[~, contour4] = contourf(v', 20);
colorbar;
title('Y-velocity Contours');
xlabel('x');
ylabel('y');

% --- Inputs ---
% u, v: Velocity fields (size NxM)
% T: Temperature field (size NxM)
% dx, dy: Grid spacings
% N, M: Number of grid points in x and y directions

%% (1) Calculate umax (max horizontal velocity on vertical mid-plane x=0.5)
x_mid_idx = round(N/2);  % Index of vertical mid-plane (x=0.5)
u_midplane = u(x_mid_idx, :);  % Horizontal velocity on x=0.5
[umax, umax_idx] = max(abs(u_midplane));  % Maximum magnitude and its index
umax_y_location = (umax_idx - 1) * dy;  % y-coordinate of umax
umax_sign = sign(u_midplane(umax_idx));  % Sign of umax (direction)

fprintf('umax = %.6f at y = %.4f (direction: %d)\n', umax, umax_y_location, umax_sign);

%% (2) Calculate vmax (max vertical velocity on horizontal mid-plane y=0.5)
y_mid_idx = round(M/2);  % Index of horizontal mid-plane (y=0.5)
v_midplane = v(:, y_mid_idx);  % Vertical velocity on y=0.5
[vmax, vmax_idx] = max(abs(v_midplane));  % Maximum magnitude and its index
vmax_x_location = (vmax_idx - 1) * dx;  % x-coordinate of vmax
vmax_sign = sign(v_midplane(vmax_idx));  % Sign of vmax (direction)

fprintf('vmax = %.6f at x = %.4f (direction: %d)\n', vmax, vmax_x_location, vmax_sign);

%% (3) Calculate Nu0 (avg Nusselt number at bottom wall y=0)
dTdy_bottom = (T(:, 2) - T(:, 1)) / dy;  % Temperature gradient at y=0 (forward difference)
Nu_local_bottom = -dTdy_bottom;  % Local Nusselt number (T_wall = T(:,1))
Nu0 = mean(Nu_local_bottom);  % Average Nusselt number at y=0

fprintf('Nu0 (avg at y=0) = %.6f\n', Nu0);

%% (4) Calculate Nu1 (avg Nusselt number at mid-plane y=0.5)
dTdy_mid = (T(:, y_mid_idx+1) - T(:, y_mid_idx-1)) / (2*dy);  % Central difference at y=0.5
Nu_local_mid = -dTdy_mid;  % Local Nusselt number at y=0.5
Nu1 = mean(Nu_local_mid);  % Average Nusselt number at y=0.5

fprintf('Nu1 (avg at y=0.5) = %.6f\n', Nu1);

%% (5) Calculate Numax (max Nusselt number at bottom wall y=0)
[Numax, Numax_idx] = max(Nu_local_bottom);  % Max Nusselt number and its index
Numax_x_location = (Numax_idx - 1) * dx;  % x-coordinate of Numax

fprintf('Numax (max at y=0) = %.6f at x = %.4f\n', Numax, Numax_x_location);
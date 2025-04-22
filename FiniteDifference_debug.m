% Dependencies
% No import needed, MATLAB has built-in plotting and array operations

% Parameters
N = 242;
dx = 1.0 / (N - 1);
M = 242;
dy = 1.0 / (M - 1);
Ra = 3.5e3;
Pr = 0.7;
CFL_crit = 2 * sqrt(2);
margin_h = 0.1;
h = margin_h * CFL_crit / (1/dx + 1/dy) / 1000;
beta = 0.000001;
iter = 700;

% Initialization
u = zeros(N, M);
v = zeros(N, M);
T = zeros(N, M);
vor = zeros(N, M);
p = zeros(N, M);

% Bottom boundary condition of T
for i = 1:N
    T(i,1) = 0.5 * cos(pi * (i-1) / (N-1)) + 1;
end

errp = inf;
errvor = inf;
iter_no = 0;

tic;

for it = 1:iter
    % Residual Vorticity
    rvor = zeros(N, M);
    for i = 2:N-1
        for j = 2:M-1
            dvorx2 = (vor(i+1,j) - 2*vor(i,j) + vor(i-1,j)) / dx^2;
            dvory2 = (vor(i,j+1) - 2*vor(i,j) + vor(i,j-1)) / dy^2;
            dvorx1 = u(i,j) * (vor(i+1,j) - vor(i-1,j)) / (2*dx);
            dvory1 = v(i,j) * (vor(i,j+1) - vor(i,j-1)) / (2*dy);
            dTx = (T(i+1,j) - T(i-1,j)) / (2*dx);

            rvor(i,j) = (dvorx2 + dvory2) * Pr + Pr * Ra * dTx - dvorx1 - dvory1;
        end
    end
    vor(2:N-1,2:M-1) = vor(2:N-1,2:M-1) + h * rvor(2:N-1,2:M-1);

    % Residual Poisson equation
    rp = zeros(N, M);
    for i = 3:N-2
        for j = 3:M-2
            rp(i,j) = vor(i,j) - ((p(i+1,j) - 2*p(i,j) + p(i-1,j)) / dx^2 ...
                        + (p(i,j+1) - 2*p(i,j) + p(i,j-1)) / dy^2);
        end
    end

    b_W = 1 / dx^2;
    b_E = b_W;
    b_S = 1 / dy^2;
    b_N = b_S;
    b_P = 2 * (b_W + b_S);

    for i = 3:N-2
        for j = 3:M-2
            p(i,j) = p(i,j) + beta * rp(i,j) / b_P;
        end
    end

    % Boundary conditions for stream function
    for j = 1:M
        p(2,j) = 0.25 * p(3,j);
        p(N-1,j) = 0.25 * p(N-2,j);
    end
    for i = 1:N
        p(i,2) = 0.25 * p(i,3);
        p(i,M-1) = 0.25 * p(i,M-2);
    end

    % Boundary conditions for vorticity
    for j = 1:M
        vor(1,j) = 3.0 * p(2,j) / dx^2 - 0.5 * vor(2,j);
        vor(N,j) = 3.0 * p(N-1,j) / dx^2 - 0.5 * vor(N-1,j);
    end
    for i = 2:N-1
        vor(i,1) = 3.0 * p(i,2) / dy^2 - 0.5 * vor(i,2);
        vor(i,M) = 3.0 * p(i,M-1) / dy^2 - 0.5 * vor(i,M-1);
    end

    % Velocity components from stream function
    u(:,[1 M]) = 0;
    u([1 N],:) = 0;
    v(:,[1 M]) = 0;
    v([1 N],:) = 0;

    for i = 2:N-1
        for j = 2:M-1
            u(i,j) = 0.5 * (p(i,j+1) - p(i,j-1)) / dy;
            v(i,j) = -0.5 * (p(i-1,j) - p(i+1,j)) / dx;
        end
    end

    % Residual temperature — Modified RK4 method
    Ti = T;

    % 1st stage
    rtemp = zeros(N, M);
    for i = 2:N-1
        for j = 2:M-1
            dTx2 = (T(i+1,j) - 2*T(i,j) + T(i-1,j)) / dx^2;
            dTy2 = (T(i,j+1) - 2*T(i,j) + T(i,j-1)) / dy^2;
            dTx1 = u(i,j) * (T(i+1,j) - T(i-1,j)) / (2*dx);
            dTy1 = v(i,j) * (T(i,j+1) - T(i,j-1)) / (2*dy);

            rtemp(i,j) = dTx2 + dTy2 - dTx1 - dTy1;
        end
    end
    Ti(2:N-1,2:M-1) = T(2:N-1,2:M-1) + 0.25 * h * rtemp(2:N-1,2:M-1);

    % 2nd stage
    rtempi = zeros(N, M);
    for i = 2:N-1
        for j = 2:M-1
            dTx2 = (Ti(i+1,j) - 2*Ti(i,j) + Ti(i-1,j)) / dx^2;
            dTy2 = (Ti(i,j+1) - 2*Ti(i,j) + Ti(i,j-1)) / dy^2;
            dTx1 = u(i,j) * (Ti(i+1,j) - Ti(i-1,j)) / (2*dx);
            dTy1 = v(i,j) * (Ti(i,j+1) - Ti(i,j-1)) / (2*dy);

            rtempi(i,j) = dTx2 + dTy2 - dTx1 - dTy1;
        end
    end
    Ti(2:N-1,2:M-1) = T(2:N-1,2:M-1) + (h/3) * rtempi(2:N-1,2:M-1);

    % 3rd stage
    rtempii = zeros(N, M);
    for i = 2:N-1
        for j = 2:M-1
            dTx2 = (Ti(i+1,j) - 2*Ti(i,j) + Ti(i-1,j)) / dx^2;
            dTy2 = (Ti(i,j+1) - 2*Ti(i,j) + Ti(i,j-1)) / dy^2;
            dTx1 = u(i,j) * (Ti(i+1,j) - Ti(i-1,j)) / (2*dx);
            dTy1 = v(i,j) * (Ti(i,j+1) - Ti(i,j-1)) / (2*dy);

            rtempii(i,j) = dTx2 + dTy2 - dTx1 - dTy1;
        end
    end
    Ti(2:N-1,2:M-1) = T(2:N-1,2:M-1) + 0.5 * h * rtempii(2:N-1,2:M-1);

    % 4th stage
    rtempiii = zeros(N, M);
    for i = 2:N-1
        for j = 2:M-1
            dTx2 = (Ti(i+1,j) - 2*Ti(i,j) + Ti(i-1,j)) / dx^2;
            dTy2 = (Ti(i,j+1) - 2*Ti(i,j) + Ti(i,j-1)) / dy^2;
            dTx1 = u(i,j) * (Ti(i+1,j) - Ti(i-1,j)) / (2*dx);
            dTy1 = v(i,j) * (Ti(i,j+1) - Ti(i,j-1)) / (2*dy);

            rtempiii(i,j) = dTx2 + dTy2 - dTx1 - dTy1;
        end
    end
    T(2:N-1,2:M-1) = T(2:N-1,2:M-1) + h * rtempiii(2:N-1,2:M-1);

    % Temperature boundary updates
    T(1,:) = (4/3) * T(2,:) - (1/3) * T(3,:);
    T(N,:) = (4/3) * T(N-1,:) - (1/3) * T(N-2,:);

    iter_no = iter_no + 1;

    % Error calculations
    errvor = sqrt(sum(sum(vor(2:N-1,2:M-1).^2)) / ((N-2)*(M-2)));
    errp = sqrt(sum(sum(p(2:N-1,2:M-1).^2)) / ((N-2)*(M-2)));
    
    fprintf('Iteration number %d, errp: %f, errvor: %f\n', iter_no, errp, errvor);
end

fprintf('My time step is: %f\n', h);
fprintf('Total time elapsed is: %f\n', h*iter);
fprintf('My errp is: %f\n', errp);
fprintf('My errvor is: %f\n', errvor);

% Plotting
figure;

subplot(2,2,1);
imagesc([0 1], [0 1], T');
axis equal tight;
colorbar;
title('Temperature Field');
xlabel('x'); ylabel('y');

subplot(2,2,2);
imagesc([0 1], [0 1], vor');
axis equal tight;
colorbar;
title('Vorticity Field');
xlabel('x'); ylabel('y');

subplot(2,2,3);
imagesc([0 1], [0 1], u');
axis equal tight;
colorbar;
title('X-velocity Field');
xlabel('x'); ylabel('y');

subplot(2,2,4);
imagesc([0 1], [0 1], v');
axis equal tight;
colorbar;
title('Y-velocity Field');
xlabel('x'); ylabel('y');

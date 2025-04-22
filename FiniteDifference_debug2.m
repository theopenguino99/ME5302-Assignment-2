clear; clc;

% Defining parameters
N = 61;
M = 61;
dx = 1.0 / (N - 1);
dy = 1.0 / (M - 1);
Ra = 3.5e3;
Pr = 0.7;
CFL_crit = 2 * sqrt(2);
margin_h = 0.1;
h = margin_h * CFL_crit / (1/dx + 1/dy)/1;
beta = 0.01;
iter = 100000;

% Initialisation
u = zeros(N, M);
v = zeros(N, M);
T = zeros(N, M);
vor = zeros(N, M);
p = zeros(N, M);

% Bottom boundary temperature
for i = 1:N
    T(i,1) = 0.5 * cos(pi * (i-1) / (N-1)) + 1;
end

%% Residual functions
function rvor = resvor(vor, u, v, T, dx, dy, Pr, Ra, N, M)
    rvor = zeros(size(vor));
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
end

function rtemp = restemp(T, u, v, dx, dy, N, M)
    rtemp = zeros(size(T));
    for i = 2:N-1
        for j = 2:M-1
            dTx2 = (T(i+1,j) - 2*T(i,j) + T(i-1,j)) / dx^2;
            dTy2 = (T(i,j+1) - 2*T(i,j) + T(i,j-1)) / dy^2;
            dTx1 = u(i,j) * (T(i+1,j) - T(i-1,j)) / (2*dx);
            dTy1 = v(i,j) * (T(i,j+1) - T(i,j-1)) / (2*dy);

            rtemp(i,j) = dTx2 + dTy2 - dTx1 - dTy1;
        end
    end
end

function rp = resp(p, vor, dx, dy, N, M, conditions)
    rp = zeros(size(p));
    if conditions == 1
        range_x = 2:N-1;
        range_y = 2:M-1;
    elseif conditions == 2
        range_x = 3:N-2;
        range_y = 3:M-2;
    end

    for i = range_x
        for j = range_y
            rp(i,j) = vor(i,j) - ...
                     ( (p(i+1,j) - 2*p(i,j) + p(i-1,j))/dx^2 + ...
                       (p(i,j+1) - 2*p(i,j) + p(i,j-1))/dy^2 );
        end
    end
end

%% Update functions
function T = solT(T, u, v, h, beta, dx, dy, N, M)
    rT = restemp(T, u, v, dx, dy, N, M);
    a_p_T = (2/dx^2 + 2/dy^2) + 1/h;
    for i = 2:N-1
        for j = 2:M-1
            T(i,j) = T(i,j) + beta * rT(i,j) / a_p_T;
        end
    end
end

function vor = solvor(vor, u, v, T, h, beta, Pr, Ra, dx, dy, N, M)
    rvor = resvor(vor, u, v, T, dx, dy, Pr, Ra, N, M);
    a_p_vor = Pr*(2/dx^2 + 2/dy^2) + 1/h;
    for i = 2:N-1
        for j = 2:M-1
            vor(i,j) = vor(i,j) + beta * rvor(i,j) / a_p_vor;
        end
    end
end

function p = solp(p, vor, beta, dx, dy, N, M, conditions)
    rp = resp(p, vor, dx, dy, N, M, conditions);
    b_W = 1/dx^2;
    b_E = b_W;
    b_S = 1/dy^2;
    b_N = b_S;
    b_P = -2*(b_W + b_S);

    if conditions == 1
        range_x = 2:N-1;
        range_y = 2:M-1;
    elseif conditions == 2
        range_x = 3:N-2;
        range_y = 3:M-2;
    end

    for i = range_x
        for j = range_y
            p(i,j) = p(i,j) + beta * rp(i,j) / b_P;
        end
    end
end

function p = BCp(p, N, M)
    for j = 2:M-1
        p(2,j) = 0.25 * p(3,j);
        p(N-1,j) = 0.25 * p(N-2,j);
    end
    for i = 2:N-1
        p(i,2) = 0.25 * p(i,3);
        p(i,M-1) = 0.25 * p(i,M-2);
    end
end

function vor = BCvor(vor, p, dx, dy, N, M)
    for j = 1:M
        vor(1,j) = 3.0 * p(2,j) / dx^2 - 0.5 * vor(2,j);
        vor(N,j) = 3.0 * p(N-1,j) / dx^2 - 0.5 * vor(N-1,j);
    end
    for i = 2:N-1
        vor(i,1) = 3.0 * p(i,2) / dy^2 - 0.5 * vor(i,2);
        vor(i,M) = 3.0 * p(i,M-1) / dy^2 - 0.5 * vor(i,M-1);
    end
end

function T = BCT(T, N, M)
    for j = 1:M
        T(1,j) = (4/3) * T(2,j) - (1/3) * T(3,j);
        T(N,j) = (4/3) * T(N-1,j) - (1/3) * T(N-2,j);
    end
end

function [u, v] = caluv(u, v, p, dx, dy, N, M)
    u(1,:) = 0; u(N,:) = 0;
    v(1,:) = 0; v(N,:) = 0;
    u(:,1) = 0; u(:,M) = 0;
    v(:,1) = 0; v(:,M) = 0;

    for i = 2:N-1
        for j = 2:M-1
            u(i,j) = 0.5 * (p(i,j+1) - p(i,j-1)) / dy;
            v(i,j) = 0.5 * (p(i-1,j) - p(i+1,j)) / dx;
        end
    end
end

%% Main solver loop
errp = inf;
errvor = inf;
iter_no = 0;

tic
while iter_no < iter
    vor = solvor(vor, u, v, T, h, beta, Pr, Ra, dx, dy, N, M);
    p = solp(p, vor, beta, dx, dy, N, M, 2);
    p = BCp(p, N, M);
    vor = BCvor(vor, p, dx, dy, N, M);
    [u, v] = caluv(u, v, p, dx, dy, N, M);
    T = solT(T, u, v, h, beta, dx, dy, N, M);
    T = BCT(T, N, M);

    iter_no = iter_no + 1;

    errvor = sqrt(sum(sum(vor(2:N-1,2:M-1).^2)) / (N-2)/(M-2));
    errp = sqrt(sum(sum(p(2:N-1,2:M-1).^2)) / (N-2)/(M-2));

    disp(['Iteration ', num2str(iter_no), ' errp: ', num2str(errp), ' errvor: ', num2str(errvor)]);
end
elapsed_time = toc;

disp(['Time step h: ', num2str(h)]);
disp(['Elapsed simulation time: ', num2str(h*iter)]);
disp(['Elapsed wall time: ', num2str(elapsed_time), ' seconds']);

%% Plots
figure;
subplot(2,2,1);
imagesc(T'); colorbar; title('Temperature'); axis equal tight;

subplot(2,2,2);
imagesc(vor'); colorbar; title('Vorticity'); axis equal tight;

subplot(2,2,3);
imagesc(u'); colorbar; title('U-velocity'); axis equal tight;

subplot(2,2,4);
imagesc(v'); colorbar; title('V-velocity'); axis equal tight;

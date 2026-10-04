function [S_Lx, S_Tx, S_Ly, S_Ty] = structure_functions(ux, uy, p_orders,nmax)
% STRUCTURE_FUNCTIONS Absolute longitudinal and transverse increment moments.
% ux and uy use [nx,ny] ordering. p_orders lists moment orders; nmax is optional.
% Rows correspond to integer separation in cells; columns correspond to orders.
% Odd orders are absolute moments, not signed energy-flux statistics.

    % Inputs:
    % ux, uy  : nx-by-ny periodic velocity fields; rows represent x and columns y.
    % p_orders: Moment orders, e.g. [2, 3, 4, 6]; all moments use absolute increments.
    %
    % Outputs:
    % S_Lx, S_Tx, S_Ly, S_Ty: Arrays with separation r in rows and orders from p_orders in columns.

    % 1. Rows represent x and columns represent y.
    [nx, ny] = size(ux);
    num_p = length(p_orders);

    % Subtract the mean velocity to obtain fluctuations.
    ux_prime = ux - mean(ux(:), 'omitnan');
    uy_prime = uy - mean(uy(:), 'omitnan');

    % Default maximum separation is half the grid length in each direction.
    if exist('nmax','var')==1
        rmax_x=nmax;
        rmax_y=nmax;
    else
        rmax_x = floor(nx / 2);
        rmax_y = floor(ny / 2);
    end

    % Allocate the result arrays.
    S_Lx = zeros(rmax_x, num_p);
    S_Tx = zeros(rmax_x, num_p);
    S_Ly = zeros(rmax_y, num_p);
    S_Ty = zeros(rmax_y, num_p);

    % =========================================================
    % 2. Compute moments for separations along x (row shifts).
    % =========================================================
    for r = 1:rmax_x
        % [-r, 0] shifts rows by r cells without shifting columns.
        ux_shifted_x = circshift(ux_prime, [-r, 0]);
        uy_shifted_x = circshift(uy_prime, [-r, 0]);

        du_L_x = ux_shifted_x - ux_prime; % Longitudinal increment for an x separation.
        du_T_x = uy_shifted_x - uy_prime; % Transverse increment for an x separation.

        for k = 1:num_p
            p = p_orders(k);
            S_Lx(r, k) = mean(abs(du_L_x(:)).^p, 'omitnan');
            S_Tx(r, k) = mean(abs(du_T_x(:)).^p, 'omitnan');
            % S_Lx(r, k) = mean(real(du_L_x(:)).^p, 'omitnan');
            % S_Tx(r, k) = mean(real(du_T_x(:)).^p, 'omitnan');
        end
    end

    % =========================================================
    % 3. Compute moments for separations along y (column shifts).
    % =========================================================
    for r = 1:rmax_y
        % [0, -r] shifts columns by r cells without shifting rows.
        ux_shifted_y = circshift(ux_prime, [0, -r]);
        uy_shifted_y = circshift(uy_prime, [0, -r]);

        du_L_y = uy_shifted_y - uy_prime; % Longitudinal increment for a y separation.
        du_T_y = ux_shifted_y - ux_prime; % Transverse increment for a y separation.

        for k = 1:num_p
            p = p_orders(k);
            S_Ly(r, k) = mean(abs(du_L_y(:)).^p, 'omitnan');
            S_Ty(r, k) = mean(abs(du_T_y(:)).^p, 'omitnan');
            % S_Ly(r, k) = mean(real(du_L_y(:)).^p, 'omitnan');
            % S_Ty(r, k) = mean(real(du_T_y(:)).^p, 'omitnan');
        end
    end
end

function [k_bin, Sc_1d, Sn_1d, vortices, spacing_info] = compute_vortex_structure_factor(Phi, Lx, Ly)
% COMPUTE_VORTEX_STRUCTURE_FACTOR Charge and number structure factors.
% Phi uses [Ny, Nx] ordering (rows=y, columns=x), unlike ndgrid simulation fields.
% Lx and Ly are domain lengths. FFT powers are normalized by the vortex count.
% Returns radial bin centers, two spectra, [x,y,charge] positions, and spacing estimates.

% Compute vortex structure factors from a periodic order parameter Phi(x,y).
%
% Inputs:
%   Phi : Complex Ny-by-Nx order parameter with periodic boundaries.
%   Lx  : Periodic domain length along x; dx is computed internally as Lx/Nx.
%   Ly  : Periodic domain length along y; dy is computed internally as Ly/Ny.
%
% Outputs:
%   k_bin   : Radial wavenumber bin centers.
%   Sc_1d   : Charge structure factor S_c(k).
%   Sn_1d   : Vortex-number structure factor S_n(k).
%   vortices: Detected vortex positions and charges [x, y, charge].

    [Ny, Nx] = size(Phi);
    dx=Lx/Nx;dy=Ly/Ny;
    theta = angle(Phi); % Extract the phase in (-pi, pi].

    %% 1. Detect topological defects on periodic plaquettes.
    theta_x  = circshift(theta, [0, -1]);  % Neighbor at the next column (x direction).
    theta_y  = circshift(theta, [-1, 0]);  % Neighbor at the next row (y direction).
    theta_xy = circshift(theta, [-1, -1]); % Diagonal plaquette neighbor.

    % Wrap plaquette phase increments into [-pi, pi).
    d1 = mod(theta_x - theta + pi, 2*pi) - pi;
    d2 = mod(theta_xy - theta_x + pi, 2*pi) - pi;
    d3 = mod(theta_y - theta_xy + pi, 2*pi) - pi;
    d4 = mod(theta - theta_y + pi, 2*pi) - pi;

    charge_grid = round((d1 + d2 + d3 + d4) / (2*pi));

    % Extract vortex coordinates.
    [iy, ix] = find(abs(charge_grid) > 0.5);
    charges = charge_grid(sub2ind(size(charge_grid), iy, ix));
    Nv = length(charges);

    if Nv == 0
        warning('No vortices detected.');
        k_bin = []; Sc_1d = []; Sn_1d = []; vortices = []; spacing_info = [];
        return;
    end

    x_v = (ix - 0.5) * dx;
    y_v = (iy - 0.5) * dy;
    vortices = [x_v, y_v, charges];

    %% 2. Estimate vortex spacing from density and nearest neighbors.

    % (a) Spacing estimated from the global vortex density.
    Area = Lx * Ly;
    rho = Nv / Area;
    d_density = 1 / sqrt(rho);
    k_density = 2 * pi / d_density;

    % (b) Mean nearest-neighbor distance with periodic boundaries.
    nnd = zeros(Nv, 1);
    for i = 1:Nv
        % Use minimum-image periodic distances.
        dx_vec = abs(x_v - x_v(i));
        dx_vec = min(dx_vec, Lx - dx_vec); % Minimum-image distance across a periodic boundary.
        dy_vec = abs(y_v - y_v(i));
        dy_vec = min(dy_vec, Ly - dy_vec); % Minimum-image distance across a periodic boundary.

        dist_sq = dx_vec.^2 + dy_vec.^2;
        dist_sq(i) = Inf; % Exclude self distances.
        nnd(i) = sqrt(min(dist_sq));
    end
    d_nnd = mean(nnd);
    k_nnd = 2 * pi / d_nnd;

    %% 3. Compute structure factors using the 2-D FFT.
    W_c = charge_grid;       % Topological charge density.
    W_n = abs(charge_grid);  % Vortex-number density.

    Sc_2D = fftshift(abs(fft2(W_c)).^2 / Nv);
    Sn_2D = fftshift(abs(fft2(W_n)).^2 / Nv);

    %% 4. Construct wavenumbers and average over radial bins.
    dkx = 2*pi / Lx;
    dky = 2*pi / Ly;

    kx = (-floor(Nx/2) : ceil(Nx/2)-1) * dkx;
    ky = (-floor(Ny/2) : ceil(Ny/2)-1) * dky;
    [KX, KY] = meshgrid(kx, ky);
    K_mag = hypot(KX, KY);

    dk = min(dkx, dky);
    k_max = min(max(abs(kx)), max(abs(ky)));
    k_edges = 0:dk:k_max;
    k_bin = 0.5 * (k_edges(1:end-1) + k_edges(2:end));
    Num_bins = length(k_bin);

    Sc_1d = zeros(1, Num_bins);
    Sn_1d = zeros(1, Num_bins);

    for b = 1:Num_bins
        mask = (K_mag >= k_edges(b)) & (K_mag < k_edges(b+1));
        if any(mask(:))
            Sc_1d(b) = mean(Sc_2D(mask));
            Sn_1d(b) = mean(Sn_2D(mask));
        end
    end

    % Collect spacing estimates.
    spacing_info.rho = rho;
    spacing_info.d_density = d_density;
    spacing_info.d_nnd = d_nnd;
    spacing_info.k_density = k_density;
    spacing_info.k_nnd = k_nnd;

    %% 5. Optional result visualization.
    % figure('Name', 'Vortex Structure Factor (PBC)', 'Color', 'w', 'Position', [100 100 1000 800]);
    % theme(gcf,"light")
    % % (1) Vortex positions.
    % subplot(2, 2, 1);
    % scatter(x_v(charges>0), y_v(charges>0), 15, 'r', 'filled'); hold on;
    % scatter(x_v(charges<0), y_v(charges<0), 15, 'b', 'filled');
    % axis equal tight; box on;
    % % Set axes to the physical domain.
    % xlim([0, Nx*dx]); ylim([0, Ny*dy]);
    % title(sprintf('Vortex Positions (N = %d)', Nv));
    % xlabel('x'); ylabel('y');
    % legend('+\kappa', '-\kappa', 'Location', 'northeastoutside');
    %
    % % (2) Two-dimensional charge structure factor.
    % subplot(2, 2, 3);
    % imagesc(kx, ky, log10(Sc_2D + 1e-6));
    % axis square; set(gca, 'YDir', 'normal'); colorbar;
    % title('2D S_c(k_x, k_y) [Log Scale]');
    % xlabel('k_x'); ylabel('k_y');
    % % set(gca,'XScale','log','YScale','log');
    % subplot(2, 2, 4);
    % imagesc(kx, ky, log10(Sn_2D + 1e-6));
    % axis square; set(gca, 'YDir', 'normal'); colorbar;
    % title('2D S_n(k_x, k_y) [Log Scale]');
    % xlabel('k_x'); ylabel('k_y');
    % % set(gca,'XScale','log','YScale','log');
    %
    % % (3) Radially averaged structure factors on logarithmic axes.
    % subplot(2, 2, 2);
    % loglog(k_bin, Sc_1d, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Charge S_c(k)'); hold on;
    % loglog(k_bin, Sn_1d, 'b--s', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Number S_n(k)');
    % grid on; box on;
    % title('1D Structure Factor S(k)');
    % xlabel('k'); ylabel('S(k)');
    % legend('Location', 'best');
end

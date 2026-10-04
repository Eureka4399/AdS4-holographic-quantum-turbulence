function [centroids, cluster_info] = find_vortex_dpc_pbc(vortices, Lx, Ly, dc, gamma_threshold)
% FIND_VORTEX_DPC_PBC Locate density peaks using minimum-image distances.
% Legacy input convention is [y, x, charge]; outputs are [x, y] peak positions.
% Charges are not used in clustering. Two diagnostic plots are created.

% FIND_VORTEX_DPC_PBC Identify density peaks in a periodic vortex distribution.
%
% Inputs:
%   vortices        : Nv-by-3 array. Legacy implementation reads columns as [y, x, charge].
%   Lx, Ly          : Periodic domain lengths (e.g. 500, 500).
%   dc              : (Optional) Gaussian density-kernel bandwidth; default is 10.
%   gamma_threshold : (Optional) Density-peak threshold for gamma = rho * delta.
%                     If omitted or empty, use mean(gamma) + 3*std(gamma).
%
% Outputs:
%   centroids    : Nc-by-2 positions [x_c, y_c] of selected density peaks.
%   cluster_info : Structure with peak spacing, density rho, distance delta, and gamma.

    x_v = vortices(:, 2);
    y_v = vortices(:, 1);
    Nv  = length(x_v);

    if nargin < 4 || isempty(dc)
        % Default Gaussian kernel bandwidth.
        dc = 10;
    end

    %% 1. Compute the Nv-by-Nv minimum-image distance matrix.
    dx_mat = abs(x_v - x_v');
    dx_mat = min(dx_mat, Lx - dx_mat);

    dy_mat = abs(y_v - y_v');
    dy_mat = min(dy_mat, Ly - dy_mat);

    D_pbc = hypot(dx_mat, dy_mat);

    %% 2. Estimate local vortex density using a Gaussian kernel.
    % rho_i = sum_j exp(-(d_ij / dc)^2)
    rho = sum(exp(-(D_pbc ./ dc).^2), 2) - 1; % Remove the unit self contribution.

    %% 3. Find each vortex distance delta to a higher-density point.
    delta = zeros(Nv, 1);
    nneigh = zeros(Nv, 1); % Store the index of a higher-density neighbor.

    [~, ordrho] = sort(rho, 'descend');

    % For the highest-density point, use the maximum distance to another point.
    delta(ordrho(1)) = max(D_pbc(ordrho(1), :));
    nneigh(ordrho(1)) = 0;

    for i = 2:Nv
        curr_idx = ordrho(i);
        higher_density_indices = ordrho(1:i-1);
        [delta(curr_idx), min_pos] = min(D_pbc(curr_idx, higher_density_indices));
        nneigh(curr_idx) = higher_density_indices(min_pos);
    end

    %% 4. Construct the density-peak indicator gamma = rho * delta.
    gamma = rho .* delta;

    % Select peak centers using the supplied or default gamma threshold.
    if nargin < 5 || isempty(gamma_threshold)
        % Select values at least three standard deviations above the mean.
        gamma_threshold = mean(gamma) + 3 * std(gamma);
    end

    center_indices = find(gamma >= gamma_threshold);
    Nc = length(center_indices);

    if Nc == 0
        warning('No significant density peaks found; consider lowering gamma_threshold.');
        centroids = []; cluster_info = [];
        return;
    end

    centroids = [x_v(center_indices), y_v(center_indices)];

    %% 5. Measure nearest-neighbor spacing between selected peak centers.
    if Nc > 1
        dx_c = abs(centroids(:, 1) - centroids(:, 1)');
        dx_c = min(dx_c, Lx - dx_c);

        dy_c = abs(centroids(:, 2) - centroids(:, 2)');
        dy_c = min(dy_c, Ly - dy_c);

        D_centroids = hypot(dx_c, dy_c);
        D_centroids(1:Nc+1:end) = Inf; % Exclude self distances.

        nnd_cluster = min(D_centroids, [], 2);
    else
        nnd_cluster = NaN;
    end

    %% 6. Collect output statistics.
    cluster_info.num_clusters = Nc;
    cluster_info.center_indices = center_indices;
    cluster_info.rho = rho;
    cluster_info.delta = delta;
    cluster_info.gamma = gamma;
    cluster_info.mean_nnd_cluster = mean(nnd_cluster);
    cluster_info.median_nnd_cluster = median(nnd_cluster);

    %% 7. Plot the density-peak decision graph and selected centers.
    figure('Name', 'DPC Vortex Cluster Centroids', 'Color', 'w', 'Position', [100, 100, 1000, 350]);
    theme(gcf,'light')

    % (1) Decision Graph (Finding isolated points in the upper right corner)
    subplot(1, 2, 1);
    plot(rho, delta, 'o', 'MarkerSize', 5, 'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', 'none'); hold on;
    plot(rho(center_indices), delta(center_indices), 'r*', 'MarkerSize', 10, 'LineWidth', 1.5);
    grid on; box on;
    xlabel('Local density $\rho$','Interpreter','latex', 'FontSize', 14);
    ylabel('Distance to higher density points $\delta$','Interpreter','latex', 'FontSize', 14);
    % title('Decision Graph (Red stars at upper right denote centroids)','Interpreter','latex');

    % (2) Centroid positions in physical space
    subplot(1, 2, 2);
    scatter(x_v, y_v, 8, rho, 'filled'); hold on; % Colored by local density \rho
    colormap(gca, 'jet'); c=colorbar;
    title(c, '$\rho$', 'FontSize', 14, 'Interpreter', 'latex');
    plot(centroids(:, 1), centroids(:, 2), 'kp', 'MarkerSize', 12, ...
        'MarkerFaceColor', 'w', 'LineWidth', 1.5);
    xlim([0, Lx]); ylim([0, Ly]);
    xticks([100, 200, 300, 400]);
    yticks([100, 200, 300, 400]);
    axis equal tight; box on; grid on;
    xlabel('$x$','Interpreter','latex', 'FontSize', 14); ylabel('$y$','Interpreter','latex', 'FontSize', 14);
    % title(sprintf('High-Density Cluster Centroids (Count: %d, Mean spacing: %.2f)', Nc, cluster_info.mean_nnd_cluster),'Interpreter','latex');
end

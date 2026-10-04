function [smooth_vortices, smooth_antivortices] = find_vortices(O, threshold, window_radius)
% FIND_VORTICES Refine periodic vortex detections in matrix-index units.
% O is complex; threshold is a normalized density cutoff; window_radius is in cells.
% Outputs use [row, column] indices and require explicit conversion to physical lengths.
% This routine is used for configurations with equal vortex and antivortex counts.

% Refine phase-winding detections using a low-density mask centroid.
% Inputs:
%   O - Complex nx-by-ny order-parameter array.
%   threshold - Threshold on density normalized by its global maximum (e.g. 0.2-0.5).
%   window_radius - Search radius in grid cells; 2 or 3 gives a 5-by-5 or 7-by-7 window.

    % 1. Detect coarse plaquette positions using the local phase-winding routine.
    [v_coarse, av_coarse, ~] = find_vortices_old(O);

    % Normalize the density by its global maximum.
    density = abs(O).^2/max(abs(O).^2,[],'all');

    % 2. Refine vortex and antivortex positions.
    if isempty(v_coarse)==0
        for k=1:length(v_coarse(:,1))
            smooth_vortices(k,:) = refine_positions(v_coarse(k,:), density, threshold, window_radius);
            smooth_antivortices(k,:) = refine_positions(av_coarse(k,:), density, threshold, window_radius);
        end
    else
        smooth_vortices=[];
        smooth_antivortices=[];
    end
end

function refined_pos = refine_positions(coarse_pos, density, threshold, radius)
% REFINE_POSITIONS_PBC Compute a low-density mask centroid with periodic boundaries.

    if isempty(coarse_pos)
        refined_pos = [];
        return;
    end

    [nx, ny] = size(density);
    num_vortices = size(coarse_pos, 1);
    refined_pos = zeros(num_vortices, 2);

    % Relative index offsets across the local search window.
    [dC, dR] = meshgrid(-radius:radius, -radius:radius);

    for k = 1:num_vortices
        r_c = coarse_pos(k, 1);
        c_c = coarse_pos(k, 2);

        % Wrap local indices across the periodic boundaries.
        % Subtract and restore 1 to preserve MATLAB indexing when wrapping.
        rows = mod((r_c - radius : r_c + radius) - 1, nx) + 1;
        cols = mod((c_c - radius : c_c + radius) - 1, ny) + 1;

        % Extract the local window, including any periodic boundary crossing.
        local_density = density(rows, cols);

        mask = local_density < threshold;

        if sum(mask(:)) > 0
            % weight = (threshold - local_density) .* mask;
            weight=mask;
            sum_weight = sum(weight(:));

            % Compute the centroid of the binary low-density mask.
            offset_r = sum(dR(:) .* weight(:)) / sum_weight;
            offset_c = sum(dC(:) .* weight(:)) / sum_weight;

            % Add the offsets and wrap the refined indices into the domain.
            final_r = mod((r_c + offset_r) - 1, nx) + 1;
            final_c = mod((c_c + offset_c) - 1, ny) + 1;

            refined_pos(k, :) = [final_r, final_c];
        else
            refined_pos(k, :) = [r_c + 0.5, c_c + 0.5];
        end
    end
end

function [vortices, antivortices, charge_matrix] = find_vortices_old(O)
% FIND_VORTICES Detect vortices and antivortices in a complex 2-D order parameter.
% Inputs:
%   O - Complex nx-by-ny order-parameter array.
% Outputs:
%   vortices     - Positive-vortex matrix indices [row, col] (charge +1).
%   antivortices - Negative-vortex matrix indices [row, col] (charge -1).
%   charge_matrix- nx-by-ny plaquette charges, including periodic edge plaquettes.

    % Obtain the four vertices of each periodic plaquette.
    % p1: lower left, p2: lower right, p3: upper right, p4: upper left.
    p1 = O;                                 % (i, j)
    p2 = circshift(O, [-1, 0]);             % (i+1, j)
    p3 = circshift(O, [-1, -1]);            % (i+1, j+1)
    p4 = circshift(O, [0, -1]);             % (i, j+1)

    % Compute wrapped phase increments from products with complex conjugates.
    d1 = angle(p2 .* conj(p1));
    d2 = angle(p3 .* conj(p2));
    d3 = angle(p4 .* conj(p3));
    d4 = angle(p1 .* conj(p4));

    % Sum phase increments around each plaquette.
    W = d1 + d2 + d3 + d4;

    % Round the winding divided by 2*pi to obtain integer charges.
    charge_matrix = round(W / (2*pi));

    % Extract positions with charge +1.
    [v_row, v_col] = find(charge_matrix == 1);
    vortices = [v_row, v_col];

    % Extract positions with charge -1.
    [av_row, av_col] = find(charge_matrix == -1);
    antivortices = [av_row, av_col];

    % Returned coordinates use matrix indices rather than physical lengths.
    % The unrefined plaquette center is at (row+0.5, col+0.5).
end

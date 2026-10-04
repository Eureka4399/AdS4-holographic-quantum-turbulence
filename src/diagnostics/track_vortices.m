function sorted_data = track_vortices(raw_data, L, max_jump)
% TRACK_VORTICES Greedily match existing tracks with periodic distances.
% raw_data has shape [frame, particle, coordinate]; use NaN for missing positions.
% L defaults to 50, max_jump to 0.1*L, in the same units as the coordinates.
% Tracks absent in one frame are not restarted; new births are not assigned tracks.

    % raw_data: Coordinate array [frame, particle, coordinate]; absent positions are NaN.
    % L: Scalar periodic domain length (default 50).
    % max_jump: Maximum allowed displacement between successive frames.
    if nargin < 2, L = 50; end
    if nargin < 3, max_jump = 0.1*L; end % Default matching radius is 10% of the domain length.

    [num_frames, num_particles, num_dims] = size(raw_data);
    sorted_data = NaN(num_frames, num_particles, num_dims);
    sorted_data(1, :, :) = raw_data(1, :, :);

    for t = 2:num_frames
        prev_pos = reshape(sorted_data(t-1, :, :), [num_particles, num_dims]);
        curr_pos = reshape(raw_data(t, :, :), [num_particles, num_dims]);

        % 1. Compute periodic distances between prior and current positions.
        dist_matrix = Inf(num_particles, num_particles);
        for i = 1:num_particles
            if any(isnan(prev_pos(i, :))), continue; end % Skip tracks absent in the previous frame.
            for j = 1:num_particles
                if ~any(isnan(curr_pos(j, :)))
                    d_pos = prev_pos(i, :) - curr_pos(j, :);
                    d_pos = d_pos - L .* round(d_pos ./ L); % Apply the minimum-image periodic displacement.
                    dist_matrix(i, j) = sum(d_pos.^2);
                end
            end
        end

        % 2. Greedily match globally sorted distances, smallest first.
        matched_indices = zeros(num_particles, 1);
        used_curr = false(num_particles, 1);

        % Sort all pairwise squared distances.
        [sorted_dists, sort_idx] = sort(dist_matrix(:));

        for k = 1:length(sort_idx)
            % Stop when remaining squared distances exceed max_jump squared.
            if sorted_dists(k) > max_jump^2
                break;
            end

            % Recover prior-track and current-position indices from the linear index.
            [p_idx, c_idx] = ind2sub([num_particles, num_particles], sort_idx(k));

            % Match only if both indices are unused.
            if matched_indices(p_idx) == 0 && ~used_curr(c_idx)
                matched_indices(p_idx) = c_idx;
                used_curr(c_idx) = true;
            end
        end

        % 3. Store the matched current-frame positions.
        for i = 1:num_particles
            idx = matched_indices(i);
            if idx > 0
                sorted_data(t, i, :) = curr_pos(idx, :);
            else
                % Unmatched tracks are marked missing; this alone does not prove annihilation.
                sorted_data(t, i, :) = NaN;
            end
        end
    end
end

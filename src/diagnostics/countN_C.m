function [n, np, nm, C] = countN_C(x1)
% COUNTN_C Count vortices and compute nearest-neighbor sign correlation.
% C averages sign_i*sign_neighbor and is NaN when fewer than two vortices exist.
% Detection is periodic; neighbor distances use Euclidean matrix indices.

    % Compute the same periodic phase-winding masks used by countN.
    x1 = gather(x1);
    [x2, x3, x4] = deal(circshift(x1, [-1, 0]), circshift(x1, [-1, -1]), circshift(x1, [0, -1]));
    w = 1/(2*pi) * (angle(x1./x2) + angle(x2./x3) + angle(x3./x4) + angle(x4./x1));

    Lp = w >= 1-1e-4 & w <= 1+1e-4;
    Lm = w <= -1+1e-4 & w >= -1-1e-4;

    % ======= Step 1: Extract positive and negative vortex indices (row, column). =======
    [row_p, col_p] = find(Lp);
    [row_m, col_m] = find(Lm);

    np = length(row_p);
    nm = length(row_m);
    n  = np + nm; % Total vortex count.

    % ======= Step 2: Compute the nearest-neighbor sign correlation C. =======
    if n < 2
        % Return NaN when fewer than two vortices make neighbor statistics undefined.
        C = NaN;
        return;
    end

    % 1. Combine vortex positions into an N-by-2 index array.
    pos = [row_p, col_p; row_m, col_m];

    % 2. Combine vortex signs (+1 and -1) into an N-by-1 vector.
    gamma = [ones(np, 1); -ones(nm, 1)];

    % 3. Compute Euclidean index distances; this legacy statistic does not wrap boundaries.
    % Use implicit expansion to construct pairwise distances.
    D = sqrt((pos(:, 1) - pos(:, 1)').^2 + (pos(:, 2) - pos(:, 2)').^2);

    % 4. Exclude self distances by setting the diagonal to infinity.
    D(1:n+1:end) = Inf;

    % 5. Find the nearest-neighbor index in each row.
    [~, nearest_idx] = min(D, [], 2);

    % 6. Multiply each sign by its nearest-neighbor sign.
    % Equal signs give +1 and opposite signs give -1.
    c = gamma .* gamma(nearest_idx);

    % 7. Average the sign products to obtain C.
    C = mean(c);
end

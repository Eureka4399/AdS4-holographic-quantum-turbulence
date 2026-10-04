function [S2, rx, ry] = structure_function_2d(ux, uy)
% STRUCTURE_FUNCTION_2D Periodic second-order velocity structure function.
% ux and uy are finite real fields with shape [nx,ny]. rx and ry are index lags.
% NaNs propagate through the FFT even though the mean uses omitnan.

    % Inputs:
    % ux, uy  : nx-by-ny velocity fields.
    %
    % Outputs:
    % S2      : nx-by-ny structure function at index lags (rx(i), ry(j)).
    % rx, ry  : Index-lag vectors with the zero lag shifted to the array center.

    [nx, ny] = size(ux);

    % 1. Subtract the mean velocity to obtain fluctuations.
    ux_prime = ux - mean(ux(:), 'omitnan');
    uy_prime = uy - mean(uy(:), 'omitnan');

    % 2. Compute total fluctuation variance R(0).
    R0 = mean(ux_prime(:).^2 + uy_prime(:).^2, 'omitnan');

    % 3. Compute the periodic spatial autocorrelation using a 2-D FFT.
    F_ux = fft2(ux_prime);
    F_uy = fft2(uy_prime);

    % The inverse power-spectrum transform gives the autocorrelation.
    R_r = real(ifft2(abs(F_ux).^2 + abs(F_uy).^2)) / (nx * ny);

    % 4. Use S_2(r) = 2 * [R(0) - R(r)].
    S2_raw = 2 * (R0 - R_r);

    % 5. Shift zero lag to the center for plotting.
    S2 = fftshift(S2_raw);

    % 6. Build lag coordinates in grid-cell units.
    rx = -floor(nx/2) : (ceil(nx/2) - 1);
    ry = -floor(ny/2) : (ceil(ny/2) - 1);
end

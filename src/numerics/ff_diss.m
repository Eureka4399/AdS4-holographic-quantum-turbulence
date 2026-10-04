function w = ff_diss(x,y,gamma)
% FF_DISS Dampen masked Fourier coefficients by exp(-gamma).
% x is a field or stack, y is a Fourier mask, and gamma is the damping strength.
% Unlike ff, the inverse transform retains its complex component.

% Work in Fourier space, independently for each trailing field slice.
b=fft2(x);
% Dampen only the selected modes while leaving the others unchanged.
b(y)=b(y)*exp(-gamma);
% Retain the complex inverse transform so scalar phases are preserved.
w=ifft2(b);
end

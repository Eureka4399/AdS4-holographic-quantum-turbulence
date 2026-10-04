function w = ff(x,y)
% FF Zero masked Fourier coefficients and return a real field.
% x is a 2-D field or stack; y is a logical mask compatible with fft2(x).
% The imaginary part is discarded. Filter complex fields componentwise.

% Transform the first two dimensions; trailing dimensions represent field slices.
b=fft2(x);
% Remove the modes selected by the simulation's Fourier cutoff mask.
b(y)=0;
% This filter is intended for real components of the evolved fields.
w=real(ifft2(b));
end

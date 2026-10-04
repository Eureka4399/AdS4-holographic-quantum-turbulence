function [n,np,nm] = countN(x1)
% COUNTN Count periodic plaquettes with winding +1 and -1.
% Input x1 is a complex 2-D order parameter; outputs are total, positive, negative.
% Exact zeros can make phase ratios undefined; no core regularization is applied.

% Bring a GPU input to host memory before detecting discrete plaquette charges.
x1=gather(x1);
% Visit each plaquette and include its neighbors across periodic boundaries.
[x2,x3,x4]=deal(circshift(x1,[-1,0]),circshift(x1,[-1,-1]),circshift(x1,[0,-1]));
% Sum wrapped phase increments and normalize the winding by one full turn.
w=1/(2*pi)*(angle(x1./x2)+angle(x2./x3)+angle(x3./x4)+angle(x4./x1));
% A narrow tolerance distinguishes unit charges from floating-point roundoff.
Lp=w>=1-1e-4&w<=1+1e-4;  Lm=w<=-1+1e-4&w>=-1-1e-4;
Lall= Lp+Lm;%Vortex occupancy mask.
% Report the total and the two charge signs separately.
[a0,~]=find(Lall==1);
n=length(a0);
[ap,~]=find(Lp==1);
np=length(ap);
[am,~]=find(Lm==1);
nm=length(am);
end

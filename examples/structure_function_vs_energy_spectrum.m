% STRUCTURE_FUNCTION_VS_ENERGY_SPECTRUM Explore a Bessel-kernel integral.
% This standalone exploratory example uses no saved simulation data.

% Inspect the Bessel kernel for one separation before integrating over wavenumber.
r=0.1;
fun=@(k) (1-2*besselj(1,k*r)./(k*r)).*exp(-(k-pi/1).^2*1e1);
k=0.01:0.01:10;
figure(5),clf
hold on
plot((k),(fun(k)),'b')
% plot(log(k),-5/3*log(k),'r')
hold off

%%
% Repeat the integral for increasing separations and compare with quadratic scaling.
A=zeros(1,100);
for kk=1:100
    r=0.1*kk;a=0.1;
    fun=@(k) (1-2*besselj(1,k*r)./(k*r)).*exp(-(k-pi/1).^2*1e1);
    A(kk)=integral(fun,0.01,5,'RelTol',0,'AbsTol',1e-12);
end
figure(5),clf
hold on
plot(log(1:100),log(A),'b')
plot(log(1:100),2*log(1:100)+log(A(1)),'r')
hold off

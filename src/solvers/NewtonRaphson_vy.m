function [z1,z2,VY1,it,error] = NewtonRaphson_vy(nz,zh,small,D,Z1,Z2,vy1,vy)
% NEWTONRAPHSON_VY Solve stationary radial profiles with imposed flow.
% Z1, Z2, and vy1 are nz-by-1 initial guesses; vy is the boundary flow parameter.
% Returns updated profiles, iteration count, and final update norm.
% The stopping residual tolerance is 1e-6; the iteration limit is 1000.
% This solver opens diagnostic figures and prints iteration progress.

it_max = 1000;
tol =1e-6;

% Map descending Chebyshev nodes onto the horizon-to-boundary radial interval.
[zzcheb,Dz]=chebdif(nz,1); %[~,prex]=FD(1,0);[~,prez]=FD(nz,0);
lz=Dz;
zi=D; zf=1-small;
dlz=zf-zi;

pz=2/dlz;
zz=(zzcheb-1)*dlz/2+zf*ones(nz,1);
z=zz;
Lz=pz*lz;

func_Lz=@(x) Lz*x;

% Evaluate the blackening factor and the regular horizon limit of Z2/f.
f=1-(zz/zh).^3;df=-3*zz.^2/zh^3;
pZ2=func_Lz(Z2);Z2_f=zeros(nz,1);Z2_f_temp=Z2_f;
Z2_f(2:nz)=Z2(2:nz)./(1-(zz(2:nz)/zh).^3);Z2_f(1)=-pZ2(1)/3;
vy1=vy1-vy1(nz);
Vy1=vy+vy1;
%%%%%%%%%% with vy %%%%%%%%%%%%%%%%%%%%%%%%%%
% Assemble the three coupled stationary equations in profile-block order.
fun=@(Z1,Z2,Vy1,Z2_f) [f.*(Lz^2*Z1)+df.*(Lz*Z1)-Vy1.^2.*Z1+Z2.*Z2_f.*Z1-z.*Z1; ...
                         f.*(Lz^2*Z2)-2*Z1.^2.*Z2; ...
                         -f.*(Lz^2*Vy1)-df.*(Lz*Vy1)+2*Vy1.*Z1.^2;];

% Use the analytic block Jacobian for a simultaneous Newton update.
dfun=@(Z1,Z2,Vy1,Z2_f) [diag(f)*Lz^2+diag(df)*Lz+diag(Z2.*Z2_f-Vy1.^2-z),   diag(2*Z2_f.*Z1),  diag(-2*Vy1.*Z1); ...
                          diag(-4*Z1.*Z2),   diag(f)*Lz^2-diag(2*Z1.^2),  zeros(nz); ...
                          diag(4*Vy1.*Z1), zeros(nz),-diag(f)*Lz^2-diag(df)*Lz+diag(2*Z1.^2);];

zz1=Z1;
zz2=Z2;
zz4=Vy1;

figure(1),plot(zz,zz1);
figure(2),plot(zz,zz2);
figure(3),plot(zz,zz4);


BCz=[zeros(2,(nz-1) ),eye(2 ),zeros(2 ,(2*nz-1) );
     zeros(1,(2*nz-1) ),eye(1),zeros(1,1*nz );
     zeros(1,(3*nz-1) ),eye(1);];

% Replace selected Jacobian rows with endpoint conditions before each Newton solve.

K = 0;  a=0.1;
while K<it_max
     if K==0
        Fun=fun(Z1,Z2,Vy1,Z2_f);dFun=dfun(Z1,Z2,Vy1,Z2_f);
        k=[1*(nz-1)+1:1*(nz+1),1*(2*nz-1)+1:2*1*nz,1*(3*nz-1)+1:3*1*nz];
        Fun(k,1)=0;
        dFun(k,:)=BCz;
    elseif K>0
        Fun=Fun_next;dFun=dfun(Z1,Z2,Vy1,Z2_f);
        k=[1*(nz-1)+1:1*(nz+1),1*(2*nz-1)+1:2*1*nz,1*(3*nz-1)+1:3*1*nz];
        dFun(k,:)=BCz;
     end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y=-dFun\Fun;
% Update all three profiles together, then recompute the nonlinear residual.
% y=-gmres(dFun,Fun,10,1e-4,10);
err0=max(abs(Fun));
    Z1_temp = Z1 + y(1:1*nz,1);
    Z2_temp = Z2 + y(1*nz+1:2*1*nz,1);
    Vy1_temp = Vy1 + y(2*1*nz+1:3*1*nz,1);
    pAt=func_Lz(Z2_temp);
    Z2_f_temp(1+1:end)=Z2_temp(1+1:end)./(1-(z(1+1:end)/zh).^3);
    Z2_f_temp(1:1)=-pAt(1)/3;
    Fun_next=fun(Z1_temp,Z2_temp,Vy1_temp,Z2_f_temp);
    Fun_next(k,1)=0;
    err2=max(abs(Fun_next));

    [Z1,Z2,Vy1,Z2_f]=deal(Z1_temp,Z2_temp,Vy1_temp,Z2_f_temp);
    err2
    err = norm(y,inf);
    [K,err]
    % Stop at the residual tolerance; preserve the update norm as the output error.
    if isnan(err)
        fprintf('Error is an NaN. \n')
        break
%     elseif err<tol&&max(err2)<tol
    elseif max(err2)<tol
        break;
%     elseif err<tol
%         break
    end
%%%%%%%% Plot the Newton iterates. %%%%%%%%%%%%%%
figure(1),plot(zz,Z1);
figure(2),plot(zz,Z2);
figure(3),plot(zz,Vy1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    K = K+1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z1=Z1;z2=Z2;VY1=Vy1; it = K;error=err;

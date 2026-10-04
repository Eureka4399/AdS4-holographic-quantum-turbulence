% STATINARY_STATE_WITH_V Historical stationary-state and thermodynamics notebook.
% Run selected sections after setup_project; parameters and dependencies differ by section.
% The original filename is retained for traceability.

% Enable compatibility before creating or loading GPU-backed fields.
enable_gpu

nz=30;
%  vy0=-5*2*pi/20;
% mu=5;
for kk=1
    for KK=1
%     mu=mu0-0.1*(KK-1);
    % Load the dataset selected by this section using a repository-relative location.
    load(hqt_data_path('reference', ['mu=',num2str(6),',nz=',num2str(nz),',vy=',num2str(0),'_20.mat']))
%     load(hqt_data_path('reference', ['mu=4.07,nz=',num2str(nz),',vy=',num2str(0.02*kk),'_20.mat']))
%     mu=mu0-0.1*(KK);
mu=5;
[zzcheb,Dz]=chebdif(nz,1);
lz=Dz;
small=0;        % the distance to horizon;
D=0;          % the distance to infinite boundary

zi=D; zf=1-small;               % INPUT of end points of z-direction
dlz=zf-zi;
pz=2/dlz;

zz=(zzcheb-1)*dlz/2+zf*ones(nz,1); %coordinate trans. to real coordinate
%---define real Derivative with coefficients
Lz=pz*lz; %Lzz=pz.^2*lzz;
Lz1=Lz(1:nz-1,1:nz-1)\eye(nz-1);
Lz2=Lz(2:nz,2:nz)\eye(nz-1);
LLz=Lz^2;Lzz=LLz(2:nz-1,2:nz-1)\eye(nz-2);
% END OF UNIVERSAL BASICS

% Define derivatives and radial integration operators with boundary conditions.
func_Lz=@(x) Lz*x;     % Apply the radial derivative along the z dimension.
func_Lzr=@(x) [[Lz1,zeros(nz-1,1)]*x;zeros(1)];
func_Lzr2=@(x) [zeros(1);[zeros(nz-1,1),Lz2]*x];

%Define the scalar and gauge-field evolution operators.
zh=1;e=1;
% vy=-2*pi/20*(0.02*kk-0.01);
vy=-2*pi/20*0;
f=1-(zz/zh).^3;df=-3*zz.^2/zh^3;
%%Solve the stationary initial state by Newton iteration.%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% z1=0.1*zz.^0.5;z2=mu.*(1-zz);
% vy1=vy*ones(nz,1);

Z2=Z2/6*mu;
[z1,z2,vy1]=deal(double(Z1),double(Z2),double(Vy1));
[Z1,Z2,Vy1,it,error] = NewtonRaphson_vy(nz,zh,small,D,z1,z2,vy1,vy);
 rhoi=-func_Lz(Z2);
%  func_Lz(Z1)
%  save(hqt_data_path('reference', ['mu=',num2str(mu),',nz=',num2str(nz),',vy=',num2str(-vy/2/pi*20),'_20.mat']),'mu','rhoi','Z1','Z2','Vy1')
 save(hqt_data_path('reference', ['mu=',num2str(mu),',nz=',num2str(nz),',vy=',num2str(-vy/2/pi*20),'_20.mat']),'mu','rhoi','Z1','Z2','Vy1')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end
%% free energy
nz=30;
[zzcheb,Dz]=chebdif(nz,1);
lz=Dz;
small=0;        % the distance to horizon;
D=0;          % the distance to infinite boundary
zi=D; zf=1-small;               % INPUT of end points of z-direction
dlz=zf-zi;
pz=2/dlz;
zz=(zzcheb-1)*dlz/2+zf*ones(nz,1); %coordinate trans. to real coordinate
%---define real Derivative with coefficients
Lz=pz*lz; %Lzz=pz.^2*lzz;
Lz1=Lz(1:nz-1,1:nz-1)\eye(nz-1);
Lz2=Lz(2:nz,2:nz)\eye(nz-1);
LLz=Lz^2;Lzz=LLz(2:nz-1,2:nz-1)\eye(nz-2);
% END OF UNIVERSAL BASICS
% Define derivatives and radial integration operators with boundary conditions.
func_Lz=@(x) Lz*x;     % Apply the radial derivative along the z dimension.
func_Lzr=@(x) [[Lz1,zeros(nz-1,1)]*x;zeros(1)];
func_Lzr2=@(x) [zeros(1);[zeros(nz-1,1),Lz2]*x];
%Define the scalar and gauge-field evolution operators.
zh=1;e=1;
f=1-(zz/zh).^3;df=-3*zz.^2/zh^3;

vy0=-0*2*pi/20;mu0=6;
n1=0;
n2=75;
Free_energy0=zeros(n1+1,n2+1);
for kk=0:n2
    for KK=0
    vy=-2*pi/20*0.1*kk+vy0;
    mu=mu0+0.1*KK;
    % Load the dataset selected by this section using a repository-relative location.
    load(hqt_data_path('reference', ['mu=',num2str(mu),',nz=',num2str(nz),',vy=',num2str(-vy/2/pi*20),'_20.mat']))
    Z5=zeros(30,1);Z4=vy-Vy1;rho=-func_Lz(Z2);rho=rho(nz);
%     %%%% fix rho %%%%
%     pZ2 = func_Lz(Z2);Z2_f=zeros(nz,1);
%     Z2_f(2:nz) = Z2(2:nz)./(1-(zz(2:nz)/zh).^3);
%     Z2_f(1) = -pZ2(1)/3;
%     psi = (Z1(:));
%     phi = (Z2(:));
%     phi_f = (Z2_f(:));
%     z_1d = (zz(:));
%     bulk_term =psi.^2.*phi.*phi_f;
%     integral_bulk_term = Lz(1:nz-1,1:nz-1)\bulk_term(1:nz-1);
%     boundary_term = mu*rho/2;
%     F = boundary_term+integral_bulk_term(1);
%     Free_energy0(KK+1,kk+1)=F(1)-rho^2/2;
    %%%% fix mu %%%%
    F=free_energy(Z1,Z2,Z4,vy);
    Free_energy0(KK+1,kk+1)=F(1)+mu^2/2;
    %%%%%%%%%%%%%%%%
    end
end
figure(1),clf
plot(Free_energy0)
%% Free energy response
nz=30;
vy0=-0.3*2*pi/20;mu0=4.07;
Free_energy0=zeros(11,11);
J=zeros(11,11);
for kk=0:1
    for KK=0:1
    vy=-2*pi/20*0.001*kk+vy0;
    mu=mu0+0.01*KK;
    % Load the dataset selected by this section using a repository-relative location.
    load(hqt_data_path('processed', 'thermodynamics', ['mu=',num2str(mu),',nz=',num2str(nz),',vy=',num2str(-vy/2/pi*20),'_20.mat']))
    Z5=zeros(30,1);Z4=vy-Vy1;
    F=free_energy_1(Z1,Z5,Z2,Z4,vy,0,0);
    Free_energy0(KK+1,kk+1)=F(1);
    J0=func_Lz(Z4);
    J(KK+1,kk+1)=J0(nz);
    end
end
% figure(1),clf,mesh(J)
% figure(2),clf,plot((J(1,2:end)-J(1,1:end-1))/(-0.001*pi/10))
n=(Free_energy0(2,1)-Free_energy0(1,1))/1e-2;
chi_nn=(Free_energy0(1,1)+Free_energy0(3,1)-2*Free_energy0(2,1))/1e-4;
chi_nh=((Free_energy0(2,2)-Free_energy0(1,2))/1e-2-(Free_energy0(2,1)-Free_energy0(1,1))/1e-2)/(-0.001*2*pi/20);
% Load the dataset selected by this section using a repository-relative location.
load(hqt_data_path('processed', 'thermodynamics', ['mu=',num2str(mu0),',nz=',num2str(nz),',vy=',num2str(-vy0/2/pi*20),'_20.mat']))
Z4=vy-Vy1;
J1=func_Lz(Z4);
J1=J1(nz);
% Load the dataset selected by this section using a repository-relative location.
load(hqt_data_path('processed', 'thermodynamics', ['mu=',num2str(mu0),',nz=',num2str(nz),',vy=',num2str(-vy0/2/pi*20+0.001),'_20.mat']))
Z4=vy-Vy1;
J2=func_Lz(Z4);
J2=J2(nz);
ns=(J2-J1)/(-0.001*pi/10);
% Slope_logV_N_S=0.5037;
% Slope_logV_N_S=1.627;
Slope_logV_N_S=0.4781;
R=(Slope_logV_N_S/(ns*chi_nn/(chi_nn-ns*chi_nh^2))/pi/2)^0.5; % superfluid_velocity=vy/mu;
%%
nx=100; nz=30;
[x,Dx]=FD(nx,1);
% [x,Dx]=chebdif(nx,1);
[z,Dz]=chebdif(nz,1);
%[zzcheb,yycheb]=meshgrid(z,y);      %meshgrid ordering differs from the ndgrid convention used below.
[xxcheb,zzcheb]=ndgrid(x,z);         %Use ndgrid ordering: the first index represents x.

lx=Dx;  lz=Dz;
small=0;        % the distance to horizon;
D=0;          % the distance to infinite boundary
L0=50;

xi=0; xf=L0;               % INPUT of end points of x-direction                      % INPUT of end points of y-direction
zi=D; zf=1-small;               % INPUT of end points of z-direction
dlz=zf-zi;
px=2/L0;%2/dlx;
pz=2/dlz;

xx=(xxcheb-1)*L0/2+xf*ones(nx,nz); %coordinate trans. to real coordinate
zz=(zzcheb-1)*dlz/2+zf*ones(nx,nz); %coordinate trans. to real coordinate
%---define real Derivative with coefficients
Lx=px*lx; %Lxx=px.^2*lxx;
Lz=pz*lz; %Lzz=pz.^2*lzz;
Lz1=Lz(1:nz-1,1:nz-1)\eye(nz-1);
Lz2=Lz(2:nz,2:nz)\eye(nz-1);
LLz=Lz^2;Lzz=LLz(2:nz-1,2:nz-1)\eye(nz-2);
xy0=zeros(1,nx);

func_Lx=@(x) reshape(Lx*reshape(x,[nx,nz]),[nx,nz]);      % Apply the x derivative along the first array dimension.
func_Lz=@(x) permute(reshape(Lz*reshape(permute(x,[2,1]),[nz,nx]),[nz,nx]),[2,1]);     % Apply the radial derivative along the z dimension.
f1=@(x) reshape(permute(x,[2,1]),[nz,nx]);
f2=@(x) permute(reshape(x,[nz,nx]),[2,1]);
func_Lzr=@(x) f2([[Lz1,zeros(nz-1,1)]*f1(x);xy0]);

% load(hqt_data_path('reference', 'mu=6,nu=-1,L=5,s=1.mat'))
% Load the dataset selected by this section using a repository-relative location.
load(hqt_data_path('reference', 'mu=4.07,nu=-1,L=50,s=1.mat'))
zh=1;
pZ2=func_Lz(at);Z2_f=zeros(nx,nz);
Z2_f(:,2:nz)=at(:,2:nz)./(1-(zz(:,2:nz)/zh).^3);Z2_f(:,1)=-pZ2(:,1)/3;
% Integrate the radial potential to obtain the phase used in the field initialization.
theta=-func_Lzr(Z2_f);
psi=psi.*exp(1i*theta);
O=func_Lz(psi);
figure(1),clf,hold on
Jtheta=func_Lz(atheta)./xx;
plot(xx(:,nz),abs(O(:,nz)),'r')
% plot(xx(:,nz),abs(Jtheta(:,nz)),'b')
xlim([0,L0])
hold off
xL='$r$';yL='$|\mathcal{O}|$';
% xL='$r$';yL='$J_\theta$';
xlabel(xL,'Interpreter','latex','fontsize',14)
ylabel(yL,'Interpreter','latex','fontsize',14)

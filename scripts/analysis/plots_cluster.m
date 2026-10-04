% PLOTS_CLUSTER Historical checkpoint analysis and plotting notebook.
% Call setup_project first, then run selected MATLAB sections in dependency order.
% Checkpoint inputs under data/raw/ are intentionally omitted from Git.
% Set grid, domain, time range, and run-folder suffix to match the generated run.
% Figures/movies go to outputs/; processed arrays go to data/processed/.

% diary(hqt_output_path('logs', 'progress_read.log'))
% Enable compatibility before creating or loading GPU-backed fields.
enable_gpu

gpuDevice()
mu=6;vy_load=-2*pi/20*7;
subFolder = hqt_data_path('raw', ['check_point_driven_mu=',num2str(mu),',v=',num2str(-vy_load*20/2/pi),'_20_large']);
fileName = fullfile(subFolder, ['check_point_t=',num2str(100),'.mat']);load(fileName)
NN=size(Z1);
nx=NN(1); ny=NN(2); nz=NN(3);
% Build periodic spectral differentiation matrices on the spatial grid.
[x,Dx]=fourdif(nx,1);
[y,Dy]=fourdif(ny,1);
[z,Dz]=chebdif(nz,1);
% Use x/y/z array ordering before mapping the collocation points to physical coordinates.
[xxcheb,yycheb,zzcheb]=ndgrid(x,y,z);         %Use ndgrid ordering: the first index represents x.

lx=Dx; ly=Dy;  lz=Dz;
small=gpuArray(0);        % the distance to horizon;
D=gpuArray(0);          % the distance to infinite boundary
L=gpuArray((nx-1)/2);
L0=L;

% Construct the Fourier-mode cutoff used to suppress unresolved spatial modes.
fx=false(nx,1,1);
fx(round(round(nx/3)):round(nx/3*2))=1;
fy=false(1,ny,1);
fy(round(round(ny/3)):round(ny/3*2))=1;
fxyz=gpuArray(repmat(fx|fy,[1,1,nz]));

%---transform to real coordinates
xi=0; xf=L0;               % INPUT of end points of x-direction
yi=0; yf=L;                    % INPUT of end points of y-direction
zi=D; zf=1-small;               % INPUT of end points of z-direction
dlz=zf-zi;
px=2.*pi/L0;%2/dlx;
py=2.*pi/L;%2/dly;
pz=2/dlz;

xx=(xxcheb)*L0/2/pi; %coordinate trans. to real coordinate
yy=(yycheb*L)/2/pi; %coordinate trans. to real coordinate
zz=(zzcheb-1)*dlz/2+zf*ones(nx,ny,nz); %coordinate trans. to real coordinate
%---define real Derivative with coefficients
Lx=px*lx; %Lxx=px.^2*lxx;
Ly=py*ly; %Lyy=py.^2*lyy;
Lz=pz*lz; %Lzz=pz.^2*lzz;
Lz1=Lz(1:nz-1,1:nz-1)\eye(nz-1);
Lz2=Lz(2:nz,2:nz)\eye(nz-1);
LLz=Lz^2;Lzz=LLz(2:nz-1,2:nz-1)\eye(nz-2);
xy0=gpuArray(zeros(1,ny*nx));
xy1=gpuArray(ones(1,ny*nx));
% END OF UNIVERSAL BASICS

% Define derivatives and radial integration operators with boundary conditions.
func_Lx=@(x) reshape(Lx*reshape(x,[nx,ny*nz]),[nx,ny,nz]);      % Apply the x derivative along the first array dimension.
func_Ly=@(x) permute(reshape(Ly*reshape(permute(x,[2,1,3]),[ny,nx*nz]),[ny,nx,nz]),[2,1,3]); % Apply the y derivative along the second array dimension.
func_Lz=@(x) permute(reshape(Lz*reshape(permute(x,[3,2,1]),[nz,ny*nx]),[nz,ny,nx]),[3,2,1]);     % Apply the radial derivative along the z dimension.
f1=@(x) reshape(permute(x,[3,2,1]),[nz,ny*nx]);
f2=@(x) permute(reshape(x,[nz,ny,nx]),[3,2,1]);
func_Lzr=@(x) f2([[Lz1,zeros(nz-1,1)]*f1(x);xy0]);
func_Lzr2=@(x) f2([xy0;[zeros(nz-1,1),Lz2]*f1(x)]);
% END OF UNIVERSAL BASICS
%Define the scalar and gauge-field evolution operators.
zh=1;e=1;
f=1-(zz/zh).^3;df=-3*zz.^2/zh^3;
F1=@(y1,y2,y3,y4) func_Lzr(1i.*y2.*(func_Lz(y1))+0.5i.*y1.*func_Lz(y2)+0.5.*(1-(zz/zh).^3).*func_Lz(func_Lz(y1))-0.5.*3.*zz.^2./zh^3.*func_Lz(y1)-0.5.*zz.*y1-0.5.*y3.*y3.*y1-0.5.*y4.*y4.*y1+1.*0.5.*func_Lx(func_Lx(y1))+ ...
    1.*0.5.*func_Ly(func_Ly((y1)))-1.0.*0.5i.*y1.*(func_Lx(y3)+func_Ly(y4))-1i.*(y3.*func_Lx(y1)+y4.*func_Ly(y1)));
F3=@(y1,y2,y3,y4) real(func_Lzr(0.5.*func_Lz(func_Lx(y2)+(1-(zz/zh).^3).*func_Lz(y3))+0.5.*func_Ly(func_Ly(y3)-func_Lx(y4))-0.5i.*(conj(y1).*func_Lx(y1)-y1.*func_Lx(conj(y1)))-y3.*y1.*conj(y1)));
F4=@(y1,y2,y3,y4) real(func_Lzr(0.5.*func_Lz(func_Ly(y2)+(1-(zz/zh).^3).*func_Lz(y4))+0.5.*func_Lx(-func_Ly(y3)+func_Lx(y4))-0.5i.*(conj(y1).*func_Ly(y1)-y1.*func_Ly(conj(y1)))-y4.*y1.*conj(y1)));
F2=@(y1,y2,y3,y4) (1-zz.^3).*func_Lz(func_Lx(y3)+func_Ly(y4));
Constraint=@(y1,y2,y3,y4) func_Lz(func_Lx(y3)+func_Ly(y4)-func_Lz(y2))-2*imag((conj(y1).*func_Lz(y1)));

tmin=0;
tmax=2000;
cmap = turbo((tmax-tmin)/10+1);
NofV=zeros(1,tmax/10);
O=zeros(nx,ny,tmax/10);
vx=zeros(nx,ny,tmax/10);
vy=zeros(nx,ny,tmax/10);
% Nx=(nx-1)/2;
for t=tmin+10:10:tmax
    t
    fileName = fullfile(subFolder, ['check_point_t=',num2str(t),'.mat']);load(fileName)
    % Differentiate the bulk scalar; its final radial slice is the boundary order parameter.
    O1=func_Lz(Z1);
    % Construct the density-weighted boundary velocity observable for this analysis.
    VX_temp=abs(O1)./max(abs(O1(:,:,end))).*real(-1i*(func_Lx(O1)-func_Lx(abs(O1)).*O1./abs(O1))./O1); VX_temp=VX_temp(:,:,end);
    VY_temp=abs(O1)./max(abs(O1(:,:,end))).*real(-1i*(func_Ly(O1)-func_Ly(abs(O1)).*O1./abs(O1))./O1); VY_temp=VY_temp(:,:,end);
    [NofV(round(t/10)),~]=countN(reshape(O1(:,:,nz),[nx,ny]));
    O(:,:,round(t/10))=O1(:,:,nz);
    vx(:,:,round(t/10))=VX_temp;
    vy(:,:,round(t/10))=VY_temp;
end
save(hqt_data_path('processed', ['driven_mu=',num2str(mu),',v=',num2str(-vy_load*20/2/pi),'_20.mat']),'NofV','O','vx','vy', '-v7.3')
%% movie show
t=2000;
fileName = fullfile(subFolder, ['check_point_t=',num2str(t),'.mat']);load(fileName)
Nx=(nx-1)/2;

% Differentiate the bulk scalar; its final radial slice is the boundary order parameter.
O1=func_Lz(Z1);
fig1=figure('Visible', 'off');
c_blue = [0.0,0.0,0.8];  % #2166AC (blue endpoint)
c_mid  = [247, 247, 247] / 255;  % #F7F7F7 (white/gray midpoint)
c_red  = [0.8,0.0,0.0];  % #B2182B (red endpoint)
N = 256;
x = [1, round(N/2), N]; % Colormap endpoint and midpoint indices.
xi = (1:N)';            % Colormap interpolation samples.
my_cmap = interp1(x, [c_blue; c_mid; c_red], xi, 'linear');
colormap(my_cmap);
set(0,"CurrentFigure",fig1)
set(fig1,'Position',[100 200 800 280]);
clf
tiledlayout(1,2,'TileSpacing','compact','Padding','compact')%,'Padding','compact'
ax1=nexttile;
s=surf(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),abs(O1((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz)),'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([L0/2-L/2,L0/2+L/2]);
view(0,90);
h_cb1 = colorbar;
h_cb1.FontSize = 12;
ylabel(h_cb1, '$\mathcal{O}$', 'FontSize', 14, 'Interpreter', 'latex');
clim([0,gather(max(abs(O1(:,:,nz)),[],'all'))])
h_cb1.TickLabelInterpreter = 'latex';
ax2=nexttile;
ss=mesh(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),angle(O1((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz)),'FaceColor','interp' );
ss.EdgeColor = 'none';
xlim([L0/2-L/2,L0/2+L/2]);
view(0,90);
c = colorbar;
c.FontSize = 12;
ylabel(c, '$\theta$', 'FontSize', 14, 'Interpreter', 'latex');
clim([-pi,pi])
c.Ticks = [-pi, 0, pi];
c.TickLabels = {'$-\pi$', '0', '$\pi$'};
c.TickLabelInterpreter = 'latex';
axis([ax1,ax2], 'off');
saveas(gcf,hqt_output_path('figures', ['movie_driven_mu=',num2str(mu),',v=',num2str(-vy_load*20/2/pi),'_20.png']))

VX_temp=real(-1i*(conj(O1).*func_Lx(O1)-O1.*func_Lx(conj(O1)))./(2*abs(O1)));
VY_temp=real(-1i*(conj(O1).*func_Ly(O1)-O1.*func_Ly(conj(O1)))./(2*abs(O1)));
ux=reshape(VX_temp(:,:,nz),[nx,ny]);
uy=reshape(VY_temp(:,:,nz),[nx,ny]);
ux = ux - mean(ux(:));
uy = uy - mean(uy(:));
%%%%%%%%%%%%%%% Helmholtz decomposition %%%%%%%%%%%%%%%%%%%%%%%%%
% Use the divergence as the source of the Helmholtz potential in Fourier space.
rhs=func_Lx(VX_temp)+func_Ly(VY_temp);
[kx,ky]=ndgrid((-(nx-1)/2:1:(nx-1)/2)*2*pi/L0,(-(ny-1)/2:1:(ny-1)/2)*2*pi/L);
kx((nx+1)/2,(ny+1)/2)=1;ky((nx+1)/2,(ny+1)/2)=1;
rhs_fft=fftshift(fftshift(fft2(rhs),2),1);
rhs_fft((nx+1)/2,(ny+1)/2)=0;
% Remove the zero mode and solve the spatial Poisson equation mode by mode.
Phi_fft=ifftshift(ifftshift(-rhs_fft./(kx.^2+ky.^2),2),1);
Phi_hd=ifft2(Phi_fft);
VX_c=(func_Lx(Phi_hd));VY_c=(func_Ly(Phi_hd));
VX_c=VX_c(:,:,nz);VY_c=VY_c(:,:,nz);
VX_i=ux-VX_c;VY_i=uy-VY_c;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if mod(nx, 2) == 0
    kx = (2*pi/L0) * [0:nx/2-1 , -nx/2:-1];
    ky = (2*pi/L ) * [0:ny/2-1 , -ny/2:-1];
else
    kx = (2*pi/L0) * [0:(nx-1)/2, -(nx-1)/2:-1];
    ky = (2*pi/L ) * [0:(ny-1)/2, -(ny-1)/2:-1];
end
[KX,KY] = meshgrid(kx,ky);
K = sqrt(KX.^2 + KY.^2);
kmax = floor(max(K(:)));
dk = 2*pi/L0;
% Define radial wavenumber shells for the isotropic spectrum.
kbins = 0:dk:kmax;
kplot = 0.5*(kbins(1:end-1) + kbins(2:end));

uxhat = fft2(ux);
uyhat = fft2(uy);
% Form the two-dimensional kinetic-energy spectrum before radial shell averaging.
E2d = 0.5 * (abs(uxhat).^2 + abs(uyhat).^2);
Ek = zeros(length(kbins)-1,1);
for i = 1:length(kbins)-1
    mask = (K >= kbins(i)) & ...
        (K <  kbins(i+1));
    % Ek(i) = sum(E2d(mask));
    Ek(i) = kbins(i)*mean(E2d(mask));
end

uxchat = fft2(VX_c);
uychat = fft2(VY_c);
E2dc = 0.5 * (abs(uxchat).^2 + abs(uychat).^2);
Ekc = zeros(length(kbins)-1,1);
for i = 1:length(kbins)-1
    mask = (K >= kbins(i)) & ...
        (K <  kbins(i+1));
    % Ekc(i) = sum(E2dc(mask));
    Ekc(i) = kbins(i)*mean(E2dc(mask));
end

uxihat = fft2(VX_i);
uyihat = fft2(VY_i);
E2di = 0.5 * (abs(uxihat).^2 + abs(uyihat).^2);
Eki = zeros(length(kbins)-1,1);
for i = 1:length(kbins)-1
    mask = (K >= kbins(i)) & ...
        (K <  kbins(i+1));
    % Eki(i) = sum(E2di(mask));
    Eki(i) = kbins(i)*mean(E2di(mask));
end

nmax=(nx-1)/2;
% nmax=50;
p_orders=2;
% Measure absolute velocity-increment moments at the requested separations and orders.
[S_Lx, S_Tx, S_Ly, S_Ty] = structure_functions(ux, uy, p_orders, nmax);
[S_Lxi, S_Txi, S_Lyi, S_Tyi] = structure_functions(VX_i, VY_i, p_orders, nmax);
[S_Lxc, S_Txc, S_Lyc, S_Tyc] = structure_functions(VX_c, VY_c, p_orders, nmax);
%% Spectrums and structure functions averaged in time
Ek_tol=0*Ek;Ekc_tol=0*Ekc;Eki_tol=0*Eki;
S_Lx_tol=0*S_Lx; S_Tx_tol=0*S_Tx; S_Ly_tol=0*S_Ly; S_Ty_tol=0*S_Ty;
S_Lxi_tol=0*S_Lxi; S_Txi_tol=0*S_Txi; S_Lyi_tol=0*S_Lyi; S_Tyi_tol=0*S_Tyi;
S_Lxc_tol=0*S_Lxc; S_Txc_tol=0*S_Txc; S_Lyc_tol=0*S_Lyc; S_Tyc_tol=0*S_Tyc;
ux1=0;uy1=0;
Pi_k_tol=0;Pi_ki_tol=0;Pi_kc_tol=0;
Sc_1d=0;Sn_1d=0;spacing_info=[];
spacing_info.d_density = 0;
spacing_info.d_nnd = 0;
spacing_info.k_density = 0;
spacing_info.k_nnd = 0;

Sc_1d_tol=zeros(500,201)';Sn_1d_tol=zeros(500,201)';
tmin=1000;tmax=2000;dt=10;
for t=tmin:dt:tmax
% for t=2000
    t
O1=O(:,:,round(t/10));
VX_temp=repmat(vx(:,:,round(t/10)),[1,1,nz]);
VY_temp=repmat(vy(:,:,round(t/10)),[1,1,nz]);
ux=reshape(VX_temp(:,:,nz),[nx,ny]);
uy=reshape(VY_temp(:,:,nz),[nx,ny]);
ux = ux - mean(ux(:));
uy = uy - mean(uy(:));
%%%%%%%%%%%%%%% Helmholtz decomposition %%%%%%%%%%%%%%%%%%%%%%%%%
% Use the divergence as the source of the Helmholtz potential in Fourier space.
rhs=func_Lx(VX_temp)+func_Ly(VY_temp);
[kx,ky]=ndgrid((-(nx-1)/2:1:(nx-1)/2)*2*pi/L0,(-(ny-1)/2:1:(ny-1)/2)*2*pi/L);
kx((nx+1)/2,(ny+1)/2)=1;ky((nx+1)/2,(ny+1)/2)=1;
rhs_fft=fftshift(fftshift(fft2(rhs),2),1);
rhs_fft((nx+1)/2,(ny+1)/2)=0;
% Remove the zero mode and solve the spatial Poisson equation mode by mode.
Phi_fft=ifftshift(ifftshift(-rhs_fft./(kx.^2+ky.^2),2),1);
Phi_hd=ifft2(Phi_fft);
VX_c=(func_Lx(Phi_hd));VY_c=(func_Ly(Phi_hd));
VX_c=VX_c(:,:,nz);VY_c=VY_c(:,:,nz);
VX_i=ux-VX_c;VY_i=uy-VY_c;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if mod(nx, 2) == 0
    kx = (2*pi/L0) * [0:nx/2-1 , -nx/2:-1];
    ky = (2*pi/L ) * [0:ny/2-1 , -ny/2:-1];
else
    kx = (2*pi/L0) * [0:(nx-1)/2, -(nx-1)/2:-1];
    ky = (2*pi/L ) * [0:(ny-1)/2, -(ny-1)/2:-1];
end
[KX,KY] = meshgrid(kx,ky);
K = sqrt(KX.^2 + KY.^2);
kmax = floor(max(K(:)));
dk = 2*pi/L0;
% Define radial wavenumber shells for the isotropic spectrum.
kbins = 0:dk:kmax;
kplot = 0.5*(kbins(1:end-1) + kbins(2:end));

uxhat = fft2(ux);
uyhat = fft2(uy);
% Form the two-dimensional kinetic-energy spectrum before radial shell averaging.
E2d = 0.5 * (abs(uxhat).^2 + abs(uyhat).^2);
Ek = zeros(length(kbins)-1,1);
for i = 1:length(kbins)-1
    mask = (K >= kbins(i)) & ...
        (K <  kbins(i+1));
    % Ek(i) = sum(E2d(mask));
    Ek(i) = kbins(i)*mean(E2d(mask));
end

uxchat = fft2(VX_c);
uychat = fft2(VY_c);
E2dc = 0.5 * (abs(uxchat).^2 + abs(uychat).^2);
Ekc = zeros(length(kbins)-1,1);
for i = 1:length(kbins)-1
    mask = (K >= kbins(i)) & ...
        (K <  kbins(i+1));
    % Ekc(i) = sum(E2dc(mask));
    Ekc(i) = kbins(i)*mean(E2dc(mask));
end

uxihat = fft2(VX_i);
uyihat = fft2(VY_i);
E2di = 0.5 * (abs(uxihat).^2 + abs(uyihat).^2);
Eki = zeros(length(kbins)-1,1);
for i = 1:length(kbins)-1
    mask = (K >= kbins(i)) & ...
        (K <  kbins(i+1));
    % Eki(i) = sum(E2di(mask));
    Eki(i) = kbins(i)*mean(E2di(mask));
end

nmax=(nx-1)/2;
% nmax=50;
p_orders=2;
% Measure absolute velocity-increment moments at the requested separations and orders.
[S_Lx, S_Tx, S_Ly, S_Ty] = structure_functions(ux, uy, p_orders, nmax);
Ek_tol=Ek_tol+Ek;Ekc_tol=Ekc_tol+Ekc;Eki_tol=Eki_tol+Eki;
S_Lx_tol=S_Lx_tol+S_Lx; S_Tx_tol=S_Tx_tol+S_Tx; S_Ly_tol=S_Ly_tol+S_Ly; S_Ty_tol=S_Ty_tol+S_Ty;

[k_bin, Sc_1d_temp, Sn_1d_temp, vortices, spacing_info_temp] = compute_vortex_structure_factor(O1(:,:,1), L0, L);
Sc_1d=Sc_1d+Sc_1d_temp; Sn_1d=Sn_1d+Sn_1d_temp;
spacing_info.d_density=spacing_info.d_density+spacing_info_temp.d_density;
spacing_info.d_nnd=spacing_info.d_nnd+spacing_info_temp.d_nnd;
spacing_info.k_density=spacing_info.k_density+spacing_info_temp.k_density;
spacing_info.k_nnd=spacing_info.k_nnd+spacing_info_temp.k_nnd;
Sc_1d_tol(round(t/10),:)=Sc_1d_temp;Sn_1d_tol(round(t/10),:)=Sn_1d_temp;
end
Ek=Ek_tol/((tmax-tmin)/dt+1);Ekc=Ekc_tol/((tmax-tmin)/dt+1);Eki=Eki_tol/((tmax-tmin)/dt+1);
S_Lx=S_Lx_tol/((tmax-tmin)/dt+1); S_Tx=S_Tx_tol/((tmax-tmin)/dt+1); S_Ly=S_Ly_tol/((tmax-tmin)/dt+1); S_Ty=S_Ty_tol/((tmax-tmin)/dt+1);
Sc_1d=Sc_1d/((tmax-tmin)/dt+1);Sn_1d=Sn_1d/((tmax-tmin)/dt+1);
save(hqt_data_path('processed', ['average_mu=',num2str(mu),',v=',num2str(-vy_load*20/2/pi),'_20.mat']),'Sc_1d','Sn_1d','vortices','spacing_info','k_bin','Ek','Ekc','Eki','S_Lx','S_Ly','S_Tx','S_Ty','kplot','Sc_1d_tol','Sn_1d_tol', '-v7.3')
%% energy spectrum & structure function
figure('Visible', 'off'),clf,
set(gcf,'Position',[200 300 1100 300]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact')
% superfluid energy spectrum
nexttile
plot(kplot(2:end), Ek(2:end), 'LineWidth',2);
xlabel('$k$','Interpreter','latex','FontSize',14);
ylabel('$E(k)$','Interpreter','latex','FontSize',14);
% title('Superfluid Energy Spectrum');
grid on;
hold on
% C1 = 2.5e5;  C2 = 9e5; C3=2.5e5; C4=1.5e10; % nx=401
plot(kplot, 1*Ek(90)/kplot(90)^(-5/3)*kplot.^(-5/3), 'r--', 'LineWidth',2);
plot(kplot, 0.5*Ek(end)/kplot(end)^(-3)*kplot.^(-3), 'k--', 'LineWidth',2);
plot(kplot, 1*Ek(25)/kplot(25)^(-1)*kplot.^(-1), 'b--', 'LineWidth',2);
plot(kplot, 1.5*Ek(2)/kplot(2)*kplot.^(1), 'g--', 'LineWidth',2);
hold off
xlim([min(kplot(2:end)),max(kplot(2:end))])
ylim([min(Ek(2:end)),5*max(Ek(2:end))])
set(gca,'XScale','log','YScale','log');
nexttile
plot(kplot(2:end), Ekc(2:end), 'LineWidth',2);
xlabel('$k$','Interpreter','latex','FontSize',14);
ylabel('$E_c(k)$','Interpreter','latex','FontSize',14);
% title('Compressible Superfluid Energy Spectrum');
grid on;
hold on
% C1 = 8e7;  C2 = 4e8; C3=1e8; % nx=401
% plot(kplot, Ekc(90)/kplot(90)^(-5/3)*kplot.^(-5/3), 'r--');
plot(kplot, Ekc(90)/kplot(90)^(-2)*kplot.^(-2), 'm--', 'LineWidth',2);
plot(kplot, Ekc(end)/kplot(end)^(-3)*kplot.^(-3), 'k--', 'LineWidth',2);
plot(kplot, Ekc(25)/kplot(25)^(-1)*kplot.^(-1), 'b--', 'LineWidth',2);
plot(kplot, 1.5*Ekc(2)/kplot(2)*kplot.^(1), 'g--', 'LineWidth',2);
hold off
xlim([min(kplot(2:end)),max(kplot(2:end))])
ylim([min(Ekc(2:end)),5*max(Ekc(2:end))])
set(gca,'XScale','log','YScale','log');
nexttile
plot(kplot(2:end), Eki(2:end), 'LineWidth',2);
xlabel('$k$','Interpreter','latex','FontSize',14);
ylabel('$E_i(k)$','Interpreter','latex','FontSize',14);
% title('Incompressible Superfluid Energy Spectrum');
grid on;
hold on
% C1 = 1.8e8;  C2 = 7e8; C3=1.8e8; C4=7e9; % nx=401
plot(kplot, Eki(90)/kplot(90)^(-5/3)*kplot.^(-5/3), 'r--', 'LineWidth',2);
plot(kplot, 0.7*Eki(end)/kplot(end)^(-3)*kplot.^(-3), 'k--', 'LineWidth',2);
plot(kplot, Eki(25)/kplot(25)^(-1)*kplot.^(-1), 'b--', 'LineWidth',2);
plot(kplot, 1.5*Eki(2)/kplot(2)*kplot.^(1), 'g--', 'LineWidth',2);
hold off
xlim([min(kplot(2:end)),max(kplot(2:end))])
ylim([min(Eki(2:end)),5*max(Eki(2:end))])
set(gca,'XScale','log','YScale','log');
exportgraphics(gcf,hqt_output_path('figures', ['spectrum_average_mu=',num2str(mu),',v=',num2str(-vy_load*20/2/pi),'_20.pdf']),ContentType="Image",Resolution=300)

S_L=(S_Lx+S_Ly+S_Tx+S_Ty)/2;
figure('Visible', 'off'),clf
set(gcf,'Position',[400 300 700 300]);
t = tiledlayout(1, 2, 'Padding','compact');
% title(t, 'Compressible component structure function', 'FontSize', 14);
nexttile
grid on, box on
hold on
for p=1
h = plot(xx(1:nmax), S_L(:,p), 'ko');
current_color = h.Color;
plot(xx(1:nmax), 1.7*S_L(3,p)*xx(1:nmax).^(p/3),'r--','LineWidth',2);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_L(2:end,1),[],'all'),2*max(S_L(:,1),[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('$r$','Interpreter','latex','FontSize',14);
ylabel('$S_2$','Interpreter','latex','FontSize',14);
nexttile
grid on, box on
hold on
for p=1
h = plot(xx(1:nmax), S_L(:,p), 'ko');
current_color = h.Color;
plot(xx(1:nmax), 0.078*log(xx(1:nmax)/xx(10))+S_L(10,p),'b--','LineWidth',2);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_L(:,1),[],'all'),max(1.2*S_L(:,1),[],'all')])
set(gca,'XScale','log');
xlabel('$r$','Interpreter','latex','FontSize',14);
ylabel('$S_2$','Interpreter','latex','FontSize',14);
exportgraphics(gcf,hqt_output_path('figures', ['structure_function_mu=',num2str(mu),',v=',num2str(-vy_load*20/2/pi),'_20.pdf']),ContentType="Image",Resolution=300)
%% structure factor
for kk=1:3
if kk==1
    mu=5;vy_load=-2*pi/20*3.8;
    % Load the dataset selected by this section using a repository-relative location.
    load(hqt_data_path('processed', ['average_mu=',num2str(mu),',v=',num2str(-vy_load*20/2/pi),'_20']))
elseif kk==2
    mu=8;vy_load=-2*pi/20*13.5;
    % Load the dataset selected by this section using a repository-relative location.
    load(hqt_data_path('processed', ['average_mu=',num2str(mu),',v=',num2str(-vy_load*20/2/pi),'_20']))
elseif kk==3
    mu=6;vy_load=-2*pi/20*7;
    % Load the dataset selected by this section using a repository-relative location.
    load(hqt_data_path('processed', 'average2.mat'))
end

Sc_1d=Sc_1d/((tmax-tmin)/dt+1);Sn_1d=Sn_1d/((tmax-tmin)/dt+1);
x_v=vortices(:,2);y_v=vortices(:,1);charges=vortices(:,3);
d_density=spacing_info.d_density/((tmax-tmin)/dt+1);
d_nnd=spacing_info.d_nnd/((tmax-tmin)/dt+1);
k_density=spacing_info.k_density/((tmax-tmin)/dt+1);
k_nnd=spacing_info.k_nnd/((tmax-tmin)/dt+1);

% f=@(a,x) a(1).*x+a(2);
% % f=@(a,x) a(2).*x.^a(1);
% a0=[2,0.1];
% opts = statset('nlinfit');opts.TolFun=1e-13;
% % a1 = nlinfit((k_bin(2:7)),(Sc_1d(2:7)),f,a0);
% a1 = nlinfit(log(k_bin(2:7)),log(Sc_1d(2:7)),f,a0);

f=@(a,x) 2.5.*x+a(1);
% f=@(a,x) a(2).*x.^a(1);
a0=[0.1];
opts = statset('nlinfit');opts.TolFun=1e-13;
% a1 = nlinfit((k_bin(2:7)),(Sc_1d(2:7)),f,a0);
% Fit the selected model to the plotted spectrum or structure-factor range.
a1 = nlinfit(log(k_bin(2:7)),log(Sc_1d(2:7)),f,a0);

figure('Visible', 'off'),theme('light'),clf
set(gcf, 'Position', [100 100 800 350])
tiledlayout(1,2,'Padding','compact')%  'TileSpacing','compact',  ,'Padding','compact'
ax1=nexttile;
box on;
scatter(x_v(charges>0), y_v(charges>0), 7, 'r', 'filled'); hold on;
scatter(x_v(charges<0), y_v(charges<0), 7, 'b', 'filled');
axis equal tight; box on;
xlim([0, L0]); ylim([0, L]);
% title(sprintf('Vortex Positions'),'Interpreter','latex','FontSize',12);
% xlabel('x'); ylabel('y');
% legend('+1', '-1', 'Location', 'northeastoutside');
xticks([]);
yticks([]);
xlabel('$x$','Interpreter','latex','FontSize', 14);
ylabel('$y$','Interpreter','latex','FontSize', 14);
text( -0.12, 1, '(c)', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
nexttile
plot(k_bin(2:end), Sc_1d(2:end), 'r-o', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', '$S_c(k)$'); hold on;
plot(k_bin(2:end), Sn_1d(2:end), 'b--o', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', '$S_n(k)$');
plot(k_bin(1:10),exp(a1(1))*k_bin(1:10).^2.5,'k', 'LineWidth', 1.2, 'DisplayName', ['$k^{',num2str(2.5,2),'}$'])
% xline(2*pi/distance, 'k.', 'LineWidth', 1.2, 'DisplayName', '$k_{d}$','Interpreter','latex');
grid on; box on;
set(gca,'XScale','log','YScale','log');
ylim([0.3*min([Sc_1d(2:end),Sn_1d(2:end)]),1.5*max([Sc_1d(2:end),Sn_1d(2:end)])])
% title('1D Structure Factor $S(k)$','Interpreter','latex','FontSize',12);
xlabel('$k$','Interpreter','latex','FontSize',12); ylabel('$S(k)$','Interpreter','latex','FontSize',12);
legend('Location', 'best','FontSize',12,'Interpreter','latex');
% axis(ax1, 'off');
text( -0.12, 1, '(d)', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
exportgraphics(gcf,hqt_output_path('figures', ['structure_factor_mu=',num2str(mu),',v=',num2str(-vy_load*20/2/pi),'_20.pdf']),ContentType="Image",Resolution=300)
end

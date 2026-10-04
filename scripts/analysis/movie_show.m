% MOVIE_SHOW Historical checkpoint analysis and plotting notebook.
% Call setup_project first, then run selected MATLAB sections in dependency order.
% Checkpoint inputs under data/raw/ are intentionally omitted from Git.
% Set grid, domain, time range, and run-folder suffix to match the generated run.
% Figures/movies go to outputs/; processed arrays go to data/processed/.

% Enable compatibility before loading checkpoint fields saved as gpuArray objects.
enable_gpu
mu=6;vy=-2*pi/20*7;
subFolder = hqt_data_path('raw', ['check_point_driven_mu=',num2str(mu),',v=',num2str(-vy*20/2/pi),'_20']);
if ~exist(subFolder, 'dir')
    mkdir(subFolder);
    fprintf('Creating new subfolder');
end
fileName = fullfile(subFolder, ['check_point_t=',num2str(10),'.mat']);load(fileName)
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

fig1=figure(1); theme(gcf,"light")
fig2=figure(2); theme(gcf,"light"), clf, hold on, box on
% fig3=figure(3); theme(gcf,"light"), clf, hold on, box on
set(fig1,'Position',[800 0 550 1000]);

% Initialize the animation state before the active GIF-writing block below.
gifname = hqt_output_path('movies', 'evolution_driven.gif');
delaytime = 0.05;
firstFrame = true;

tmin=0;
tmax=2000;
cmap = turbo((tmax-tmin)/10+1);
NofV=zeros(1,tmax/10);
Nx=(nx-1)/2;
for t=10:10:tmax
    t
    fileName = fullfile(subFolder, ['check_point_t=',num2str(t),'.mat']);load(fileName)

    % Differentiate the bulk scalar; its final radial slice is the boundary order parameter.
    O1=func_Lz(Z1);
    % Construct the density-weighted boundary velocity observable for this analysis.
    VX_temp=abs(O1)./max(abs(O1(:,:,end))).*real(-1i*(func_Lx(O1)-func_Lx(abs(O1)).*O1./abs(O1))./O1); VX_temp=VX_temp(:,:,end);
    VY_temp=abs(O1)./max(abs(O1(:,:,end))).*real(-1i*(func_Ly(O1)-func_Ly(abs(O1)).*O1./abs(O1))./O1); VY_temp=VY_temp(:,:,end);
    [NofV(t/10),~]=countN(reshape(O1(:,:,nz),[nx,ny]));
    set(0,"CurrentFigure",fig1)
    clf
    tiledlayout(2,1,'TileSpacing','compact')%,'Padding','compact'
    nexttile
    s=surf(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),abs(O1((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz)),'FaceColor','interp' );
    s.EdgeColor = 'none';
    xlim([L0/2-L/2,L0/2+L/2]);
    view(0,90);
    %         caxis([0,O])
    colorbar;
    Xlabel= '$x$'; Ylabel= '$y$';
    Title=['$\mathcal{O}$, t=',num2str(round(t))];
    xlabel(Xlabel,'Interpreter','latex')
    ylabel(Ylabel,'Interpreter','latex')
    title(Title,'Interpreter','latex')
    %         figure(2),
    nexttile
    ss=mesh(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),angle(O1((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz)),'FaceColor','interp' );
    ss.EdgeColor = 'none';
    xlim([L0/2-L/2,L0/2+L/2]);
    view(0,90);
    colorbar;
    Xlabel= '$x$'; Ylabel= '$y$';
    Title=['$\theta$, t=',num2str(round(t))];
    xlabel(Xlabel,'Interpreter','latex')
    ylabel(Ylabel,'Interpreter','latex')
    title(Title,'Interpreter','latex')
    drawnow

    % ===== Write a GIF frame. =====
    frame = getframe(fig1);
    im = frame2im(frame);
    [A,map] = rgb2ind(im,256,'nodither');

    if firstFrame
        imwrite(A,map,gifname,'gif','LoopCount',inf,'DelayTime',delaytime);
        firstFrame = false;
    else
        imwrite(A,map,gifname,'gif','WriteMode','append','DelayTime',delaytime);
    end

    set(0,'CurrentFigure',fig2)
    % tiledlayout(1,2,'TileSpacing','compact')%,'Padding','compact'
    % nexttile
    ux=reshape(VX_temp(:,:,end),[nx,ny]);
    uy=reshape(VY_temp(:,:,end),[nx,ny]);
    ux = ux - mean(ux(:));
    uy = uy - mean(uy(:));
    uxhat = fft2(ux);
    uyhat = fft2(uy);
    % Form the two-dimensional kinetic-energy spectrum before radial shell averaging.
    E2d = 0.5 * (abs(uxhat).^2 + abs(uyhat).^2);
    kx = (2*pi/L0) * [0:nx/2-1 , -nx/2:-1];
    ky = (2*pi/L ) * [0:ny/2-1 , -ny/2:-1];
    [KX,KY] = meshgrid(kx,ky);
    K = sqrt(KX.^2 + KY.^2);
    kmax = floor(max(K(:)));
    dk = 2*pi/L0;
    % Define radial wavenumber shells for the isotropic spectrum.
    kbins = 0:dk:kmax;
    Ek = zeros(length(kbins)-1,1);
    for i = 1:length(kbins)-1
        mask = (K >= kbins(i)) & ...
            (K <  kbins(i+1));
        Ek(i) = sum(E2d(mask));
    end
    kplot = 0.5*(kbins(1:end-1) + kbins(2:end));
    plot(kplot(2:end), Ek(2:end), 'LineWidth',2,'Color', cmap((t-tmin)/10+1,:));
    colormap(cmap);
    clim([tmin,tmax]);
    cb = colorbar;
    cb.Label.String = 'time';
    xlabel('k');
    ylabel('E(k)');
    title('Superfluid Energy Spectrum');
    set(gca,'XScale','log','YScale','log');
    drawnow
end
figure(4),theme(gcf,"light"),box on
plot(10:10:tmax,NofV)
xlim([0,tmax])
ylim([0,max(NofV)+10])
xlabel('t','FontSize',14);
ylabel('Vortices number','FontSize',14);
%% movie show
t=3000;
fileName = fullfile(subFolder, ['check_point_t=',num2str(t),'.mat']);load(fileName)
% Differentiate the bulk scalar; its final radial slice is the boundary order parameter.
O1=func_Lz(Z1);
set(0,"CurrentFigure",fig1)
clf
tiledlayout(2,1,'TileSpacing','compact')%,'Padding','compact'
nexttile
s=surf(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),abs(O1((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz)),'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([L0/2-L/2,L0/2+L/2]);
view(0,90);
%         caxis([0,O])
colorbar;
Xlabel= '$x$'; Ylabel= '$y$';
Title=['$\mathcal{O}$, t=',num2str(round(t))];
xlabel(Xlabel,'Interpreter','latex')
ylabel(Ylabel,'Interpreter','latex')
title(Title,'Interpreter','latex')
%         figure(2),
nexttile
ss=mesh(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),angle(O1((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz)),'FaceColor','interp' );
ss.EdgeColor = 'none';
xlim([L0/2-L/2,L0/2+L/2]);
view(0,90);
colorbar;
Xlabel= '$x$'; Ylabel= '$y$';
Title=['$\theta$, t=',num2str(round(t))];
xlabel(Xlabel,'Interpreter','latex')
ylabel(Ylabel,'Interpreter','latex')
title(Title,'Interpreter','latex')

% VX_temp=real(func_Lz(Z3)); VY_temp=real(func_Lz(Z4));
% VX_temp=real(-1i*(func_Lx(O1)-func_Lx(abs(O1)).*O1./abs(O1))./O1);
% VY_temp=real(-1i*(func_Ly(O1)-func_Ly(abs(O1)).*O1./abs(O1))./O1);
% Construct the density-weighted boundary velocity observable for this analysis.
VX_temp=abs(O1)./max(abs(O1(:,:,end))).*real(-1i*(func_Lx(O1)-func_Lx(abs(O1)).*O1./abs(O1))./O1);
VY_temp=abs(O1)./max(abs(O1(:,:,end))).*real(-1i*(func_Ly(O1)-func_Ly(abs(O1)).*O1./abs(O1))./O1);
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
kx = (2*pi/L0) * [0:nx/2-1 , -nx/2:-1];
ky = (2*pi/L ) * [0:ny/2-1 , -ny/2:-1];
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
    Ek(i) = sum(E2d(mask));
end

uxchat = fft2(VX_c);
uychat = fft2(VY_c);
E2dc = 0.5 * (abs(uxchat).^2 + abs(uychat).^2);
Ekc = zeros(length(kbins)-1,1);
for i = 1:length(kbins)-1
    mask = (K >= kbins(i)) & ...
        (K <  kbins(i+1));
    Ekc(i) = sum(E2dc(mask));
end

uxihat = fft2(VX_i);
uyihat = fft2(VY_i);
E2di = 0.5 * (abs(uxihat).^2 + abs(uyihat).^2);
Eki = zeros(length(kbins)-1,1);
for i = 1:length(kbins)-1
    mask = (K >= kbins(i)) & ...
        (K <  kbins(i+1));
    Eki(i) = sum(E2di(mask));
end

figure(3),clf,theme(gcf,"light")
set(gcf,'Position',[400 300 900 600]);
tiledlayout(2,2,'TileSpacing','compact')
% superfluid energy spectrum
nexttile
plot(kplot(2:end), Ek(2:end), 'LineWidth',2);
xlabel('k');
ylabel('E(k)');
title('Superfluid Energy Spectrum');
grid on;
hold on
C1 = 1.5e7;  C2 = 7e7; C3=2e7; % nx=401
plot(kplot, C1*kplot.^(-5/3), 'r--');
plot(kplot, C2*kplot.^(-3), 'k--');
plot(kplot, C3*kplot.^(-1), 'b--');
hold off
xlim([min(kplot(2:end)),max(kplot(2:end))])
ylim([min(Ek(2:end)),5*max(Ek(2:end))])
set(gca,'XScale','log','YScale','log');
nexttile
plot(kplot(2:end), Ekc(2:end), 'LineWidth',2);
xlabel('k');
ylabel('E_c(k)');
title('Compressible Superfluid Energy Spectrum');
grid on;
hold on
C1 = 1.5e7;  C2 = 7e7; C3=2e7; % nx=401
plot(kplot, C1*kplot.^(-5/3), 'r--');
plot(kplot, C2*kplot.^(-3), 'k--');
plot(kplot, C3*kplot.^(-1), 'b--');
hold off
xlim([min(kplot(2:end)),max(kplot(2:end))])
ylim([min(Ekc(2:end)),5*max(Ekc(2:end))])
set(gca,'XScale','log','YScale','log');
nexttile
plot(kplot(2:end), Eki(2:end), 'LineWidth',2);
xlabel('k');
ylabel('E_i(k)');
title('Incompressible Superfluid Energy Spectrum');
grid on;
hold on
C1 = 1.5e7;  C2 = 7e7; C3=2e7; % nx=401
plot(kplot, C1*kplot.^(-5/3), 'r--');
plot(kplot, C2*kplot.^(-3), 'k--');
plot(kplot, C3*kplot.^(-1), 'b--');
hold off
xlim([min(kplot(2:end)),max(kplot(2:end))])
ylim([min(Eki(2:end)),5*max(Eki(2:end))])
set(gca,'XScale','log','YScale','log');
% superfluid velocity distribution
nexttile
ux1 = ux(:);
uy1 = uy(:);
Nbins = 100;
[countsx, edgesx] = histcounts(ux1, ...
    Nbins, ...
    'Normalization','pdf');
[countsy, edgesy] = histcounts(uy1, ...
    Nbins, ...
    'Normalization','pdf');
uxc = 0.5*(edgesx(1:end-1) + edgesx(2:end));
uyc = 0.5*(edgesy(1:end-1) + edgesy(2:end));
mux = mean(ux1);
muy = mean(uy1);
sigmax = std(ux1);
sigmay = std(uy1);
gaussx = 1/(sqrt(2*pi)*sigmax) ...
    .* exp(-(uxc-mux).^2/(2*sigmax^2));
gaussy = 1/(sqrt(2*pi)*sigmay) ...
    .* exp(-(uyc-muy).^2/(2*sigmay^2));

semilogy(uxc, countsx, 'r',...
    'LineWidth',2);
hold on;
semilogy(uyc, countsy, 'b',...
    'LineWidth',2);
semilogy(uxc, gaussx, 'r--', ...
    'LineWidth',2);
semilogy(uyc, gaussy, 'b--', ...
    'LineWidth',2);
xlabel('u');
ylabel('PDF');
title('Superfluid velocity distribution');
legend( ...
    'u_x numerical', ...
    'u_y numerical', ...
    'u_x Gaussian', ...
    'u_y Gaussian','Location','best');
grid on;

figure(7),clf,theme(gcf,"light")
set(gcf,'Position',[800 0 800 1200]);
tiledlayout(3,2,'TileSpacing','compact')
nexttile
s=surf(xx(:,:,nz),yy(:,:,nz),real(ux(:,:)),'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([L0/2-L/2,L0/2+L/2]);
view(0,90);
colorbar;
Xlabel= '$x$'; Ylabel= '$y$';
Title=['$v_x$, t=',num2str(round(t))];
xlabel(Xlabel,'Interpreter','latex')
ylabel(Ylabel,'Interpreter','latex')
title(Title,'Interpreter','latex')
nexttile
s=surf(xx(:,:,nz),yy(:,:,nz),real(uy(:,:)),'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([L0/2-L/2,L0/2+L/2]);
view(0,90);
colorbar;
Xlabel= '$x$'; Ylabel= '$y$';
Title=['$v_y$, t=',num2str(round(t))];
xlabel(Xlabel,'Interpreter','latex')
ylabel(Ylabel,'Interpreter','latex')
title(Title,'Interpreter','latex')
nexttile
s=surf(xx(:,:,nz),yy(:,:,nz),real(VX_i(:,:)),'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([L0/2-L/2,L0/2+L/2]);
view(0,90);
colorbar;
Xlabel= '$x$'; Ylabel= '$y$';
Title=['$(v_x)_i$, t=',num2str(round(t))];
xlabel(Xlabel,'Interpreter','latex')
ylabel(Ylabel,'Interpreter','latex')
title(Title,'Interpreter','latex')
nexttile
s=surf(xx(:,:,nz),yy(:,:,nz),real(VY_i(:,:)),'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([L0/2-L/2,L0/2+L/2]);
view(0,90);
colorbar;
Xlabel= '$x$'; Ylabel= '$y$';
Title=['$(v_y)_i$, t=',num2str(round(t))];
xlabel(Xlabel,'Interpreter','latex')
ylabel(Ylabel,'Interpreter','latex')
title(Title,'Interpreter','latex')
nexttile
s=surf(xx(:,:,nz),yy(:,:,nz),real(VX_c(:,:)),'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([L0/2-L/2,L0/2+L/2]);
view(0,90);
colorbar;
Xlabel= '$x$'; Ylabel= '$y$';
Title=['$(v_x)_c$, t=',num2str(round(t))];
xlabel(Xlabel,'Interpreter','latex')
ylabel(Ylabel,'Interpreter','latex')
title(Title,'Interpreter','latex')
nexttile
s=surf(xx(:,:,nz),yy(:,:,nz),real(VY_c(:,:)),'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([L0/2-L/2,L0/2+L/2]);
view(0,90);
colorbar;
Xlabel= '$x$'; Ylabel= '$y$';
Title=['$(v_y)_c$, t=',num2str(round(t))];
xlabel(Xlabel,'Interpreter','latex')
ylabel(Ylabel,'Interpreter','latex')
title(Title,'Interpreter','latex')
%% structure functions
nmax=(nx-1)/2;
% nmax=50;
p_orders=2:5;
% Measure absolute velocity-increment moments at the requested separations and orders.
[S_Lx, S_Tx, S_Ly, S_Ty] = structure_functions(ux, uy, p_orders, nmax);
[S_Lxi, S_Txi, S_Lyi, S_Tyi] = structure_functions(VX_i, VY_i, p_orders, nmax);
[S_Lxc, S_Txc, S_Lyc, S_Tyc] = structure_functions(VX_c, VY_c, p_orders, nmax);

figure(4),clf,theme(gcf,"light")
set(gcf,'Position',[400 300 900 600]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact');
title(t, 'Total structure function', 'FontSize', 14);
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Lx(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Lx(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Lx,[],'all'),max(S_Lx,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Tx(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Tx(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Tx,[],'all'),max(S_Tx,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Ly(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Ly(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Ly,[],'all'),max(S_Ly,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_y$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Ty(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Ty(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Ty,[],'all'),max(S_Ty,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_y$','Interpreter','latex');

figure(5),clf,theme(gcf,"light")
set(gcf,'Position',[400 300 900 600]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact');
title(t, 'Incompressible component structure function', 'FontSize', 14);
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Lxi(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Lxi(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Lxi,[],'all'),max(S_Lxi,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Txi(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Txi(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Txi,[],'all'),max(S_Txi,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Lyi(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Lyi(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Lyi,[],'all'),max(S_Lyi,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_y$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Tyi(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Tyi(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Tyi,[],'all'),max(S_Tyi,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_y$','Interpreter','latex');

figure(6),clf,theme(gcf,"light")
set(gcf,'Position',[400 300 900 600]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact');
title(t, 'Compressible component structure function', 'FontSize', 14);
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Lxc(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Lxc(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Lxc,[],'all'),max(S_Lxc,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Txc(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Txc(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Txc,[],'all'),max(S_Txc,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Lyc(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Lyc(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Lyc,[],'all'),max(S_Lyc,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_y$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Tyc(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Tyc(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Tyc,[],'all'),max(S_Tyc,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_y$','Interpreter','latex');
%% generalized structure functions
nmax=(nx-1)/2;
% nmax=50;
p_orders=2:5;
% Measure absolute velocity-increment moments at the requested separations and orders.
[S_Lx, S_Tx, S_Ly, S_Ty] = structure_functions(ux, uy, p_orders, nmax);
[S_Lxi, S_Txi, S_Lyi, S_Tyi] = structure_functions(VX_i, VY_i, p_orders, nmax);
[S_Lxc, S_Txc, S_Lyc, S_Tyc] = structure_functions(VX_c, VY_c, p_orders, nmax);

figure(4),clf,theme(gcf,"light")
set(gcf,'Position',[400 300 900 600]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact');
title(t, 'Total structure function', 'FontSize', 14);
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
plot(S_Lx(:,3), S_Lx(:,p),'.');
plot(S_Lx(:,3), S_Lx(3,p)*S_Lx(:,3).^(p/3),'--');
end
hold off
xlim([0,max(S_Lx(:,3))])
ylim([min(S_Lx,[],'all'),max(S_Lx,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
plot(S_Tx(:,3), S_Tx(:,p),'.');
plot(S_Tx(:,3), S_Tx(3,p)*S_Tx(:,3).^(p/3),'--');
end
hold off
xlim([0,max(S_Tx(:,3))])
ylim([min(S_Tx,[],'all'),max(S_Tx,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
plot(S_Ly(:,3), S_Ly(:,p),'.');
plot(S_Ly(:,3), S_Ly(3,p)*S_Ly(:,3).^(p/3),'--');
end
hold off
xlim([0,max(S_Ly(:,3))])
ylim([min(S_Ly,[],'all'),max(S_Ly,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_y$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
plot(S_Ty(:,3), S_Ty(:,p),'.');
plot(S_Ty(:,3), S_Ty(3,p)*S_Ty(:,3).^(p/3),'--');
end
hold off
xlim([0,max(S_Ty(:,3))])
ylim([min(S_Ty,[],'all'),max(S_Ty,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_y$','Interpreter','latex');

figure(5),clf,theme(gcf,"light")
set(gcf,'Position',[400 300 900 600]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact');
title(t, 'Incompressible component structure function', 'FontSize', 14);
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
plot(S_Lxi(:,3), S_Lxi(:,p),'.');
plot(S_Lxi(:,3), S_Lxi(3,p)*S_Lxi(:,3).^(p/3),'--');
end
hold off
xlim([0,max(S_Lxi(:,3))])
ylim([min(S_Lxi,[],'all'),max(S_Lxi,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
plot(S_Txi(:,3), S_Txi(:,p),'.');
plot(S_Txi(:,3), S_Txi(3,p)*S_Txi(:,3).^(p/3),'--');
end
hold off
xlim([0,max(S_Txi(:,3))])
ylim([min(S_Txi,[],'all'),max(S_Txi,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
plot(S_Lyi(:,3), S_Lyi(:,p),'.');
plot(S_Lyi(:,3), S_Lyi(3,p)*S_Lyi(:,3).^(p/3),'--');
end
hold off
xlim([0,max(S_Lyi(:,3))])
ylim([min(S_Lyi,[],'all'),max(S_Lyi,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_y$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
plot(S_Tyi(:,3), S_Tyi(:,p),'.');
plot(S_Tyi(:,3), S_Tyi(3,p)*S_Tyi(:,3).^(p/3),'--');
end
hold off
xlim([0,max(S_Tyi(:,3))])
ylim([min(S_Tyi,[],'all'),max(S_Tyi,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_y$','Interpreter','latex');

figure(6),clf,theme(gcf,"light")
set(gcf,'Position',[400 300 900 600]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact');
title(t, 'Compressible component structure function', 'FontSize', 14);
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
plot(S_Lxc(:,3), S_Lxc(:,p),'.');
plot(S_Lxc(:,3), S_Lxc(3,p)*S_Lxc(:,3).^(p/3),'--');
end
hold off
xlim([0,max(S_Lxc(:,3))])
ylim([min(S_Lxc,[],'all'),max(S_Lxc,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
plot(S_Txc(:,3), S_Txc(:,p),'.');
plot(S_Txc(:,3), S_Txc(3,p)*S_Txc(:,3).^(p/3),'--');
end
hold off
xlim([0,max(S_Txc(:,3))])
ylim([min(S_Txc,[],'all'),max(S_Txc,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
plot(S_Lyc(:,3), S_Lyc(:,p),'.');
plot(S_Lyc(:,3), S_Lyc(3,p)*S_Lyc(:,3).^(p/3),'--');
end
hold off
xlim([0,max(S_Lyc(:,3))])
ylim([min(S_Lyc,[],'all'),max(S_Lyc,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_y$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
plot(S_Tyc(:,3), S_Tyc(:,p),'.');
plot(S_Tyc(:,3), S_Tyc(3,p)*S_Tyc(:,3).^(p/3),'--');
end
hold off
xlim([0,max(S_Tyc(:,3))])
ylim([min(S_Tyc,[],'all'),max(S_Tyc,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_y$','Interpreter','latex');
%% Spectrums and structure functions averaged in time
Ek_tol=0*Ek;Ekc_tol=0*Ekc;Eki_tol=0*Eki;
S_Lx_tol=0*S_Lx; S_Tx_tol=0*S_Tx; S_Ly_tol=0*S_Ly; S_Ty_tol=0*S_Ty;
S_Lxi_tol=0*S_Lxi; S_Txi_tol=0*S_Txi; S_Lyi_tol=0*S_Lyi; S_Tyi_tol=0*S_Tyi;
S_Lxc_tol=0*S_Lxc; S_Txc_tol=0*S_Txc; S_Lyc_tol=0*S_Lyc; S_Tyc_tol=0*S_Tyc;
ux1=0;uy1=0;
Pi_k_tol=0;Pi_ki_tol=0;Pi_kc_tol=0;

tmin=2000;tmax=2000;dt=100;
for t=tmin:dt:tmax
fileName = fullfile(subFolder, ['check_point_t=',num2str(t),'.mat']);load(fileName)
% Differentiate the bulk scalar; its final radial slice is the boundary order parameter.
O1=func_Lz(Z1);
VX_temp=real(func_Lz(Z3)); VY_temp=real(func_Lz(Z4));
% VX_temp=abs(O1)./max(abs(O1(:,:,end))).*real(-1i*(func_Lx(O1)-func_Lx(abs(O1)).*O1./abs(O1))./O1);
% VY_temp=abs(O1)./max(abs(O1(:,:,end))).*real(-1i*(func_Ly(O1)-func_Ly(abs(O1)).*O1./abs(O1))./O1);
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
kx = (2*pi/L0) * [0:nx/2-1 , -nx/2:-1];
ky = (2*pi/L ) * [0:ny/2-1 , -ny/2:-1];
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
    Ek(i) = sum(E2d(mask));
end

uxchat = fft2(VX_c);
uychat = fft2(VY_c);
E2dc = 0.5 * (abs(uxchat).^2 + abs(uychat).^2);
Ekc = zeros(length(kbins)-1,1);
for i = 1:length(kbins)-1
    mask = (K >= kbins(i)) & ...
        (K <  kbins(i+1));
    Ekc(i) = sum(E2dc(mask));
end

uxihat = fft2(VX_i);
uyihat = fft2(VY_i);
E2di = 0.5 * (abs(uxihat).^2 + abs(uyihat).^2);
Eki = zeros(length(kbins)-1,1);
for i = 1:length(kbins)-1
    mask = (K >= kbins(i)) & ...
        (K <  kbins(i+1));
    Eki(i) = sum(E2di(mask));
end

ux1 =squeeze([ux1(:);ux(:)]);uy1 = [uy1(:);uy(:)];

nmax=(nx-1)/2;
% nmax=50;
p_orders=2:5;
% Measure absolute velocity-increment moments at the requested separations and orders.
[S_Lx, S_Tx, S_Ly, S_Ty] = structure_functions(ux, uy, p_orders, nmax);
[S_Lxi, S_Txi, S_Lyi, S_Tyi] = structure_functions(VX_i, VY_i, p_orders, nmax);
[S_Lxc, S_Txc, S_Lyc, S_Tyc] = structure_functions(VX_c, VY_c, p_orders, nmax);

[k_1d, Pi_k, T_k] = calc_energy_flux(ux, uy, L0, L);
[~, Pi_ki, T_ki] = calc_energy_flux(VX_i, VY_i, L0, L);
[~, Pi_kc, T_kc] = calc_energy_flux(VX_c, VY_c, L0, L);

Ek_tol=Ek_tol+Ek;Ekc_tol=Ekc_tol+Ekc;Eki_tol=Eki_tol+Eki;
S_Lx_tol=S_Lx_tol+S_Lx; S_Tx_tol=S_Tx_tol+S_Tx; S_Ly_tol=S_Ly_tol+S_Ly; S_Ty_tol=S_Ty_tol+S_Ty;
S_Lxi_tol=S_Lxi_tol+S_Lxi; S_Txi_tol=S_Txi_tol+S_Txi; S_Lyi_tol=S_Lyi_tol+S_Lyi; S_Tyi_tol=S_Tyi_tol+S_Tyi;
S_Lxc_tol=S_Lxc_tol+S_Lxc; S_Txc_tol=S_Txc_tol+S_Txc; S_Lyc_tol=S_Lyc_tol+S_Lyc; S_Tyc_tol=S_Tyc_tol+S_Tyc;
Pi_k_tol=Pi_k_tol+Pi_k;Pi_ki_tol=Pi_ki_tol+Pi_ki;Pi_kc_tol=Pi_kc_tol+Pi_kc;
end
Ek=Ek_tol/((tmax-tmin)/dt+1);Ekc=Ekc_tol/((tmax-tmin)/dt+1);Eki=Eki_tol/((tmax-tmin)/dt+1);
S_Lx=S_Lx_tol/((tmax-tmin)/dt+1); S_Tx=S_Tx_tol/((tmax-tmin)/dt+1); S_Ly=S_Ly_tol/((tmax-tmin)/dt+1); S_Ty=S_Ty_tol/((tmax-tmin)/dt+1);
S_Lxi=S_Lxi_tol/((tmax-tmin)/dt+1); S_Txi=S_Txi_tol/((tmax-tmin)/dt+1); S_Lyi=S_Lyi_tol/((tmax-tmin)/dt+1); S_Tyi=S_Tyi_tol/((tmax-tmin)/dt+1);
S_Lxc=S_Lxc_tol/((tmax-tmin)/dt+1); S_Txc=S_Txc_tol/((tmax-tmin)/dt+1); S_Lyc=S_Lyc_tol/((tmax-tmin)/dt+1); S_Tyc=S_Tyc_tol/((tmax-tmin)/dt+1);
Pi_k=Pi_k_tol/((tmax-tmin)/dt+1);Pi_ki=Pi_ki_tol/((tmax-tmin)/dt+1);Pi_kc=Pi_kc_tol/((tmax-tmin)/dt+1);

Nbins = 100;
[countsx, edgesx] = histcounts(ux1, ...
    Nbins, ...
    'Normalization','pdf');
[countsy, edgesy] = histcounts(uy1, ...
    Nbins, ...
    'Normalization','pdf');
uxc = 0.5*(edgesx(1:end-1) + edgesx(2:end));
uyc = 0.5*(edgesy(1:end-1) + edgesy(2:end));
mux = mean(ux1);
muy = mean(uy1);
sigmax = std(ux1);
sigmay = std(uy1);
gaussx = 1/(sqrt(2*pi)*sigmax) ...
    .* exp(-(uxc-mux).^2/(2*sigmax^2));
gaussy = 1/(sqrt(2*pi)*sigmay) ...
    .* exp(-(uyc-muy).^2/(2*sigmay^2));

figure(3),clf,theme(gcf,"light")
set(gcf,'Position',[400 300 900 600]);
tiledlayout(2,2,'TileSpacing','compact')
% superfluid energy spectrum
nexttile
plot(kplot(2:end), Ek(2:end), 'LineWidth',2);
xlabel('k');
ylabel('E(k)');
title('Superfluid Energy Spectrum');
grid on;
hold on
C1 = 1.5e7;  C2 = 7e7; C3=2e7; % nx=401
plot(kplot, C1*kplot.^(-5/3), 'r--');
plot(kplot, C2*kplot.^(-3), 'k--');
plot(kplot, C3*kplot.^(-1), 'b--');
hold off
xlim([min(kplot(2:end)),max(kplot(2:end))])
ylim([min(Ek(2:end)),5*max(Ek(2:end))])
set(gca,'XScale','log','YScale','log');
nexttile
plot(kplot(2:end), Ekc(2:end), 'LineWidth',2);
xlabel('k');
ylabel('E_c(k)');
title('Compressible Superfluid Energy Spectrum');
grid on;
hold on
C1 = 1.5e7;  C2 = 7e7; C3=2e7; % nx=401
plot(kplot, C1*kplot.^(-5/3), 'r--');
plot(kplot, C2*kplot.^(-3), 'k--');
plot(kplot, C3*kplot.^(-1), 'b--');
hold off
xlim([min(kplot(2:end)),max(kplot(2:end))])
ylim([min(Ekc(2:end)),5*max(Ekc(2:end))])
set(gca,'XScale','log','YScale','log');
nexttile
plot(kplot(2:end), Eki(2:end), 'LineWidth',2);
xlabel('k');
ylabel('E_i(k)');
title('Incompressible Superfluid Energy Spectrum');
grid on;
hold on
C1 = 1.5e7;  C2 = 7e7; C3=2e7; % nx=401
plot(kplot, C1*kplot.^(-5/3), 'r--');
plot(kplot, C2*kplot.^(-3), 'k--');
plot(kplot, C3*kplot.^(-1), 'b--');
hold off
xlim([min(kplot(2:end)),max(kplot(2:end))])
ylim([min(Eki(2:end)),5*max(Eki(2:end))])
set(gca,'XScale','log','YScale','log');
% superfluid velocity distribution
nexttile
semilogy(uxc, countsx, 'r',...
    'LineWidth',2);
hold on;
semilogy(uyc, countsy, 'b',...
    'LineWidth',2);
semilogy(uxc, gaussx, 'r--', ...
    'LineWidth',2);
semilogy(uyc, gaussy, 'b--', ...
    'LineWidth',2);
xlabel('u');
ylabel('PDF');
title('Superfluid velocity distribution');
legend( ...
    'u_x numerical', ...
    'u_y numerical', ...
    'u_x Gaussian', ...
    'u_y Gaussian','Location','best');
grid on;

figure(4),clf,theme(gcf,"light")
set(gcf,'Position',[400 300 900 600]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact');
title(t, 'Total structure function', 'FontSize', 14);
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Lx(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Lx(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Lx,[],'all'),max(S_Lx,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Tx(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Tx(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Tx,[],'all'),max(S_Tx,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Ly(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Ly(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Ly,[],'all'),max(S_Ly,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_y$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Ty(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Ty(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Ty,[],'all'),max(S_Ty,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_y$','Interpreter','latex');

figure(5),clf,theme(gcf,"light")
set(gcf,'Position',[400 300 900 600]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact');
title(t, 'Incompressible component structure function', 'FontSize', 14);
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Lxi(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Lxi(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Lxi,[],'all'),max(S_Lxi,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Txi(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Txi(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Txi,[],'all'),max(S_Txi,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Lyi(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Lyi(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Lyi,[],'all'),max(S_Lyi,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_y$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Tyi(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Tyi(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Tyi,[],'all'),max(S_Tyi,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_y$','Interpreter','latex');

figure(6),clf,theme(gcf,"light")
set(gcf,'Position',[400 300 900 600]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact');
title(t, 'Compressible component structure function', 'FontSize', 14);
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Lxc(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Lxc(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Lxc,[],'all'),max(S_Lxc,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Txc(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Txc(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Txc,[],'all'),max(S_Txc,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_x$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Lyc(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Lyc(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Lyc,[],'all'),max(S_Lyc,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^L_p)_y$','Interpreter','latex');
nexttile
grid on, box on
hold on
for p=1:length(p_orders)
h = plot(xx(1:nmax), S_Tyc(:,p), '.');
current_color = h.Color;
plot(xx(1:nmax), S_Tyc(3,p)*xx(1:nmax).^(p/3),'--', 'Color', current_color);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_Tyc,[],'all'),max(S_Tyc,[],'all')])
set(gca,'XScale','log','YScale','log');
xlabel('x','Interpreter','latex');
ylabel('$(S^T_p)_y$','Interpreter','latex');

figure('Color', 'w', 'Position', [100, 100, 1500, 450]);
tiledlayout(1,3,'TileSpacing','compact')
nexttile
plot(k_1d, Pi_k, 'k-', 'LineWidth', 2);
hold on;
yline(0, 'r--', 'LineWidth', 1.5);
xlabel('Wavenumber k', 'FontSize', 14);
ylabel('Energy Flux \Pi(k)', 'FontSize', 14);
% title('Energy Flux in Phenomenological Wave-space', 'FontSize', 16);
set(gca, 'XScale', 'log', 'FontSize', 12);
grid on;
box on;
nexttile
plot(k_1d, Pi_ki, 'k-', 'LineWidth', 2);
hold on;
yline(0, 'r--', 'LineWidth', 1.5);
xlabel('Wavenumber k', 'FontSize', 14);
ylabel('Energy Flux \Pi_i(k)', 'FontSize', 14);
% title('Energy Flux in Phenomenological Wave-space', 'FontSize', 16);
set(gca, 'XScale', 'log', 'FontSize', 12);
grid on;
box on;
nexttile
plot(k_1d, Pi_kc, 'k-', 'LineWidth', 2);
hold on;
yline(0, 'r--', 'LineWidth', 1.5);
xlabel('Wavenumber k', 'FontSize', 14);
ylabel('Energy Flux \Pi_c(k)', 'FontSize', 14);
% title('Energy Flux in Phenomenological Wave-space', 'FontSize', 16);
set(gca, 'XScale', 'log', 'FontSize', 12);
grid on;
box on;

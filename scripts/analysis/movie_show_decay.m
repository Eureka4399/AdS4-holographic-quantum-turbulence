% MOVIE_SHOW_DECAY Historical checkpoint analysis and plotting notebook.
% Call setup_project first, then run selected MATLAB sections in dependency order.
% Checkpoint inputs under data/raw/ are intentionally omitted from Git.
% Set grid, domain, time range, and run-folder suffix to match the generated run.
% Figures/movies go to outputs/; processed arrays go to data/processed/.

% Enable compatibility before reading GPU-backed checkpoint or averaged arrays.
enable_gpu
mu=6;vy=-2*pi/20*7;
subFolder = hqt_data_path('raw', ['check_point_driven_mu=',num2str(mu),',v=',num2str(-vy*20/2/pi),'_20_large']);
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
%Define the scalar and gauge-field evolution operators.
zh=1;e=1;
f=1-(zz/zh).^3;df=-3*zz.^2/zh^3;
F1=@(y1,y2,y3,y4) func_Lzr(1i.*y2.*(func_Lz(y1))+0.5i.*y1.*func_Lz(y2)+0.5.*(1-(zz/zh).^3).*func_Lz(func_Lz(y1))-0.5.*3.*zz.^2./zh^3.*func_Lz(y1)-0.5.*zz.*y1-0.5.*y3.*y3.*y1-0.5.*y4.*y4.*y1+1.*0.5.*func_Lx(func_Lx(y1))+ ...
    1.*0.5.*func_Ly(func_Ly((y1)))-1.0.*0.5i.*y1.*(func_Lx(y3)+func_Ly(y4))-1i.*(y3.*func_Lx(y1)+y4.*func_Ly(y1)));
F3=@(y1,y2,y3,y4) real(func_Lzr(0.5.*func_Lz(func_Lx(y2)+(1-(zz/zh).^3).*func_Lz(y3))+0.5.*func_Ly(func_Ly(y3)-func_Lx(y4))-0.5i.*(conj(y1).*func_Lx(y1)-y1.*func_Lx(conj(y1)))-y3.*y1.*conj(y1)));
F4=@(y1,y2,y3,y4) real(func_Lzr(0.5.*func_Lz(func_Ly(y2)+(1-(zz/zh).^3).*func_Lz(y4))+0.5.*func_Lx(-func_Ly(y3)+func_Lx(y4))-0.5i.*(conj(y1).*func_Ly(y1)-y1.*func_Ly(conj(y1)))-y4.*y1.*conj(y1)));
F2=@(y1,y2,y3,y4) (1-zz.^3).*func_Lz(func_Lx(y3)+func_Ly(y4));
Constraint=@(y1,y2,y3,y4) func_Lz(func_Lx(y3)+func_Ly(y4)-func_Lz(y2))-2*imag((conj(y1).*func_Lz(y1)));


fig1=figure(1); theme(gcf,"light")
fig2=figure(2); theme(gcf,"light"), clf, hold on, box on
% fig3=figure(3); theme(gcf,"light"), clf, hold on, box on
set(fig1,'Position',[100 200 1500 650]);

% gifname = hqt_output_path('movies', 'evolution_decay_large.gif');   % GIF output path.
% delaytime = 0.05;            % Frame delay in seconds.
% firstFrame = true;

tmin=1500;
tmax=3000;
cmap = turbo((tmax-tmin)/10+1);
NofV=zeros(1,tmax/10);
Nx=(nx-1)/2;
for t=tmin+10:10:tmax
    t
    fileName = fullfile(subFolder, ['check_point_t=',num2str(t),'.mat']);load(fileName)

    % Differentiate the bulk scalar; its final radial slice is the boundary order parameter.
    O1=func_Lz(Z1);
    % VX_temp=abs(O1)./max(abs(O1(:,:,end))).*real(-1i*(func_Lx(O1)-func_Lx(abs(O1)).*O1./abs(O1))./O1); VX_temp=VX_temp(:,:,end);
    % VY_temp=abs(O1)./max(abs(O1(:,:,end))).*real(-1i*(func_Ly(O1)-func_Ly(abs(O1)).*O1./abs(O1))./O1); VY_temp=VY_temp(:,:,end);
    [NofV(t/10),~]=countN(reshape(O1(:,:,nz),[nx,ny]));
    % set(0,"CurrentFigure",fig1)
    % clf
    % tiledlayout(1,2,'TileSpacing','compact')%,'Padding','compact'
    % nexttile
    % s=surf(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),abs(O1((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz)),'FaceColor','interp' );
    % s.EdgeColor = 'none';
    % xlim([L0/2-L/2,L0/2+L/2]);
    % view(0,90);
    % %         caxis([0,O])
    % colorbar;
    % Xlabel= '$x$'; Ylabel= '$y$';
    % Title=['$\mathcal{O}$, t=',num2str(round(t))];
    % xlabel(Xlabel,'Interpreter','latex')
    % ylabel(Ylabel,'Interpreter','latex')
    % title(Title,'Interpreter','latex')
    % %         figure(2),
    % nexttile
    % ss=mesh(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),angle(O1((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz)),'FaceColor','interp' );
    % ss.EdgeColor = 'none';
    % xlim([L0/2-L/2,L0/2+L/2]);
    % view(0,90);
    % colorbar;
    % Xlabel= '$x$'; Ylabel= '$y$';
    % Title=['$\theta$, t=',num2str(round(t))];
    % xlabel(Xlabel,'Interpreter','latex')
    % ylabel(Ylabel,'Interpreter','latex')
    % title(Title,'Interpreter','latex')
    % drawnow
    %
    % % ===== Write a GIF frame. =====
    % frame = getframe(fig1);
    % im = frame2im(frame);
    % [A,map] = rgb2ind(im,256,'nodither');
    %
    % if firstFrame
    %     imwrite(A,map,gifname,'gif','LoopCount',inf,'DelayTime',delaytime);
    %     firstFrame = false;
    % else
    %     imwrite(A,map,gifname,'gif','WriteMode','append','DelayTime',delaytime);
    % end
    %
    % set(0,'CurrentFigure',fig2)
    % % tiledlayout(1,2,'TileSpacing','compact')%,'Padding','compact'
    % % nexttile
    % ux=reshape(VX_temp(:,:,end),[nx,ny]);
    % uy=reshape(VY_temp(:,:,end),[nx,ny]);
    % ux = ux - mean(ux(:));
    % uy = uy - mean(uy(:));
    % uxhat = fft2(ux);
    % uyhat = fft2(uy);
    % E2d = 0.5 * (abs(uxhat).^2 + abs(uyhat).^2);
    % if mod(nx, 2) == 0
    %     kx = (2*pi/L0) * [0:nx/2-1 , -nx/2:-1];
    %     ky = (2*pi/L ) * [0:ny/2-1 , -ny/2:-1];
    % else
    %     kx = (2*pi/L0) * [0:(nx-1)/2, -(nx-1)/2:-1];
    %     ky = (2*pi/L ) * [0:(ny-1)/2, -(ny-1)/2:-1];
    % end
    % [KX,KY] = meshgrid(kx,ky);
    % K = sqrt(KX.^2 + KY.^2);
    % kmax = floor(max(K(:)));
    % dk = 2*pi/L0;
    % kbins = 0:dk:kmax;
    % Ek = zeros(length(kbins)-1,1);
    % for i = 1:length(kbins)-1
    %     mask = (K >= kbins(i)) & ...
    %         (K <  kbins(i+1));
    %     Ek(i) = sum(E2d(mask));
    % end
    % kplot = 0.5*(kbins(1:end-1) + kbins(2:end));
    % plot(kplot(2:end), Ek(2:end), 'LineWidth',2,'Color', cmap((t-tmin)/10+1,:));
    % colormap(cmap);
    % clim([tmin,tmax]);
    % cb = colorbar;
    % cb.Label.String = 'time';
    % xlabel('k');
    % ylabel('E(k)');
    % title('Superfluid Energy Spectrum');
    % set(gca,'XScale','log','YScale','log');
    % drawnow
end
save(hqt_data_path('processed', "NofV_decay.mat"),'NofV')
%% plot NofV
% Load the dataset selected by this section using a repository-relative location.
load(hqt_data_path('processed', "NofV_decay.mat"))
tmin=0;tmax=1500;
figure(4),clf,theme(gcf,"light"),
set(gcf,'Position',[100 200 500 400]);
box on
plot(tmin+10:10:tmax,NofV(tmin/10+151:end))
xlim([tmin,tmax])
ylim([0,max(NofV(151:end))+10])
xlabel('t','FontSize',14);
ylabel('Vortices number','FontSize',14);
exportgraphics(gcf,hqt_output_path('figures', "NofV_decay.pdf"),ContentType="Image",Resolution=600)
%% movie show
enable_gpu
t=1600;
mu=6;vy=-2*pi/20*7;
subFolder = hqt_data_path('raw', ['check_point_driven_mu=',num2str(mu),',v=',num2str(-vy*20/2/pi),'_20_large']);
fileName = fullfile(subFolder, ['check_point_t=',num2str(t),'.mat']);load(fileName)
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
Nx=(nx-1)/2;

% Differentiate the bulk scalar; its final radial slice is the boundary order parameter.
O1=func_Lz(Z1);
[n, np, nm, C] = countN_C(O1(:,:,nz))
fig1=figure(1);
set(0,"CurrentFigure",fig1),theme(gcf,"light")
set(fig1,'Position',[100 200 220 200]);
clf
tiledlayout(1,1,'TileSpacing','compact','Padding','compact')%,'Padding','compact'
ax1=nexttile;
s=surf(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),abs(O1((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz)),'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([L0/2-L/2,L0/2+L/2]);
view(0,90);
clim([0,max(abs(O1(:,:,nz)),[],'all')])
% colorbar;
% Xlabel= '$x$'; Ylabel= '$y$';
Title=['$\mathcal{O}$, t=',num2str(round(t-1500))];
% xlabel(Xlabel,'Interpreter','latex')
% ylabel(Ylabel,'Interpreter','latex')
title(Title,'Interpreter','latex')
axis(ax1, 'off');
% exportgraphics(gcf,hqt_output_path('figures', ['movie_decay_t=',num2str(t),'.pdf']),ContentType="Image",Resolution=600)
% %         figure(2),
% nexttile
% ss=mesh(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),angle(O1((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz)),'FaceColor','interp' );
% ss.EdgeColor = 'none';
% xlim([L0/2-L/2,L0/2+L/2]);
% view(0,90);
% colorbar;
% Xlabel= '$x$'; Ylabel= '$y$';
% Title=['$\theta$, t=',num2str(round(t))];
% xlabel(Xlabel,'Interpreter','latex')
% ylabel(Ylabel,'Interpreter','latex')
% title(Title,'Interpreter','latex')

% VX_temp=real(func_Lz(Z3)); VY_temp=real(func_Lz(Z4));
% VX_temp=real(-1i*(func_Lx(O1)-func_Lx(abs(O1)).*O1./abs(O1))./O1);
% VY_temp=real(-1i*(func_Ly(O1)-func_Ly(abs(O1)).*O1./abs(O1))./O1);
% VX_temp=abs(O1)./max(abs(O1(:,:,end))).*real(-1i*(func_Lx(O1)-func_Lx(abs(O1)).*O1./abs(O1))./O1);
% VY_temp=abs(O1)./max(abs(O1(:,:,end))).*real(-1i*(func_Ly(O1)-func_Ly(abs(O1)).*O1./abs(O1))./O1);
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

figure(3),clf,theme(gcf,"light")
set(gcf,'Position',[100 300 1500 400]);
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
plot(kplot, Ek(90)/kplot(90)^(-5/3)*kplot.^(-5/3), 'r--', 'LineWidth',2);
plot(kplot, Ek(400)/kplot(400)^(-3)*kplot.^(-3), 'k--', 'LineWidth',2);
plot(kplot, Ek(70)/kplot(70)^(-1)*kplot.^(-1), 'b--', 'LineWidth',2);
plot(kplot, Ek(4)/kplot(4)*kplot.^(1), 'g--', 'LineWidth',2);
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
plot(kplot, Ekc(400)/kplot(400)^(-3)*kplot.^(-3), 'k--', 'LineWidth',2);
plot(kplot, Ekc(25)/kplot(25)^(-1)*kplot.^(-1), 'b--', 'LineWidth',2);
plot(kplot, Ekc(4)/kplot(4)*kplot.^(1), 'g--', 'LineWidth',2);
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
plot(kplot, Eki(400)/kplot(400)^(-3)*kplot.^(-3), 'k--', 'LineWidth',2);
plot(kplot, Eki(70)/kplot(70)^(-1)*kplot.^(-1), 'b--', 'LineWidth',2);
plot(kplot, Eki(4)/kplot(4)*kplot.^(1), 'g--', 'LineWidth',2);
hold off
xlim([min(kplot(2:end)),max(kplot(2:end))])
ylim([min(Eki(2:end)),5*max(Eki(2:end))])
set(gca,'XScale','log','YScale','log');
% exportgraphics(gcf,hqt_output_path('figures', ['spectrum_decay_t=',num2str(t),'.pdf']),ContentType="Image",Resolution=600)
% superfluid velocity distribution
% nexttile
% ux1 = ux(:);
% uy1 = uy(:);
% Nbins = 100;
% [countsx, edgesx] = histcounts(ux1, ...
%     Nbins, ...
%     'Normalization','pdf');
% [countsy, edgesy] = histcounts(uy1, ...
%     Nbins, ...
%     'Normalization','pdf');
% uxc = 0.5*(edgesx(1:end-1) + edgesx(2:end));
% uyc = 0.5*(edgesy(1:end-1) + edgesy(2:end));
% mux = mean(ux1);
% muy = mean(uy1);
% sigmax = std(ux1);
% sigmay = std(uy1);
% gaussx = 1/(sqrt(2*pi)*sigmax) ...
%     .* exp(-(uxc-mux).^2/(2*sigmax^2));
% gaussy = 1/(sqrt(2*pi)*sigmay) ...
%     .* exp(-(uyc-muy).^2/(2*sigmay^2));
%
% semilogy(uxc, countsx, 'r',...
%     'LineWidth',2);
% hold on;
% semilogy(uyc, countsy, 'b',...
%     'LineWidth',2);
% semilogy(uxc, gaussx, 'r--', ...
%     'LineWidth',2);
% semilogy(uyc, gaussy, 'b--', ...
%     'LineWidth',2);
% xlabel('u');
% ylabel('PDF');
% title('Superfluid velocity distribution');
% legend( ...
%     'u_x numerical', ...
%     'u_y numerical', ...
%     'u_x Gaussian', ...
%     'u_y Gaussian','Location','best');
% grid on;

Lx1=0;Lx2=500;Ly1=0;Ly2=500;
figure(7),clf,theme(gcf,"light")
set(gcf,'Position',[800 100 680 800]);
tiledlayout(3,2,'TileSpacing','compact','Padding','compact')
ax1=nexttile;
s=surf(xx(:,:,nz),yy(:,:,nz),real(ux(:,:)),'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([Lx1,Lx2]);ylim([Ly1,Ly2]);
view(0,90);
colorbar;
% clim([min(ux,[],'all'),max(ux,[],'all')])
Xlabel= '$x$'; Ylabel= '$y$';
Title='$v^s_x$';
xlabel(Xlabel,'Interpreter','latex')
ylabel(Ylabel,'Interpreter','latex')
title(Title,'Interpreter','latex')
ax2=nexttile;
s=surf(xx(:,:,nz),yy(:,:,nz),real(uy(:,:)),'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([Lx1,Lx2]);ylim([Ly1,Ly2]);
view(0,90);
colorbar;
% clim([min(uy,[],'all'),max(uy,[],'all')])
Xlabel= '$x$'; Ylabel= '$y$';
Title='$v^s_y$';
xlabel(Xlabel,'Interpreter','latex')
ylabel(Ylabel,'Interpreter','latex')
title(Title,'Interpreter','latex')
ax3=nexttile;
s=surf(xx(:,:,nz),yy(:,:,nz),real(VX_i(:,:)),'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([Lx1,Lx2]);ylim([Ly1,Ly2]);
view(0,90);
colorbar;
% clim([min(ux,[],'all'),max(ux,[],'all')])
Xlabel= '$x$'; Ylabel= '$y$';
Title='$(v^s_x)_i$';
xlabel(Xlabel,'Interpreter','latex')
ylabel(Ylabel,'Interpreter','latex')
title(Title,'Interpreter','latex')
ax4=nexttile;
s=surf(xx(:,:,nz),yy(:,:,nz),real(VY_i(:,:)),'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([Lx1,Lx2]);ylim([Ly1,Ly2]);
view(0,90);
colorbar;
% clim([min(uy,[],'all'),max(uy,[],'all')])
Xlabel= '$x$'; Ylabel= '$y$';
Title='$(v^s_y)_i$';
xlabel(Xlabel,'Interpreter','latex')
ylabel(Ylabel,'Interpreter','latex')
title(Title,'Interpreter','latex')
ax5=nexttile;
s=surf(xx(:,:,nz),yy(:,:,nz),real(VX_c(:,:)),'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([Lx1,Lx2]);ylim([Ly1,Ly2]);
view(0,90);
colorbar;
% clim([min(ux,[],'all'),max(ux,[],'all')])
clim([min((abs(O1(:,:,end))./max(abs(O1(:,:,end)))).^10.*real(VX_c),[],'all'),max((abs(O1(:,:,end))./max(abs(O1(:,:,end)))).^10.*real(VX_c),[],'all')])
Xlabel= '$x$'; Ylabel= '$y$';
Title='$(v^s_x)_c$';
xlabel(Xlabel,'Interpreter','latex')
ylabel(Ylabel,'Interpreter','latex')
title(Title,'Interpreter','latex')
ax6=nexttile;
s=surf(xx(:,:,nz),yy(:,:,nz),real(VY_c(:,:)),'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([Lx1,Lx2]);ylim([Ly1,Ly2]);
view(0,90);
colorbar;
% clim([min(uy,[],'all'),max(uy,[],'all')])
clim([min((abs(O1(:,:,end))./max(abs(O1(:,:,end)))).^10.*real(VY_c),[],'all'),max((abs(O1(:,:,end))./max(abs(O1(:,:,end)))).^10.*real(VY_c),[],'all')])
Xlabel= '$x$'; Ylabel= '$y$';
Title='$(v^s_y)_c$';
xlabel(Xlabel,'Interpreter','latex')
ylabel(Ylabel,'Interpreter','latex')
title(Title,'Interpreter','latex')
axis([ax1,ax2,ax3,ax4,ax5,ax6], 'off');
% exportgraphics(gcf,hqt_output_path('figures', ['velocity_decomposition_t=',num2str(t),'.pdf']),ContentType="Image",Resolution=600)

% Ot=func_Lz(F1(Z1,Z2,Z3,Z4));
% thetat=-1i*(conj(O1).*Ot-O1.*conj(Ot))./(2*abs(O1).^2).*abs(O1)./max(abs(O1(:,:,nz)),[],'all');
% mu_true=mu-thetat;mu_true=mu_true(:,:,nz);
% u_x=ux./mu_true;u_y=uy./mu_true;
% w_x=sqrt(abs(func_Lz(Z3)./VX_temp)).*VX_temp;w_x=w_x(:,:,nz).*sqrt(mu_true);
% w_y=sqrt(abs(func_Lz(Z4)./VY_temp)).*VY_temp;w_y=w_y(:,:,nz).*sqrt(mu_true);
% [k_1D, E_k, T_k, Pi_k] = energy_flux(u_x, u_y, w_x, w_y, L0, L);
% figure(10),clf
% plot(k_1D, E_k, 'LineWidth',2);
% xlabel('$k$','Interpreter','latex');
% ylabel('$E(k)$','Interpreter','latex');
% title('Superfluid Energy Spectrum');
% grid on;
% hold on
% % C1 = 2.5e5;  C2 = 9e5; C3=2.5e5; C4=1.5e10; % nx=401
% plot(k_1D, E_k(90)/k_1D(90)^(-5/3)*k_1D.^(-5/3), 'r--');
% plot(k_1D, E_k(end)/k_1D(end)^(-3)*k_1D.^(-3), 'k--');
% plot(k_1D, E_k(25)/k_1D(25)^(-1)*k_1D.^(-1), 'b--');
% plot(k_1D, E_k(2)/k_1D(2)*k_1D.^(1), 'g--');
% hold off
% xlim([min(k_1D(2:end)),max(k_1D(2:end))])
% ylim([min(E_k(2:end)),5*max(E_k(2:end))])
% set(gca,'XScale','log','YScale','log');
% figure(11),plot(k_1D,Pi_k)
% grid on;
% set(gca,'XScale','log');
% xlim([min(k_1D(2:end)),max(k_1D(2:end))])
% ylim([min(Pi_k(2:end)),max(Pi_k(2:end))])
% xlabel('$k$','Interpreter','latex');
% ylabel('$\Pi(k)$','Interpreter','latex');
%% structure functions
nmax=(nx-1)/2;
% nmax=50;
p_orders=3;
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
%% plot movie
enable_gpu
fig1=figure(1);
set(0,"CurrentFigure",fig1),theme(gcf,"light")
set(fig1,'Position',[100 200 700 200]);
clf
tiledlayout(1,3,'TileSpacing','compact','Padding','compact')%,'Padding','compact'
for t=[1600,2000,3000]
mu=6;vy=-2*pi/20*7;
subFolder = hqt_data_path('raw', ['check_point_driven_mu=',num2str(mu),',v=',num2str(-vy*20/2/pi),'_20_large']);
fileName = fullfile(subFolder, ['check_point_t=',num2str(t),'.mat']);load(fileName)
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
Nx=(nx-1)/2;

% Differentiate the bulk scalar; its final radial slice is the boundary order parameter.
O1=func_Lz(Z1);
% [n, np, nm, C] = countN_C(O1(:,:,nz))

ax1=nexttile;
s=surf(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),abs(O1((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz)),'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([L0/2-L/2,L0/2+L/2]);
view(0,90);
colormap('jet')
clim([0,max(abs(O1(:,:,nz)),[],'all')])
if t==3000
colorbar;
end
% Xlabel= '$x$'; Ylabel= '$y$';
Title=['$\mathcal{O}$, t=',num2str(round(t-1500))];
% xlabel(Xlabel,'Interpreter','latex')
% ylabel(Ylabel,'Interpreter','latex')
title(Title,'Interpreter','latex')
axis(ax1, 'off');
end
exportgraphics(gcf,hqt_output_path('figures', ['movie_decay.pdf']),ContentType="Image",Resolution=600)

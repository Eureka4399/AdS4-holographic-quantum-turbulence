% PLOTS_FOR_PAPER Historical checkpoint analysis and plotting notebook.
% Call setup_project first, then run selected MATLAB sections in dependency order.
% Checkpoint inputs under data/raw/ are intentionally omitted from Git.
% Set grid, domain, time range, and run-folder suffix to match the generated run.
% Figures/movies go to outputs/; processed arrays go to data/processed/.

%% load %%%%%%%%%%%%%%%%%%%%
enable_gpu
t=3000;
mu=6;vy_load=-2*pi/20*7;
subFolder = hqt_data_path('raw', ['check_point_driven_mu=',num2str(mu),',v=',num2str(-vy_load*20/2/pi),'_20_large_2']);
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

% Load the dataset selected by this section using a repository-relative location.
load(hqt_data_path('processed', "driven_2.mat"))
% Load the dataset selected by this section using a repository-relative location.
load(hqt_data_path('processed', 'average2.mat'))
%% figure 1
fig1=figure(1); theme(gcf,"light"), clf
% tl =tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
set(fig1,'Position',[200 300 400 320]);

% ax1=nexttile(tl);
ax1 = gca;
hold on, box on
tmin=0;
tmax=3000;
cmap = turbo((tmax-tmin)/10+1);
for t=tmin+10:10:tmax
    % for t=[tmin+10,tmax]
    t
    VX_temp=vx(:,:,round(t/10));
    VY_temp=vy(:,:,round(t/10));

    % set(0,'CurrentFigure',fig1)
    ux=reshape(VX_temp(:,:,end),[nx,ny]);
    uy=reshape(VY_temp(:,:,end),[nx,ny]);
    ux = ux - mean(ux(:));
    uy = uy - mean(uy(:));
    uxhat = fft2(ux);
    uyhat = fft2(uy);
    % Form the two-dimensional kinetic-energy spectrum before radial shell averaging.
    E2d = 0.5 * (abs(uxhat).^2 + abs(uyhat).^2);
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
    Ek = zeros(length(kbins)-1,1);
    for i = 1:length(kbins)-1
        mask = (K >= kbins(i)) & ...
            (K <  kbins(i+1));
        Ek(i) = sum(E2d(mask));
    end
    kplot = 0.5*(kbins(1:end-1) + kbins(2:end));
    plot(kplot(2:end), Ek(2:end), 'LineWidth',2,'Color', cmap((t-tmin)/10+1,:));
    xlim([min(kplot(2:end),[],'all'),max(kplot(2:end),[],'all')])
    colormap(ax1, cmap);             % Attach to ax1.
    clim(ax1, [tmin, tmax]);
    cb = colorbar(ax1);              % Attach to ax1.
    title(cb, '$t$', 'FontSize', 14, 'Interpreter', 'latex');
    xlabel('$k$','Interpreter','latex','FontSize',14);
    ylabel('$E(k)$','Interpreter','latex','FontSize',14);
    % title('Superfluid Energy Spectrum');
    set(gca,'XScale','log','YScale','log');
    drawnow
end
hold off
ax1 = gca;
cb.Ruler.Exponent = 3;
cb.Ticks=[0,1000,2000,3000];
% cb.TickLabels={'0','1','2','3'};
cb.TickLabelInterpreter = 'latex';
cb.FontSize=11;
drawnow; % Update the layout before reading the axes position.
pos1 = ax1.Position; % Read [left, bottom, width, height] relative to the figure.
set(gcf, 'NextPlot', 'add'); % Retain overlapping axes when adding the inset.
rel_x = 0.25; rel_y = 0.1; rel_w = 0.5; rel_h = 0.4;
inset_pos = [pos1(1) + pos1(3)*rel_x, pos1(2) + pos1(4)*rel_y, pos1(3)*rel_w, pos1(4)*rel_h];
margin_x = 0.04; margin_y = 0.07;
card_pos = [inset_pos(1) - pos1(3)*margin_x, inset_pos(2) - pos1(4)*margin_y, ...
    inset_pos(3) * (1 + margin_x + 0.07), inset_pos(4) * (1 + margin_y + 0.12)];
h_card = axes('Position', card_pos);
h_card.Color = 'w';                   % White background.
h_card.XTick = []; h_card.YTick = [];  % Hide ticks.
h_card.XColor = [0.85, 0.85, 0.85];   % Light gray border.
h_card.YColor = [0.85, 0.85, 0.85];
box(h_card, 'on');
h_inset = axes('Position', inset_pos);
h_inset.Color = 'none'; % Transparent inset background.
h_inset.FontSize = 8;
h_inset.TickDir = 'in';
box(h_inset, 'on');
plot(h_inset, tmin+10:10:tmax, NofV(tmin/10+1:tmax/10), 'k');
xlim(h_inset, [tmin, tmax]);
ylim(h_inset, [0, max(NofV)+400]);
xticks(h_inset, [0, 1500, 3000]);
xticklabels(h_inset, {'0', '1.5', '3'});
yticks(h_inset, [0, 2000]);
yticklabels(h_inset, {'0', '2'});
text(h_inset, 0.01, 0.9, '$N(t)\,(\times 10^3)$', 'Units', 'normalized', 'FontSize', 9, 'Interpreter', 'latex');
text(h_inset, 0.65, 0.1, '$t\,(\times 10^3)$', 'Units', 'normalized', 'FontSize', 9, 'Interpreter', 'latex');
title(ax1, ' ', 'FontSize', 14);
text(ax1,0.03, 0.95, '(a)', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
% exportgraphics(gcf,hqt_output_path('figures', ['figure1a.pdf']),ContentType="Image",Resolution=600)

% ax3=nexttile(tl);
fig4=figure(4); theme(gcf,"light"), clf
% tl =tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
set(fig4,'Position',[200 300 400 320]);
ax3 = gca;
O1=O(:,:,300);
c_blue = [0.0,0.0,0.8];  % #2166AC (blue endpoint)
c_mid  = [247, 247, 247] / 255;  % #F7F7F7 (white/gray midpoint)
c_red  = [0.8,0.0,0.0];  % #B2182B (red endpoint)
N = 256;
x = [1, round(N/2), N]; % Colormap endpoint and midpoint indices.
xi = (1:N)';            % Colormap interpolation samples.
my_cmap = interp1(x, [c_blue; c_mid; c_red], xi, 'linear');
% colormap(my_cmap);
colormap(ax3, my_cmap);
s = pcolor(ax3, xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz), ...
    yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz), ...
    abs(O1((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,1)));
s.EdgeColor = 'none';
shading(ax3, 'interp'); % Interpolate surface colors.
xlim([L0/2-L/2,L0/2+L/2]);
view(0,90);
% xticks(ax3, [0,100,200,300,400,500]);
% xticklabels(ax3, {'0', '1', '2', '3', '4', '5'});
% yticks(ax3, [0,100,200,300,400,500]);
% yticklabels(ax3, {'0', '1', '2', '3', '4', '5'});
xticks([]);yticks([]);
xlabel('$x$','Interpreter','latex', 'FontSize', 14);
ylabel('$y$','Interpreter','latex', 'FontSize', 14);
% set(ax3, 'PlotBoxAspectRatioMode', 'auto', 'DataAspectRatioMode', 'auto');
h_cb1 = colorbar;
h_cb1.FontSize = 12;
title(h_cb1, '$\mathcal{O}$', 'FontSize', 14, 'Interpreter', 'latex');
clim([0,max(abs(O1(:,:,1)),[],'all')])
h_cb1.TickLabelInterpreter = 'latex';
% axis(ax3, 'off');
axis(ax3, 'tight');   % Fit the axes tightly to the data.
axis(ax3, 'equal');
text(ax3, 0.03, 0.95, '(b)', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');                   % Axes and label spacing.
title(ax3, ' ', 'FontSize', 15);
exportgraphics(gcf,hqt_output_path('figures', ['figure1b.pdf']),ContentType="Image",Resolution=600)
%% figure2
% Load the dataset selected by this section using a repository-relative location.
load(hqt_data_path('processed', 'average2.mat'))

x_v=vortices(:,2);y_v=vortices(:,1);charges=vortices(:,3);
distance=80.9252;
f=@(a,x) 2.5.*x+a(1);
% f=@(a,x) a(2).*x.^a(1);
a0=[0.1];
opts = statset('nlinfit');opts.TolFun=1e-13;
% a1 = nlinfit((k_bin(2:7)),(Sc_1d(2:7)),f,a0);
% Fit the selected model to the plotted spectrum or structure-factor range.
a1 = nlinfit(log(k_bin(2:7)),log(Sc_1d(2:7)),f,a0);

std_c=std(Sc_1d_tol(101:end,2:end));std_n=std(Sn_1d_tol(101:end,2:end));

figure(2),clf
theme(gcf,"light")
set(gcf, 'Position', [100 100 400 350])
% tiledlayout(1,2,'Padding','compact')%  'TileSpacing','compact',  ,'Padding','compact'
% nexttile
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
text( -0.12, 1, '(a)', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
% exportgraphics(gcf,hqt_output_path('figures', "figure2a.pdf"),ContentType="Image",Resolution=600)

% nexttile
figure(6),clf
theme(gcf,"light")
set(gcf, 'Position', [100 100 400 350])
hold on;
% errorbar(k_bin(2:end), Sc_1d(2:end),std_c, 'r-o', 'LineWidth', 1, 'MarkerSize', 4, 'DisplayName', '$S_c(k)$');
% errorbar(k_bin(2:end), Sn_1d(2:end),std_n, 'b-o', 'LineWidth', 1, 'MarkerSize', 4, 'DisplayName', '$S_n(k)$');

k_data  = k_bin(2:end);
Sc_data = Sc_1d(2:end);
Sn_data = Sn_1d(2:end);
err_c   = std_c(1:end);
err_n   = std_n(1:end);
N = length(k_data);
idx = unique(round(logspace(0, log10(N), 50)));
plot(k_data, Sc_data, 'r-', 'LineWidth', 1, 'HandleVisibility', 'off');
plot(k_data, Sn_data, 'b-', 'LineWidth', 1, 'HandleVisibility', 'off');
errorbar(k_data(idx), Sc_data(idx), err_c(idx), 'ro', 'LineWidth', 1, 'MarkerSize', 4, 'DisplayName', '$S_c(k)$');
errorbar(k_data(idx), Sn_data(idx), err_n(idx), 'bo', 'LineWidth', 1, 'MarkerSize', 4, 'DisplayName', '$S_n(k)$');

plot(k_bin(1:10),exp(a1(1))*k_bin(1:10).^2.5,'k', 'LineWidth', 1.2, 'DisplayName', ['$k^{',num2str(2.5,2),'}$'], 'HandleVisibility', 'off')
xline(2*pi/distance, 'k--','DisplayName', '$k_{d}$','Interpreter','latex', 'HandleVisibility', 'off');
box on;
set(gca,'XScale','log','YScale','log');
ylim([0.3*min([Sc_1d(2:end),Sn_1d(2:end)]),1.5*max([Sc_1d(2:end),Sn_1d(2:end)])])
% title('1D Structure Factor $S(k)$','Interpreter','latex','FontSize',12);
xlabel('$k$','Interpreter','latex','FontSize',14); ylabel('$S(k)$','Interpreter','latex','FontSize',14);
legend('Location', 'best','FontSize',12,'Interpreter','latex');
text( -0.12, 1, '(b)', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
text( 0.3, 0.3, '$k_d$', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
text( 0.09, 0.5, '$k^{2.5}$', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
exportgraphics(gcf,hqt_output_path('figures', "figure2b.pdf"),ContentType="Image",Resolution=600)

% exportgraphics(gcf,hqt_output_path('figures', "figure2.pdf"),ContentType="Image",Resolution=600)

%% figure3
% nexttile(tl)
fig5=figure(5); theme(gcf,"light"), clf
% tl =tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
set(fig5,'Position',[200 300 400 320]);
% Load the dataset selected by this section using a repository-relative location.
load(hqt_data_path('processed', 'average2.mat'))
plot(kplot(2:end), Ek(2:end),'k', 'LineWidth',2);
xlabel('$k$','Interpreter','latex','FontSize',14);
ylabel('$E(k)$','Interpreter','latex','FontSize',14);
% title('Superfluid Energy Spectrum');
grid on;
hold on
% C1 = 2.5e5;  C2 = 9e5; C3=2.5e5; C4=1.5e10; % nx=401
plot(kplot, 1*Ek(90)/kplot(90)^(-5/3)*kplot.^(-5/3), 'r--', 'LineWidth',2);
% plot(kplot, 0.5*Ek(end)/kplot(end)^(-3)*kplot.^(-3), 'k--', 'LineWidth',2);
plot(kplot, 1*Ek(25)/kplot(25)^(-1)*kplot.^(-1), 'b--', 'LineWidth',2);
% plot(kplot, 1.5*Ek(2)/kplot(2)*kplot.^(1), 'g--', 'LineWidth',2);
hold off
xlim([min(kplot(2:end)),max(kplot(2:end))])
ylim([min(Ek(2:end)),5*max(Ek(2:end))])
set(gca,'XScale','log','YScale','log');
% legend('$E(k)$', '$k^{-5/3}$', '$k^{-3}$', '$k^{-1}$', ...
%     'Interpreter', 'latex', 'FontSize', 11, 'Location', 'southwest');
text(0.03, 0.95, '(a)', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');                   % Axes and label spacing.
% text(0.5, 0.8, '$k^{-5/3}$', ...
%     'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
%     'FontSize', 14, ...             % Font size.
%     'Interpreter', 'latex');                   % Axes and label spacing.
text(0.4, 0.9, '$k^{-5/3}$', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
text(0.12, 0.85, '$k^{-1}$', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
% text(0.6, 0.85, '$k^{-3}$', ...
%     'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
%     'FontSize', 14, ...             % Font size.
%     'Interpreter', 'latex');
exportgraphics(gcf,hqt_output_path('figures', ['figure3a.pdf']),ContentType="Image",Resolution=600)

fig3=figure(3);theme('light')
set(fig3,'Position',[200 300 400 320]);
% tiledlayout(1,3,'TileSpacing','compact','Padding','compact')
% nexttile
plot(kplot(2:end), Eki(2:end),'k', 'LineWidth',2);
xlabel('$k$','Interpreter','latex','FontSize',14);
ylabel('$E_i(k)$','Interpreter','latex','FontSize',14);
% title('Incompressible Superfluid Energy Spectrum');
% grid on;
hold on
% C1 = 1.8e8;  C2 = 7e8; C3=1.8e8; C4=7e9; % nx=401
plot(kplot, Eki(90)/kplot(90)^(-5/3)*kplot.^(-5/3), 'r--', 'LineWidth',2);
plot(kplot, 0.7*Eki(end)/kplot(end)^(-3)*kplot.^(-3), '--','Color',[0,0.5,0], 'LineWidth',2);
plot(kplot(5:end), Eki(25)/kplot(25)^(-1)*kplot(5:end).^(-1), 'b--', 'LineWidth',2);
plot(kplot(1:8), 1.2*Eki(2)/kplot(2)^1.5*kplot(1:8).^(1.5), '--','Color',[235, 100, 55]/255, 'LineWidth',2);
hold off
xlim([min(kplot(2:end)),max(kplot(2:end))])
ylim([min(Eki(2:end)),5*max(Eki(2:end))])
set(gca,'XScale','log','YScale','log');
% legend('$E_i(k)$', '$k^{-5/3}$', '$k^{-3}$', '$k^{-1}$', ...
%     'Interpreter', 'latex', 'FontSize', 11, 'Location', 'southwest');
text(0.35, 0.9, '$k^{-5/3}$', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
text(0.12, 0.85, '$k^{-1}$', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
text(0.6, 0.85, '$k^{-3}$', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
text(0.1, 0.6, '$k^{1.5}$', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
text( 0.03, 0.95, '(b)', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
exportgraphics(gcf,hqt_output_path('figures', "figure3b.pdf"),ContentType="Image",Resolution=600)

% nexttile
fig7=figure(7);theme('light')
set(fig7,'Position',[200 300 400 320]);
plot(kplot(2:end), Ekc(2:end),'k', 'LineWidth',2);
xlabel('$k$','Interpreter','latex','FontSize',14);
ylabel('$E_c(k)$','Interpreter','latex','FontSize',14);
% title('Compressible Superfluid Energy Spectrum');
% grid on;
hold on
% C1 = 8e7;  C2 = 4e8; C3=1e8; % nx=401
% plot(kplot, Ekc(90)/kplot(90)^(-5/3)*kplot.^(-5/3), 'r--');
plot(kplot, Ekc(90)/kplot(90)^(-2)*kplot.^(-2), 'm--', 'LineWidth',2);
% plot(kplot, Ekc(end)/kplot(end)^(-3)*kplot.^(-3), 'k--', 'LineWidth',2);
% plot(kplot, Ekc(25)/kplot(25)^(-1)*kplot.^(-1), 'b--', 'LineWidth',2);
% plot(kplot, 1.5*Ekc(2)/kplot(2)*kplot.^(1), 'g--', 'LineWidth',2);
hold off
xlim([min(kplot(2:end)),max(kplot(2:end))])
ylim([min(Ekc(2:end)),5*max(Ekc(2:end))])
set(gca,'XScale','log','YScale','log');
% legend('$E_c(k)$', '$k^{-2}$', ...
%     'Interpreter', 'latex', 'FontSize', 11, 'Location', 'southwest');
text(0.5, 0.8, '$k^{-2}$', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
text( 0.03, 0.95, '(c)', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
exportgraphics(gcf,hqt_output_path('figures', "figure3c.pdf"),ContentType="Image",Resolution=600)

S_L=(S_Lx+S_Ly+S_Tx+S_Ty)/2;nmax=500;
% nexttile
fig8=figure(8);clf,theme('light')
set(fig8,'Position',[200 300 400 320]);
box on
hold on
for p=1
    h = plot(xx(1:nmax), S_L(:,p), 'ko');
    current_color = h.Color;
    plot(xx(1:nmax), 1.7*S_L(3,p)*xx(1:nmax).^(2/3),'r--','LineWidth',2);
    plot(xx(1:nmax), 0.078*log(xx(1:nmax)/xx(10))+S_L(10,p),'b-.','LineWidth',2);
end
hold off
xlim([0,xx(nmax,1,1)])
ylim([min(S_L(:,1),[],'all'),max(1.2*S_L(:,1),[],'all')])
set(gca,'XScale','log');
xlabel('$r$','Interpreter','latex','FontSize',14);
ylabel('$S_2$','Interpreter','latex','FontSize',14);
xticks([1,10,100]);
% legend('$S_2$', '$\propto r^{2/3}$', '$\propto \ln(r)$', ...
%     'Interpreter', 'latex', 'FontSize', 11, 'Location', 'southeast');
text(0.18, 0.8, '$r^{2/3}$', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
text(0.45, 0.8, '$\mathrm{ln} r$', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');
text( 0.03, 0.95, '(d)', ...
    'Units', 'normalized', ...      % Use normalized coordinates in [0, 1].
    'FontSize', 14, ...             % Font size.
    'Interpreter', 'latex');

exportgraphics(gcf,hqt_output_path('figures', "figure3d.pdf"),ContentType="Image",Resolution=600)

%% figure S2
% Load the dataset selected by this section using a repository-relative location.
load(hqt_data_path('processed', 'average2.mat'));[centroids, cluster_info] = find_vortex_dpc_pbc(vortices, 500,500,10);
% exportgraphics(gcf,hqt_output_path('figures', "vortex_cluster.pdf"),ContentType="Image",Resolution=600)
%% figure S1
t=3000;
ux=reshape(vx(:,:,round(t/10)),[nx,ny]);
uy=reshape(vy(:,:,round(t/10)),[nx,ny]);
ux = ux - mean(ux(:));
uy = uy - mean(uy(:));
uxhat = fft2(ux);
uyhat = fft2(uy);
% Form the two-dimensional kinetic-energy spectrum before radial shell averaging.
E2d = 0.5 * (abs(uxhat).^2 + abs(uyhat).^2);
if mod(nx, 2) == 0
    kx = (2*pi/L0) * [0:nx/2-1 , -nx/2:-1];
    ky = (2*pi/L ) * [0:ny/2-1 , -ny/2:-1];
else
    kx = (2*pi/L0) * [0:(nx-1)/2, -(nx-1)/2:-1];
    ky = (2*pi/L ) * [0:(ny-1)/2, -(ny-1)/2:-1];
end
[KX,KY] = meshgrid(kx,ky);

figure(7),theme('light')
set(gcf, 'Position', [100 100 400 350])
mesh(KX, KY, log(abs(E2d)));
% xlim([min(kplot(2:end),[],'all'),max(kplot(2:end),[],'all')])
view(0,90)
axis equal tight; box on;
clim([7,max(log(abs(E2d)),[],'all')])
xlim([2/3*min(KX,[],'all'),2/3*max(KX,[],'all')])
ylim([2/3*min(KY,[],'all'),2/3*max(KY,[],'all')])
colormap("jet");
cb = colorbar;
title(cb, '$\mathrm{ln}\Phi_{ii}(\mathbf{k})$', 'FontSize', 14, 'Interpreter', 'latex');
xlabel('$k_x$','Interpreter','latex','FontSize',14);
ylabel('$k_y$','Interpreter','latex','FontSize',14);
xticks([-4,-2,0,2,4]);
yticks([-4,-2,0,2,4]);
% set(gca,'XScale','log','YScale','log');
% exportgraphics(gcf,hqt_output_path('figures', "figureS1a.pdf"),ContentType="Image",Resolution=600)

nmax=(nx-1)/2;
% Use periodic autocorrelations to evaluate the two-dimensional second-order structure function.
[S2, rx, ry] = structure_function_2d(ux, uy);
figure(8),theme('light')
set(gcf, 'Position', [100 100 400 350])
mesh(xx(:,:,1)-L0/2, yy(:,:,1)-L/2, S2);
% xlim([min(kplot(2:end),[],'all'),max(kplot(2:end),[],'all')])
view(0,90)
axis equal tight; box on;
% clim([7,max(log(abs(E2d)),[],'all')])
xlim(250*[-1,1])
ylim(250*[-1,1])
colormap("jet");
cb = colorbar;
title(cb, '$S_2(\mathbf{r})$', 'FontSize', 14, 'Interpreter', 'latex');
xlabel('$r_x$','Interpreter','latex','FontSize',14);
ylabel('$r_y$','Interpreter','latex','FontSize',14);
xticks([-200,-100,0,100,200]);
yticks([-200,-100,0,100,200]);
% set(gca,'XScale','log','YScale','log');
% exportgraphics(gcf,hqt_output_path('figures', "figureS1b.pdf"),ContentType="Image",Resolution=600)
%% figure S3
mu=6;vy_load=-2*pi/20*7;
subFolder = hqt_data_path('raw', ['check_point_driven_mu=',num2str(mu),',v=',num2str(-vy_load*20/2/pi),'_20_large']);
figure(9);clf,theme('light')
set(gcf,'Position',[200 300 1100 350]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact')
c_blue = [0.0,0.0,0.8];  % #2166AC (blue endpoint)
c_mid  = [247, 247, 247] / 255;  % #F7F7F7 (white/gray midpoint)
c_red  = [0.8,0.0,0.0];  % #B2182B (red endpoint)
N = 256;
x = [1, round(N/2), N]; % Colormap endpoint and midpoint indices.
xi = (1:N)';            % Colormap interpolation samples.
my_cmap = interp1(x, [c_blue; c_mid; c_red], xi, 'linear');
colormap(my_cmap);
for t=[1600,2000,3000]
    fileName = fullfile(subFolder, ['check_point_t=',num2str(t),'.mat']);load(fileName)
    % Differentiate the bulk scalar; its final radial slice is the boundary order parameter.
    O1=func_Lz(Z1);
    % [n, np, nm, C] = countN_C(O1(:,:,nz))
    ax1=nexttile;
    s=surf(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),abs(O1((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz)),'FaceColor','interp' );
    s.EdgeColor = 'none';
    xlim([L0/2-L/2,L0/2+L/2]);
    view(0,90);
    colormap(my_cmap)
    clim([0,max(abs(O1(:,:,nz)),[],'all')])
    if t==3000
        cb=colorbar;
    end
    Xlabel= '$x$'; Ylabel= '$y$';
    Title=['$t=',num2str(round(t-1500)),'$'];
    xlabel(Xlabel,'Interpreter','latex','FontSize',14)
    ylabel(Ylabel,'Interpreter','latex','FontSize',14)
    title(Title,'Interpreter','latex','FontSize',14)
    % axis(ax1, 'off');
    xticks([]);yticks([]);
end
title(cb,'$\mathcal{O}$','Interpreter','latex','FontSize',14);
exportgraphics(gcf,hqt_output_path('figures', ['movie_decay.pdf']),ContentType="Image",Resolution=600)

mu=6;vy_load=-2*pi/20*0;
subFolder = hqt_data_path('raw', ['check_point_driven_mu=',num2str(mu),',v=',num2str(-vy_load*20/2/pi),'_20_lattice_large']);
figure(10);clf,theme('light')
set(gcf,'Position',[200 300 1100 350]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact')
c_blue = [0.0,0.0,0.8];  % #2166AC (blue endpoint)
c_mid  = [247, 247, 247] / 255;  % #F7F7F7 (white/gray midpoint)
c_red  = [0.8,0.0,0.0];  % #B2182B (red endpoint)
N = 256;
x = [1, round(N/2), N]; % Colormap endpoint and midpoint indices.
xi = (1:N)';            % Colormap interpolation samples.
my_cmap = interp1(x, [c_blue; c_mid; c_red], xi, 'linear');
colormap(my_cmap);
for t=[40,300,1000]
    fileName = fullfile(subFolder, ['check_point_t=',num2str(t),'.mat']);load(fileName)
    % Differentiate the bulk scalar; its final radial slice is the boundary order parameter.
    O1=func_Lz(Z1);
    % [n, np, nm, C] = countN_C(O1(:,:,nz))
    ax1=nexttile;
    s=surf(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz),abs(O1((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz)),'FaceColor','interp' );
    s.EdgeColor = 'none';
    xlim([L0/2-L/2,L0/2+L/2]);
    view(0,90);
    colormap(my_cmap)
    clim([0,max(abs(O1(:,:,nz)),[],'all')])
    if t==1000
        cb=colorbar;
    end
    Xlabel= '$x$'; Ylabel= '$y$';
    Title=['$t=',num2str(round(t)),'$'];
    xlabel(Xlabel,'Interpreter','latex','FontSize',14)
    ylabel(Ylabel,'Interpreter','latex','FontSize',14)
    title(Title,'Interpreter','latex','FontSize',14)
    % axis(ax1, 'off');
    xticks([]);yticks([]);
end
title(cb,'$\mathcal{O}$','Interpreter','latex','FontSize',14);
exportgraphics(gcf,hqt_output_path('figures', ['movie_lattice.pdf']),ContentType="Image",Resolution=600)
%% figure S4
% Load the dataset selected by this section using a repository-relative location.
load(hqt_data_path('processed', "NofV_decay.mat"))
tmin=0;tmax=1500;
figure(11),clf,theme(gcf,"light"),
set(gcf, 'Position', [100 100 400 350])
box on
plot(tmin+10:10:tmax,NofV(tmin/10+151:end))
xlim([tmin,tmax])
ylim([0,max(NofV(151:end))+10])
xlabel('$t$','FontSize',14,'Interpreter','latex');
ylabel('$N(t)$','FontSize',14,'Interpreter','latex');
exportgraphics(gcf,hqt_output_path('figures', "NofV_decay.pdf"),ContentType="Image",Resolution=600)

% Load the dataset selected by this section using a repository-relative location.
load(hqt_data_path('processed', "NofV_lattice.mat"))
tmin=0;tmax=3000;
figure(12),clf,theme(gcf,"light"),
set(gcf, 'Position', [100 100 400 350])
box on
plot(tmin+10:10:tmax,NofV(tmin/10+1:end))
xlim([tmin,tmax])
ylim([0,max(NofV)+10])
xlabel('$t$','FontSize',14,'Interpreter','latex');
ylabel('$N(t)$','FontSize',14,'Interpreter','latex');
exportgraphics(gcf,hqt_output_path('figures', "NofV_lattice.pdf"),ContentType="Image",Resolution=600)
%% figure S5
%%%%%%%%%%%%%%% Helmholtz decomposition %%%%%%%%%%%%%%%%%%%%%%%%%
rhs=func_Lx(repmat(vx(:,:,300),[1,1,nz]))+func_Ly(repmat(vy(:,:,300),[1,1,nz]));
[kx,ky]=ndgrid((-(nx-1)/2:1:(nx-1)/2)*2*pi/L0,(-(ny-1)/2:1:(ny-1)/2)*2*pi/L);
kx((nx+1)/2,(ny+1)/2)=1;ky((nx+1)/2,(ny+1)/2)=1;
rhs_fft=fftshift(fftshift(fft2(rhs),2),1);
rhs_fft((nx+1)/2,(ny+1)/2)=0;
% Remove the zero mode and solve the spatial Poisson equation mode by mode.
Phi_fft=ifftshift(ifftshift(-rhs_fft./(kx.^2+ky.^2),2),1);
Phi_hd=ifft2(Phi_fft);
VX_c=real(func_Lx(Phi_hd));VY_c=real(func_Ly(Phi_hd));
VX_c=VX_c(:,:,nz);VY_c=VY_c(:,:,nz);
VX_i=vx(:,:,300)-VX_c;VY_i=vy(:,:,300)-VY_c+2.2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(13),clf,theme(gcf,"light"),
set(gcf, 'Position', [100 100 1200 640])
tiledlayout(2,3,'TileSpacing','compact','Padding','compact')
c_blue = [0.0,0.0,0.8];  % #2166AC (blue endpoint)
c_mid  = [247, 247, 247] / 255;  % #F7F7F7 (white/gray midpoint)
c_red  = [0.8,0.0,0.0];  % #B2182B (red endpoint)
N = 256;
x = [1, round(N/2), N]; % Colormap endpoint and midpoint indices.
xi = (1:N)';            % Colormap interpolation samples.
my_cmap = interp1(x, [c_blue; c_mid; c_red], xi, 'linear');
colormap(my_cmap);
ax1=nexttile;
s=surf(xx(:,:,nz),yy(:,:,nz),vx(:,:,300),'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([0,200]);ylim([0,200]);
view(0,90);
colormap(my_cmap)
cb=colorbar;
Xlabel= '$x$'; Ylabel= '$y$';
xlabel(Xlabel,'Interpreter','latex','FontSize',14)
ylabel(Ylabel,'Interpreter','latex','FontSize',14)
title(ax1,'a','Color','w','FontSize',14)
xticks([]);yticks([]);
title(cb,'$v^s_x$','Interpreter','latex','FontSize',14);
nexttile
s=surf(xx(:,:,nz),yy(:,:,nz),VX_i,'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([0,200]);ylim([0,200]);
view(0,90);
colormap(my_cmap)
cb=colorbar;
Xlabel= '$x$'; Ylabel= '$y$';
xlabel(Xlabel,'Interpreter','latex','FontSize',14)
ylabel(Ylabel,'Interpreter','latex','FontSize',14)
xticks([]);yticks([]);
title(cb,'$(v^s_x)_i$','Interpreter','latex','FontSize',14);
nexttile
s=surf(xx(:,:,nz),yy(:,:,nz),VX_c,'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([0,200]);ylim([0,200]);
view(0,90);
colormap(my_cmap)
cb=colorbar;
Xlabel= '$x$'; Ylabel= '$y$';
xlabel(Xlabel,'Interpreter','latex','FontSize',14)
ylabel(Ylabel,'Interpreter','latex','FontSize',14)
xticks([]);yticks([]);
title(cb,'$(v^s_x)_c$','Interpreter','latex','FontSize',14);

nexttile
s=surf(xx(:,:,nz),yy(:,:,nz),vy(:,:,300)+2.2,'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([0,200]);ylim([0,200]);
view(0,90);
colormap(my_cmap)
cb=colorbar;
Xlabel= '$x$'; Ylabel= '$y$';
xlabel(Xlabel,'Interpreter','latex','FontSize',14)
ylabel(Ylabel,'Interpreter','latex','FontSize',14)
xticks([]);yticks([]);
title(cb,'$v^s_y$','Interpreter','latex','FontSize',14);
nexttile
s=surf(xx(:,:,nz),yy(:,:,nz),VY_i,'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([0,200]);ylim([0,200]);
view(0,90);
colormap(my_cmap)
cb=colorbar;
Xlabel= '$x$'; Ylabel= '$y$';
xlabel(Xlabel,'Interpreter','latex','FontSize',14)
ylabel(Ylabel,'Interpreter','latex','FontSize',14)
xticks([]);yticks([]);
title(cb,'$(v^s_y)_i$','Interpreter','latex','FontSize',14);
nexttile
s=surf(xx(:,:,nz),yy(:,:,nz),VY_c,'FaceColor','interp' );
s.EdgeColor = 'none';
xlim([0,200]);ylim([0,200]);
view(0,90);
colormap(my_cmap)
cb=colorbar;
Xlabel= '$x$'; Ylabel= '$y$';
xlabel(Xlabel,'Interpreter','latex','FontSize',14)
ylabel(Ylabel,'Interpreter','latex','FontSize',14)
xticks([]);yticks([]);
title(cb,'$(v^s_y)_c$','Interpreter','latex','FontSize',14);
exportgraphics(gcf,hqt_output_path('figures', "velocity_decomposition.pdf"),ContentType="Image",Resolution=600)
%% figure S6
mu=6;vy_load=-2*pi/20*7;
subFolder = hqt_data_path('raw', ['check_point_driven_mu=',num2str(mu),',v=',num2str(-vy_load*20/2/pi),'_20_large']);
figure(14);clf,theme('light')
set(gcf, 'Position', [100 100 1200 960])
tiledlayout(3,3,'TileSpacing','compact','Padding','compact')
for t=[1600,2000,3000]
    fileName = fullfile(subFolder, ['check_point_t=',num2str(t),'.mat']);load(fileName)
    % Differentiate the bulk scalar; its final radial slice is the boundary order parameter.
    O1=func_Lz(Z1);
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

    nexttile
    plot(kplot(2:end), Ek(2:end),'k', 'LineWidth',1.5);
    xlabel('$k$','Interpreter','latex','FontSize',14);
    ylabel('$E(k)$','Interpreter','latex','FontSize',14);
    % title('Superfluid Energy Spectrum');
    grid on;
    hold on
    % C1 = 2.5e5;  C2 = 9e5; C3=2.5e5; C4=1.5e10; % nx=401
    plot(kplot, Ek(90)/kplot(90)^(-5/3)*kplot.^(-5/3), 'r--', 'LineWidth',1.5);
    plot(kplot, Ek(400)/kplot(400)^(-3)*kplot.^(-3), '--','Color',[0,0.5,0], 'LineWidth',1.5);
    plot(kplot, Ek(70)/kplot(70)^(-1)*kplot.^(-1), 'b--', 'LineWidth',1.5);
    % plot(kplot, Ek(4)/kplot(4)*kplot.^(1), 'g--', 'LineWidth',2);
    hold off
    xlim([min(kplot(2:end)),max(kplot(2:end))])
    ylim([min(Ek(2:end)),5*max(Ek(2:end))])
    set(gca,'XScale','log','YScale','log');
    nexttile
    plot(kplot(2:end), Ekc(2:end),'k', 'LineWidth',1.5);
    xlabel('$k$','Interpreter','latex','FontSize',14);
    ylabel('$E_c(k)$','Interpreter','latex','FontSize',14);
    % title('Compressible Superfluid Energy Spectrum');
    grid on;
    hold on
    % C1 = 8e7;  C2 = 4e8; C3=1e8; % nx=401
    % plot(kplot, Ekc(90)/kplot(90)^(-5/3)*kplot.^(-5/3), 'r--');
    plot(kplot, Ekc(90)/kplot(90)^(-2)*kplot.^(-2), 'm--', 'LineWidth',1.5);
    % plot(kplot, Ekc(400)/kplot(400)^(-3)*kplot.^(-3), 'k--', 'LineWidth',2);
    % plot(kplot, Ekc(25)/kplot(25)^(-1)*kplot.^(-1), 'b--', 'LineWidth',2);
    % plot(kplot, Ekc(4)/kplot(4)*kplot.^(1), 'g--', 'LineWidth',2);
    hold off
    xlim([min(kplot(2:end)),max(kplot(2:end))])
    ylim([min(Ekc(2:end)),5*max(Ekc(2:end))])
    set(gca,'XScale','log','YScale','log');
    nexttile
    plot(kplot(2:end), Eki(2:end),'k', 'LineWidth',1.5);
    xlabel('$k$','Interpreter','latex','FontSize',14);
    ylabel('$E_i(k)$','Interpreter','latex','FontSize',14);
    % title('Incompressible Superfluid Energy Spectrum');
    grid on;
    hold on
    % C1 = 1.8e8;  C2 = 7e8; C3=1.8e8; C4=7e9; % nx=401
    plot(kplot, Eki(90)/kplot(90)^(-5/3)*kplot.^(-5/3), 'r--', 'LineWidth',1.5);
    plot(kplot, Eki(400)/kplot(400)^(-3)*kplot.^(-3), '--', 'Color',[0,0.5,0],'LineWidth',1.5);
    plot(kplot, Eki(70)/kplot(70)^(-1)*kplot.^(-1), 'b--', 'LineWidth',1.5);
    % plot(kplot, Eki(4)/kplot(4)*kplot.^(1), 'g--', 'LineWidth',2);
    hold off
    xlim([min(kplot(2:end)),max(kplot(2:end))])
    ylim([min(Eki(2:end)),5*max(Eki(2:end))])
    set(gca,'XScale','log','YScale','log');
end
exportgraphics(gcf,hqt_output_path('figures', ['spectrum_decay.pdf']),ContentType="Image",Resolution=600)
%% figure S7
mu=6;vy_load=-2*pi/20*0;
subFolder = hqt_data_path('raw', ['check_point_driven_mu=',num2str(mu),',v=',num2str(-vy_load*20/2/pi),'_20_lattice_large']);
figure(15);clf,theme('light')
set(gcf, 'Position', [100 100 1200 960])
tiledlayout(3,3,'TileSpacing','compact','Padding','compact')
for t=[300,1000,1400]
    fileName = fullfile(subFolder, ['check_point_t=',num2str(t),'.mat']);load(fileName)
    % Differentiate the bulk scalar; its final radial slice is the boundary order parameter.
    O1=func_Lz(Z1);
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

    nexttile
    plot(kplot(2:end), Ek(2:end),'k', 'LineWidth',1.5);
    xlabel('$k$','Interpreter','latex','FontSize',14);
    ylabel('$E(k)$','Interpreter','latex','FontSize',14);
    % title('Superfluid Energy Spectrum');
    grid on;
    hold on
    % C1 = 2.5e5;  C2 = 9e5; C3=2.5e5; C4=1.5e10; % nx=401
    plot(kplot, Ek(90)/kplot(90)^(-5/3)*kplot.^(-5/3), 'r--', 'LineWidth',1.5);
    plot(kplot, Ek(400)/kplot(400)^(-3)*kplot.^(-3), '--','Color',[0,0.5,0], 'LineWidth',1.5);
    plot(kplot, Ek(70)/kplot(70)^(-1)*kplot.^(-1), 'b--', 'LineWidth',1.5);
    % plot(kplot, Ek(4)/kplot(4)*kplot.^(1), 'g--', 'LineWidth',2);
    hold off
    xlim([min(kplot(2:end)),max(kplot(2:end))])
    ylim([min(Ek(2:end)),5*max(Ek(2:end))])
    set(gca,'XScale','log','YScale','log');
    nexttile
    plot(kplot(2:end), Ekc(2:end),'k', 'LineWidth',1.5);
    xlabel('$k$','Interpreter','latex','FontSize',14);
    ylabel('$E_c(k)$','Interpreter','latex','FontSize',14);
    % title('Compressible Superfluid Energy Spectrum');
    grid on;
    hold on
    % C1 = 8e7;  C2 = 4e8; C3=1e8; % nx=401
    % plot(kplot, Ekc(90)/kplot(90)^(-5/3)*kplot.^(-5/3), 'r--');
    plot(kplot, Ekc(90)/kplot(90)^(-2)*kplot.^(-2), 'm--', 'LineWidth',1.5);
    % plot(kplot, Ekc(400)/kplot(400)^(-3)*kplot.^(-3), 'k--', 'LineWidth',2);
    % plot(kplot, Ekc(25)/kplot(25)^(-1)*kplot.^(-1), 'b--', 'LineWidth',2);
    % plot(kplot, Ekc(4)/kplot(4)*kplot.^(1), 'g--', 'LineWidth',2);
    hold off
    xlim([min(kplot(2:end)),max(kplot(2:end))])
    ylim([min(Ekc(2:end)),5*max(Ekc(2:end))])
    set(gca,'XScale','log','YScale','log');
    nexttile
    plot(kplot(2:end), Eki(2:end),'k', 'LineWidth',1.5);
    xlabel('$k$','Interpreter','latex','FontSize',14);
    ylabel('$E_i(k)$','Interpreter','latex','FontSize',14);
    % title('Incompressible Superfluid Energy Spectrum');
    grid on;
    hold on
    % C1 = 1.8e8;  C2 = 7e8; C3=1.8e8; C4=7e9; % nx=401
    plot(kplot, Eki(90)/kplot(90)^(-5/3)*kplot.^(-5/3), 'r--', 'LineWidth',1.5);
    plot(kplot, Eki(400)/kplot(400)^(-3)*kplot.^(-3), '--', 'Color',[0,0.5,0],'LineWidth',1.5);
    plot(kplot, Eki(70)/kplot(70)^(-1)*kplot.^(-1), 'b--', 'LineWidth',1.5);
    % plot(kplot, Eki(4)/kplot(4)*kplot.^(1), 'g--', 'LineWidth',2);
    hold off
    xlim([min(kplot(2:end)),max(kplot(2:end))])
    ylim([min(Eki(2:end)),5*max(Eki(2:end))])
    set(gca,'XScale','log','YScale','log');
end
exportgraphics(gcf,hqt_output_path('figures', ['spectrum_lattice.pdf']),ContentType="Image",Resolution=600)
%% structure factor
for kk=1
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

    % Sc_1d=Sc_1d/((tmax-tmin)/dt+1);Sn_1d=Sn_1d/((tmax-tmin)/dt+1);
    x_v=vortices(:,2);y_v=vortices(:,1);charges=vortices(:,3);

    std_c=std(Sc_1d_tol(101:end,2:end));std_n=std(Sn_1d_tol(101:end,2:end));

    [centroids, cluster_info] = find_vortex_dpc_pbc(vortices, 500,500,sqrt(500^2/length(x_v)));
    distance=cluster_info.mean_nnd_cluster;

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

    figure(16),theme('light'),clf
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

    nexttile
    hold on;
    % plot(k_bin(2:end), Sc_1d(2:end), 'r-o', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', '$S_c(k)$'); hold on;
    % plot(k_bin(2:end), Sn_1d(2:end), 'b--o', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', '$S_n(k)$');
    k_data  = k_bin(2:end);
    Sc_data = Sc_1d(2:end);
    Sn_data = Sn_1d(2:end);
    err_c   = std_c(1:end);
    err_n   = std_n(1:end);
    N = length(k_data);
    idx = unique(round(logspace(0, log10(N), 50)));
    plot(k_data, Sc_data, 'r-', 'LineWidth', 1, 'HandleVisibility', 'off');
    plot(k_data, Sn_data, 'b-', 'LineWidth', 1, 'HandleVisibility', 'off');
    errorbar(k_data(idx), Sc_data(idx), err_c(idx), 'ro', 'LineWidth', 1, 'MarkerSize', 4, 'DisplayName', '$S_c(k)$');
    errorbar(k_data(idx), Sn_data(idx), err_n(idx), 'bo', 'LineWidth', 1, 'MarkerSize', 4, 'DisplayName', '$S_n(k)$');

    plot(k_bin(1:10),exp(a1(1))*k_bin(1:10).^2.5,'k', 'LineWidth', 1.2, 'DisplayName', ['$k^{',num2str(2.5,2),'}$'])
    xline(2*pi/distance, 'k--', 'LineWidth', 1, 'DisplayName', '$k_{d}$','Interpreter','latex');
    grid on; box on;
    set(gca,'XScale','log','YScale','log');
    ylim([0.3*min([Sc_1d(2:end),Sn_1d(2:end)]),1.5*max([Sc_1d(2:end),Sn_1d(2:end)])])
    % title('1D Structure Factor $S(k)$','Interpreter','latex','FontSize',12);
    xlabel('$k$','Interpreter','latex','FontSize',12); ylabel('$S(k)$','Interpreter','latex','FontSize',12);
    legend('Location', 'best','FontSize',12,'Interpreter','latex');
    % axis(ax1, 'off');

    % exportgraphics(gcf,hqt_output_path('figures', ['structure_factor_mu=',num2str(mu),',v=',num2str(-vy_load*20/2/pi),'_20.pdf']),ContentType="Image",Resolution=300)
end
% [centroids, cluster_info] = find_vortex_dpc_pbc(vortices, 500,500,10);
% exportgraphics(gcf,hqt_output_path('figures', "vortex_cluster.pdf"),ContentType="Image",Resolution=600)
%% movie
for kk=3
    if kk==1
        mu=6;vy_load=-2*pi/20*7;
        % Load the dataset selected by this section using a repository-relative location.
        load(hqt_data_path('processed', 'driven_2.mat'))
    elseif kk==2
        mu=8;vy_load=-2*pi/20*13.5;
        % Load the dataset selected by this section using a repository-relative location.
        load(hqt_data_path('processed', ['driven_mu=',num2str(mu),',v=',num2str(-vy_load*20/2/pi),'_20']))
    elseif kk==3
        mu=5;vy_load=-2*pi/20*3.8;
        % Load the dataset selected by this section using a repository-relative location.
        load(hqt_data_path('processed', ['driven_mu=',num2str(mu),',v=',num2str(-vy_load*20/2/pi),'_20']))
    end

    O0=max(abs(O(:)));

    fig1 = figure(1);
    theme(gcf,"light");
    set(fig1, 'Position', [100 200 800 600]);
    ax3 = gca;

    c_blue = [0.0, 0.0, 0.8];       % #2166AC (blue endpoint)
    c_mid  = [247, 247, 247] / 255; % #F7F7F7 (white/gray midpoint)
    c_red  = [0.8, 0.0, 0.0];       % #B2182B (red endpoint)
    N = 256;
    x = [1, round(N/2), N];         % Colormap endpoint and midpoint indices.
    xi = (1:N)';                    % Colormap interpolation samples.
    my_cmap = interp1(x, [c_blue; c_mid; c_red], xi, 'linear');
    colormap(ax3, my_cmap);

    % ==================== 1. Initialize VideoWriter. ====================
    mp4name = hqt_output_path('movies', ['movie', num2str(kk), '.mp4']);   % MP4 output path.                     % Frame delay in seconds.
    fps = round(20);                 % Video frame rate (20 fps).

    % Configure the movie writer; closing it later finalizes the encoded frames.
    v = VideoWriter(mp4name, 'MPEG-4');         % Create the MPEG-4 video writer.
    v.FrameRate = fps;                          % Set the video frame rate.
    v.Quality = 100;                             % Set video quality (1-100).
    open(v);                                    % Open the writer.
    % ===============================================================

    tmin = 0;
    tmax = 2000;
    dt = 10;
    Nx = (nx-1)/2;

    for t = tmin+10 : 10 : tmax
        t
        O1 = O(:,:,round(t/dt));
        set(0, "CurrentFigure", fig1);

        s = surf(ax3, xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz), ...
            yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz), ...
            abs(O1((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,1)));
        s.EdgeColor = 'none';
        shading(ax3, 'interp'); % Interpolate surface colors.
        colormap(ax3, my_cmap);

        view(0,90);
        xticks([]); yticks([]);
        xlabel('$x$','Interpreter','latex', 'FontSize', 14);
        ylabel('$y$','Interpreter','latex', 'FontSize', 14);
        title(['$t=',num2str(t),'$'],'Interpreter','latex', 'FontSize', 14);

        h_cb1 = colorbar;
        h_cb1.FontSize = 12;
        title(h_cb1, '$\mathcal{O}$', 'FontSize', 14, 'Interpreter', 'latex');
        clim([0,O0]);
        % clim([0,11]);
        h_cb1.TickLabelInterpreter = 'latex';
        axis(ax3, 'off');
        axis(ax3, 'tight');   % Fit the axes tightly to the data.
        axis(ax3, 'equal');   % Axes and label spacing.
        drawnow;

        % ==================== 2. Write MP4 frames. ====================
        frame = getframe(fig1);
        writeVideo(v, frame);  % Write the captured figure frame.
        % =======================================================
    end

    % ==================== 3. Close the video stream. ====================
    close(v);  % Finalize the file before playback.
    % ===========================================================
end

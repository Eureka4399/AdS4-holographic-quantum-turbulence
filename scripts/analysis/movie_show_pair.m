% MOVIE_SHOW_PAIR Historical checkpoint analysis and plotting notebook.
% Call setup_project first, then run selected MATLAB sections in dependency order.
% Checkpoint inputs under data/raw/ are intentionally omitted from Git.
% Set grid, domain, time range, and run-folder suffix to match the generated run.
% Figures/movies go to outputs/; processed arrays go to data/processed/.

%% movie show
enable_gpu
nx=101; ny=nx; nz=20; nz0=30;
% Build periodic spectral differentiation matrices on the spatial grid.
[x,Dx]=fourdif(nx,1);
[y,Dy]=fourdif(ny,1);
[z,Dz]=chebdif(nz,1);
[z0,Dz0]=chebdif(nz0,1);
% Use x/y/z array ordering before mapping the collocation points to physical coordinates.
[xxcheb,yycheb,zzcheb]=ndgrid(x,y,z);         %Use ndgrid ordering: the first index represents x.

lx=Dx; ly=Dy;  lz=Dz;
small=gpuArray(0);        % the distance to horizon;
D=gpuArray(0);          % the distance to infinite boundary
L=gpuArray((nx-1)/4);
L0=L;

% Construct the Fourier-mode cutoff used to suppress unresolved spatial modes.
fx=false(nx,1,1);
fx(round(round(nx/3)):round(nx/3*2))=1;
fy=false(1,ny,1);
fy(round(round(ny/3)):round(ny/3*2))=1;
fxyz=repmat(fx|fy,[1,1,nz]);

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
Z=(z0-1)*dlz/2+zf;
%---define real Derivative with coefficients
Lx=px*lx; %Lxx=px.^2*lxx;
Ly=py*ly; %Lyy=py.^2*lyy;
Lz=pz*lz; %Lzz=pz.^2*lzz;
Lz0=pz*Dz0;
Lz1=Lz(1:nz-1,1:nz-1)\eye(nz-1);
Lz2=Lz(2:nz,2:nz)\eye(nz-1);
LLz=Lz^2;Lzz=LLz(2:nz-1,2:nz-1)\eye(nz-2);
xy0=zeros(1,ny*nx);
xy1=ones(1,ny*nx);
% END OF UNIVERSAL BASICS

% Define derivatives and radial integration operators with boundary conditions.
func_Lx=@(x) reshape(Lx*reshape(x,[nx,ny*nz]),[nx,ny,nz]);      % Apply the x derivative along the first array dimension.
func_Ly=@(x) permute(reshape(Ly*reshape(permute(x,[2,1,3]),[ny,nx*nz]),[ny,nx,nz]),[2,1,3]); % Apply the y derivative along the second array dimension.
func_Lz=@(x) permute(reshape(Lz*reshape(permute(x,[3,2,1]),[nz,ny*nx]),[nz,ny,nx]),[3,2,1]);     % Apply the radial derivative along the z dimension.
f1=@(x) reshape(permute(x,[3,2,1]),[nz,ny*nx]);
f2=@(x) permute(reshape(x,[nz,ny,nx]),[3,2,1]);
func_Lzr=@(x) f2([[Lz1,zeros(nz-1,1)]*f1(x);xy0]);
func_Lzr2=@(x) f2([xy0;[zeros(nz-1,1),Lz2]*f1(x)]);

%Define the scalar and gauge-field evolution operators.
zh=1;e=1;
f=1-(zz/zh).^3;df=-3*zz.^2/zh^3;
F1=@(y1,y2,y3,y4) func_Lzr(1i.*y2.*(func_Lz(y1))+0.5i.*y1.*func_Lz(y2)+0.5.*(1-(zz/zh).^3).*func_Lz(func_Lz(y1))-0.5.*3.*zz.^2./zh^3.*func_Lz(y1)-0.5.*zz.*y1-0.5.*y3.*y3.*y1-0.5.*y4.*y4.*y1+1.*0.5.*func_Lx(func_Lx(y1))+ ...
    1.*0.5.*func_Ly(func_Ly((y1)))-1.0.*0.5i.*y1.*(func_Lx(y3)+func_Ly(y4))-1i.*(y3.*func_Lx(y1)+y4.*func_Ly(y1)));
F3=@(y1,y2,y3,y4) real(func_Lzr(0.5.*func_Lz(func_Lx(y2)+(1-(zz/zh).^3).*func_Lz(y3))+0.5.*func_Ly(func_Ly(y3)-func_Lx(y4))-0.5i.*(conj(y1).*func_Lx(y1)-y1.*func_Lx(conj(y1)))-y3.*y1.*conj(y1)));
F4=@(y1,y2,y3,y4) real(func_Lzr(0.5.*func_Lz(func_Ly(y2)+(1-(zz/zh).^3).*func_Lz(y4))+0.5.*func_Lx(-func_Ly(y3)+func_Lx(y4))-0.5i.*(conj(y1).*func_Ly(y1)-y1.*func_Ly(conj(y1)))-y4.*y1.*conj(y1)));
F2=@(y1,y2,y3,y4) (1-zz.^3).*func_Lz(func_Lx(y3)+func_Ly(y4));
Constraint=@(y1,y2,y3,y4) func_Lz(func_Lx(y3)+func_Ly(y4)-func_Lz(y2))-2*imag((conj(y1).*func_Lz(y1)));

for mu=7.5
    % mu=3+0.05*kk;
    % subFolder = hqt_data_path('raw', ['check_point_pair_mu=',num2str(mu),',v=',num2str(0.4),'_20']);
    subFolder = hqt_data_path('raw', ['check_point_pair_mu=',num2str(mu),',v=',num2str(0),'_20']);

    fig1=figure(1); theme(gcf,"light")
    % fig2=figure(2); theme(gcf,"light"), clf, hold on, box on
    % fig3=figure(3); theme(gcf,"light"), clf, hold on, box on
    set(fig1,'Position',[800 400 800 320]);

    gifname = hqt_output_path('movies', ['pair_mu=',num2str(mu),'.gif']);   % GIF output path.
    delaytime = 0.033;            % Frame delay in seconds.
    firstFrame = true;

    tmin=0;
    % tmax=800;
    if mu==5
        tmax=1000;
    elseif mu==6
        tmax=1475;
    elseif mu==7.5
        tmax=2460;
    else
        mu
        pause
    end
    dt=1;
    vortices=nan(round(tmax/dt),2,2);
    antivortices=nan(round(tmax/dt),2,2);
    % for t=tmin+dt:dt:tmax
    for t=1:775
        t
        fileName = fullfile(subFolder, ['check_point_t=',num2str(t),'.mat']);load(fileName)
        % Differentiate the bulk scalar; its final radial slice is the boundary order parameter.
        O1=func_Lz(Z1);
        [vortices_temp, antivortices_temp] = find_vortices(reshape(O1(:,:,end),[nx,ny]),0.5,5);
        if isempty([vortices_temp, antivortices_temp])==0
            if length(vortices_temp(:))==4
                [vortices(round(t/dt),:,:), antivortices(round(t/dt),:,:)]=deal(vortices_temp, antivortices_temp);
            elseif length(vortices_temp(:))==2
                [vortices(round(t/dt),1,:), antivortices(round(t/dt),1,:)]=deal(vortices_temp, antivortices_temp);
            end
        end
        set(0,"CurrentFigure",fig1)
        clf
        tiledlayout(1,2,'TileSpacing','compact')%,'Padding','compact'
        nexttile
        s=surf(xx(:,:,nz),yy(:,:,nz),abs(O1(:,:,nz)),'FaceColor','interp' );
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
        ss=mesh(xx(:,:,nz),yy(:,:,nz),angle(O1(:,:,nz)),'FaceColor','interp' );
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
    end
    % save(hqt_data_path('processed', ['pair_mu=',num2str(mu),'.mat']),'vortices','antivortices')
    save(hqt_data_path('processed', ['pair_mu=',num2str(mu),'_2.mat']),'vortices','antivortices')
end
%% vortex distance one pair
mu=5;
% load(hqt_data_path('processed', ['pair_mu=',num2str(mu),'.mat']));vortices=vortices/4;antivortices=antivortices/4;
% Load the dataset selected by this section using a repository-relative location.
load(hqt_data_path('processed', ['pair_mu=',num2str(mu),'_2.mat']));vortices=vortices/4;antivortices=antivortices/4;
% vortices=track_vortices(vortices);antivortices=track_vortices(antivortices);

L=25;
dy1 = vortices(:, 1, 1) - antivortices(:, 1, 1);
dy1 = dy1 - L * round(dy1 / L);
distance1 = abs(dy1);
distance1(isnan(distance1)) = [];
distance1(distance1==0) = [];

dt=1;
t1=length(distance1);
T1=1:dt:t1;
% T1=t1-T0:dt:t1;
f=@(a,t) a(1)*(a(2)-t).^0.5;
a01=[1,t1];
lb1 = [-Inf, max(T1)];ub1 = [Inf, Inf];
opts = optimoptions('lsqcurvefit', 'FunctionTolerance', 1e-13);
% Fit the selected vortex-separation model within the explicit parameter bounds.
a1 = lsqcurvefit(f, a01, T1, gather(distance1(round(min(T1)/dt):round(max(T1)/dt))).', lb1, ub1, opts);
figure(2),theme(gcf,"light"),set(gcf,'Position',[100 100 480 400]);
clf,box on,hold on
plot(T1,distance1(round(min(T1)/dt):round(max(T1)/dt)).','r.')
plot(T1,f(a1,T1),'b-');
hold off
xlabel('t')
ylabel('distance')
% Construct the annotation text.
strings = ['$',num2str(a1(1)),'\times\sqrt{',num2str(a1(2)),'-t}$'];
% Add the text annotation.
annotation('textbox', [0.2, 0.1, 0.3, 0.2], 'LineStyle', '-', 'LineWidth', 2, 'String', strings,'Interpreter','latex','fontsize',14,'EdgeColor','none')
exportgraphics(gcf,hqt_output_path('figures', ['distance_mu=',num2str(mu),'.pdf']),ContentType="Image",Resolution=600)
figure(3),theme(gcf,"light"),set(gcf,'Position',[600 100 480 400]);
clf,box on,hold on
plot(abs(T1-a1(2)),distance1(round(min(T1)/dt):round(max(T1)/dt)),'r.')
plot(abs(T1-a1(2)),f(a1,T1),'b-');
hold off
xlabel('t')
ylabel('distance')
set(gca,'XScale','log','YScale','log');
% Construct the annotation text.
strings = ['$',num2str(a1(1)),'\times\sqrt{',num2str(a1(2)),'-t}$'];
% Add the text annotation.
annotation('textbox', [0.3, 0.15, 0.3, 0.2], 'LineStyle', '-', 'LineWidth', 2, 'String', strings,'Interpreter','latex','fontsize',14,'EdgeColor','none')
exportgraphics(gcf,hqt_output_path('figures', ['distance_log_mu=',num2str(mu),'.pdf']),ContentType="Image",Resolution=600)
figure(4),theme(gcf,"light"),set(gcf,'Position',[1100 100 480 400]);
clf,box on,hold on
plot(vortices(:,1,1),vortices(:,1,2),'.')
plot(antivortices(:,1,1),antivortices(:,1,2),'.')
hold off
xlim([0,L])
ylim([0,L])
xlabel('x')
ylabel('y')
exportgraphics(gcf,hqt_output_path('figures', ['trajectory_mu=',num2str(mu),'.pdf']),ContentType="Image",Resolution=600)
%% vortex distance two pairs
mu=7.5;
% Load the dataset selected by this section using a repository-relative location.
load(hqt_data_path('processed', ['pair_mu=',num2str(mu),'.mat']));vortices=vortices/4;antivortices=antivortices/4;
% load(hqt_data_path('processed', ['pair_mu=',num2str(mu),'_2.mat']));vortices=vortices/4;antivortices=antivortices/4;
vortices=track_vortices(vortices);antivortices=track_vortices(antivortices);

L=50;
dy1 = vortices(:, 1, 1) - antivortices(:, 1, 1);
dy1 = dy1 - L * round(dy1 / L);
distance1 = (abs(dy1));
distance1(isnan(distance1)) = [];
dy2 = vortices(:, 2, 1) - antivortices(:, 2, 1);
dy2 = dy2 - L * round(dy2 / L);
distance2 = (abs(dy2));
distance2(isnan(distance2)) = [];

t1=length(distance1);t2=length(distance2);dt=1;
% T1=1:dt:t1;T2=1:dt:t2;
T1=t1-400:dt:t1;T2=t2-400:dt:t2;
% T1=t1-400:dt:t1-100;T2=t2-400:dt:t2-100;
f=@(a,t) a(1)*(a(2)-t).^0.5;
a01=[1,t1];a02=[1,t2];
% opts = statset('nlinfit');opts.TolFun=1e-13;
% a1 = nlinfit(T,distance(round(min(T)/dt):round(max(T)/dt)).',f,a0);
lb1 = [-Inf, max(T1)];ub1 = [Inf, Inf];
lb2 = [-Inf, max(T2)];ub2 = [Inf, Inf];
opts = optimoptions('lsqcurvefit', 'FunctionTolerance', 1e-13);
% Fit the selected vortex-separation model within the explicit parameter bounds.
a1 = lsqcurvefit(f, a01, T1, distance1(round(min(T1)/dt):round(max(T1)/dt)).', lb1, ub1, opts);
a2 = lsqcurvefit(f, a02, T2, distance2(round(min(T2)/dt):round(max(T2)/dt)).', lb2, ub2, opts);
figure(2),theme(gcf,"light"),clf,box on,hold on
plot(T1,distance1(round(min(T1)/dt):round(max(T1)/dt)),'r.')
plot(T1,f(a1,T1),'r-');
plot(T2,distance2(round(min(T2)/dt):round(max(T2)/dt)),'b.')
plot(T2,f(a2,T2),'b-');
hold off
xlabel('t')
ylabel('distance')
% exportgraphics(gcf,hqt_output_path('figures', ['distance_mu=',num2str(mu),'.pdf']),ContentType="Image",Resolution=600)
figure(3),theme(gcf,"light"),clf,box on,hold on
plot(abs(T1-a1(2)),distance1(round(min(T1)/dt):round(max(T1)/dt)),'r.')
plot(abs(T1-a1(2)),f(a1,T1),'r-');
plot(abs(T2-a2(2)),distance2(round(min(T2)/dt):round(max(T2)/dt)),'b.')
plot(abs(T2-a2(2)),f(a2,T2),'b-');
hold off
set(gca,'XScale','log','YScale','log');
exportgraphics(gcf,hqt_output_path('figures', ['distance_log_mu=',num2str(mu),'.pdf']),ContentType="Image",Resolution=600)
figure(4),theme(gcf,"light"),clf,box on,hold on
plot(vortices(:,1,1),vortices(:,1,2),'.')
plot(antivortices(:,2,1),antivortices(:,2,2),'.')
plot(vortices(:,2,1),vortices(:,2,2),'.')
plot(antivortices(:,1,1),antivortices(:,1,2),'.')
hold off
xlim([0,50])
ylim([0,50])
xlabel('x')
ylabel('y')
% exportgraphics(gcf,hqt_output_path('figures', ['trajectory_mu=',num2str(mu),'.pdf']),ContentType="Image",Resolution=600)


% STATISTICS Historical checkpoint-based spectral diagnostics.
% Run this script by full path after setup_project; it is excluded from the MATLAB path
% to avoid conflicts with functions named statistics. The old checkpoints are omitted.

% Enable compatibility before creating or loading GPU-backed fields.
enable_gpu

nx=401; ny=401; nz=20;
% [x,Dx,D6]=FD(nx,1);
% Build periodic spectral differentiation matrices on the spatial grid.
[x,Dx]=fourdif(nx,1);
[y,Dy]=fourdif(ny,1);
% [y,Dy]=FD(ny,1);
[z,Dz]=chebdif(nz,1);
%[zzcheb,yycheb]=meshgrid(z,y);      %meshgrid ordering differs from the ndgrid convention used below.
% Use x/y/z array ordering before mapping the collocation points to physical coordinates.
[xxcheb,yycheb,zzcheb]=ndgrid(x,y,z);         %Use ndgrid ordering: the first index represents x.

lx=Dx; ly=Dy;  lz=Dz;
small=0;        % the distance to horizon;
D=0;          % the distance to infinite boundary
L=200;L0=L;
% L=90;L0=90;

%---transform to real coordinates
xi=0; xf=L0;               % INPUT of end points of x-direction
yi=0; yf=L;                    % INPUT of end points of y-direction
zi=D; zf=1-small;               % INPUT of end points of z-direction
dlz=zf-zi;
px=2*pi/L0;%2/dlx;
py=2.*pi/L;%2/dly;
pz=2/dlz;

% xx=(xxcheb-1)*L0/2+xf*ones(nx,ny,nz); %coordinate trans. to real coordinate
xx=(xxcheb*L0)/2/pi;
yy=(yycheb*L)/2/pi; %coordinate trans. to real coordinate
zz=(zzcheb-1)*dlz/2+zf*ones(nx,ny,nz); %coordinate trans. to real coordinate
%---define real Derivative with coefficients
Lx=px*lx; %Lxx=px.^2*lxx;
Ly=py*ly; %Lyy=py.^2*lyy;
Lz=pz*lz; %Lzz=pz.^2*lzz;
Lz1=Lz(1:nz-1,1:nz-1)\eye(nz-1);
Lz2=Lz(2:nz,2:nz)\eye(nz-1);
LLz=Lz^2;Lzz=LLz(2:nz-1,2:nz-1)\eye(nz-2);
xy0=zeros(1,ny*nx);
xy1=ones(1,ny*nx);
func_Lx=@(x) reshape(Lx*reshape(x,[nx,ny*nz]),[nx,ny,nz]);      % Apply the x derivative along the first array dimension.
func_Ly=@(x) permute(reshape(Ly*reshape(permute(x,[2,1,3]),[ny,nx*nz]),[ny,nx,nz]),[2,1,3]); % Apply the y derivative along the second array dimension.
func_Lz=@(x) permute(reshape(Lz*reshape(permute(x,[3,2,1]),[nz,ny*nx]),[nz,ny,nx]),[3,2,1]);     % Apply the radial derivative along the z dimension.
f1=@(x) reshape(permute(x,[3,2,1]),[nz,ny*nx]);
f2=@(x) permute(reshape(x,[nz,ny,nx]),[3,2,1]);
func_Lzr=@(x) f2([[Lz1,zeros(nz-1,1)]*f1(x);xy0]);
func_Lzr2=@(x) f2([xy0;[zeros(nz-1,1),Lz2]*f1(x)]);
% END OF UNIVERSAL BASICS

Nx=200;
% for k=100:100:3400
for k=3400
    % Load the dataset selected by this section using a repository-relative location.
    load(hqt_data_path('raw', ['check_mu=6_v=3_t=',num2str(k),'.mat']))
    % Differentiate the bulk scalar; its final radial slice is the boundary order parameter.
    O1=func_Lz(Z1);
    O1=gather(O1);
%     VX_temp=(-1i*(func_Lx(O1)-func_Lx(abs(O1)).*(O1./abs(O1))));VX_temp=gather(VX_temp-mean(VX_temp(:,:,nz),'all'));
%     VY_temp=(-1i*(func_Ly(O1)-func_Ly(abs(O1)).*(O1./abs(O1))));VY_temp=gather(VY_temp-mean(VY_temp(:,:,nz),'all'));
%     VX_temp=-1i*func_Lx(O1./abs(O1)).*conj(O1);VX_temp=gather(VX_temp-mean(VX_temp(:,:,nz),'all'));
%     VY_temp=-1i*func_Ly(O1./abs(O1)).*conj(O1);VY_temp=gather(VY_temp-mean(VY_temp(:,:,nz),'all'));
%     VX_temp=0.5i*(func_Lx(O1).*conj(O1)-func_Lx(conj(O1)).*O1)./abs(O1).^2;VX_temp=gather(VX_temp-mean(VX_temp(:,:,nz),'all'));
%     VY_temp=0.5i*(func_Ly(O1).*conj(O1)-func_Ly(conj(O1)).*O1)./abs(O1).^2;VY_temp=gather(VY_temp-mean(VY_temp(:,:,nz),'all'));
%     VX_temp=0.5i*(func_Lx(O1).*conj(O1)-func_Lx(conj(O1)).*O1)./abs(O1).^2.*O1;VX_temp=gather(VX_temp-mean(VX_temp(:,:,nz),'all'));
%     VY_temp=0.5i*(func_Ly(O1).*conj(O1)-func_Ly(conj(O1)).*O1)./abs(O1).^2.*O1;VY_temp=gather(VY_temp-mean(VY_temp(:,:,nz),'all'));
%     O1=reshape(gather(O1(:,:,nz)),[nx,ny]);
    VX_temp=-real(func_Lz(Z3));VX_temp=gather(VX_temp-mean(VX_temp(:,:,nz),'all'));
    VY_temp=-real(func_Lz(Z4));VY_temp=gather(VY_temp-mean(VY_temp(:,:,nz),'all'));
%     [NofV,~,~]=countN(O1(:,:,nz));
    %%%%% Helmholtz decomposition %%%%%%%%%%%%%%%%%%%%%%%%%
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
    % Subtract the longitudinal component to obtain the incompressible velocity.
    VX_i=VX_temp-VX_c;VY_i=VY_temp-VY_c;
%     A=gather(func_Lx(VX_i)+func_Ly(VY_i));
%     B=gather(func_Ly(VX_c)-func_Lx(VY_c));
    A=gather(func_Lx(VX_temp)+func_Ly(VY_temp));
    B=gather(func_Ly(VX_temp)-func_Lx(VY_temp));
    figure(5)
    tiledlayout(1,2,'TileSpacing','compact')%,'Padding','compact'
    nexttile
    mesh(abs(A(:,:,nz)))
    colorbar
    nexttile
    mesh(abs(B(:,:,nz)))
    colorbar
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    figure(2),
    clf
    tiledlayout(4,2,'TileSpacing','compact')%,'Padding','compact'
    nexttile
    s=surf(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz),abs(O1((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz)),'FaceColor','interp' );
    s.EdgeColor = 'none';
    xlim([L0/2-L/2,L0/2+L/2]);
    view(0,90);
    colorbar;
    Xlabel= '$x$'; Ylabel= '$y$';
    Title=['$\mathcal{O}$, t=',num2str(k)];
    xlabel(Xlabel,'Interpreter','latex')
    ylabel(Ylabel,'Interpreter','latex')
    title(Title,'Interpreter','latex')
    nexttile
    ss=mesh(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz),angle(O1((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz)),'FaceColor','interp' );
    ss.EdgeColor = 'none';
    xlim([L0/2-L/2,L0/2+L/2]);
    view(0,90);
    colorbar;
    Xlabel= '$x$'; Ylabel= '$y$';
    Title=['$\theta$, t=',num2str(k)];
    xlabel(Xlabel,'Interpreter','latex')
    ylabel(Ylabel,'Interpreter','latex')
    title(Title,'Interpreter','latex')
    nexttile
    ss=mesh(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz),real(VX_temp((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz)),'FaceColor','interp' );
    ss.EdgeColor = 'none';
    xlim([L0/2-L/2,L0/2+L/2]);
    view(0,90);
    colorbar;
    Xlabel= '$x$'; Ylabel= '$y$';
    Title=['$J_x$'];
    xlabel(Xlabel,'Interpreter','latex','fontsize',14)
    ylabel(Ylabel,'Interpreter','latex','fontsize',14)
    title(Title,'Interpreter','latex','fontsize',14)
    nexttile
    s=surf(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz),real(VY_temp((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz)),'FaceColor','interp' );
    s.EdgeColor = 'none';
    xlim([L0/2-L/2,L0/2+L/2]);
    view(0,90);
    colorbar;
    Xlabel= '$x$'; Ylabel= '$y$';
    Title=['$J_y$'];
    xlabel(Xlabel,'Interpreter','latex','fontsize',14)
    ylabel(Ylabel,'Interpreter','latex','fontsize',14)
    title(Title,'Interpreter','latex','fontsize',14)
    nexttile
    ss=mesh(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz),real(VX_c((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz)),'FaceColor','interp' );
    ss.EdgeColor = 'none';
    xlim([L0/2-L/2,L0/2+L/2]);
    view(0,90);
    colorbar;
    Xlabel= '$x$'; Ylabel= '$y$';
    Title=['$J_x^c$'];
    xlabel(Xlabel,'Interpreter','latex','fontsize',14)
    ylabel(Ylabel,'Interpreter','latex','fontsize',14)
    title(Title,'Interpreter','latex','fontsize',14)
    nexttile
    s=surf(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz),real(VY_c((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz)),'FaceColor','interp' );
    s.EdgeColor = 'none';
    xlim([L0/2-L/2,L0/2+L/2]);
    view(0,90);
    colorbar;
    Xlabel= '$x$'; Ylabel= '$y$';
    Title=['$J_y^c$'];
    xlabel(Xlabel,'Interpreter','latex','fontsize',14)
    ylabel(Ylabel,'Interpreter','latex','fontsize',14)
    title(Title,'Interpreter','latex','fontsize',14)
    nexttile
    ss=mesh(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz),real(VX_i((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz)),'FaceColor','interp' );
    ss.EdgeColor = 'none';
    xlim([L0/2-L/2,L0/2+L/2]);
    view(0,90);
    colorbar;
    Xlabel= '$x$'; Ylabel= '$y$';
    Title=['$J_x^i$'];
    xlabel(Xlabel,'Interpreter','latex','fontsize',14)
    ylabel(Ylabel,'Interpreter','latex','fontsize',14)
    title(Title,'Interpreter','latex','fontsize',14)
    nexttile
    s=surf(xx((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz),yy((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz),real(VY_i((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,(nx-1)/2-Nx+1:(nx-1)/2+Nx+1,nz)),'FaceColor','interp' );
    s.EdgeColor = 'none';
    xlim([L0/2-L/2,L0/2+L/2]);
    view(0,90);
    colorbar;
    Xlabel= '$x$'; Ylabel= '$y$';
    Title=['$J_y^i$'];
    xlabel(Xlabel,'Interpreter','latex','fontsize',14)
    ylabel(Ylabel,'Interpreter','latex','fontsize',14)
    title(Title,'Interpreter','latex','fontsize',14)

    n_i=2;n_p=5;
    figure(3),
    clf
    tiledlayout((n_p-n_i+1),2,'TileSpacing','compact')%,'Padding','compact'
    figure(5),
    clf
    tiledlayout((n_p-n_i+1),6,'TileSpacing','compact')%,'Padding','compact'
    %     e_p=zeros(1,n_p-n_i+1);
    for p=n_i:n_p
        nx_max=100;ny_max=nx_max;
        DVX=zeros(1,nx_max);DVY=zeros(1,ny_max);DVX3=zeros(1,nx_max);DVY3=zeros(1,ny_max);
        VX_temp2=zeros(size(VX_temp));VY_temp2=zeros(size(VY_temp));
        DVX_c=zeros(1,nx_max);DVY_c=zeros(1,ny_max);DVX3_c=zeros(1,nx_max);DVY3_c=zeros(1,ny_max);
        VX_c2=zeros(size(VX_c));VY_c2=zeros(size(VY_c));
        DVX_i=zeros(1,nx_max);DVY_i=zeros(1,ny_max);DVX3_i=zeros(1,nx_max);DVY3_i=zeros(1,ny_max);
        VX_i2=zeros(size(VX_i));VY_i2=zeros(size(VY_i));
        for k1=1:nx_max
            VX_temp2(1:k1,:)=VX_temp(end-k1+1:end,:);
            VX_temp2(k1+1:end,:)=VX_temp(1:end-k1,:);
            DVX(k1)=mean(abs(VX_temp2-VX_temp).^p,'all');
            DVX3(k1)=mean(abs(VX_temp2-VX_temp).^3,'all');
            VX_c2(1:k1,:)=VX_c(end-k1+1:end,:);
            VX_c2(k1+1:end,:)=VX_c(1:end-k1,:);
            DVX_c(k1)=mean(abs(VX_c2-VX_c).^p,'all');
            DVX3_c(k1)=mean(abs(VX_c2-VX_c).^3,'all');
            VX_i2(1:k1,:)=VX_i(end-k1+1:end,:);
            VX_i2(k1+1:end,:)=VX_i(1:end-k1,:);
            DVX_i(k1)=mean(abs(VX_i2-VX_i).^p,'all');
            DVX3_i(k1)=mean(abs(VX_i2-VX_i).^3,'all');
        end
        for k2=1:ny_max
            VY_temp2(:,1:k2)=VY_temp(:,end-k2+1:end);
            VY_temp2(:,k2+1:end)=VY_temp(:,1:end-k2);
            DVY(k2)=mean(abs(VY_temp2-VY_temp).^p,'all');
            DVY3(k2)=mean(abs(VY_temp2-VY_temp).^3,'all');
            VY_c2(:,1:k2)=VY_c(:,end-k2+1:end);
            VY_c2(:,k2+1:end)=VY_c(:,1:end-k2);
            DVY_c(k2)=mean(abs(VY_c2-VY_c).^p,'all');
            DVY3_c(k2)=mean(abs(VY_c2-VY_c).^3,'all');
            VY_i2(:,1:k2)=VY_i(:,end-k2+1:end);
            VY_i2(:,k2+1:end)=VY_i(:,1:end-k2);
            DVY_i(k2)=mean(abs(VY_i2-VY_i).^p,'all');
            DVY3_i(k2)=mean(abs(VY_i2-VY_i).^3,'all');
        end

        nx_min=5;ny_min=nx_min;
        XX=(nx_min:nx_max)/2;YY=(ny_min:ny_max)/2;
%         f = @(a,x) a(1)+a(2)*x;
%         a0 =[0,p/3];
%         a1 = nlinfit(log(XX),log(DVX(nx_min:end)),f,a0);
%         log_DVX_fit=f(a1,log(XX));
%         a2 = nlinfit(log(YY),log(DVY(ny_min:end)),f,a0);
%         log_DVY_fit=f(a2,log(YY));
%         %     [a1(2)/(p/3),a2(2)/(p/3)]
%         %         e_p(p)=a1(2)-p/3;

        figure(3)
        nexttile
        hold on
        plot(log(XX),log(DVX(nx_min:end)),'bo')
%         plot(log(XX),log_DVX_fit,'b-')
        plot(log(XX),p/3*log(XX)+log(DVX(nx_min))-p/3*log(XX(1)),'b--')
        plot(log(XX),log(DVX_c(nx_min:end)),'ro')
        plot(log(XX),log(DVX_i(nx_min:end)),'ko')
        hold off
        xL='$\mathrm{ln}x$';yL='$\mathrm{ln}(\Delta J_x)^p$';
        xlabel(xL,'Interpreter','latex','fontsize',14)
        ylabel(yL,'Interpreter','latex','fontsize',14)
        Title=['$p=',num2str(p),'$'];
        title(Title,'Interpreter','latex','fontsize',14)
        nexttile
        hold on
        plot(log(YY),log(DVY(ny_min:end)),'bo')
%         plot(log(YY),log_DVY_fit,'b-')
        plot(log(YY),p/3*log(YY)+log(DVY(ny_min))-p/3*log(YY(1)),'b--')
        plot(log(YY),log(DVY_c(ny_min:end)),'ro')
        plot(log(YY),log(DVY_i(ny_min:end)),'ko')
        hold off
        xL='$\mathrm{ln}y$';yL='$\mathrm{ln}(\Delta J_y)^p$';
        xlabel(xL,'Interpreter','latex','fontsize',14)
        ylabel(yL,'Interpreter','latex','fontsize',14)
        Title=['$p=',num2str(p),'$'];
        title(Title,'Interpreter','latex','fontsize',14)

        %%%%%% generalized self similarity %%%%%%
        figure(5)
        nexttile
        hold on
        plot(log(DVX3(nx_min:end))/3,log(DVX(nx_min:end)),'bo')
        plot(log(DVX3(nx_min:end))/3,p*log(DVX3(nx_min:end))/3+log(DVX(nx_min))-p*log(DVX3(nx_min))/3,'b--')
        hold off
        xL='$\mathrm{ln}S_3/3$';yL='$\mathrm{ln}(\Delta J_x)^p$';
        xlabel(xL,'Interpreter','latex','fontsize',14)
        ylabel(yL,'Interpreter','latex','fontsize',14)
        Title=['$p=',num2str(p),'$'];
        title(Title,'Interpreter','latex','fontsize',14)
        nexttile
        hold on
        plot(log(DVX3_c(nx_min:end))/3,log(DVX_c(nx_min:end)),'ro')
        plot(log(DVX3_c(nx_min:end))/3,p*log(DVX3_c(nx_min:end))/3+log(DVX_c(nx_min))-p*log(DVX3_c(nx_min))/3,'b--')
        hold off
        xL='$\mathrm{ln}S_3^c/3$';yL='$\mathrm{ln}(\Delta J_x^c)^p$';
        xlabel(xL,'Interpreter','latex','fontsize',14)
        ylabel(yL,'Interpreter','latex','fontsize',14)
        Title=['$p=',num2str(p),'$'];
        title(Title,'Interpreter','latex','fontsize',14)
        nexttile
        hold on
        plot(log(DVX3_i(nx_min:end))/3,log(DVX_i(nx_min:end)),'ko')
        plot(log(DVX3_i(nx_min:end))/3,p*log(DVX3_i(nx_min:end))/3+log(DVX_i(nx_min))-p*log(DVX3_i(nx_min))/3,'b--')
        hold off
        xL='$\mathrm{ln}S_3^i/3$';yL='$\mathrm{ln}(\Delta J_x^i)^p$';
        xlabel(xL,'Interpreter','latex','fontsize',14)
        ylabel(yL,'Interpreter','latex','fontsize',14)
        Title=['$p=',num2str(p),'$'];
        title(Title,'Interpreter','latex','fontsize',14)
        nexttile
        hold on
        plot(log(DVY3(ny_min:end))/3,log(DVY(ny_min:end)),'bo')
        plot(log(DVY3(ny_min:end))/3,p/3*log(DVY3(ny_min:end))+log(DVY(ny_min))-p/3*log(DVY3(ny_min)),'b--')
        hold off
        xL='$\mathrm{ln}S_3/3$';yL='$\mathrm{ln}(\Delta J_y)^p$';
        xlabel(xL,'Interpreter','latex','fontsize',14)
        ylabel(yL,'Interpreter','latex','fontsize',14)
        Title=['$p=',num2str(p),'$'];
        title(Title,'Interpreter','latex','fontsize',14)
        nexttile
        hold on
        plot(log(DVY3_c(ny_min:end))/3,log(DVY_c(ny_min:end)),'ro')
        plot(log(DVY3_c(ny_min:end))/3,p/3*log(DVY3_c(ny_min:end))+log(DVY_c(ny_min))-p/3*log(DVY3_c(ny_min)),'b--')
        hold off
        xL='$\mathrm{ln}S_3^c/3$';yL='$\mathrm{ln}(\Delta J_y^c)^p$';
        xlabel(xL,'Interpreter','latex','fontsize',14)
        ylabel(yL,'Interpreter','latex','fontsize',14)
        Title=['$p=',num2str(p),'$'];
        title(Title,'Interpreter','latex','fontsize',14)
        nexttile
        hold on
        plot(log(DVY3_i(ny_min:end))/3,log(DVY_i(ny_min:end)),'ko')
        plot(log(DVY3_i(ny_min:end))/3,p/3*log(DVY3_i(ny_min:end))+log(DVY_i(ny_min))-p/3*log(DVY3_i(ny_min)),'b--')
        hold off
        xL='$\mathrm{ln}S_3^i/3$';yL='$\mathrm{ln}(\Delta J_y^i)^p$';
        xlabel(xL,'Interpreter','latex','fontsize',14)
        ylabel(yL,'Interpreter','latex','fontsize',14)
        Title=['$p=',num2str(p),'$'];
        title(Title,'Interpreter','latex','fontsize',14)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    %     figure(2),plot(e_p)
    %     xL='$p$';yL='$\epsilon_p$';
    %     xlabel(xL,'Interpreter','latex','fontsize',14)
    %     ylabel(yL,'Interpreter','latex','fontsize',14)

    %%%% Energy Spectrum
    VX_k=ifft2((VX_temp(:,:,nz)));VY_k=ifft2((VY_temp(:,:,nz)));
    Ek_k=abs(VX_k).^2+abs(VY_k).^2;Ek_k=double(fftshift(fftshift(Ek_k,1),2));
    VX_c_k=ifft2((VX_c(:,:,nz)));VY_c_k=ifft2((VY_c(:,:,nz)));
    Ek_c_k=abs(VX_c_k).^2+abs(VY_c_k).^2;Ek_c_k=double(fftshift(fftshift(Ek_c_k,1),2));
    VX_i_k=ifft2((VX_i(:,:,nz)));VY_i_k=ifft2((VY_i(:,:,nz)));
    Ek_i_k=abs(VX_i_k).^2+abs(VY_i_k).^2;Ek_i_k=double(fftshift(fftshift(Ek_i_k,1),2));
    KX=(-(nx-1)/2:1:(nx-1)/2)*2*pi/L0;KY=(-(ny-1)/2:1:(ny-1)/2)*2*pi/L;
    [KX,KY]=meshgrid(KX,KY);
    Theta=(1:20)/20*2*pi;Kabs=(1:(nx-1)/2-20)*2*pi/L0;
    KX_polar=Kabs'*cos(Theta);KY_polar=Kabs'*sin(Theta);
    Ek_k_polar=griddata(KX,KY,Ek_k,KX_polar,KY_polar);
    Ek_c_k_polar=griddata(KX,KY,Ek_c_k,KX_polar,KY_polar);
    Ek_i_k_polar=griddata(KX,KY,Ek_i_k,KX_polar,KY_polar);
    Ek=Kabs.*abs(mean(Ek_k_polar,2)');
    Ek_c=Kabs.*abs(mean(Ek_c_k_polar,2)');
    Ek_i=Kabs.*abs(mean(Ek_i_k_polar,2)');
    figure(4),clf
    tiledlayout(2,1,'TileSpacing','compact')%,'Padding','compact'
    nexttile
    hold on
    plot(log(Kabs),log(Ek),'bo')
    plot(log(Kabs),-5/3*log(Kabs)+log(Ek(1))+5/3*log(Kabs(1)),'b-')
    plot(log(Kabs),-3*log(Kabs)+log(Ek(1))+3*log(Kabs(1)),'b--')
    plot(log(Kabs),-1*log(Kabs)+log(Ek(1))+1*log(Kabs(1)),'b-.')
    plot(log(Kabs),log(Ek_c),'ro')
    plot(log(Kabs),log(Ek_i),'ko')
    hold off
    xL='$\mathrm{ln}k$';yL='$\mathrm{ln}E_k$';
    xlabel(xL,'Interpreter','latex','fontsize',14)
    ylabel(yL,'Interpreter','latex','fontsize',14)
    xlim([min(log(Kabs)),log(Kabs(end-50))])
    nexttile
    hold on
    plot((Kabs),(Ek),'bo')
    plot((Kabs),(Ek_c),'ro')
    plot((Kabs),(Ek_i),'ko')
    hold off
    xL='$k$';yL='$E_k$';
    xlabel(xL,'Interpreter','latex','fontsize',14)
    ylabel(yL,'Interpreter','latex','fontsize',14)
end


% TEVOL_DRIVEN Historical driven-flow evolution with GIF output.
% Requires the mu=6, nz=30, drive-index=7 stationary reference and a MATLAB GPU.
% Defaults: nx=ny=161, nz=10, dt=0.02, final time=2000.
% Edit parameters in this script before running; setup_project must be called first.
% Fourier filtering follows the original simulation implementation.

% Enable compatibility before creating or loading GPU-backed fields.
enable_gpu

for K=1
    nx=161; ny=nx; nz=10; nz0=30;
    % Build periodic spectral differentiation matrices on the spatial grid.
    [x,Dx]=fourdif(nx,1);
    [y,Dy]=fourdif(ny,1);
    % [y,Dy]=FD(ny,1);
    [z,Dz]=chebdif(nz,1);
    [z0,Dz0]=chebdif(nz0,1);
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
    % Invert the radial derivative on interior blocks to impose endpoint values.
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
    % Z1 is the complex scalar; Z2, Z3, and Z4 are the temporal, x, and y gauge fields.
    zh=1;e=1;
    f=1-(zz/zh).^3;df=-3*zz.^2/zh^3;
    F1=@(y1,y2,y3,y4) func_Lzr(1i.*y2.*(func_Lz(y1))+0.5i.*y1.*func_Lz(y2)+0.5.*(1-(zz/zh).^3).*func_Lz(func_Lz(y1))-0.5.*3.*zz.^2./zh^3.*func_Lz(y1)-0.5.*zz.*y1-0.5.*y3.*y3.*y1-0.5.*y4.*y4.*y1+1.*0.5.*func_Lx(func_Lx(y1))+ ...
        1.*0.5.*func_Ly(func_Ly((y1)))-1.0.*0.5i.*y1.*(func_Lx(y3)+func_Ly(y4))-1i.*(y3.*func_Lx(y1)+y4.*func_Ly(y1)));
    F3=@(y1,y2,y3,y4) func_Lzr(0.5.*func_Lz(func_Lx(y2)+(1-(zz/zh).^3).*func_Lz(y3))+0.5.*func_Ly(func_Ly(y3)-func_Lx(y4))-0.5i.*(conj(y1).*func_Lx(y1)-y1.*func_Lx(conj(y1)))-y3.*y1.*conj(y1));
    F4=@(y1,y2,y3,y4) func_Lzr(0.5.*func_Lz(func_Ly(y2)+(1-(zz/zh).^3).*func_Lz(y4))+0.5.*func_Lx(-func_Ly(y3)+func_Lx(y4))-0.5i.*(conj(y1).*func_Ly(y1)-y1.*func_Ly(conj(y1)))-y4.*y1.*conj(y1));
    F2=@(y1,y2,y3,y4) (1-zz.^3).*func_Lz(func_Lx(y3)+func_Ly(y4));
    Constraint=@(y1,y2,y3,y4) func_Lz(func_Lx(y3)+func_Ly(y4)-func_Lz(y2))-2*imag((conj(y1).*func_Lz(y1)));

    mu=6;kk=7;
    % Load the dataset selected by this section using a repository-relative location.
    load(hqt_data_path('reference', ['mu=',num2str(mu),',nz=',num2str(nz0),',vy=',num2str(kk),'_20.mat'])) % mu=',num2str(mu),',
    zz0=reshape(zz(1,1,:),[nz,1]);% J=0-(-Lz0(nz0,:)*Vy1);
    % Interpolate the stationary profiles from the reference radial grid to this run.
    Z1=interp1(Z,Z1,zz0,'spline');Z2=interp1(Z,Z2,zz0,'spline');Vy1=interp1(Z,Vy1,zz0,'spline');rhoi=interp1(Z,rhoi,zz0,'spline');
    L=(nx-1)/2;L0=L;
    vy=-2*pi/20*kk;
    % vy0=-2*pi/20*16;tau=10;
    z1=repmat(reshape(Z1,[1,1,nz]),[nx,ny,1]);z2=repmat(reshape(Z2,[1,1,nz]),[nx,ny,1]);
    vy1=repmat(reshape(Vy1,[1,1,nz]),[nx,ny,1]);rho=repmat(reshape(rhoi,[1,1,nz]),[nx,ny,1]);
    pZ2=func_Lz(z2);Z2_f=zeros(nx,ny,nz);
    Z2_f(:,:,2:nz)=z2(:,:,2:nz)./(1-(zz(:,:,2:nz)/zh).^3);Z2_f(:,:,1)=-pZ2(:,:,1)/3;
    % Integrate the radial potential to obtain the phase used in the field initialization.
    theta=-func_Lzr(Z2_f);
    z4=0*vy-vy1;
    % z4=-vy*ones(nx,ny,nz);
    z1=z1.*exp(1i*(theta+0*vy.*yy));
    z3=func_Lx(theta);
    rhoi=repmat(rho(:,:,nz),[1,1,nz]);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % r=((xx-L0/2).^2+(yy-L/2).^2).^0.5;
    % sigma=0.5;
    % nv=6;XX=0.95*L0*rand(nv,nv)+0.05*L0;YY=0.95*L*rand(nv,nv)+0.05*L;
    % % nn=4;
    % % [XX,YY]=meshgrid(1/(2*nn)*L:1/(nn)*L:(2*nn-1)/(2*nn)*L,1/(2*nn)*L:1/(nn)*L:(2*nn-1)/(2*nn)*L);
    % % rng(1)
    % % XX=XX+1*randn(nn,nn);YY=YY+1*randn(nn,nn);
    % % % XX(nn/2,nn/2)=XX(nn/2,nn/2)+10;
    % Profile=0;
    % for x_index=1:length(XX(1,:))
    %     for y_index=1:length(YY(:,1))
    %         Profile=Profile+exp(-((xx-XX(y_index,x_index)).^2+(yy-YY(y_index,x_index)).^2)/sigma^2);
    %     end
    % end
    % Phase_Plus=1;Phase_Minus=1;
    % for x_index=1:length(XX(1,:))
    %     for y_index=1:length(YY(:,1))
    %         if x_index<=nv/2
    %         Phase_Plus=Phase_Plus.*exp(1i*atan2(yy-YY(y_index,x_index),xx-XX(y_index,x_index)));
    %         else
    %         Phase_Minus=Phase_Minus.*exp(-1i*atan2(yy-YY(y_index,x_index),xx-XX(y_index,x_index)));
    %         end
    %     end
    % end
    % % for x_index=1:length(XX(1,:))
    % %     for y_index=1:length(YY(:,1))
    % %         if mod(x_index,2)==mod(y_index,2)
    % %         Phase_Plus=Phase_Plus.*exp(1i*atan2(yy-YY(y_index,x_index),xx-XX(y_index,x_index)));
    % %         else
    % %         Phase_Minus=Phase_Minus.*exp(-1i*atan2(yy-YY(y_index,x_index),xx-XX(y_index,x_index)));
    % %         end
    % %     end
    % % end
    % Z1=z1.*(1-Profile).*Phase_Plus.*Phase_Minus;
    % Apply the chosen perturbation or vortex imprint to the stationary scalar profile.
    Z1=z1.*(1+1e-2*repmat(randn(nx,ny),[1,1,nz]));
    Z2=z2;Z3=z3;Z4=z4;O=1;

    fig1=figure(1); theme(gcf,"light")
    set(fig1,'Position',[800 100 500 800]);

    gifname = hqt_output_path('movies', 'evolution_driven2.gif');   % GIF output path.
    delaytime = 0.0333;            % Frame delay in seconds.
    firstFrame = true;


    t=0;h=gpuArray(double(0.02));%Fixed time step.
    n=0;Nx=(nx-1)/2;
    % h=0.05;
    while  t<2000

        % Four Runge-Kutta stages advance the scalar and spatial gauge fields.
        [k11,k31,k41]=deal(h.*F1(Z1,Z2,Z3,Z4 ),h.*F3(Z1,Z2,Z3,Z4 ),h.*F4(Z1,Z2,Z3,Z4 ));k11(:,:,nz)=0;k31(:,:,nz)=0;k41(:,:,nz)=0;k21=h.*F2(Z1,Z2,Z3,Z4 );
        [k12,k32,k42]=deal(h.*F1(Z1+k11/2,Z2,Z3+k31/2,Z4+k41/2 ),h.*F3(Z1+k11/2,Z2,Z3+k31/2,Z4+k41/2 ),h.*F4(Z1+k11/2,Z2,Z3+k31/2,Z4+k41/2 ));k12(:,:,nz)=0;k32(:,:,nz)=0;k42(:,:,nz)=0;k22=h.*F2(Z1+k11/2,Z2,Z3+k31/2,Z4+k41/2 );
        [k13,k33,k43]=deal(h.*F1(Z1+k12/2,Z2,Z3+k32/2,Z4+k42/2 ),h.*F3(Z1+k12/2,Z2,Z3+k32/2,Z4+k42/2 ),h.*F4(Z1+k12/2,Z2,Z3+k32/2,Z4+k42/2 ));k13(:,:,nz)=0;k33(:,:,nz)=0;k43(:,:,nz)=0;k23=h.*F2(Z1+k12/2,Z2,Z3+k32/2,Z4+k42/2 );
        [k14,k34,k44]=deal(h.*F1(Z1+k13,Z2,Z3+k33,Z4+k43 )      ,h.*F3(Z1+k13,Z2,Z3+k33,Z4+k43 )      ,h.*F4(Z1+k13,Z2,Z3+k33,Z4+k43 ))      ;k14(:,:,nz)=0;k34(:,:,nz)=0;k44(:,:,nz)=0;k24=h.*F2(Z1+k13,Z2,Z3+k33,Z4+k43 );
        [k1,k3,k4]=deal((1/6).*(k11+2.*k12+2.*k13+k14),(1/6).*(k31+2.*k32+2.*k33+k34),(1/6).*(k41+2.*k42+2.*k43+k44));k2=(1/6).*(k21+2.*k22+2.*k23+k24);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        t=t+h;n=n+1;
        % Apply the Runge-Kutta increments and update the boundary charge integration data.
        [Z1,Z3,Z4]=deal(Z1+k1,Z3+k3,Z4+k4);rhoi=rhoi-repmat(k2(:,:,nz),[1,1,nz]);
        cons=func_Lz(func_Lx(Z3)+func_Ly(Z4))+1i.*(conj(Z1).*(func_Lz(Z1))-Z1.*(func_Lz(conj(Z1))));% func_Lz(func_Lx(Z3)+func_Ly(Z4))
        % Reconstruct the temporal potential from the radial constraint and charge data.
        Z2=func_Lzr(func_Lzr(cons)-rhoi)+mu;
        % Restore the prescribed AdS-boundary values after each time step.
        Z1(:,:,nz)=0;Z3(:,:,nz)=0;Z2(:,:,nz)=mu;Z4(:,:,nz)=-vy;
        %     if t<tau||t==tau
        % %         Z4(:,:,nz)=-vy-(vy0-vy)/tau*t;
        %         Z4=func_Lzr(J/tau*t*ones(nx,ny,nz))-vy-(vy0-vy)/tau*t;
        %     else
        %         Z4(:,:,nz)=-vy0;
        %     end

        % Differentiate the bulk scalar; its final radial slice is the boundary order parameter.
        O1=func_Lz(Z1); O0=double(gather(max(max(abs(O1(:,:,nz))))));
        winding=round(mean(sum(angle(circshift(O1(:,:,end),-1,2)./O1(:,:,end)),1)/(2*pi)));
        % Remove the measured global phase winding along the periodic y direction.
        Z1=Z1.*exp(-1i*2*pi*yy/L*(winding+0));

        [Z1,Z2,Z3,Z4]=deal(ff(Z1,fxyz),ff(Z2,fxyz),ff(Z3,fxyz),ff(Z4,fxyz));%Apply the Fourier filter.

        if   rem(n/50,1)==0
            t
            % Differentiate the bulk scalar; its final radial slice is the boundary order parameter.
            O1=func_Lz(Z1); O0=double(gather(max(max(abs(O1(:,:,nz))))));
            %         OO=abs(O12((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz))-abs(O22((nx-1)/2-Nx+1:(nx-1)/2+Nx+1,:,nz));
            %         figure(3),plot(reshape(abs(k1(1,1,:)),[nz,1]))
            %         figure(4),plot(reshape(abs(Z1(1,1,:)),[nz,1]))
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
            %          figure(2),
            %         clf
            %         s=surf(reshape(xx(:,100,:),[nx,nz]),reshape(zz(:,100,:),[nx,nz]),reshape(abs(Z1(:,100,:)),[nx,nz]),'FaceColor','interp' );
            %         s.EdgeColor = 'none';
            % %         xlim([L0/2-L/2,L0/2+L/2]);
            %         view(0,90);
            % %         caxis([0,O])
            %         colorbar;

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
        end
        % if   rem(n/50,1)==0
        %     save(hqt_data_path('raw', ['check_mu=',num2str(mu),'_v=',num2str(-vy/(2*pi)*20),'_t=',num2str(t), '_',num2str(K),'.mat']),'Z1','Z2','Z3','Z4','rhoi');
        % end
        %     if max(max(max(isnan([Z1;Z2;Z3;Z4]))))>0
        %         t
        %         pause
        %     end

    end
end

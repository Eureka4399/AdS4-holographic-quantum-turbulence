function [x,D1,D6,D2,Dv1,Dv2]=FD(N,xi,xm)
% FD Legacy finite-difference and dissipation matrices.
% Outputs are x, D1, D6, D2, Dv1, and Dv2; derivatives act on columns.
% The legacy coordinate interval is [-1, 1].

 %This code calculate 4th order finite difference matrices for 1st and 2nd derivatives.
 %Dv1,2 for advect term dtPhi+VdxPhi=0, 1 for V>0, 2 for V<0

 if nargin<4
      xi=-1;
      xm=1;
  end
% Set the uniform spacing and collocation coordinates for the legacy stencils.
dx=(xm-xi)/(N-1);
x=xi-dx+dx.*(1:N)';

%%%%%%%%%%%%%%%%%%%%%%Fourth order%%%%%%%%%%%%%%%%%%%%
d1=zeros(N,N);
d2=zeros(N,N);
for h=3:N-2
   d1(h-2,h)=1/(12*dx);
   d1(h+2,h)=-1/(12*dx);
   d1(h-1,h)=-2/(3*dx);
   d1(h+1,h)=2/(3*dx);
   d2(h-2,h)=-1/(12*dx^2);
   d2(h+2,h)=-1/(12*dx^2);
   d2(h-1,h)=4/(3*dx^2);
   d2(h+1,h)=4/(3*dx^2);
   d2(h,h)=-5/(2*dx^2);
end
d1(1,2)=-3/(12*dx);d1(2,2)=-10/(12*dx);d1(3,2)=18/(12*dx);d1(4,2)=-6/(12*dx);d1(5,2)=1/(12*dx);
d2(1,2)=10/(12*dx^2);d2(2,2)=-15/(12*dx^2);d2(3,2)=-4/(12*dx^2);d2(4,2)=14/(12*dx^2);d2(5,2)=-6/(12*dx^2);d2(6,2)=1/(12*dx^2);%second column
d1(N-4,N-1)=-1/(12*dx);d1(N-3,N-1)=6/(12*dx);
d1(N-2,N-1)=-18/(12*dx);d1(N-1,N-1)=10/(12*dx);
d1(N,N-1)=3/(12*dx);
d2(N-5,N-1)=1/(12*dx^2);d2(N-4,N-1)=-6/(12*dx^2);
d2(N-3,N-1)=14/(12*dx^2);d2(N-2,N-1)=-4/(12*dx^2);
d2(N-1,N-1)=-15/(12*dx^2);d2(N,N-1)=10/(12*dx^2);%last second column
d1(1,1)=-25/(12*dx);d1(2,1)=48/(12*dx);d1(3,1)=-36/(12*dx);d1(4,1)=16/(12*dx);d1(5,1)=-3/(12*dx);
d1(N-4,N)=3/(12*dx);d1(N-3,N)=-16/(12*dx);
d1(N-2,N)=36/(12*dx);d1(N-1,N)=-48/(12*dx);
d1(N,N)=25/(12*dx);
d2(1,1)=45/(12*dx^2);d2(2,1)=-154/(12*dx^2);d2(3,1)=214/(12*dx^2);d2(4,1)=-156/(12*dx^2);d2(5,1)=61/(12*dx^2);d2(6,1)=-10/(12*dx^2);
d2(N-5,N)=-10/(12*dx^2);d2(N-4,N)=61/(12*dx^2);
d2(N-3,N)=-156/(12*dx^2);d2(N-2,N)=214/(12*dx^2);
d2(N-1,N)=-154/(12*dx^2);d2(N,N)=45/(12*dx^2);

% Transpose coefficient columns so matrices act on column-vector field samples.
D1=d1';D2=d2';
% Reuse centered stencils with wrapped indexing at the two spatial endpoints.
D1(2,:)=circshift(D1(3,:),-1);D1(1,:)=circshift(D1(3,:),-2);D1(end-1,:)=circshift(D1(end-2,:),1);D1(end,:)=circshift(D1(end-2,:),2);
D2(2,:)=circshift(D2(3,:),-1);D2(1,:)=circshift(D2(3,:),-2);D2(end-1,:)=circshift(D2(end-2,:),1);D2(end,:)=circshift(D2(end-2,:),2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%% artificial dissipation (6th order) %%%%%%%%%%%%%%%%%%
d6=zeros(N,N);
for h=4:N-3
   d6(h-3,h)=1/(dx^6);
   d6(h-2,h)=-6/(dx^6);
   d6(h-1,h)=15/(dx^6);
   d6(h,h)=-20/(dx^6);
   d6(h+1,h)=15/(dx^6);
   d6(h+2,h)=-6/(dx^6);
   d6(h+2,h)=1/(dx^6);
end
d6(:,1)=d6(:,4);d6(:,2)=d6(:,4);d6(:,3)=d6(:,4);
d6(:,N)=d6(:,N-3);d6(:,N-1)=d6(:,N-3);d6(:,N-2)=d6(:,N-3);
D6=d6';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%% advect term %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build the directional advection operators for the two velocity signs.
dv1=zeros(N,N);
dv2=zeros(N,N);
for h=4:N-1
   dv1(h-3,h)=-1/(12*dx);
   dv1(h-2,h)=6/(12*dx);
   dv1(h-1,h)=-18/(12*dx);
   dv1(h,h)=10/(12*dx);
    dv1(h+1,h)=3/(12*dx);
end
dv1(1,1)=-25/(12*dx);dv1(2,1)=48/(12*dx);dv1(3,1)=-36/(12*dx);dv1(4,1)=16/(12*dx);dv1(5,1)=-3/(12*dx);
dv1(1,2)=-3/(12*dx);dv1(2,2)=-10/(12*dx);dv1(3,2)=18/(12*dx);dv1(4,2)=-6/(12*dx);dv1(5,2)=1/(12*dx);
dv1(:,3)=d1(:,3);
dv1(N-4,N)=3/(12*dx);dv1(N-3,N)=-16/(12*dx);
dv1(N-2,N)=36/(12*dx);dv1(N-1,N)=-48/(12*dx);
dv1(N,N)=25/(12*dx);
for h=2:N-3
   dv2(h-1,h)=-3/(12*dx);
   dv2(h,h)=-10/(12*dx);
   dv2(h+1,h)=18/(12*dx);
   dv2(h+2,h)=-6/(12*dx);
   dv2(h+3,h)=1/(12*dx);
end
dv2(1,1)=-25/(12*dx);dv2(2,1)=48/(12*dx);dv2(3,1)=-36/(12*dx);dv2(4,1)=16/(12*dx);dv2(5,1)=-3/(12*dx);
dv2(:,N-2)=d1(:,N-2);
dv2(N-4,N-1)=-1/(12*dx);dv2(N-3,N-1)=6/(12*dx);
dv2(N-2,N-1)=-18/(12*dx);dv2(N-1,N-1)=10/(12*dx);
dv2(N,N-1)=3/(12*dx);
dv2(N-4,N)=3/(12*dx);dv2(N-3,N)=-16/(12*dx);
dv2(N-2,N)=36/(12*dx);dv2(N-1,N)=-48/(12*dx);
dv2(N,N)=25/(12*dx);

Dv1=dv1';Dv2=dv2';

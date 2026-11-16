function K=q4_ke_2D(h,nu,E0)
% Standard bilinear plane-stress Q4, BL BR TR TL, exact 2x2 quadrature.
if nargin<3,E0=1;end
D=E0/(1-nu^2)*[1 nu 0;nu 1 0;0 0 (1-nu)/2];
K=zeros(8);sg=[-1 -1;1 -1;1 1;-1 1];
for x=[-1 1]/sqrt(3)
 for y=[-1 1]/sqrt(3)
  dx=sg(:,1)'.*(1+y*sg(:,2)')/(2*h(1));
  dy=sg(:,2)'.*(1+x*sg(:,1)')/(2*h(2));
  B=zeros(3,8);B(1,1:2:8)=dx;B(2,2:2:8)=dy;B(3,1:2:8)=dy;B(3,2:2:8)=dx;
  K=K+(B'*D*B)*prod(h)/4;
 end
end
K=.5*(K+K');
end

function levels=q4_setup_2D(grid,side,nu,E0)
% Only preconditioner levels are rediscretized; target Q4 is unchanged.
sides=["left","right","bottom","top"];
bc=find(sides==string(side));assert(isscalar(bc),'q4:Boundary','Invalid fixed edge.');
grid=double(grid(:)');stride=1;levels={};
for k=1:24
 shape=ceil(grid/stride);tail=grid-(shape-1)*stride;Ke=zeros(8,8,4);
 for t=0:3
  h=stride*[1 1];for d=1:2,if bitget(t,d),h(d)=tail(d);end,end
  Ke(:,:,t+1)=q4_ke_2D(h,nu,E0);
 end
 levels{k}=struct('grid',shape,'original_grid',grid,'stride',stride,'edge',1,'bc',bc,'Ke',Ke);
 if 2*prod(shape+1)<=256,break;end
 stride=2*stride;
end
end

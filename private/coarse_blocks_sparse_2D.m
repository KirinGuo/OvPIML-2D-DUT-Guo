function A=coarse_blocks_sparse_2D(K)
% Explicit scalar conversion for diagnostics or requested CPU direct solve.
% The accelerated default never calls this function.
B=gather(K.blocks);nx=K.nslx;ny=K.nsly;ns=nx*ny;
[dy,dx]=deal([0 1 -1 0 1]',[0 0 1 1 1]');
owner=0:ns-1;ry=mod(owner,ny);rx=floor(owner/ny);
cx=rx+dx;cy=ry+dy;valid=cx>=0&cx<nx&cy>=0&cy<ny;
ri=repmat(owner,5,1);ci=cy+ny*cx;ri=ri(valid);ci=ci(valid);
V=reshape(B,64,5*ns);V=V(:,valid(:));
ii=reshape(8*ri(:)',1,1,[])+(1:8)';jj=reshape(8*ci(:)',1,1,[])+reshape(1:8,1,8);
ii=repmat(ii,1,8,1);jj=repmat(jj,8,1,1);
A=sparse(ii(:),jj(:),V(:),8*ns,8*ns);
off=ri~=ci;A=A+sparse(reshape(jj(:,:,off),[],1),reshape(ii(:,:,off),[],1),reshape(V(:,off),[],1),8*ns,8*ns);
end

function result=mechanics2d_resident(cfg)
% Native GPU basis construction, assembly, coarse solve and Q4 correction.
g=gpuDevice;previousPolicy=g.CachePolicy;g.CachePolicy='minimum';cleanup=onCleanup(@()restore_policy(g,previousPolicy));wait(g);total=tic;timing=struct();
t=tic;S=struct('nslx',cfg.nslx,'nsly',cfg.nsly,'ns',cfg.nslx*cfg.nsly,'nelxIs',cfg.nelxIs,'layer',cfg.layer,'nelx',cfg.nslx*(cfg.nelxIs-1)+1,'nely',cfg.nsly*(cfg.nelxIs-1)+1);
nx=S.nelx;ny=S.nely;nd=2*(nx+1)*(ny+1);nC=8*S.ns;assert(8*nd<=double(intmax('int32')),'mechanics2d:Size','Packed basis exceeds GPU element limit.');
if isempty(cfg.rho),[x,y]=meshgrid(((1:nx)-.5)/nx,((1:ny)-.5)/ny);rho=.55+.35*sin(4*pi*x).*cos(3*pi*y);clear x y
elseif isscalar(cfg.rho),rho=repmat(double(cfg.rho),ny,nx);else,rho=double(cfg.rho);end
validateattributes(rho,{'double'},{'real','finite','size',[ny nx],'>=',0,'<=',1});
bc=find(strcmpi(cfg.fixed_end,{'left','right','bottom','top'}));assert(isscalar(bc));
switch bc
 case 1,fixedNodes=(1:ny+1)';loadNode=nx*(ny+1)+floor(ny/2)+1;owners=0:S.nsly-1;channels=[1 2 7 8];
 case 2,fixedNodes=nx*(ny+1)+(1:ny+1)';loadNode=floor(ny/2)+1;owners=(S.nslx-1)*S.nsly+(0:S.nsly-1);channels=3:6;
 case 3,fixedNodes=1+(0:nx)'*(ny+1);loadNode=(floor(nx/2)+1)*(ny+1);owners=0:S.nsly:(S.nslx-1)*S.nsly;channels=1:4;
 case 4,fixedNodes=(1:nx+1)'*(ny+1);loadNode=floor(nx/2)*(ny+1)+1;owners=S.nsly-1:S.nsly:S.ns-1;channels=5:8;
end
if ~isempty(cfg.load_node_1based)
 loadNode=double(cfg.load_node_1based);
 assert(~ismember(loadNode,fixedNodes),'V9:Load','The point load must have a free y degree of freedom.');
end
fixed=reshape(8*owners+channels',[],1);fixedFine=reshape([2*fixedNodes'-1;2*fixedNodes'],[],1);rhog=gpuArray(rho);ag=(cfg.Emin+(cfg.E0-cfg.Emin)*rhog.^cfg.penal)/cfg.E0;wait(g);timing.mesh=toc(t);
t=tic;
[Pg,binfo,basisTiming]=v9_basis_gpu(rho,ag,S,cfg);
wait(g);timing.basis=toc(t);clear rhog
if cfg.verbose,fprintf('[V9] raw basis: %.6fs, %d owners\n',timing.basis,binfo.owners);drawnow;end
timing.plan=0;
t=tic;D=cfg.E0/(1-cfg.nu^2)*[1 cfg.nu 0;cfg.nu 1 0;0 0 (1-cfg.nu)/2];eK=Ele_stiff_mat(D);[K,ainfo]=assemble_Kreduce_structured_2D(S,eK,ag,Pg);wait(g);timing.assembly=toc(t);
t=tic;grid=double([S.nslx S.nsly S.nelxIs-1 S.layer bc 1]);F=mechanics2d_basis_gpu_v1_mex('load',Pg,grid,double([loadNode-1 1 cfg.force]));F(fixed)=0;wait(g);timing.load=toc(t);
t=tic;shape=double([S.nslx S.nsly S.nslx S.nsly 1 bc]);
if cfg.verbose,fprintf('[resident_stage] coarse solve begin: DOFs=%d\n',nC);drawnow;end
if cfg.coarse_solver=="cpu-direct"
 A=coarse_blocks_sparse_2D(K);free=setdiff((1:nC)',fixed);dd=zeros(nC,1);tt=tic;factor=decomposition(A(free,free),'chol');setup=toc(tt);dd(free)=factor\gather(F(free));d=gpuArray(dd);coarse=struct('backend','CPU-direct-explicit','flag',0,'iterations',0,'true_relres',norm(A(free,:)*dd-gather(F(free)))/max(gather(norm(F)),eps),'setup_seconds',setup,'solve_seconds',toc(t)-setup);
else
 [d,coarse]=mechanics2d_blocks_v1_cuda_mex('solve',K.blocks,shape,F,struct('tol',double(cfg.coarse_tol),'maxit',double(cfg.coarse_maxit),'reserve_bytes',512*2^20));
end
wait(g);timing.solve=toc(t);assert(coarse.flag==0&&coarse.true_relres<=max(1e-7,cfg.coarse_tol),'mechanics2d:Coarse','Coarse system failed.');
if cfg.verbose,fprintf('[resident_stage] coarse ready: %.6fs iterations=%d r=%.9g\n',timing.solve,coarse.iterations,coarse.true_relres);drawnow;end
if ~cfg.keep_reduced,K=[];end
t=tic;Uov=mechanics2d_basis_gpu_v1_mex('recover',Pg,grid,d);wait(g);timing.recover=toc(t);clear Pg
t=tic;U0=mechanics2d_fields_cuda_mex('map',Uov,[nx ny S.nslx S.nsly S.nelxIs-1]);clear Uov;assert(gather(max(abs(U0(fixedFine))))==0,'V9:Clamp','Clamped BVP yielded nonzero physical wall.');
work=cfg.force*gather(U0(2*loadNode));reducedWork=gather(dot(d,F));workError=abs(work-reducedWork)/max(abs(work),eps);assert(workError<2e-6,'mechanics2d:Work','Native physical virtual-work mismatch %.3e.',workError);wait(g);timing.physical_map=toc(t);timing.seed_ready=toc(total);
t=tic;levels=q4_setup_2D([nx ny],cfg.fixed_end,cfg.nu,cfg.E0);fg=zeros(nd,1,'double','gpuArray');fg(2*loadNode)=cfg.force;wait(g);timing.fine_setup=toc(t);
opts=struct('rtol',cfg.fine_tol,'maxit',cfg.fine_maxit,'check_every',1,'smooth_steps',2,'reserve_bytes',512*2^20,'verbose',double(cfg.verbose),'true_stop_below',double(cfg.fine_stop_below),'recursive_stop_below',0,'stop_file','');
budget=~isempty(cfg.correction_steps);
if budget
 opts.maxit=cfg.correction_steps; opts.true_stop_below=0;
end
if cfg.correction
 t=tic;[U,fine]=mechanics2d_q4_cuda_mex('solve',levels,ag,fg,U0,opts);wait(g);timing.correction=toc(t);if budget
 assert(fine.iterations==cfg.correction_steps || (fine.true_relres<=cfg.fine_tol && fine.iterations<cfg.correction_steps),'V9:Fine','Correction budget mismatch.');
 else
 assert(fine.true_relres<cfg.fine_stop_below,'V9:Fine','Target residual not reached.');
 end
else,U=U0;timing.correction=0;fine=struct('flag',-1,'iterations',0,'true_relres',NaN,'initial_true_relres',NaN);end
timing.corrected_ready=toc(total);
t=tic;raw=levels(1);raw{1}.bc=0;au=mechanics2d_q4_cuda_mex('matvec',raw,ag,U,opts);r=fg-au;r(fixedFine)=0;trueRes=gather(norm(r))/max(gather(norm(fg)),eps);energy=.5*gather(dot(U,au));compliance=cfg.force*gather(U(2*loadNode));wall=gather(max(abs(U(fixedFine))));assert(wall==0&&gather(all(isfinite(U))));if cfg.correction && ~budget,assert(trueRes<cfg.fine_stop_below);end;wait(g);timing.certificate=toc(t);clear au r ag fg
t=tic;U=gather(U);if cfg.save_initial,U0=gather(U0);else,U0=[];end;d=gather(d);if cfg.keep_reduced,F=gather(F);free=setdiff((1:nC)',fixed);else,F=[];free=[];end;wait(g);timing.download=toc(t);timing.total=toc(total);
result=struct('U',U,'U_initial',U0,'ux',reshape(U(1:2:end),ny+1,nx+1),'uy',reshape(U(2:2:end),ny+1,nx+1),'rho',rho,'compliance',compliance,'strain_energy',energy, ...
 'max_displacement',max(hypot(U(1:2:end),U(2:2:end))),'relative_residual',coarse.true_relres,'fine_relative_residual',trueRes,'fixed_wall_max',wall,'ndof',nd,'ncoarse',nC, ...
 'basis',binfo,'basis_timing',basisTiming,'assembly',ainfo,'coarse',coarse,'fine',fine,'timing',timing,'gpu',string(g.Name),'config',cfg,'K_reduce',K,'F_reduce',F,'d',d,'free_coarse',free,'physical_work_consistency',workError);
if cfg.verbose,fprintf('[V9 raw8 Q4] fine=%d coarse=%d initialR=%.6g finalR=%.6g correction=%d total=%.6fs\n',nd,nC,fine.initial_true_relres,trueRes,fine.iterations,timing.total);disp(timing);disp(basisTiming);end
if cfg.plot,figure('Color','w');tiledlayout(1,2);nexttile;imagesc(rho);axis image xy;colorbar;nexttile;imagesc(result.uy);axis image xy;colorbar;end
end
function restore_policy(g,previous),g.CachePolicy=previous;end

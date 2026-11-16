function report=run_random_density(sizeName,edgeMode,output,withReference,saveFields,caseName)
% Portable V9 example; default is the original smooth 120,626,288-DOF beam.
% run_random_density                 120,626,288 DOFs, reference and L2 errors.
% run_random_density('small')             33,488 DOFs, quick installation check.
% The optional sixth argument 'iid' selects the previous independent random case.
if nargin<1,sizeName="120m";end
if nargin<2,edgeMode="fno";end
root=fileparts(mfilename('fullpath'));
if nargin<3 || strlength(string(output))==0
    output=fullfile(root,'results',string(datetime('now','Format','yyyyMMdd_HHmmss_SSS')));
end
if nargin<4,withReference=true;end
if nargin<5,saveFields=false;end
if nargin<6,caseName="smooth";end
sizeName=string(sizeName);edgeMode=string(edgeMode);caseName=string(caseName);
assert(ismember(sizeName,["small","10m","120m"]),'Unknown size.');
assert(ismember(edgeMode,["fno","exact"]),'edgeMode must be fno or exact.');
assert(ismember(caseName,["smooth","iid"]),'caseName must be smooth or iid.');
assert(~isfile(fullfile(output,'result.json')),'Use a fresh output directory.');
if ~isfolder(output),mkdir(output);end
env=v9_environment;gpu=gpuDevice;
previousThreads=maxNumCompThreads(8);threadCleanup=onCleanup(@()maxNumCompThreads(previousThreads));
switch sizeName
    case "small",owners=[20 10];
    case "10m",owners=[382 192];
    case "120m",owners=[1220 610];
end
nx=9*owners(1)+1;ny=9*owners(2)+1;
if caseName=="smooth"
    h=1;loadY=floor(ny/2);Emin=1e-6;
    rho=.55+.35*(cos(3*pi*(((1:ny)-.5)/ny))'*sin(4*pi*(((1:nx)-.5)/nx)));
    materialInfo="Analytic smooth density: rho=0.55+0.35*cos(3*pi*(j-0.5)/ny)*sin(4*pi*(i-0.5)/nx). E/E0=1e-6+(1-1e-6)*rho.";
    warmRho=[];
else
    h=181/nx;loadY=floor(ny*45/91);Emin=0;
    rho=random_material(ny,nx,91732);
    materialInfo="Independent Uniform[0.01,1] fine-cell E/E0, with solid rectangle x/Lx in [32/181,78/181], y/Ly in [18/91,50/91].";
    warmRho=random_material(91,181,7301);
end
loadNode=nx*(ny+1)+loadY+1;loadDof=2*loadNode;
if edgeMode=="exact",basis="internal-fno";else,basis="fno";end
cfg=struct('nslx',owners(1),'nsly',owners(2),'basis',basis,'load_node_1based',loadNode, ...
    'rho',rho,'Emin',Emin,'correction_steps',3,'inference_batch',512);
% Warm an independent small problem, including the chosen exact-edge route.
warm=struct('nslx',20,'nsly',10,'basis',basis,'rho',warmRho,'Emin',Emin,'correction_steps',3);
mechanics2d(warm);
started=tic;sol=mechanics2d(cfg);wall=toc(started);
assert(sol.ndof==2*(nx+1)*(ny+1) && sol.fixed_wall_max==0 && sol.fine.iterations==3);
report=struct('case',caseName,'size',sizeName,'grid_elements',[nx ny],'dofs',sol.ndof,'owners',owners, ...
    'domain_xy',h*[nx ny],'element_side',h,'thickness',1,'nu',.3,'E0',1,'Emin',Emin,'penal',1, ...
    'material',materialInfo,'material_sha256_f64',array_hash(sol.rho), ...
    'load_node_1based',loadNode,'load_xy',h*[nx loadY],'load_vector',[0 -1], ...
    'fixed_edge',"x=0, ux=uy=0",'edge_mode',edgeMode,'basis',sol.basis, ...
    'coarse_relative_residual',sol.relative_residual,'corrected_fine_relative_residual',sol.fine_relative_residual, ...
    'initial_load_uy',sol.U_initial(loadDof),'corrected_load_uy',sol.U(loadDof), ...
    'fixed_wall_max',sol.fixed_wall_max,'virtual_work_error',sol.physical_work_consistency, ...
    'gpu',string(gpu.Name),'matlab',version,'python',env.python,'timing_seconds',sol.timing,'complete_call_seconds',wall, ...
    'timing_scope',"One warmed mechanics2d call, including fresh bases, reduced assembly/solve, three Q4 corrections, certification and host return. Material creation, warmup, reference, metrics and writes excluded.");
if caseName=="iid",report.seed=91732;report.rng="MATLAB mt19937ar, column-major";end
U0=sol.U_initial;U3=sol.U;rho=sol.rho;
if withReference
    [Uref,reference]=fine_reference(rho,loadDof,Emin,caseName);
    effectiveMaterial=Emin+(1-Emin)*rho;
    report.reference=reference;
    report.initial=field_errors(U0,Uref,effectiveMaterial,loadDof,h);
    report.corrected=field_errors(U3,Uref,effectiveMaterial,loadDof,h);
    fprintf('Full-domain displacement L2: %.8f%% -> %.8f%%\n',100*report.initial.displacement_l2,100*report.corrected.displacement_l2);
    fprintf('Cell energy-density L2:      %.6f%% -> %.6f%%\n',100*report.initial.energy_density_l2,100*report.corrected.energy_density_l2);
end
if saveFields
    if withReference,save(fullfile(output,'fields.mat'),'rho','U0','U3','Uref','-v7.3');
    else,save(fullfile(output,'fields.mat'),'rho','U0','U3','-v7.3');end
end
fid=fopen(fullfile(output,'result.json'),'w','n','UTF-8');assert(fid>=0);
closeFile=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(report,'PrettyPrint',true));
fprintf('DOFs %d, AI %.6f s, AI + 3 steps %.6f s, complete call %.6f s\n', ...
    report.dofs,sol.timing.seed_ready,sol.timing.corrected_ready,wall);
fprintf('Result: %s\n',fullfile(output,'result.json'));
end

function rho=random_material(ny,nx,seed)
stream=RandStream('mt19937ar','Seed',seed);
rho=.01+.99*rand(stream,ny,nx);
xx=((1:nx)-.5)/nx;yy=((1:ny)-.5)/ny;
rho(yy>=18/91 & yy<50/91,xx>=32/181 & xx<78/181)=1;
end

function [u,receipt]=fine_reference(rho,loadDof,Emin,caseName)
[ny,nx]=size(rho);levels=q4_setup_2D([nx ny],"left",.3,1);
material=Emin+(1-Emin)*gpuArray(rho(:));f=zeros(2*(ny+1)*(nx+1),1,'double','gpuArray');f(loadDof)=-1;
opts=struct('rtol',1e-9,'maxit',1500,'check_every',15,'smooth_steps',2,'reserve_bytes',512*2^20, ...
    'verbose',0,'true_stop_below',0,'recursive_stop_below',0,'stop_file','');
if caseName=="smooth",opts.maxit=300;opts.check_every=20;end
started=tic;[v,stats]=mechanics2d_q4_cuda_mex('solve',levels,material,f,[],opts);
raw=levels(1);raw{1}.bc=0;kv=mechanics2d_q4_cuda_mex('matvec',raw,material,v,opts);
r=f-kv;r(1:2*(ny+1))=0;rr=gather(norm(r));wall=max(abs(gather(v(1:2*(ny+1)))));
assert(stats.flag==0 && rr<=1e-9 && wall==0,'Fine reference failed: %.6g',rr);
reaction=sum(reshape(gather(kv(1:2*(ny+1))),2,[]),2)';u=gather(v);
receipt=struct('solver',"FP64 physical Q4, zero-initial geometric multigrid PCG",'iterations',stats.iterations, ...
    'true_relative_residual',rr,'seconds',toc(started),'load_uy',u(loadDof),'reaction_xy',reaction);
end

function m=field_errors(u,ref,rho,loadDof,h)
[ny,nx]=size(rho);ke=q4_ke_2D([h h],.3,1);num=0;den=0;acc=zeros(1,5);
for start=1:2^20:numel(u)
    ids=start:min(start+2^20-1,numel(u));r=ref(ids);e=u(ids)-r;
    num=num+dot(e,e);den=den+dot(r,r);
end
offset=[0 1 2*(ny+1) 2*(ny+1)+1 2*(ny+2) 2*(ny+2)+1 2 3]';
for start=1:48:nx
    x=start:min(start+47,nx);ids=reshape(2*((0:ny-1)'+(ny+1)*(x-1))+1,1,[])+offset;
    mat=reshape(rho(:,x),1,[]);q=u(ids);qr=ref(ids);
    q(1:2:end,:)=q(1:2:end,:)-q(1,:);q(2:2:end,:)=q(2:2:end,:)-q(2,:);
    qr(1:2:end,:)=qr(1:2:end,:)-qr(1,:);qr(2:2:end,:)=qr(2:2:end,:)-qr(2,:);dq=q-qr;
    w=.5*mat.*sum(q.*(ke*q),1);wr=.5*mat.*sum(qr.*(ke*qr),1);we=.5*mat.*sum(dq.*(ke*dq),1);
    diff=(w-wr)/h^2;density=wr/h^2;
    acc=acc+[dot(diff,diff),dot(density,density),sum(we),sum(w),sum(wr)];
end
m=struct('displacement_l2',sqrt(num/den),'energy_density_l2',sqrt(acc(1)/acc(2)), ...
    'displacement_energy_norm',sqrt(acc(3)/acc(5)),'integrated_energy',acc(4),'reference_energy',acc(5), ...
    'load_uy',u(loadDof),'load_uy_relative_error',abs(u(loadDof)/ref(loadDof)-1));
end

function value=array_hash(a)
md=java.security.MessageDigest.getInstance('SHA-256');
for start=1:2^20:numel(a)
    ids=start:min(start+2^20-1,numel(a));md.update(typecast(double(a(ids)),'uint8'));
end
value=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
end

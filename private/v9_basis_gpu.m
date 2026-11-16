function [Pg,info,timing]=v9_basis_gpu(~,ag,S,cfg)
% Native 100k inference on the physical GPU material, with raw columns scattered on device.
started=tic;
root=fileparts(fileparts(mfilename('fullpath')));
bundle=fullfile(root,'gpu_basis','models');
sides=["left","right","bottom","top"]; bc=find(sides==cfg.fixed_end);
grid=double([S.nslx S.nsly S.nelxIs-1 S.layer bc 1]);
v9_environment;
persistent modelRoot manifest physicalMap prototypes
if isempty(modelRoot) || ~strcmp(modelRoot,bundle)
    manifest=jsondecode(fileread(fullfile(bundle,'export.json')));
    map=load(fullfile(bundle,'owner_maps.mat'),'signed_map');physicalMap=map.signed_map;
    uni=load(fullfile(bundle,'uniform_templates.mat'),'prototypes');prototypes=uni.prototypes;
    modelRoot=bundle;
end
tt=tic;
assert(gather(all(isfinite(ag(:)) & ag(:)>0)),'V9:Material','Q4 requires finite positive relative stiffness.');
gmap=gpuArray(physicalMap);
v9_native_basis_mex('begin',grid);
cleanup=onCleanup(@()v9_native_basis_mex('abort'));
uniform=logical(gather(v9_native_basis_mex('classify',ag,grid)));uniform=uniform(:);
sx=2+double((1:S.nslx)==S.nslx)-double((1:S.nslx)==1);
sy=2+double((1:S.nsly)==S.nsly)-double((1:S.nsly)==1);
types=uint8(sx+3*(3-sy'));types=types(:); assert(numel(types)==S.ns);
clampTypes={[1 4 7],[3 6 9],[7 8 9],[1 2 3]};
clamp=ismember(types,clampTypes{bc});
corner=ismember(types,[1 3 7 9]);
numeric=~uniform & (corner|clamp|(cfg.basis=="internal-fno" & types~=5));
internal=~uniform & ~numeric & types==5;
edge=~uniform & ~numeric & types~=5;
timing.classify_and_route=toc(tt);
routes=struct();
for tid=1:9
    routes.(sprintf('type_%d',tid))=sum(types==tid);
end
routes=set_count(routes,'exact_clamped',nnz(numeric & clamp));
routes=set_count(routes,'exact_corner',nnz(numeric & ~clamp & corner));
routes=set_count(routes,'exact_edge',nnz(numeric & ~clamp & ~corner));
routes=set_count(routes,'fno_internal',nnz(internal));
routes=set_count(routes,'fno_edge',nnz(edge));

t=tic; nTemplates=0;
for tid=1:9
    ids=find(uniform & types==tid)-1;
    if isempty(ids),continue;end
    slot=1+bc*ismember(tid,clampTypes{bc});
    template=prototypes(:,:,tid,slot);
    v9_native_basis_mex('put',gpuArray(template),gpuArray(int32(ids)),zeros(numel(ids),1,'int32','gpuArray'));
    nTemplates=nTemplates+1;
end
wait(gpuDevice);timing.uniform_templates=toc(t);
routes=set_count(routes,'uniform_build',nTemplates);
routes=set_count(routes,'uniform_reuse',nnz(uniform)-nTemplates);
t=tic;
if any(numeric)
    teacher=exact_teacher(root);
    for tid=1:9
        ids=find(numeric & types==tid)-1;
        if isempty(ids),continue;end
        side="none";if ismember(tid,clampTypes{bc}),side=cfg.fixed_end;end
        % Real boundaries have zero halo along the physical side.
        left=mod(tid-1,3)==0; right=mod(tid-1,3)==2;
        low=tid>=7; high=tid<=3;
        dy=-6*(~low):9+6*(~high); dx=-6*(~left):9+6*(~right);
        offsets=reshape(dy',[],1,1)+reshape(S.nely*dx,1,[],1);
        for first=1:cfg.inference_batch:numel(ids)
            block=ids(first:min(numel(ids),first+cfg.inference_batch-1));
            ox=floor(block/S.nsly);oy=mod(block,S.nsly);
            address=1+reshape(9*oy+9*S.nely*ox,1,1,[])+offsets;
            win=reshape(gather(ag(address)),numel(dy),numel(dx),[]);
            page=single(teacher.solve_pages(py.numpy.asarray(win),int32(tid),char(side)));
            v9_native_basis_mex('put',gpuArray(page),gpuArray(int32(block)),gpuArray(int32(0:numel(block)-1)));
        end
    end
end
wait(gpuDevice);timing.exact_windows_cpu_solve_and_put=toc(t);
timing.exact_and_template=timing.exact_windows_cpu_solve_and_put+timing.uniform_templates;
checkpoints=struct();passes=struct();
t=tic;
for kind=["internal","edge"]
    if kind=="internal", sel=internal; else,sel=edge;end
    ids=find(sel)-1;
    if isempty(ids),continue;end
    passes.(kind)=v9_native_basis_mex('infer',ag,gpuArray(int32(ids)),char(kind), ...
        char(fullfile(bundle,kind+".ts")),double(cfg.inference_batch),gmap);
    m=manifest.models.(kind);
    checkpoints.(kind)=struct('file',m.file,'epoch',m.epoch,'training_sha256',m.training_sha256, ...
        'checkpoint_sha256',m.source_sha256,'script_sha256',m.script_sha256);
end
[Pg,~]=v9_native_basis_mex('finish'); wait(gpuDevice);
timing.batched_fno_and_decode=toc(t);
timing.gpu_gather=0; timing.forward=0; timing.gpu_scatter=0;
for key=string(fieldnames(passes))'
    p=passes.(key);
    timing.gpu_gather=timing.gpu_gather+p.gather_seconds;
    timing.forward=timing.forward+p.forward_seconds;
    timing.gpu_scatter=timing.gpu_scatter+p.scatter_seconds;
end
timing.total=toc(started);
info=struct('owners',S.ns,'routes',routes,'checkpoints',checkpoints,'basis_mode',cfg.basis, ...
    'backend',"native-cuda-libtorch-raw8",'original_columns',8,'corner_normalization',false, ...
    'uniformity',"all physical cells in clipped oversampling window", ...
    'uniform_templates',"precomputed exact uniform Q4 per orientation and clamp", ...
    'fixed_end',cfg.fixed_end,'numerical_fallbacks',0,'timing',timing,'inference',passes);
info.learned_models=fieldnames(checkpoints);
end

function s=set_count(s,name,n)
if n>0,s.(name)=n;end
end

function teacher=exact_teacher(root)
env=v9_environment; exe=env.python;
p=pyenv;
if p.Status~="Loaded"
    pyenv(Version=exe,ExecutionMode="OutOfProcess");
elseif p.ExecutionMode~="OutOfProcess" || ~strcmpi(p.Executable,exe)
    error('V9:PythonEnv','Use the selected out-of-process Python in a new MATLAB session.');
end
if count(py.sys.path,root)==0,insert(py.sys.path,int32(0),root);end
teacher=py.importlib.import_module('gpu_basis.exact');
end


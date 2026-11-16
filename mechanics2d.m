function result=mechanics2d(cfg)
% Raw-eight FNO bases, physical Q4 assembly, and GPU multigrid corrections.
if nargin==0,cfg=struct();end
base=struct('nslx',20,'nsly',10,'nelxIs',10,'layer',6,'nu',.3, ...
    'E0',1,'Emin',0,'penal',1,'rho',[],'fixed_end',"left",'force',-1,'load_node_1based',[], ...
    'basis',"fno",'inference_batch',512,'coarse_solver',"auto",'coarse_tol',1e-8,'coarse_maxit',5000, ...
    'correction',true,'correction_steps',3,'fine_tol',1e-10,'fine_maxit',3000,'fine_stop_below',.1, ...
    'save_initial',true,'keep_reduced',false,'verbose',false,'plot',false);
assert(isstruct(cfg)&&isscalar(cfg),'Expected a scalar configuration.');
unknown=setdiff(fieldnames(cfg),fieldnames(base));
assert(isempty(unknown),'Unknown option: %s',strjoin(unknown,', '));
for name=string(fieldnames(base))'
    if ~isfield(cfg,name),cfg.(name)=base.(name);end
end
cfg.basis=lower(string(cfg.basis));cfg.fixed_end=lower(string(cfg.fixed_end));
cfg.coarse_solver=lower(string(cfg.coarse_solver));
assert(ismember(cfg.basis,["fno","internal-fno"]),'basis must be fno or internal-fno.');
assert(ismember(cfg.fixed_end,["left","right","bottom","top"]),'Invalid fixed edge.');
assert(ismember(cfg.coarse_solver,["auto","gpu-mg","cpu-direct"]),'Invalid coarse solver.');
assert(cfg.nelxIs==10 && cfg.layer==6 && cfg.nu==.3,'Frozen model geometry is core=10, halo=6, nu=0.3.');
for name=["nslx","nsly"],validateattributes(cfg.(name),{'numeric'},{'scalar','integer','>=',2});end
for name=["inference_batch","coarse_maxit","fine_maxit"],validateattributes(cfg.(name),{'numeric'},{'scalar','integer','positive'});end
validateattributes(cfg.inference_batch,{'numeric'},{'<=',4096});
for name=["coarse_tol","fine_tol","fine_stop_below"],validateattributes(cfg.(name),{'numeric'},{'scalar','positive','<=',1});end
validateattributes(cfg.force,{'numeric'},{'scalar','real','finite'});
assert(cfg.E0>0&&isfinite(cfg.E0)&&cfg.Emin>=0&&cfg.Emin<cfg.E0&&cfg.penal>0&&isfinite(cfg.penal));
if ~isempty(cfg.load_node_1based)
    validateattributes(cfg.load_node_1based,{'numeric'},{'scalar','integer','>=',1,'<=',(9*cfg.nslx+2)*(9*cfg.nsly+2)});
end
if ~isempty(cfg.correction_steps)
    validateattributes(cfg.correction_steps,{'numeric'},{'scalar','integer','nonnegative'});
    if cfg.correction_steps==0,cfg.correction=false;end
end
result=mechanics2d_resident(cfg);
end

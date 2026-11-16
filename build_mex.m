function build_mex
% Rebuild every CUDA module against this machine's MATLAB and local PyTorch.
env=v9_environment;root=env.root;target=fullfile(root,'private');
toolkit=getenv('V9_NVCC_DIR');
if ~isempty(toolkit),setenv('MW_NVCC_PATH',toolkit);end
common={'-R2018a','NVCCFLAGS=-std=c++17 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_80,code=compute_80', ...
    'CXXFLAGS="$CXXFLAGS /std:c++17 /EHsc"','-outdir',target};
for name=["mechanics2d_basis_gpu_v1","mechanics2d_blocks_v1_cuda","mechanics2d_fields_cuda","mechanics2d_q4_cuda"]
    libs={};
    if contains(name,"blocks") || contains(name,"q4"),libs={'-lcublas'};end
    mexcuda(common{:},libs{:},'-output',char(name+"_mex"),fullfile(target,'cuda',name+".cu"));
end
torch=char(env.torch);
mexcuda(common{:},['-I' fullfile(torch,'include')], ...
    ['-I' fullfile(torch,'include','torch','csrc','api','include')],['-L' fullfile(torch,'lib')], ...
    '-ltorch','-ltorch_cpu','-ltorch_cuda','-lc10','-lc10_cuda', ...
    '-output','v9_native_basis_mex',fullfile(root,'gpu_basis','v9_native_basis_mex.cu'));
end

// Resident raw-eight Q4/FNO construction. The original packed Phi storage
// and exact-template scatter are shared with the Q4 assembly implementation.
#define MECHANICS2D_BASIS_GPU_EMBED
#include "../private/cuda/mechanics2d_basis_gpu_v1.cu"
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <vector>

namespace v9 {
using namespace basisgpu;
using Clock = std::chrono::steady_clock;

struct Window { int y0,x0,ny,nx; };
__device__ Window bounds(Grid g, int owner) {
    const int ix=owner/g.sy, iy=owner%g.sy;
    const int x0=max(0,ix*g.stride-g.layer), y0=max(0,iy*g.stride-g.layer);
    const int x1=min(g.nx,(ix+1)*g.stride+1+g.layer);
    const int y1=min(g.ny,(iy+1)*g.stride+1+g.layer);
    return {y0,x0,y1-y0,x1-x0};
}

__global__ void classifyClipped(const double* E, Grid g, int* flags) {
    const int owner=blockIdx.x;
    const auto w=bounds(g,owner);
    const double first=E[w.y0+size_t(g.ny)*w.x0];
    bool different=false;
    for(int i=threadIdx.x;i<w.ny*w.nx;i+=blockDim.x) {
        const int y=w.y0+i%w.ny,x=w.x0+i/w.ny;
        different |= E[y+size_t(g.ny)*x]!=first;
    }
    const int any=__syncthreads_or(different);
    if(threadIdx.x==0) flags[owner]=!any;
}

__global__ void extractNative(const double* E,const int* ids,Grid g,bool edge,
                              float* inputs,int* fault) {
    const int b=blockIdx.x,owner=ids[b];
    if(owner<0||owner>=g.ns) { if(threadIdx.x==0) atomicOr(fault,64); return; }
    const int tid=type(g,owner);
    if((edge && tid!=2 && tid!=4 && tid!=6 && tid!=8) || (!edge && tid!=5)) {
        if(threadIdx.x==0) atomicOr(fault,64); return;
    }
    const auto w=bounds(g,owner);
    const int h=edge?16:22, width=22;
    for(int i=threadIdx.x;i<h*width;i+=blockDim.x) {
        const int cy=i/width,cx=i%width;
        int x=cx,y=cy;
        if(tid==2){x=w.nx-1-cx;y=w.ny-1-cy;}
        else if(tid==4){x=cy;y=w.ny-1-cx;}
        else if(tid==6){x=w.nx-1-cy;y=cx;}
        inputs[size_t(b)*h*width+i]=float(E[w.y0+y+size_t(g.ny)*(w.x0+x)]);
    }
}

__global__ void scatterNative(const float* outputs,const int* ids,const int* lookup,
                              Grid g,bool edge,float* phi,int* filled,int* fault) {
    const int b=blockIdx.x,owner=ids[b];
    if(owner<0||owner>=g.ns) return;
    const int tid=type(g,owner),nf=edge?180:162;
    if((edge && tid!=2 && tid!=4 && tid!=6 && tid!=8) || (!edge && tid!=5)) return;
    for(int j=threadIdx.x;j<nf*8;j+=blockDim.x) {
        const int row=j%nf,col=j/nf;
        const int code=lookup[row+200*(col+8*(tid-1))];
        if(code==0||code>nf*8||code< -nf*8){atomicOr(fault,128);continue;}
        float val=outputs[size_t(b)*nf*8+abs(code)-1];
        if(code<0) val=-val;
        if(!isfinite(val)) atomicOr(fault,4);
        phi[globalDof(g,owner,row)+size_t(g.nd)*col]=val;
    }
    if(threadIdx.x==0 && atomicCAS(filled+owner,0,1)!=0)atomicOr(fault,16);
}

struct Events {
    std::vector<cudaEvent_t> marks;
    ~Events(){for(auto e:marks)cudaEventDestroy(e);}
    void mark(cudaStream_t stream){
        cudaEvent_t e;ck(cudaEventCreate(&e));marks.push_back(e);ck(cudaEventRecord(e,stream));
    }
    double span(size_t a,size_t b) const{
        float t=0;ck(cudaEventElapsedTime(&t,marks[a],marks[b]));return .001*double(t);
    }
};
struct Loaded {
    std::unique_ptr<torch::jit::Module> module;
    std::filesystem::file_time_type modified;
    uintmax_t size=0;
};
std::map<std::pair<int,std::string>,Loaded> cache;
bool exitRegistered=false;
void release(){cache.clear();c10::cuda::CUDACachingAllocator::emptyCache();}
void onExit(){try{basisgpu::cleanup();release();}catch(...){}}

std::string modelPath(const mxArray* argument){
    req(mxIsChar(argument),"Expected TorchScript path as a character vector");
    char* utf8=mxArrayToUTF8String(argument);req(utf8!=nullptr,"Invalid model path");
    std::string result(utf8);mxFree(utf8);return result;
}

torch::jit::Module& loadModule(const std::string& path,int device,bool& hit) {
    const auto file=std::filesystem::u8path(path);
    req(std::filesystem::is_regular_file(file),"TorchScript file not found");
    const auto stamp=std::filesystem::last_write_time(file);
    const auto size=std::filesystem::file_size(file);
    auto& entry=cache[{device,path}];
    hit=entry.module && entry.modified==stamp && entry.size==size;
    if(!hit){
        std::ifstream source(file,std::ios::binary);
        req(source.is_open(),"Unable to open TorchScript archive");
        auto model=torch::jit::load(source,torch::Device(torch::kCUDA,device));
        model.eval();
        entry.module=std::make_unique<torch::jit::Module>(std::move(model));
        entry.modified=stamp;entry.size=size;
        if(!exitRegistered){mexAtExit(onExit);exitRegistered=true;}
    }
    return *entry.module;
}

void set(mxArray* s,const char* k,double value){mxSetField(s,0,k,mxCreateDoubleScalar(value));}
void infer(int nlhs,mxArray** out,int nrhs,const mxArray** in) {
    req(nrhs==7 && nlhs==1 && bool(session),"infer expects begin + E, owner IDs, kind, path, batch, signed map");
    auto started=Clock::now();
    const auto g=session->g;req(g.stride==9 && g.layer==6,"100k models require stride=9, halo=6");
    Ref material(in[1],mxDOUBLE_CLASS),ids(in[2],mxINT32_CLASS),lookup(in[6],mxINT32_CLASS);
    const auto kind=str(in[3]),path=modelPath(in[4]);
    req(kind=="edge"||kind=="internal","kind must be internal/edge");
    req(material.n()==size_t(g.nx)*g.ny && lookup.n()==200*8*9,"Wrong material or D4 map shape");
    req(ids.n()>0 && ids.n()<=size_t(g.ns),"Empty or oversized owner set");
    const int limit=asint(*vec(in[5],1),1,4096);
    const int total=int(ids.n()),batch=std::min(total,limit),h=kind=="edge"?16:22;
    const bool edge=kind=="edge";
    int device=0;ck(cudaGetDevice(&device));
    // This exported ATen CUDA symbol keeps torch_cuda.dll in the Windows
    // import table so the CUDA dispatcher is registered before jit::load.
    (void)at::cuda::getCurrentDeviceProperties();
    auto ts=c10::cuda::getCurrentCUDAStream(device);
    cudaStream_t stream=ts.stream();
    c10::InferenceMode noGrad;
    torch::jit::setTensorExprFuserEnabled(false);
    ck(cudaDeviceSynchronize());
    bool hit=false;auto loadStart=Clock::now();
    auto& model=loadModule(path,device,hit);ck(cudaDeviceSynchronize());
    const double loadSeconds=std::chrono::duration<double>(Clock::now()-loadStart).count();
    const auto options=torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA,device);
    auto storage=torch::empty({batch,h,22},options);
    Events times;
    for(int first=0;first<total;first+=batch){
        const int count=std::min(batch,total-first);
        const int* selected=static_cast<const int*>(ids.p())+first;
        auto input=storage.narrow(0,0,count);
        times.mark(stream);
        extractNative<<<count,128,0,stream>>>(static_cast<const double*>(material.p()),selected,g,edge,
            input.data_ptr<float>(),session->flag.p);
        ck(cudaGetLastError()); times.mark(stream);
        auto value=model.forward({input}).toTensor().contiguous();
        req(value.scalar_type()==torch::kFloat32 && value.is_cuda()&&value.get_device()==device&&value.dim()==4 &&
            value.size(0)==count && value.size(1)==16 && value.size(2)==(edge?10:9) && value.size(3)==9,
            "Frozen FNO output must be [B,16,redY,9] FP32 on the current GPU");
        times.mark(stream);
        scatterNative<<<count,256,0,stream>>>(value.data_ptr<float>(),selected,static_cast<const int*>(lookup.p()),
            g,edge,static_cast<float*>(session->Phi.p()),session->filled.p,session->flag.p);
        ck(cudaGetLastError()); times.mark(stream);
    }
    ck(cudaStreamSynchronize(stream));
    double extract=0,forward=0,scatter=0;
    for(size_t i=0;i<times.marks.size();i+=4){
        extract+=times.span(i,i+1);forward+=times.span(i+1,i+2);scatter+=times.span(i+2,i+3);
    }
    const char* names[]={"owner_count","batches","cache_hit","load_seconds","gather_seconds","forward_seconds","scatter_seconds","total_seconds"};
    auto stat=mxCreateStructMatrix(1,1,8,names);
    set(stat,"owner_count",total);set(stat,"batches",double(times.marks.size()/4));
    set(stat,"cache_hit",hit);set(stat,"load_seconds",loadSeconds);set(stat,"gather_seconds",extract);
    set(stat,"forward_seconds",forward);set(stat,"scatter_seconds",scatter);
    set(stat,"total_seconds",std::chrono::duration<double>(Clock::now()-started).count());
    out[0]=stat;
}
}

void mexFunction(int nlhs,mxArray** out,int nrhs,const mxArray** in){
    try{
        using namespace basisgpu;
        req(nrhs>=1,"Expected a command");const auto cmd=str(in[0]);mxInitGPU();
        if(cmd=="classify"){
            req(nlhs==1 && nrhs==3,"flags=classify(E,grid)");
            const Grid g=grid(in[2]);Ref material(in[1],mxDOUBLE_CLASS);
            req(material.n()==size_t(g.ny)*g.nx,"Wrong physical material shape");
            Out flags(g.ns,1,mxINT32_CLASS);v9::classifyClipped<<<g.ns,128>>>(static_cast<const double*>(material.p()),g,static_cast<int*>(flags.p()));
            ck(cudaGetLastError());out[0]=flags.give();return;
        }
        if(cmd=="infer"){v9::infer(nlhs,out,nrhs,in);return;}
        if(cmd=="release"||cmd=="trim"){
            req(nlhs==0&&nrhs==1,"Lifecycle operation takes no arguments or outputs");
            ck(cudaDeviceSynchronize());
            if(cmd=="release")v9::release();else c10::cuda::CUDACachingAllocator::emptyCache();
            return;
        }
        req(cmd=="begin"||cmd=="abort"||cmd=="put"||cmd=="finish","Unknown V9 native raw basis command");
        basisgpu::run(nlhs,out,nrhs,in);
    }catch(const std::exception& e){mexErrMsgIdAndTxt("V9:NativeFNO","%s",e.what());}
}

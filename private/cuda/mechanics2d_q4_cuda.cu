/* 2D plane-stress Q4 port, 2026-10-25. See port_provenance.json. */
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mechanics2d_q4 {
void req(bool v,const std::string& s){if(!v)throw std::runtime_error(s);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void cb(cublasStatus_t s){req(s==CUBLAS_STATUS_SUCCESS,"cuBLAS failed: "+std::to_string(int(s)));}
int blocks(size_t n){return int(std::min<size_t>(4096,(n+255)/256));}
template<class T> struct Buf {
    T* p=nullptr;size_t n=0;
    Buf()=default;Buf(const Buf&)=delete;Buf& operator=(const Buf&)=delete;
    ~Buf(){if(p)cudaFree(p);} void alloc(size_t m){req(!p,"Buffer allocated twice");n=m;if(m)ck(cudaMalloc((void**)&p,m*sizeof(T)));}
    void zero(){if(n)ck(cudaMemset(p,0,n*sizeof(T)));}
};
struct Ref {
    const mxGPUArray* p=nullptr;
    explicit Ref(const mxArray* a){req(mxIsGPUArray(a),"Inputs must be gpuArray");p=mxGPUCreateFromMxArray(a);
        req(mxGPUGetClassID(p)==mxDOUBLE_CLASS&&mxGPUGetComplexity(p)==mxREAL,"Inputs must be real FP64 gpuArray");}
    ~Ref(){if(p)mxGPUDestroyGPUArray(p);} const double* data()const{return (const double*)mxGPUGetDataReadOnly(p);}
    size_t count()const{return mxGPUGetNumberOfElements(p);}
};
struct Out {
    mxGPUArray* p=nullptr;
    explicit Out(size_t n){mwSize d[2]={n,1};p=mxGPUCreateGPUArray(2,d,mxDOUBLE_CLASS,mxREAL,MX_GPU_DO_NOT_INITIALIZE);}
    ~Out(){if(p)mxGPUDestroyGPUArray(p);}double* data(){return (double*)mxGPUGetData(p);}
};
const mxArray* field(const mxArray* a,const char* n){auto* f=mxGetField(a,0,n);req(f,std::string("Missing field ")+n);return f;}
double scalar(const mxArray* a){req(a&&!mxIsGPUArray(a)&&mxIsNumeric(a)&&!mxIsComplex(a)&&mxGetNumberOfElements(a)==1,"Expected CPU scalar");double x=mxGetScalar(a);req(std::isfinite(x),"Scalar must be finite");return x;}
double opt(const mxArray* o,const char* n,double d){auto* f=mxGetField(o,0,n);return f?scalar(f):d;}
int asInt(double d,int lo,int hi){req(d>=lo&&d<=hi&&d==floor(d),"Integer option out of range");return int(d);}
std::string text(const mxArray* a){req(mxIsChar(a),"Command must be char");char* s=mxArrayToString(a);req(s,"Invalid string");std::string r(s);mxFree(s);return r;}

struct Shape {int nx,ny,ox,oy,stride,bc,n,ne;double edge;};
__host__ __device__ int node(Shape s,int x,int y){return y+(s.ny+1)*x;}
__host__ __device__ bool fixedFace(Shape s,int x,int y){return (s.bc==1&&x==0)||(s.bc==2&&x==s.nx)||(s.bc==3&&y==0)||(s.bc==4&&y==s.ny);}
__device__ bool fixedDof(Shape s,int d){int a=d/2;return fixedFace(s,a/(s.ny+1),a%(s.ny+1));}
__host__ __device__ int corner(int x,int y){return y?(x?2:3):(x?1:0);}
__host__ __device__ int exNode(int a){return a==1||a==2;}
__host__ __device__ int eyNode(int a){return a>=2;}
__host__ __device__ int type(Shape s,int x,int y){return (x==s.nx-1?1:0)+(y==s.ny-1?2:0);}
__host__ __device__ double width(int e,int stride,int original,double edge){return double((e+1)*stride<original?stride:original-e*stride)*edge;}
__global__ void validate(const double* a,size_t n,bool positive,int* bad){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)if(!isfinite(a[i])||(positive&&a[i]<=0))atomicExch(bad,1);}
__global__ void clearAction(Shape s,const double* x,double* y){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(s.n);i+=size_t(gridDim.x)*blockDim.x)y[i]=fixedDof(s,int(i))?x[i]:0;}
__constant__ double cKe0[64];
struct Dot2 {
    double sum,correction;
    __device__ __forceinline__ Dot2():sum(0.0),correction(0.0){}
    __device__ __forceinline__ void add(double a,double x){
        const double product=__dmul_rn(a,x);
        const double productError=__fma_rn(a,x,-product);
        const double next=__dadd_rn(sum,product);
        const double z=__dsub_rn(next,sum);
        const double sumError=__dadd_rn(__dsub_rn(sum,__dsub_rn(next,z)),__dsub_rn(product,z));
        correction=__dadd_rn(correction,__dadd_rn(productError,sumError));
        sum=next;
    }
    __device__ __forceinline__ double value()const{return __dadd_rn(sum,correction);}
};

template<bool Finest,bool Accurate>
__global__ void __launch_bounds__(128) actionColourElements(Shape s,const double* __restrict__ rho,const double* __restrict__ ke,const double* __restrict__ x,double* __restrict__ y,int ox,int oy,int cx,int cy){
 for(size_t e=size_t(blockIdx.x)*blockDim.x+threadIdx.x;e<size_t(cx)*cy;e+=size_t(gridDim.x)*blockDim.x){
  int ey=2*int(e%cy)+oy,ex=2*int(e/cy)+ox;double v[8];int base[4];bool fx[8];
  #pragma unroll
  for(int a=0;a<4;a++){base[a]=2*node(s,ex+exNode(a),ey+eyNode(a));
   #pragma unroll
   for(int c=0;c<2;c++){fx[2*a+c]=fixedDof(s,base[a]+c);v[2*a+c]=fx[2*a+c]?0:x[base[a]+c];}}
  double r=rho[ey+s.ny*ex];const double* k=ke+64*type(s,ex,ey);
  #pragma unroll
  for(int i=0;i<8;i++){if(fx[i])continue;double sum=0;Dot2 dot;
   #pragma unroll
   for(int j=0;j<8;j++){double a=Finest?cKe0[i+8*j]:__ldg(k+i+8*j);if(Accurate)dot.add(a,v[j]);else sum+=a*v[j];}
   if(Accurate)y[base[i/2]+i%2]=__fma_rn(r,dot.value(),y[base[i/2]+i%2]);else y[base[i/2]+i%2]+=r*sum;}
 }
}
__global__ void smoothFromZero(Shape s,const double* r,const double* diag,double* z,double omega){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(s.n);i+=size_t(gridDim.x)*blockDim.x)
        z[i]=fixedDof(s,int(i))?0:__fma_rn(__dmul_rn(omega,diag[i]),r[i],0.0);
}
__global__ void diagonal(Shape s,const double* rho,const double* ke,const double* absrow,double* inv,double* bound,int* bad){
 for(size_t d=size_t(blockIdx.x)*blockDim.x+threadIdx.x;d<size_t(s.n);d+=size_t(gridDim.x)*blockDim.x){int a=int(d)/2,c=int(d)%2,iy=a%(s.ny+1),ix=a/(s.ny+1);
  if(fixedDof(s,int(d))){inv[d]=1;continue;}double dg=0,ab=0;
  for(int x=0;x<2;x++)for(int y=0;y<2;y++){int ex=ix-x,ey=iy-y;if(ex<0||ey<0||ex>=s.nx||ey>=s.ny)continue;
   int t=type(s,ex,ey),row=2*corner(x,y)+c;double r=rho[ey+s.ny*ex];dg+=r*ke[64*t+row+8*row];ab+=r*absrow[8*t+row];}
  if(!(dg>0)||!isfinite(dg)||!isfinite(ab)){atomicExch(bad,1);inv[d]=0;continue;}inv[d]=1/dg;atomicMax((unsigned long long*)bound,__double_as_longlong(ab/dg));}
}
__global__ void averageRho(Shape f,Shape c,const double* rf,double* rc){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(c.ne);i+=size_t(gridDim.x)*blockDim.x){int y=int(i%c.ny),x=int(i/c.ny);double sum=0,vol=0;
  for(int dx=0;dx<2;dx++)for(int dy=0;dy<2;dy++){int fx=2*x+dx,fy=2*y+dy;if(fx>=f.nx||fy>=f.ny)continue;double v=width(fx,f.stride,f.ox,f.edge)*width(fy,f.stride,f.oy,f.edge);sum+=v*rf[fy+f.ny*fx];vol+=v;}rc[i]=sum/vol;}
}
__device__ void interp(int i,int nf,int original,int stride,int& lo,int& hi,double& wh){int nc=(nf+1)/2;if(i==nf){lo=hi=nc;wh=0;return;}lo=i/2;hi=lo+1;int q=min(i*stride,original),l=min(lo*stride*2,original),h=min(hi*stride*2,original);wh=double(q-l)/double(h-l);}
__device__ double weight(int i,int coarse,int nf,int original,int stride){int lo,hi;double w;interp(i,nf,original,stride,lo,hi,w);return coarse==lo?1-w:(coarse==hi?w:0);}
__global__ void restrictTranspose(Shape f,Shape c,const double* r,const double* az,double* out){
 for(size_t d=size_t(blockIdx.x)*blockDim.x+threadIdx.x;d<size_t(c.n);d+=size_t(gridDim.x)*blockDim.x){int a=int(d)/2,q=int(d)%2,y=a%(c.ny+1),x=a/(c.ny+1);if(fixedDof(c,int(d))){out[d]=0;continue;}double sum=0;
  for(int ix=max(0,2*x-1);ix<=min(f.nx,2*x+1);ix++)for(int iy=max(0,2*y-1);iy<=min(f.ny,2*y+1);iy++){int j=2*node(f,ix,iy)+q;if(fixedDof(f,j))continue;double w=weight(ix,x,f.nx,f.ox,f.stride)*weight(iy,y,f.ny,f.oy,f.stride);sum+=w*(r[j]-az[j]);}out[d]=sum;}
}
__global__ void prolongAdd(Shape f,Shape c,const double* in,double* out){
 for(size_t d=size_t(blockIdx.x)*blockDim.x+threadIdx.x;d<size_t(f.n);d+=size_t(gridDim.x)*blockDim.x){int a=int(d)/2,q=int(d)%2,iy=a%(f.ny+1),ix=a/(f.ny+1);if(fixedDof(f,int(d)))continue;int lx,hx,ly,hy;double wx,wy;
  interp(ix,f.nx,f.ox,f.stride,lx,hx,wx);interp(iy,f.ny,f.oy,f.stride,ly,hy,wy);double sum=0;
  for(int x=0;x<2;x++)for(int y=0;y<2;y++){int cx=x?hx:lx,cy=y?hy:ly;double w=(x?wx:1-wx)*(y?wy:1-wy);int j=2*node(c,cx,cy)+q;if(!fixedDof(c,j))sum+=w*in[j];}out[d]+=sum;}
}
__global__ void smooth(Shape s,const double* r,const double* az,const double* diag,double* z,double omega){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(s.n);i+=size_t(gridDim.x)*blockDim.x)
        z[i]=fixedDof(s,int(i))?0:z[i]+omega*diag[i]*(r[i]-az[i]);
}
__global__ void initialize(Shape s,const double* x0,double* x){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(s.n);i+=size_t(gridDim.x)*blockDim.x)x[i]=fixedDof(s,int(i))?0:(x0?x0[i]:0);
}
__global__ void residual(Shape s,const double* b,const double* ax,double* r){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(s.n);i+=size_t(gridDim.x)*blockDim.x)r[i]=fixedDof(s,int(i))?0:b[i]-ax[i];
}
__global__ void direction(const double* z,double* p,size_t n,double beta){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)p[i]=z[i]+beta*p[i];
}
__global__ void update(double* x,double* r,const double* p,const double* ap,size_t n,double alpha){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x){x[i]+=alpha*p[i];r[i]-=alpha*ap[i];}
}

__global__ void elementEnergy(Shape s,const double* ke,const double* u,double* ce){
 for(size_t e=size_t(blockIdx.x)*blockDim.x+threadIdx.x;e<size_t(s.ne);e+=size_t(gridDim.x)*blockDim.x){int ey=int(e%s.ny),ex=int(e/s.ny);double v[8];
  for(int a=0;a<4;a++){int d=2*node(s,ex+exNode(a),ey+eyNode(a));for(int c=0;c<2;c++)v[2*a+c]=u[d+c];}const double* k=ke+64*type(s,ex,ey);Dot2 total;
  for(int i=0;i<8;i++){Dot2 q;for(int j=0;j<8;j++)q.add(k[i+8*j],v[j]);total.add(v[i],q.value());}ce[e]=total.value();}
}
struct Level {Shape s{};const double* rho=nullptr;std::vector<double> hostKe;Buf<double> ownedRho,ke,absrow,diag,r,z,tmp,chol;double omega=0;};
struct BlasOwner {cublasHandle_t h=nullptr;~BlasOwner(){if(h)cublasDestroy(h);}operator cublasHandle_t()const{return h;}};
struct Engine {
 std::vector<std::unique_ptr<Level>> L;BlasOwner blas;int sweeps;bool finestUniform=false;double nativeBytes=0;size_t freeBefore=0,total=0;
 explicit Engine(const mxArray* cells,const double* rho,const mxArray* opts,bool solve):sweeps(asInt(opt(opts,"smooth_steps",2),1,8)){
  req(mxIsCell(cells)&&mxGetNumberOfElements(cells)>0,"levels must be nonempty CPU cell");size_t count=solve?mxGetNumberOfElements(cells):1;req(count<=24,"Too many levels");
  for(size_t k=0;k<count;k++){auto l=std::make_unique<Level>();const mxArray* a=mxGetCell(cells,k);req(a&&mxIsStruct(a),"Malformed level");auto grid=field(a,"grid"),orig=field(a,"original_grid");req(mxIsDouble(grid)&&mxGetNumberOfElements(grid)==2&&mxIsDouble(orig)&&mxGetNumberOfElements(orig)==2,"Expected grid pairs");const double* g=mxGetPr(grid);const double* o=mxGetPr(orig);auto& s=l->s;
   s.nx=asInt(g[0],1,INT_MAX);s.ny=asInt(g[1],1,INT_MAX);s.ox=asInt(o[0],1,INT_MAX);s.oy=asInt(o[1],1,INT_MAX);s.stride=asInt(scalar(field(a,"stride")),1,INT_MAX/2);s.bc=asInt(scalar(field(a,"bc")),0,4);s.edge=scalar(field(a,"edge"));req(s.edge>0,"edge must be positive");
   double nd=2.0*(s.nx+1)*(s.ny+1),ne=double(s.nx)*s.ny;req(nd<=INT_MAX&&ne<=INT_MAX,"Grid exceeds int32");s.n=int(nd);s.ne=int(ne);req(s.nx==(s.ox+s.stride-1)/s.stride&&s.ny==(s.oy+s.stride-1)/s.stride,"Endpoints mismatch");
   if(k){auto f=L[k-1]->s;req(s.stride==2*f.stride&&s.ox==f.ox&&s.oy==f.oy&&s.bc==f.bc&&s.edge==f.edge,"Inconsistent levels");}else req(s.stride==1,"Finest stride must be one");
   auto* ka=field(a,"Ke");req(mxIsDouble(ka)&&!mxIsComplex(ka)&&!mxIsGPUArray(ka)&&mxGetNumberOfElements(ka)==64*4,"Ke must have four FP64 Q4 templates");l->hostKe.assign(mxGetPr(ka),mxGetPr(ka)+64*4);
   for(int t=0;t<4;t++)for(int j=0;j<8;j++)for(int i=0;i<8;i++){double v=l->hostKe[t*64+i+8*j],w=l->hostKe[t*64+j+8*i];req(std::isfinite(v)&&std::abs(v-w)<=1e-12*std::max(1.,std::abs(v)),"Nonfinite/nonsymmetric Ke");}
   nativeBytes+=8.0*(64*4+8*4+(solve?4*s.n:0)+(k?s.ne:0));L.push_back(std::move(l));}
  nativeBytes+=8.0*L[0]->s.n*(solve?2:1)+(solve?8.0*L.back()->s.n*L.back()->s.n:0)+1048576;if(solve)req(L.back()->s.n<=1024,"Terminal matrix too large");
  ck(cudaMemGetInfo(&freeBefore,&total));double reserve=opt(opts,"reserve_bytes",1073741824.),budget=opt(opts,"memory_budget_bytes",double(freeBefore));req(reserve>=0&&budget>0,"Invalid budget");req(nativeBytes<=std::min(budget,std::max(0.,double(freeBefore)-reserve)*.85),"Q4 native memory budget exceeded");cb(cublasCreate(&blas.h));Buf<int> bad;bad.alloc(1);bad.zero();
  const auto& k0=L[0]->hostKe;finestUniform=true;for(int t=1;t<4&&finestUniform;t++)finestUniform=std::memcmp(k0.data(),k0.data()+64*t,64*sizeof(double))==0;if(finestUniform)ck(cudaMemcpyToSymbol(cKe0,k0.data(),64*sizeof(double)));
  validate<<<blocks(L[0]->s.ne),256>>>(rho,L[0]->s.ne,true,bad.p);int hbad=0;ck(cudaMemcpy(&hbad,bad.p,sizeof(int),cudaMemcpyDeviceToHost));req(!hbad,"rho must be positive finite");
  for(size_t k=0;k<L.size();k++){auto& l=*L[k];auto s=l.s;l.ke.alloc(64*4);ck(cudaMemcpy(l.ke.p,l.hostKe.data(),64*4*sizeof(double),cudaMemcpyHostToDevice));if(k){l.ownedRho.alloc(s.ne);l.rho=l.ownedRho.p;averageRho<<<blocks(s.ne),256>>>(L[k-1]->s,s,L[k-1]->rho,l.ownedRho.p);}else l.rho=rho;
   if(solve){l.diag.alloc(s.n);l.r.alloc(s.n);l.z.alloc(s.n);l.tmp.alloc(s.n);l.absrow.alloc(8*4);std::vector<double> sums(8*4,0);for(int t=0;t<4;t++)for(int i=0;i<8;i++)for(int j=0;j<8;j++)sums[t*8+i]+=std::abs(l.hostKe[t*64+i+8*j]);ck(cudaMemcpy(l.absrow.p,sums.data(),sums.size()*sizeof(double),cudaMemcpyHostToDevice));Buf<double> bound;bound.alloc(1);bound.zero();diagonal<<<blocks(s.n),256>>>(s,l.rho,l.ke.p,l.absrow.p,l.diag.p,bound.p,bad.p);double beta=0;ck(cudaMemcpy(&beta,bound.p,sizeof(double),cudaMemcpyDeviceToHost));ck(cudaMemcpy(&hbad,bad.p,sizeof(int),cudaMemcpyDeviceToHost));req(!hbad&&std::isfinite(beta)&&beta>0,"Invalid smoothing bound");l.omega=std::min(.6,1.8/beta);}}
  if(solve)terminal();ck(cudaGetLastError());
 }
 void action(int k,const double* x,double* y,bool accurate){auto& l=*L[k];auto s=l.s;clearAction<<<blocks(s.n),256>>>(s,x,y);bool finest=k==0&&finestUniform;
  for(int xc=0;xc<2;xc++)for(int yc=0;yc<2;yc++){int cx=(s.nx+1-xc)/2,cy=(s.ny+1-yc)/2;size_t n=size_t(cx)*cy;if(!n)continue;int grid=int(std::min<size_t>(1<<20,(n+127)/128));
   if(accurate){if(finest)actionColourElements<true,true><<<grid,128>>>(s,l.rho,l.ke.p,x,y,xc,yc,cx,cy);else actionColourElements<false,true><<<grid,128>>>(s,l.rho,l.ke.p,x,y,xc,yc,cx,cy);}else{if(finest)actionColourElements<true,false><<<grid,128>>>(s,l.rho,l.ke.p,x,y,xc,yc,cx,cy);else actionColourElements<false,false><<<grid,128>>>(s,l.rho,l.ke.p,x,y,xc,yc,cx,cy);}}
 }
 void terminal(){auto& l=*L.back();auto s=l.s;int n=s.n;std::vector<double> rho(s.ne),a(size_t(n)*n,0);ck(cudaMemcpy(rho.data(),l.rho,s.ne*sizeof(double),cudaMemcpyDeviceToHost));
  for(int x=0;x<s.nx;x++)for(int y=0;y<s.ny;y++){const double* ke=l.hostKe.data()+64*type(s,x,y);double r=rho[y+s.ny*x];int dof[8];bool f[8];for(int j=0;j<8;j++){int v=j/2;dof[j]=2*node(s,x+exNode(v),y+eyNode(v))+j%2;f[j]=fixedFace(s,x+exNode(v),y+eyNode(v));}for(int j=0;j<8;j++)if(!f[j])for(int i=0;i<8;i++)if(!f[i])a[dof[i]+size_t(n)*dof[j]]+=r*ke[i+8*j];}
  for(int d=0;d<n;d++)if(fixedFace(s,d/2/(s.ny+1),(d/2)%(s.ny+1)))a[d+size_t(n)*d]=1;
  for(int j=0;j<n;j++){double v=a[j+size_t(n)*j];for(int k=0;k<j;k++)v-=a[j+size_t(n)*k]*a[j+size_t(n)*k];req(v>0&&std::isfinite(v),"Terminal not SPD");a[j+size_t(n)*j]=std::sqrt(v);for(int i=j+1;i<n;i++){double q=a[i+size_t(n)*j];for(int k=0;k<j;k++)q-=a[i+size_t(n)*k]*a[j+size_t(n)*k];a[i+size_t(n)*j]=q/a[j+size_t(n)*j];}for(int i=0;i<j;i++)a[i+size_t(n)*j]=0;}l.chol.alloc(a.size());ck(cudaMemcpy(l.chol.p,a.data(),a.size()*sizeof(double),cudaMemcpyHostToDevice));
 }
    void cycle(int k=0){
        auto& l=*L[k];auto s=l.s;
        if(k==int(L.size())-1){ck(cudaMemcpy(l.z.p,l.r.p,s.n*sizeof(double),cudaMemcpyDeviceToDevice));cb(cublasDtrsv(blas,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,s.n,l.chol.p,s.n,l.z.p,1));cb(cublasDtrsv(blas,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,s.n,l.chol.p,s.n,l.z.p,1));return;}
        smoothFromZero<<<blocks(s.n),256>>>(s,l.r.p,l.diag.p,l.z.p,l.omega);
        for(int i=1;i<sweeps;i++){action(k,l.z.p,l.tmp.p,false);smooth<<<blocks(s.n),256>>>(s,l.r.p,l.tmp.p,l.diag.p,l.z.p,l.omega);}
        action(k,l.z.p,l.tmp.p,false);auto& c=*L[k+1];restrictTranspose<<<blocks(c.s.n),256>>>(s,c.s,l.r.p,l.tmp.p,c.r.p);cycle(k+1);
        prolongAdd<<<blocks(s.n),256>>>(s,c.s,c.z.p,l.z.p);
        for(int i=0;i<sweeps;i++){action(k,l.z.p,l.tmp.p,false);smooth<<<blocks(s.n),256>>>(s,l.r.p,l.tmp.p,l.diag.p,l.z.p,l.omega);}
    }
    double norm(const double* x,int n){double v;cb(cublasDnrm2(blas,n,x,1,&v));return v;}
    double dot(const double* x,const double* y,int n){double v;cb(cublasDdot(blas,n,x,1,y,1,&v));return v;}
};
void put(mxArray* a,const char* name,double v){mxSetField(a,0,name,mxCreateDoubleScalar(v));}
mxArray* stats(Engine& e,int flag,int it,double rel,double initial,const std::vector<double>& history,double setup,double solve,double compliance,double energy){
    const char* f[]={"flag","iterations","true_relres","initial_true_relres","history","history_columns","setup_seconds","solve_seconds","native_budget_bytes","free_bytes_before_native","total_device_bytes","level_dofs","smoother_omega","compliance","strain_energy","backend","target_operator","preconditioner","precision","accumulation"};
    mxArray* o=mxCreateStructMatrix(1,1,20,f);put(o,"flag",flag);put(o,"iterations",it);put(o,"true_relres",rel);put(o,"initial_true_relres",initial);put(o,"setup_seconds",setup);put(o,"solve_seconds",solve);put(o,"native_budget_bytes",e.nativeBytes);put(o,"free_bytes_before_native",double(e.freeBefore));put(o,"total_device_bytes",double(e.total));put(o,"compliance",compliance);put(o,"strain_energy",energy);
    size_t rows=history.size()/5;mxArray* h=mxCreateDoubleMatrix(rows,5,mxREAL);for(size_t i=0;i<rows;i++)for(int j=0;j<5;j++)mxGetPr(h)[i+rows*j]=history[i*5+j];mxSetField(o,0,"history",h);
    mxSetField(o,0,"history_columns",mxCreateString("iteration, recursive_relres, true_relres, compliance, strain_energy"));
    mxArray* n=mxCreateDoubleMatrix(1,e.L.size(),mxREAL);mxArray* w=mxCreateDoubleMatrix(1,e.L.size(),mxREAL);for(size_t i=0;i<e.L.size();i++){mxGetPr(n)[i]=e.L[i]->s.n;mxGetPr(w)[i]=e.L[i]->omega;}mxSetField(o,0,"level_dofs",n);mxSetField(o,0,"smoother_omega",w);
    mxSetField(o,0,"backend",mxCreateString("mechanics2d-q4-matrixfree-dot2-firsttrue"));mxSetField(o,0,"target_operator",mxCreateString("standard conforming plane-stress Q4; xy interleaved; y-x"));mxSetField(o,0,"preconditioner",mxCreateString("symmetric geometric V-cycle; volume-averaged rediscretization; R=P transpose"));mxSetField(o,0,"precision",mxCreateString("FP64"));
    mxSetField(o,0,"accumulation",mxCreateString("target Ap/Ax/raw matvec and element energy: FMA-Dot2; colour scatter: fixed-order FMA; V-cycle: original ordinary FP64; unchanged Ke and constraints"));return o;
}
void run(int nlhs,mxArray** out,int nrhs,const mxArray** in){
    req(nrhs>=5&&nlhs<=2,"Use mex('solve',levels,rho,b,x0,opts), mex('matvec',levels,rho,u,opts), or mex('apply',levels,rho,r,opts)");std::string command=text(in[0]);bool solving=command=="solve",applying=command=="apply",energyCommand=command=="energy";req(solving||applying||command=="matvec"||energyCommand,"Unknown command");req(nrhs==(solving?6:5),"Wrong number of inputs");
    const mxArray* opts=in[solving?5:4];req(mxIsStruct(opts)&&mxGetNumberOfElements(opts)==1,"opts must be scalar struct");Ref rho(in[2]),b(in[3]);std::unique_ptr<Ref> x0;
    if(solving&&mxGetNumberOfElements(in[4]))x0=std::make_unique<Ref>(in[4]);
    req(mxIsCell(in[1])&&mxGetNumberOfElements(in[1])>0,"levels must be nonempty cell");
    const mxArray* first=mxGetCell(in[1],0);req(first&&mxIsStruct(first),"Invalid first level");
    const mxArray* ga=field(first,"grid");req(mxIsDouble(ga)&&!mxIsGPUArray(ga)&&mxGetNumberOfElements(ga)==2,"Invalid grid triple");
    const double* g=mxGetPr(ga);double ne=g[0]*g[1],nd=2*(g[0]+1)*(g[1]+1);
    req(std::isfinite(ne)&&std::isfinite(nd)&&ne==double(rho.count())&&nd==double(b.count())&&(!x0||nd==double(x0->count())),"rho/b/x0 length mismatch");
    const auto start=std::chrono::steady_clock::now();Engine e(in[1],rho.data(),opts,solving||applying);auto& top=*e.L[0];auto s=top.s;
    Buf<int> bad;bad.alloc(1);bad.zero();validate<<<blocks(b.count()),256>>>(b.data(),b.count(),false,bad.p);if(x0)validate<<<blocks(x0->count()),256>>>(x0->data(),x0->count(),false,bad.p);int invalid=0;ck(cudaMemcpy(&invalid,bad.p,sizeof(int),cudaMemcpyDeviceToHost));req(!invalid,"Nonfinite input vector");
    if(energyCommand){Out ce(s.ne);elementEnergy<<<blocks(s.ne),256>>>(s,top.ke.p,b.data(),ce.data());ck(cudaGetLastError());ck(cudaDeviceSynchronize());if(nlhs)out[0]=mxGPUCreateMxArrayOnGPU(ce.p);return;}
    Out x(s.n);if(!solving){
        if(applying){initialize<<<blocks(s.n),256>>>(s,b.data(),top.r.p);e.cycle();ck(cudaMemcpy(x.data(),top.z.p,s.n*sizeof(double),cudaMemcpyDeviceToDevice));}
        else e.action(0,b.data(),x.data(),true);
        ck(cudaDeviceSynchronize());if(nlhs)out[0]=mxGPUCreateMxArrayOnGPU(x.p);if(nlhs>1)out[1]=stats(e,0,0,0,0,{},0,0,0,0);return;
    }
    double tol=opt(opts,"rtol",1e-8);req(tol>0&&tol<1,"rtol must be in (0,1)");int maxit=asInt(opt(opts,"maxit",1000),0,100000),every=asInt(opt(opts,"check_every",5),1,100000);bool verbose=opt(opts,"verbose",1)!=0;
    const double recursiveStop=opt(opts,"recursive_stop_below",0);req(std::isfinite(recursiveStop)&&recursiveStop>=0,"Invalid recursive stopping threshold");
    const double trueStop=opt(opts,"true_stop_below",0);req(std::isfinite(trueStop)&&trueStop>=0,"Invalid true stopping threshold");req(!(trueStop>0&&recursiveStop>0),"True and recursive early stop are mutually exclusive");
    double lastRecursive=INFINITY,previousRecursive=INFINITY,previousTrue=INFINITY;
    std::string stopFile;const mxArray* stop=mxGetField(opts,0,"stop_file");if(stop)stopFile=text(stop);
    Buf<double> p;p.alloc(s.n);p.zero();initialize<<<blocks(s.n),256>>>(s,x0?x0->data():nullptr,x.data());
    // Masked RHS norm without retaining an additional fine vector.
    initialize<<<blocks(s.n),256>>>(s,b.data(),top.tmp.p);double bn=e.norm(top.tmp.p,s.n),den=bn>0?bn:1;
    ck(cudaDeviceSynchronize());double setup=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();const auto solveStart=std::chrono::steady_clock::now();
    if(bn==0)ck(cudaMemset(x.data(),0,s.n*sizeof(double)));
    std::vector<double> history;double rel=INFINITY,recursive=INFINITY,compliance=0,energy=0,oldrz=0;int flag=1,it=0;bool restart=true;
    auto check=[&](int iteration,double rec){
        e.action(0,x.data(),top.tmp.p,true);energy=.5*e.dot(x.data(),top.tmp.p,s.n);compliance=e.dot(x.data(),b.data(),s.n);
        residual<<<blocks(s.n),256>>>(s,b.data(),top.tmp.p,top.tmp.p);previousTrue=rel;rel=e.norm(top.tmp.p,s.n)/den;
        if(iteration==0)rec=rel;lastRecursive=rec;history.insert(history.end(),{double(iteration),rec,rel,compliance,energy});
        if(!std::isfinite(rel)||!std::isfinite(rec)||!std::isfinite(energy)||!std::isfinite(compliance))flag=4;
        else if(trueStop>0){if(rel<trueStop)flag=6;}
        else if(recursiveStop>0){if(rec<recursiveStop)flag=5;}
        else if(rel<=tol)flag=0;
        if(!stopFile.empty()){FILE* f=std::fopen(stopFile.c_str(),"rb");if(f){std::fclose(f);flag=2;}}
        if(verbose){double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-solveStart).count();mexPrintf("[fine_q4] iter=%d recursive=%.12e true=%.12e energy=%.12e compliance=%.12e elapsed=%.3fs flag=%d\n",iteration,rec,rel,energy,compliance,elapsed,flag);std::fflush(stdout);}
    };
    check(0,0);double initial=rel;recursive=rel;previousRecursive=rel;ck(cudaMemcpy(top.r.p,top.tmp.p,s.n*sizeof(double),cudaMemcpyDeviceToDevice));
    for(it=0;it<maxit&&flag==1;){
        e.cycle();double rz=e.dot(top.r.p,top.z.p,s.n);if(!(rz>0)||!std::isfinite(rz)){flag=4;break;}
        double beta=restart?0:rz/oldrz;if(!std::isfinite(beta)||beta<0){flag=4;break;}
        direction<<<blocks(s.n),256>>>(top.z.p,p.p,s.n,beta);e.action(0,p.p,top.tmp.p,true);double pq=e.dot(p.p,top.tmp.p,s.n);if(!(pq>0)||!std::isfinite(pq)){flag=4;break;}
        double alpha=rz/pq;if(!std::isfinite(alpha)){flag=4;break;}update<<<blocks(s.n),256>>>(x.data(),top.r.p,p.p,top.tmp.p,s.n,alpha);++it;oldrz=rz;restart=false;
        previousRecursive=recursive;recursive=e.norm(top.r.p,s.n)/den;if(!std::isfinite(recursive)){flag=4;break;}
        if(trueStop>0||it%every==0||it==maxit||(recursiveStop>0?recursive<recursiveStop:recursive<=tol)){check(it,recursive);if(flag==1&&trueStop==0&&recursiveStop==0&&recursive<=tol){ck(cudaMemcpy(top.r.p,top.tmp.p,s.n*sizeof(double),cudaMemcpyDeviceToDevice));restart=true;}}
    }
    if(history.empty()||int(history[history.size()-5])!=it){int oldflag=flag;check(it,recursive);if(oldflag==4)flag=4;}
    ck(cudaGetLastError());ck(cudaDeviceSynchronize());double solve=std::chrono::duration<double>(std::chrono::steady_clock::now()-solveStart).count();
    if(nlhs)out[0]=mxGPUCreateMxArrayOnGPU(x.p);if(nlhs>1){
        out[1]=stats(e,flag,it,rel,initial,history,setup,solve,compliance,energy);
        mxAddField(out[1],"recursive_relres");put(out[1],"recursive_relres",lastRecursive);
        mxAddField(out[1],"previous_recursive_relres");put(out[1],"previous_recursive_relres",previousRecursive);
        mxAddField(out[1],"recursive_stop_below");put(out[1],"recursive_stop_below",recursiveStop);
        mxAddField(out[1],"true_stop_below");put(out[1],"true_stop_below",trueStop);
        mxAddField(out[1],"previous_true_relres");put(out[1],"previous_true_relres",previousTrue);
        mxAddField(out[1],"stop_reason");mxSetField(out[1],0,"stop_reason",mxCreateString(flag==6?"first_true_strictly_below_threshold":flag==5?"first_recursive_strictly_below_threshold":flag==0?"true_residual_target_met":flag==2?"STOP_requested":"not_converged"));
    }
}
}
void mexFunction(int nlhs,mxArray* out[],int nrhs,const mxArray* in[]){try{mxInitGPU();mechanics2d_q4::run(nlhs,out,nrhs,in);}catch(const std::exception& e){mexErrMsgIdAndTxt("mechanics2d_q4:Failure","%s",e.what());}}

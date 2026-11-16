/* Structured owner-pair assembly and exact block-Galerkin MGCG, 2026-10-25.
 * Assembly follows stream_assemble_structured_cuda_3D.cu: one CTA owns a
 * unique upper owner pair, visits only its support intersection, and writes
 * one block. No COO, sorting, sparse summation, global plan, or atomics.
 * Five FP64 8x8 blocks per owner; FP32/FP64 input basis is storage only.
 * The original sixteen 2-D integration-domain matrices are used unchanged.
 */
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace block2d {
void req(bool b,const char* s){if(!b)throw std::runtime_error(s);}
void ck(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
void cb(cublasStatus_t s){req(s==CUBLAS_STATUS_SUCCESS,"cuBLAS failure");}
int launch(size_t n){return int(std::min<size_t>(8192,(n+255)/256));}
double elapsed(std::chrono::steady_clock::time_point t){return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();}
struct Ref {
 const mxGPUArray* a=nullptr;
 Ref(const mxArray* x,mxClassID type=mxDOUBLE_CLASS){req(mxIsGPUArray(x),"Expected gpuArray input");a=mxGPUCreateFromMxArray(x);auto c=mxGPUGetClassID(a);req((type==mxUNKNOWN_CLASS?(c==mxSINGLE_CLASS||c==mxDOUBLE_CLASS):c==type)&&mxGPUGetComplexity(a)==mxREAL&&!mxGPUIsSparse(a),"Invalid GPU type/complexity/sparsity");}
 ~Ref(){if(a)mxGPUDestroyGPUArray(a);}const double* d()const{return (const double*)mxGPUGetDataReadOnly(a);}size_t n()const{return mxGPUGetNumberOfElements(a);}
};
struct Out {
 mxGPUArray* a=nullptr;
 Out(size_t r,size_t c=1){mwSize dims[2]={r,c};a=mxGPUCreateGPUArray(2,dims,mxDOUBLE_CLASS,mxREAL,MX_GPU_DO_NOT_INITIALIZE);}
 ~Out(){if(a)mxGPUDestroyGPUArray(a);}double* d(){return (double*)mxGPUGetData(a);}mxArray* give(){return mxGPUCreateMxArrayOnGPU(a);}
};
template<class T>struct Buf {
 T* p=nullptr;size_t n=0;
 Buf()=default;Buf(const Buf&)=delete;Buf& operator=(const Buf&)=delete;
 ~Buf(){if(p)cudaFree(p);}void alloc(size_t m){req(!p,"Repeated allocation");n=m;if(m)ck(cudaMalloc((void**)&p,m*sizeof(T)));}
 void zero(){if(n)ck(cudaMemset(p,0,n*sizeof(T)));}
};
double scalar(const mxArray* a){req(a&&!mxIsGPUArray(a)&&mxIsDouble(a)&&!mxIsComplex(a)&&mxGetNumberOfElements(a)==1,"Expected CPU double scalar");double x=mxGetScalar(a);req(std::isfinite(x),"Nonfinite scalar");return x;}
double opt(const mxArray* a,const char* n,double d){auto p=mxGetField(a,0,n);return p?scalar(p):d;}
int integer(double v,int lo,int hi){req(std::isfinite(v)&&v==floor(v)&&v>=lo&&v<=hi,"Integer out of range");return int(v);}
std::string command(const mxArray* a){req(mxIsChar(a),"Command must be char");char* p=mxArrayToString(a);req(p,"Invalid command");std::string s(p);mxFree(p);return s;}
const double* vector(const mxArray* a,size_t n){req(a&&!mxIsGPUArray(a)&&mxIsDouble(a)&&!mxIsComplex(a)&&!mxIsSparse(a)&&mxGetNumberOfElements(a)==n,"Invalid CPU dimension vector");return mxGetPr(a);}

struct AssemblyGrid {int sx,sy,stride,nx,ny,nd;};
__device__ __forceinline__ int axisState(int e,int n,int stride){
 int last=(n-1)*stride;
 if(e>=stride&&e<=last&&e%stride==0)return 2;
 if(e>=stride+1&&e<=last+1&&(e-1)%stride==0)return 4;
 if(e>=stride+2&&e<=last+2&&(e-2)%stride==0)return 3;
 return 1;
}
__device__ __forceinline__ int nodeOffset(int state,int side,int high){
 if(state==1)return high;if(state==2)return 2*high;if(state==3)return 2*high-1;
 return side?2*high:2*high-1;
}
__device__ __forceinline__ size_t fineRow(int d,int ex,int ey,int sx,int sy,int bx,int by,AssemblyGrid g){
 int node=d/2,highX=node==1||node==2,highY=node>=2;
 int x=ex+nodeOffset(sx,bx,highX)-1,y=ey+nodeOffset(sy,by,highY)-1;
 return 2ULL*(y+size_t(g.ny+1)*x)+d%2;
}
__device__ __forceinline__ int integrationType(int sx,int sy){
 // sx/ sy states 1,2,3,4; actual MATLAB sixteen-type order (zero-based).
 const int map[16]={0,4,3,6,1,9,7,11,2,10,8,13,5,14,12,15};
 return map[4*(sx-1)+sy-1];
}
template<class Phi>__global__ void assemble(const Phi* __restrict__ phi,const double* __restrict__ alpha,const double* __restrict__ ke,AssemblyGrid g,double* __restrict__ B){
 int owner=blockIdx.x,slot=blockIdx.y,ry=owner%g.sy+1,rx=owner/g.sy+1;
 int dx=slot>=2,dy=slot==1?1:(slot>=2?slot-3:0),cx=rx+dx,cy=ry+dy;
 int t=threadIdx.x,row=t%8,col=t/8;size_t out=64ULL*(slot+5ULL*owner)+t;
 if(cx<1||cx>g.sx||cy<1||cy>g.sy){B[out]=0;return;}
 int xlo=dx?min(rx,cx)*g.stride+1:(rx-1)*g.stride+1,xhi=dx?xlo:rx*g.stride+1;
 int ylo=dy?min(ry,cy)*g.stride+1:(ry-1)*g.stride+1,yhi=dy?ylo:ry*g.stride+1;
 __shared__ double pi[64],pj[64],k[64],tmp[64];double acc=0;
 for(int ex=xlo;ex<=xhi;ex++){
  int sx=axisState(ex,g.sx,g.stride),bix=sx==4&&ex==(rx-1)*g.stride+1,bjx=sx==4&&ex==(cx-1)*g.stride+1;
  for(int ey=ylo;ey<=yhi;ey++){
   int sy=axisState(ey,g.sy,g.stride),biy=sy==4&&ey==(ry-1)*g.stride+1,bjy=sy==4&&ey==(cy-1)*g.stride+1;
   // In 2-D the expanded y groups are ordered upper,lower, then x.
   int nsy=1+(sy==4),si=(sy==4?1-biy:0)+nsy*bix,sj=(sy==4?1-bjy:0)+nsy*bjx;
   size_t ir=fineRow(row,ex,ey,sx,sy,bix,biy,g),jr=fineRow(row,ex,ey,sx,sy,bjx,bjy,g);
   pi[t]=double(phi[ir+size_t(g.nd)*col]);pj[t]=slot?double(phi[jr+size_t(g.nd)*col]):pi[t];
   k[t]=ke[1024*integrationType(sx,sy)+(8*si+row)+32*(8*sj+col)];
   __syncthreads();double sum=0;
   #pragma unroll
   for(int q=0;q<8;q++)sum+=pi[q+8*row]*k[q+8*col];tmp[t]=sum;
   __syncthreads();sum=0;
   #pragma unroll
   for(int q=0;q<8;q++)sum+=tmp[row+8*q]*pj[q+8*col];
   acc+=alpha[ey-1+size_t(g.ny)*(ex-1)]*sum;__syncthreads();
  }
 }
 if(slot==0){tmp[t]=acc;__syncthreads();acc=.5*(acc+tmp[col+8*row]);}
 B[out]=acc;
}

struct Shape {int nx,ny,ox,oy,step,bc,n,ns;};
Shape shape(const mxArray* a){auto v=vector(a,6);Shape s{};s.nx=integer(v[0],2,INT_MAX/8);s.ny=integer(v[1],2,INT_MAX/8);s.ox=integer(v[2],2,INT_MAX/8);s.oy=integer(v[3],2,INT_MAX/8);s.step=integer(v[4],1,INT_MAX/4);s.bc=integer(v[5],0,4);req(8.0*s.nx*s.ny<=INT_MAX,"Coarse grid exceeds int32");s.ns=s.nx*s.ny;s.n=8*s.ns;req(s.nx==(s.ox-2+s.step)/s.step+1&&s.ny==(s.oy-2+s.step)/s.step+1,"Coarse endpoints/stride inconsistent");return s;}
__host__ __device__ bool fixed(Shape s,int x,int y,int q){
 if(s.bc==1&&x==0)return q<2||q>=6;
 if(s.bc==2&&x==s.nx-1)return q>=2&&q<6;
 if(s.bc==3&&y==0)return q<4;
 if(s.bc==4&&y==s.ny-1)return q>=4;
 return false;
}
__device__ __forceinline__ double blockValue(const double* B,Shape s,int ix,int iy,int jx,int jy,int r,int c){
 int dx=jx-ix,dy=jy-iy;if(abs(dx)>1||abs(dy)>1)return 0;
 int i=iy+s.ny*ix,j=jy+s.ny*jx;
 if(j<i){int v=i;i=j;j=v;dx=-dx;dy=-dy;v=r;r=c;c=v;}
 int slot=dx?3+dy:dy;
 return B[64ULL*(slot+5ULL*i)+r+8*c];
}
__global__ void action(const double* __restrict__ B,Shape s,const double* __restrict__ x,double* __restrict__ y){
 for(size_t d=size_t(blockIdx.x)*blockDim.x+threadIdx.x;d<size_t(s.n);d+=size_t(gridDim.x)*blockDim.x){
  int owner=int(d/8),q=int(d%8),iy=owner%s.ny,ix=owner/s.ny;
  if(fixed(s,ix,iy,q)){y[d]=x[d];continue;}double sum=0;
  for(int dx=-1;dx<=1;dx++)for(int dy=-1;dy<=1;dy++){
   int jx=ix+dx,jy=iy+dy;if(jx<0||jx>=s.nx||jy<0||jy>=s.ny)continue;
   int base=8*(jy+s.ny*jx);
   #pragma unroll
   for(int c=0;c<8;c++)if(!fixed(s,jx,jy,c))sum+=blockValue(B,s,ix,iy,jx,jy,q,c)*x[base+c];
  }y[d]=sum;
 }
}
__device__ __forceinline__ int coordinate(int i,int original,int step){return min(i*step,original-1);}
__device__ __forceinline__ void interp(int i,int n,int original,int step,int& lo,int& hi,double& wh){
 int nc=(n+2)/2;if(i==n-1){lo=hi=nc-1;wh=0;return;}
 lo=i/2;hi=lo+1;wh=double(coordinate(i,original,step)-coordinate(2*lo,original,step))/double(coordinate(2*hi,original,step)-coordinate(2*lo,original,step));
}
__device__ __forceinline__ double weight(int i,int ci,int n,int original,int step){int lo,hi;double w;interp(i,n,original,step,lo,hi,w);return ci==lo?1-w:(ci==hi?w:0);}
__global__ void galerkin(const double* __restrict__ B,Shape f,Shape c,double* __restrict__ C){
 int owner=blockIdx.x,slot=blockIdx.y,ry=owner%c.ny,rx=owner/c.ny,cy=ry+(slot==1?1:(slot>=2?slot-3:0)),cx=rx+(slot>=2);
 int t=threadIdx.x,r=t%8,q=t/8;size_t out=64ULL*(slot+5ULL*owner)+t;
 if(cx<0||cx>=c.nx||cy<0||cy>=c.ny){C[out]=0;return;}
 if(fixed(c,rx,ry,r)||fixed(c,cx,cy,q)){C[out]=slot==0&&r==q?1:0;return;}
 double sum=0;
 for(int ix=max(0,2*rx-1);ix<=min(f.nx-1,2*rx+1);ix++){
  double wx=weight(ix,rx,f.nx,f.ox,f.step);
  for(int iy=max(0,2*ry-1);iy<=min(f.ny-1,2*ry+1);iy++){
   if(fixed(f,ix,iy,r))continue;double wi=wx*weight(iy,ry,f.ny,f.oy,f.step);if(wi==0)continue;
   for(int jx=max(max(0,ix-1),2*cx-1);jx<=min(min(f.nx-1,ix+1),2*cx+1);jx++){
    double vx=weight(jx,cx,f.nx,f.ox,f.step);
    for(int jy=max(max(0,iy-1),2*cy-1);jy<=min(min(f.ny-1,iy+1),2*cy+1);jy++){
     if(fixed(f,jx,jy,q))continue;double wj=vx*weight(jy,cy,f.ny,f.oy,f.step);
     if(wj!=0)sum+=(wi*wj)*blockValue(B,f,ix,iy,jx,jy,r,q);
    }
   }
  }
 }
 // Diagonal products are explicitly symmetrized once, below.
 C[out]=sum;
}
__global__ void symDiagonal(double* B,int ns){
 for(int owner=blockIdx.x;owner<ns;owner+=gridDim.x){int t=threadIdx.x;if(t>=64)return;__shared__ double a[64];size_t base=320ULL*owner;a[t]=B[base+t];__syncthreads();B[base+t]=.5*(a[t]+a[t/8+8*(t%8)]);__syncthreads();}
}
__global__ void inverseDiagonal(const double* B,Shape s,double* inv,int* bad){
 int owner=blockIdx.x,t=threadIdx.x,r=t%8,c=t/8,x=owner/s.ny,y=owner%s.ny;
 __shared__ double a[64],v[64];
 a[t]=fixed(s,x,y,r)||fixed(s,x,y,c)?(r==c?1:0):B[320ULL*owner+t];v[t]=r==c?1:0;__syncthreads();
 for(int k=0;k<8;k++){
  double p=a[k+8*k];if(!(p>0)||!isfinite(p)){if(t==0)atomicExch(bad,1);p=1;}
  if(r==k){a[t]/=p;v[t]/=p;}__syncthreads();double z=a[r+8*k];__syncthreads();
  if(r!=k){a[t]-=z*a[k+8*c];v[t]-=z*v[k+8*c];}__syncthreads();
 }
 inv[64ULL*owner+t]=v[t];
}
__global__ void smoothing(Shape s,const double* inv,const double* rhs,const double* az,double* z,double omega,bool fromZero){
 for(size_t d=size_t(blockIdx.x)*blockDim.x+threadIdx.x;d<size_t(s.n);d+=size_t(gridDim.x)*blockDim.x){int owner=int(d/8),r=int(d%8),base=8*owner;if(fixed(s,owner/s.ny,owner%s.ny,r)){z[d]=0;continue;}double sum=0;
  #pragma unroll
  for(int c=0;c<8;c++)sum+=inv[64ULL*owner+r+8*c]*(rhs[base+c]-(fromZero?0:az[base+c]));z[d]=(fromZero?0:z[d])+.5*sum;
 }
}
__global__ void restrictResidual(Shape f,Shape c,const double* r,const double* az,double* rc){
 for(size_t d=size_t(blockIdx.x)*blockDim.x+threadIdx.x;d<size_t(c.n);d+=size_t(gridDim.x)*blockDim.x){int owner=int(d/8),q=int(d%8),cx=owner/c.ny,cy=owner%c.ny;if(fixed(c,cx,cy,q)){rc[d]=0;continue;}double sum=0;
  for(int x=max(0,2*cx-1);x<=min(f.nx-1,2*cx+1);x++)for(int y=max(0,2*cy-1);y<=min(f.ny-1,2*cy+1);y++)if(!fixed(f,x,y,q)){int j=8*(y+f.ny*x)+q;sum+=weight(x,cx,f.nx,f.ox,f.step)*weight(y,cy,f.ny,f.oy,f.step)*(r[j]-az[j]);}rc[d]=sum;
 }
}
__global__ void prolong(Shape f,Shape c,const double* zc,double* z){
 for(size_t d=size_t(blockIdx.x)*blockDim.x+threadIdx.x;d<size_t(f.n);d+=size_t(gridDim.x)*blockDim.x){int owner=int(d/8),q=int(d%8),x=owner/f.ny,y=owner%f.ny;if(fixed(f,x,y,q))continue;int lx,hx,ly,hy;double wx,wy;interp(x,f.nx,f.ox,f.step,lx,hx,wx);interp(y,f.ny,f.oy,f.step,ly,hy,wy);double sum=0;
  for(int a=0;a<2;a++)for(int b=0;b<2;b++){int cx=a?hx:lx,cy=b?hy:ly;if(!fixed(c,cx,cy,q))sum+=(a?wx:1-wx)*(b?wy:1-wy)*zc[8*(cy+c.ny*cx)+q];}z[d]+=sum;
 }
}
__global__ void initialize(Shape s,const double* b,double* r,double* x){
 for(size_t d=size_t(blockIdx.x)*blockDim.x+threadIdx.x;d<size_t(s.n);d+=size_t(gridDim.x)*blockDim.x){int o=int(d/8);r[d]=fixed(s,o/s.ny,o%s.ny,int(d%8))?0:b[d];if(x)x[d]=0;}
}
__global__ void residual(Shape s,const double* b,const double* ax,double* r){
 for(size_t d=size_t(blockIdx.x)*blockDim.x+threadIdx.x;d<size_t(s.n);d+=size_t(gridDim.x)*blockDim.x){int o=int(d/8);r[d]=fixed(s,o/s.ny,o%s.ny,int(d%8))?0:b[d]-ax[d];}
}
__global__ void direction(const double* z,double* p,int n,double beta){for(size_t d=size_t(blockIdx.x)*blockDim.x+threadIdx.x;d<size_t(n);d+=size_t(gridDim.x)*blockDim.x)p[d]=z[d]+beta*p[d];}
__global__ void update(double* x,double* r,const double* p,const double* ap,int n,double alpha){for(size_t d=size_t(blockIdx.x)*blockDim.x+threadIdx.x;d<size_t(n);d+=size_t(gridDim.x)*blockDim.x){x[d]+=alpha*p[d];r[d]-=alpha*ap[d];}}

struct Level {Shape s;const double* B=nullptr;Buf<double> owned,inv,r,z,tmp,chol;};
struct Blas {cublasHandle_t h=nullptr;~Blas(){if(h)cublasDestroy(h);}};
struct Engine {
 std::vector<std::unique_ptr<Level>> levels;Blas blas;double bytes=0;size_t freeBefore=0,totalBytes=0;
 Engine(const double* B,Shape fine,const mxArray* opts){
  Shape s=fine;while(true){auto l=std::make_unique<Level>();l->s=s;bytes+=8.*(64.*s.ns+3.*s.n);if(!levels.empty())bytes+=8.*320.*s.ns;levels.push_back(std::move(l));if(s.n<=512)break;s.nx=(s.nx+2)/2;s.ny=(s.ny+2)/2;s.step*=2;s.ns=s.nx*s.ny;s.n=8*s.ns;req(levels.size()<24,"Too many levels");}
  bytes+=8.*levels.back()->s.n*levels.back()->s.n;ck(cudaMemGetInfo(&freeBefore,&totalBytes));double reserve=opt(opts,"reserve_bytes",512*1048576.);req(bytes<=std::max(0.,double(freeBefore)-reserve)*.85,"Block MG native memory budget exceeded");cb(cublasCreate(&blas.h));Buf<int> bad;bad.alloc(1);bad.zero();
  for(size_t k=0;k<levels.size();k++){auto& l=*levels[k];s=l.s;if(k){l.owned.alloc(size_t(320)*s.ns);l.B=l.owned.p;auto& f=*levels[k-1];galerkin<<<dim3(s.ns,5),64>>>(f.B,f.s,s,l.owned.p);symDiagonal<<<std::min(s.ns,8192),64>>>(l.owned.p,s.ns);}else l.B=B;
   l.inv.alloc(size_t(64)*s.ns);l.r.alloc(s.n);l.z.alloc(s.n);l.tmp.alloc(s.n);inverseDiagonal<<<s.ns,64>>>(l.B,s,l.inv.p,bad.p);}
  int flag=0;ck(cudaMemcpy(&flag,bad.p,sizeof(int),cudaMemcpyDeviceToHost));req(!flag,"Nonpositive block-Jacobi pivot");terminal();ck(cudaGetLastError());
 }
 void terminal(){auto& l=*levels.back();auto s=l.s;int n=s.n;std::vector<double> blocks(size_t(320)*s.ns),A(size_t(n)*n,0);ck(cudaMemcpy(blocks.data(),l.B,blocks.size()*sizeof(double),cudaMemcpyDeviceToHost));
  for(int ix=0;ix<s.nx;ix++)for(int iy=0;iy<s.ny;iy++)for(int slot=0;slot<5;slot++){int jx=ix+(slot>=2),jy=iy+(slot==1?1:(slot>=2?slot-3:0));if(jx>=s.nx||jy<0||jy>=s.ny)continue;int oi=iy+s.ny*ix,oj=jy+s.ny*jx;
   for(int c=0;c<8;c++)for(int r=0;r<8;r++)if(!fixed(s,ix,iy,r)&&!fixed(s,jx,jy,c)){double v=blocks[64ULL*(slot+5ULL*oi)+r+8*c];int i=8*oi+r,j=8*oj+c;A[i+size_t(n)*j]=v;if(slot)A[j+size_t(n)*i]=v;}}
  for(int i=0;i<n;i++)if(fixed(s,i/8/s.ny,i/8%s.ny,i%8))A[i+size_t(n)*i]=1;
  for(int j=0;j<n;j++){double v=A[j+size_t(n)*j];for(int k=0;k<j;k++)v-=A[j+size_t(n)*k]*A[j+size_t(n)*k];req(v>0&&std::isfinite(v),"Terminal Galerkin matrix not SPD");A[j+size_t(n)*j]=sqrt(v);for(int i=j+1;i<n;i++){double q=A[i+size_t(n)*j];for(int k=0;k<j;k++)q-=A[i+size_t(n)*k]*A[j+size_t(n)*k];A[i+size_t(n)*j]=q/A[j+size_t(n)*j];}for(int i=0;i<j;i++)A[i+size_t(n)*j]=0;}
  l.chol.alloc(A.size());ck(cudaMemcpy(l.chol.p,A.data(),A.size()*sizeof(double),cudaMemcpyHostToDevice));
 }
 void act(int k,const double* x,double* y){auto& l=*levels[k];action<<<launch(l.s.n),256>>>(l.B,l.s,x,y);}
 void cycle(int k=0){auto& l=*levels[k];auto s=l.s;
  if(k==int(levels.size())-1){ck(cudaMemcpy(l.z.p,l.r.p,s.n*sizeof(double),cudaMemcpyDeviceToDevice));cb(cublasDtrsv(blas.h,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,s.n,l.chol.p,s.n,l.z.p,1));cb(cublasDtrsv(blas.h,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,s.n,l.chol.p,s.n,l.z.p,1));return;}
  smoothing<<<launch(s.n),256>>>(s,l.inv.p,l.r.p,nullptr,l.z.p,.5,true);act(k,l.z.p,l.tmp.p);smoothing<<<launch(s.n),256>>>(s,l.inv.p,l.r.p,l.tmp.p,l.z.p,.5,false);
  act(k,l.z.p,l.tmp.p);auto& c=*levels[k+1];restrictResidual<<<launch(c.s.n),256>>>(s,c.s,l.r.p,l.tmp.p,c.r.p);cycle(k+1);prolong<<<launch(s.n),256>>>(s,c.s,c.z.p,l.z.p);
  for(int j=0;j<2;j++){act(k,l.z.p,l.tmp.p);smoothing<<<launch(s.n),256>>>(s,l.inv.p,l.r.p,l.tmp.p,l.z.p,.5,false);}
 }
 double norm(const double* x,int n){double v;cb(cublasDnrm2(blas.h,n,x,1,&v));return v;}
 double dot(const double* x,const double* y,int n){double v;cb(cublasDdot(blas.h,n,x,1,y,1,&v));return v;}
};
void put(mxArray* a,const char* field,double value){mxSetField(a,0,field,mxCreateDoubleScalar(value));}
mxArray* statistics(Engine& e,int it,int flag,double rel,double setup,double solve,const std::vector<double>& history){
 const char* fields[]={"backend","precision","preconditioner","iterations","flag","true_relres","setup_seconds","solve_seconds","level_dofs","history","native_budget_bytes","free_bytes_before_native","scalar_sparse_created","galerkin"};auto a=mxCreateStructMatrix(1,1,14,fields);
 mxSetField(a,0,"backend",mxCreateString("cuda-structured-block8-exact-Galerkin-MGCG"));mxSetField(a,0,"precision",mxCreateString("FP64"));mxSetField(a,0,"preconditioner",mxCreateString("two pre/post block-Jacobi, omega 0.5; exact Galerkin; R=P transpose; terminal Cholesky"));mxSetField(a,0,"galerkin",mxCreateString("analytical unique neighbor block P-transpose A P; fixed channel masks on every level"));
 put(a,"iterations",it);put(a,"flag",flag);put(a,"true_relres",rel);put(a,"setup_seconds",setup);put(a,"solve_seconds",solve);put(a,"native_budget_bytes",e.bytes);put(a,"free_bytes_before_native",double(e.freeBefore));mxSetField(a,0,"scalar_sparse_created",mxCreateLogicalScalar(false));
 auto n=mxCreateDoubleMatrix(1,e.levels.size(),mxREAL);for(size_t k=0;k<e.levels.size();k++)mxGetPr(n)[k]=e.levels[k]->s.n;mxSetField(a,0,"level_dofs",n);auto h=mxCreateDoubleMatrix(history.size(),1,mxREAL);std::copy(history.begin(),history.end(),mxGetPr(h));mxSetField(a,0,"history",h);return a;
}
void run(int nlhs,mxArray** out,int nrhs,const mxArray** in){
 req(nrhs>=1,"Missing command");auto cmd=command(in[0]);mxInitGPU();
 if(cmd=="assemble"){
  req(nrhs==5&&nlhs==1,"B=blocks_mex('assemble',PhiGpu,alphaGpu,Ke32Gpu,[nslx nsly stride])");Ref phi(in[1],mxUNKNOWN_CLASS),alpha(in[2]),ke(in[3]);auto v=vector(in[4],3);AssemblyGrid g{};g.sx=integer(v[0],2,INT_MAX/8);g.sy=integer(v[1],2,INT_MAX/8);g.stride=integer(v[2],3,INT_MAX/4);req(double(g.sx)*g.sy*320<=INT_MAX,"Block array exceeds int32 element limit");req(double(g.sx)*g.stride+1<INT_MAX&&double(g.sy)*g.stride+1<INT_MAX,"Fine axis exceeds int32");g.nx=g.sx*g.stride+1;g.ny=g.sy*g.stride+1;req(2.*(g.nx+1)*(g.ny+1)<=INT_MAX,"Fine dofs exceed int32");g.nd=2*(g.nx+1)*(g.ny+1);
  req(phi.n()==size_t(g.nd)*8&&alpha.n()==size_t(g.nx)*g.ny&&ke.n()==16384,"Assembly array size mismatch");Out B(64,5ULL*g.sx*g.sy);dim3 grid(g.sx*g.sy,5);
  if(mxGPUGetClassID(phi.a)==mxSINGLE_CLASS)assemble<<<grid,64>>>((const float*)mxGPUGetDataReadOnly(phi.a),alpha.d(),ke.d(),g,B.d());else assemble<<<grid,64>>>(phi.d(),alpha.d(),ke.d(),g,B.d());ck(cudaGetLastError());out[0]=B.give();return;
 }
 req(nrhs>=3,"Missing block array/shape");Ref B(in[1]);Shape s=shape(in[2]);req(B.n()==size_t(320)*s.ns,"Block-array shape mismatch");
 if(cmd=="matvec"){
  req(nrhs==4&&nlhs==1,"y=blocks_mex('matvec',B,shape,xGpu)");Ref x(in[3]);req(x.n()==size_t(s.n),"Vector size mismatch");Out y(s.n);action<<<launch(s.n),256>>>(B.d(),s,x.d(),y.d());ck(cudaGetLastError());out[0]=y.give();return;
 }
 if(cmd=="galerkin"){
  req(nrhs==4&&nlhs==1,"C=blocks_mex('galerkin',B,fineShape,coarseShape)");Shape c=shape(in[3]);req(c.ox==s.ox&&c.oy==s.oy&&c.step==2*s.step&&c.bc==s.bc&&c.nx==(s.nx+2)/2&&c.ny==(s.ny+2)/2,"Inconsistent Galerkin levels");Out C(64,5ULL*c.ns);galerkin<<<dim3(c.ns,5),64>>>(B.d(),s,c,C.d());symDiagonal<<<std::min(c.ns,8192),64>>>(C.d(),c.ns);ck(cudaGetLastError());out[0]=C.give();return;
 }
 req((cmd=="solve"||cmd=="apply")&&nrhs==5&&nlhs==2,"[x,stats]=blocks_mex('solve'/'apply',B,shape,bGpu,opts)");Ref b(in[3]);req(b.n()==size_t(s.n),"RHS size mismatch");auto opts=in[4];req(mxIsStruct(opts)&&mxGetNumberOfElements(opts)==1,"Expected options struct");double tol=opt(opts,"tol",1e-8);int maxit=integer(opt(opts,"maxit",5000),1,100000);req(tol>0&&tol<1,"Invalid tolerance");
 ck(cudaDeviceSynchronize());auto clock=std::chrono::steady_clock::now();Engine e(B.d(),s,opts);Out x(s.n);Buf<double> p,ap;p.alloc(s.n);ap.alloc(s.n);initialize<<<launch(s.n),256>>>(s,b.d(),e.levels[0]->r.p,x.d());ck(cudaDeviceSynchronize());double setup=elapsed(clock);clock=std::chrono::steady_clock::now();auto& l=*e.levels[0];double bn=e.norm(l.r.p,s.n);int flag=1,it=0;double rel=bn?1:0;std::vector<double> hist{rel};
 if(cmd=="apply"){e.cycle();ck(cudaMemcpy(x.d(),l.z.p,s.n*sizeof(double),cudaMemcpyDeviceToDevice));flag=0;rel=0;}
 else if(bn==0){flag=0;}
 else{
  e.cycle();ck(cudaMemcpy(p.p,l.z.p,s.n*sizeof(double),cudaMemcpyDeviceToDevice));double rz=e.dot(l.r.p,l.z.p,s.n);req(rz>0&&std::isfinite(rz),"Nonpositive initial preconditioned residual");
  for(it=1;it<=maxit;it++){
   e.act(0,p.p,ap.p);double den=e.dot(p.p,ap.p,s.n);req(den>0&&std::isfinite(den),"Nonpositive PCG curvature");double alpha=rz/den;update<<<launch(s.n),256>>>(x.d(),l.r.p,p.p,ap.p,s.n,alpha);double rec=e.norm(l.r.p,s.n)/bn;hist.push_back(rec);
   if(it%20==0||rec<=tol||it==maxit){e.act(0,x.d(),ap.p);residual<<<launch(s.n),256>>>(s,b.d(),ap.p,l.tmp.p);rel=e.norm(l.tmp.p,s.n)/bn;if(rel<=tol){flag=0;break;}if(rec<=tol){ck(cudaMemcpy(l.r.p,l.tmp.p,s.n*sizeof(double),cudaMemcpyDeviceToDevice));e.cycle();ck(cudaMemcpy(p.p,l.z.p,s.n*sizeof(double),cudaMemcpyDeviceToDevice));rz=e.dot(l.r.p,l.z.p,s.n);continue;}}
   e.cycle();double next=e.dot(l.r.p,l.z.p,s.n);req(next>0&&std::isfinite(next),"Nonpositive preconditioned residual");direction<<<launch(s.n),256>>>(l.z.p,p.p,s.n,next/rz);rz=next;
  }
  if(it>maxit)it=maxit;e.act(0,x.d(),ap.p);residual<<<launch(s.n),256>>>(s,b.d(),ap.p,l.tmp.p);rel=e.norm(l.tmp.p,s.n)/bn;
 }
 ck(cudaDeviceSynchronize());double solve=elapsed(clock);out[0]=x.give();out[1]=statistics(e,it,flag,rel,setup,solve,hist);
}
}
void mexFunction(int nlhs,mxArray** out,int nrhs,const mxArray** in){try{block2d::run(nlhs,out,nrhs,in);}catch(const std::exception& e){mexErrMsgIdAndTxt("mechanics2d_blocks_v1:Failure","%s",e.what());}}

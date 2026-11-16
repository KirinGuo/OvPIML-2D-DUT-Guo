/* Fused 2D basis reconstruction and exact overlap physical nodal map.
 * Follows NCS's GPU-resident physical reconstruction; FP64 arithmetic.
 */
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>
#include <string>
void req(bool b,const char* s){if(!b)throw std::runtime_error(s);}
void ck(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
struct Ref{const mxGPUArray* a;explicit Ref(const mxArray* x){req(mxIsGPUArray(x),"Expected GPU array");a=mxGPUCreateFromMxArray(x);}~Ref(){mxGPUDestroyGPUArray(a);}const void* p(){return mxGPUGetDataReadOnly(a);}size_t n(){return mxGPUGetNumberOfElements(a);}};
struct Out{mxGPUArray* a;explicit Out(size_t n){mwSize dims[2]={n,1};a=mxGPUCreateGPUArray(2,dims,mxDOUBLE_CLASS,mxREAL,MX_GPU_DO_NOT_INITIALIZE);}~Out(){mxGPUDestroyGPUArray(a);}double* p(){return (double*)mxGPUGetData(a);}};
template<class T>__global__ void recover(const T* __restrict__ phi,const int* __restrict__ owner,const double* __restrict__ d,double* __restrict__ u,size_t nd){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<nd;i+=size_t(gridDim.x)*blockDim.x){size_t base=8*size_t(owner[i/2]-1);double value=0;
  #pragma unroll
  for(int c=0;c<8;c++)value+=double(phi[i+nd*c])*d[base+c];u[i]=value;}
}
__device__ bool interfaceNode(int i,int ns,int stride){int k=i/stride,r=i%stride;return k>0&&k<ns&&(r==0||r==1);}
__global__ void physical(const double* __restrict__ u,double* __restrict__ v,int nx,int ny,int nslx,int nsly,int stride){
 size_t nd=2*size_t(nx+1)*(ny+1);
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<nd;i+=size_t(gridDim.x)*blockDim.x){int q=int(i%2),node=int(i/2),y=node%(ny+1),x=node/(ny+1);bool ix=interfaceNode(x,nslx,stride),iy=interfaceNode(y,nsly,stride);
  int xl=ix?x-1:x,xh=ix?x+1:x,yl=iy?y-1:y,yh=iy?y+1:y;
  double left=iy?.5*(u[2*(size_t(yl)+(ny+1)*size_t(xl))+q]+u[2*(size_t(yh)+(ny+1)*size_t(xl))+q]):u[2*(size_t(y)+(ny+1)*size_t(xl))+q];
  if(ix){double right=iy?.5*(u[2*(size_t(yl)+(ny+1)*size_t(xh))+q]+u[2*(size_t(yh)+(ny+1)*size_t(xh))+q]):u[2*(size_t(y)+(ny+1)*size_t(xh))+q];v[i]=.5*(left+right);}else v[i]=left;
 }
}
void run(int nlhs,mxArray** out,int nrhs,const mxArray** in){
 req(nlhs==1&&nrhs>=3&&mxIsChar(in[0]),"Use fields_mex('reconstruct',Phi,owner,d) or fields_mex('map',u,grid)");char* c=mxArrayToString(in[0]);std::string cmd(c);mxFree(c);Ref a(in[1]);req(mxGPUGetComplexity(a.a)==mxREAL,"Real input required");
 if(cmd=="reconstruct"){
  req(nrhs==4,"Expected four reconstruction inputs");Ref owner(in[2]),d(in[3]);req(mxGPUGetClassID(owner.a)==mxINT32_CLASS&&mxGPUGetClassID(d.a)==mxDOUBLE_CLASS,"Expected int32 owner / FP64 coarse vector");size_t nd=2*owner.n();req(a.n()==8*nd,"Phi dimensions mismatch");Out u(nd);int blocks=int(std::min<size_t>(65535,(nd+255)/256));
  if(mxGPUGetClassID(a.a)==mxSINGLE_CLASS)recover<<<blocks,256>>>((const float*)a.p(),(const int*)owner.p(),(const double*)d.p(),u.p(),nd);else{req(mxGPUGetClassID(a.a)==mxDOUBLE_CLASS,"Phi must be FP32/FP64");recover<<<blocks,256>>>((const double*)a.p(),(const int*)owner.p(),(const double*)d.p(),u.p(),nd);}ck(cudaGetLastError());ck(cudaDeviceSynchronize());out[0]=mxGPUCreateMxArrayOnGPU(u.a);
 }else{
  req(cmd=="map"&&nrhs==3&&mxIsDouble(in[2])&&mxGetNumberOfElements(in[2])==5,"Expected grid [nx ny nslx nsly stride]");req(mxGPUGetClassID(a.a)==mxDOUBLE_CLASS,"Physical input must be FP64");const double* g=mxGetPr(in[2]);int nx=int(g[0]),ny=int(g[1]),nsx=int(g[2]),nsy=int(g[3]),stride=int(g[4]);req(nx==nsx*stride+1&&ny==nsy*stride+1&&stride>=3&&a.n()==2*size_t(nx+1)*(ny+1),"Physical map dimensions mismatch");Out v(a.n());int blocks=int(std::min<size_t>(65535,(a.n()+255)/256));physical<<<blocks,256>>>((const double*)a.p(),v.p(),nx,ny,nsx,nsy,stride);ck(cudaGetLastError());ck(cudaDeviceSynchronize());out[0]=mxGPUCreateMxArrayOnGPU(v.a);
 }
}
void mexFunction(int nlhs,mxArray** out,int nrhs,const mxArray** in){try{mxInitGPU();run(nlhs,out,nrhs,in);}catch(const std::exception& e){mexErrMsgIdAndTxt("mechanics2d_fields:Failure","%s",e.what());}}

/* GPU-resident 2D material windows, checked FP64 corner solve, D4 scatter,
 * physical load restriction and analytical-owner reconstruction.
 * Private session owns Phi until finish: no MATLAB input is mutated.
 */
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <memory>
#include <stdexcept>
#include <string>

namespace basisgpu {
void req(bool p,const char* m){if(!p)throw std::runtime_error(m);}
void ck(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
int blocks(size_t n){return int(std::min<size_t>(65535,(n+255)/256));}
struct Ref{const mxGPUArray* a=nullptr;Ref(const mxArray* x,mxClassID c){req(mxIsGPUArray(x),"Expected GPU array");a=mxGPUCreateFromMxArray(x);req(mxGPUGetClassID(a)==c&&mxGPUGetComplexity(a)==mxREAL&&!mxGPUIsSparse(a),"GPU array type mismatch");}~Ref(){if(a)mxGPUDestroyGPUArray(a);}const void* p()const{return mxGPUGetDataReadOnly(a);}size_t n()const{return mxGPUGetNumberOfElements(a);}};
struct Out{mxGPUArray* a=nullptr;Out(size_t n,size_t m,mxClassID c,bool zero=false){mwSize d[2]={n,m};a=mxGPUCreateGPUArray(2,d,c,mxREAL,zero?MX_GPU_INITIALIZE_VALUES:MX_GPU_DO_NOT_INITIALIZE);}~Out(){if(a)mxGPUDestroyGPUArray(a);}void* p(){return mxGPUGetData(a);}mxArray* give(){return mxGPUCreateMxArrayOnGPU(a);}};
template<class T>struct Buf{T* p=nullptr;~Buf(){if(p)cudaFree(p);}void alloc(size_t n){ck(cudaMalloc((void**)&p,n*sizeof(T)));}void zero(size_t n){ck(cudaMemset(p,0,n*sizeof(T)));}};
std::string str(const mxArray* x){req(mxIsChar(x),"Command must be char");char* p=mxArrayToString(x);req(p,"Invalid command");std::string s(p);mxFree(p);return s;}
const double* vec(const mxArray* x,size_t n){req(!mxIsGPUArray(x)&&mxIsDouble(x)&&!mxIsComplex(x)&&mxGetNumberOfElements(x)==n,"CPU vector type/size mismatch");return mxGetPr(x);}
int asint(double x,int lo,int hi){req(std::isfinite(x)&&x==floor(x)&&x>=lo&&x<=hi,"Integer out of range");return int(x);}
struct Grid{int sx,sy,stride,layer,nx,ny,ns,nd,bc;double fill;};
Grid grid(const mxArray* x){auto v=vec(x,6);Grid g{};g.sx=asint(v[0],2,10000000);g.sy=asint(v[1],2,10000000);g.stride=asint(v[2],3,1000);g.layer=asint(v[3],0,1000);g.bc=asint(v[4],1,4);g.fill=v[5];req(g.fill>0&&g.fill<=1,"Invalid material fill");req(2.*(g.sx*g.stride+2.)*(g.sy*g.stride+2.)<=INT_MAX/8.,"Phi exceeds int32 GPU element limit");g.nx=g.sx*g.stride+1;g.ny=g.sy*g.stride+1;g.ns=g.sx*g.sy;g.nd=2*(g.nx+1)*(g.ny+1);return g;}
__device__ int type(Grid g,int id){int x=id/g.sy,y=id%g.sy,sx=x==0?1:(x==g.sx-1?3:2),sy=y==0?1:(y==g.sy-1?3:2);return sx+3*(3-sy);}
__device__ bool clamp(Grid g,int id){int x=id/g.sy,y=id%g.sy;return (g.bc==1&&x==0)||(g.bc==2&&x==g.sx-1)||(g.bc==3&&y==0)||(g.bc==4&&y==g.sy-1);}
__device__ double sample(const double* rho,Grid g,int id,int wy,int wx){
 int y=(id%g.sy)*g.stride-g.layer+wy,x=(id/g.sy)*g.stride-g.layer+wx;
 if(x>=0&&x<g.nx&&y>=0&&y<g.ny)return rho[y+size_t(g.ny)*x];
 if(clamp(g,id)){
  if(g.bc==1&&x<0&&y>=0&&y<g.ny)x=-x-1;
  else if(g.bc==2&&x>=g.nx&&y>=0&&y<g.ny)x=2*g.nx-x-1;
  else if(g.bc==3&&y<0&&x>=0&&x<g.nx)y=-y-1;
  else if(g.bc==4&&y>=g.ny&&x>=0&&x<g.nx)y=2*g.ny-y-1;
  if(x>=0&&x<g.nx&&y>=0&&y<g.ny)return rho[y+size_t(g.ny)*x];
 }return g.fill;
}
__global__ void classify(const double* rho,Grid g,int* flags){int id=blockIdx.x,os=g.stride+1+2*g.layer;double first=sample(rho,g,id,0,0);bool different=false;for(int t=threadIdx.x;t<os*os;t+=blockDim.x)different|=sample(rho,g,id,t%os,t/os)!=first;int any=__syncthreads_or(different);if(threadIdx.x==0)flags[id]=any?0:1;}
__global__ void windows(const double* rho,Grid g,const int* ids,int batch,const int* perm,float mean,float sd,float* X){
 int os=g.stride+1+2*g.layer;size_t n=size_t(os)*os*batch;
 for(size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;k<n;k+=size_t(gridDim.x)*blockDim.x){int b=int(k/(os*os)),p=int(k%(os*os)),id=ids[b];if(id<0){X[k]=0;continue;}int canonical=p/os+os*(p%os),source=perm[canonical+os*os*(type(g,id)-1)];float value=float(sample(rho,g,id,source%os,source/os));X[k]=__fdiv_rn(__fsub_rn(value,mean),sd);}
}
__device__ int redCount(int owner,int ns,int stride){return stride+2-(owner>0)-(owner<ns-1);}
__device__ int redCoord(int i,int owner,int ns,int stride){int q=i+(owner>0&&i>0);if(owner<ns-1&&q>=stride)q++;return owner*stride+q;}
__device__ size_t globalDof(Grid g,int owner,int d){int ox=owner/g.sy,oy=owner%g.sy,ny=redCount(oy,g.sy,g.stride),node=d/2,x=redCoord(node/ny,ox,g.sx,g.stride),y=redCoord(node%ny,oy,g.sy,g.stride);return 2ULL*(y+size_t(g.ny+1)*x)+d%2;}
__device__ int ownerAxis(int x,int ns,int stride){int k=x/stride,r=x%stride;if(k>0&&k<ns&&r==1)k--;return min(k,ns-1);}
__device__ void maxPositive(double* target,double x){atomicMax((unsigned long long*)target,__double_as_longlong(x));}
__device__ void minPositive(double* target,double x){atomicMin((unsigned long long*)target,__double_as_longlong(x));}
__global__ void invert(const float* raw,int nf,const int* ids,int batch,const int* corners,double* inverse,int* flag,double* checks){
 int b=blockIdx.x,t=threadIdx.x,r=t%8,c=t/8;if(ids[b]<0)return;
 __shared__ double A[64],V[64],N[64];__shared__ int pivot;
 double orig=double(raw[corners[r]+nf*(c+8*b)]);A[t]=orig;N[t]=orig*orig;V[t]=r==c?1:0;__syncthreads();
 for(int k=0;k<8;k++){
  if(t==0){pivot=k;double best=fabs(A[k+8*k]);for(int q=k+1;q<8;q++)if(fabs(A[q+8*k])>best){best=fabs(A[q+8*k]);pivot=q;}if(!isfinite(best)||best<1e-300){atomicOr(flag,1);pivot=k;}}__syncthreads();
  int source=r==k?pivot:(r==pivot?k:r);double av=A[source+8*c],vv=V[source+8*c];__syncthreads();A[t]=av;V[t]=vv;__syncthreads();
  double p=A[k+8*k];if(fabs(p)<1e-300||!isfinite(p))p=1;if(r==k){A[t]/=p;V[t]/=p;}__syncthreads();double z=A[r+8*k];__syncthreads();if(r!=k){A[t]-=z*A[k+8*c];V[t]-=z*V[k+8*c];}__syncthreads();
 }inverse[64*b+t]=V[t];
 if(t==0){double nt=0,ni=0;for(int k=0;k<64;k++){nt+=N[k];ni+=V[k]*V[k];}double lower=1/sqrt(nt*ni);if(!isfinite(lower)||lower<1e-10)atomicOr(flag,2);else minPositive(checks,lower);}
}
__global__ void decode(const float* raw,int nf,const int* ids,int batch,Grid g,const double* inverse,const int* rowMap,const int* rowSign,const int* colMap,const int* colSign,const double* rr,const double* dr,const int* corners,float* Phi,int* filled,int* flag,double* checks){
 int b=blockIdx.x,owner=ids[b],r=threadIdx.x;if(owner<0)return;int tid=type(g,owner)-1;
 __shared__ double err[256],den[256],ce[256];double e=0,n=0,corner=0;
 if(r<nf){double v[8];
  #pragma unroll
  for(int c=0;c<8;c++){double s=0;
   #pragma unroll
   for(int k=0;k<8;k++)s+=double(raw[r+nf*(k+8*b)])*inverse[64*b+k+8*c];v[c]=s;
   int targetr=rowMap[r+200*tid],targetc=colMap[c+8*tid];double out=s*rowSign[r+200*tid]*colSign[c+8*tid];Phi[globalDof(g,owner,targetr)+size_t(g.nd)*targetc]=float(out);if(!isfinite(s))atomicOr(flag,4);
  }
  for(int q=0;q<3;q++){double s=0;for(int c=0;c<8;c++)s+=v[c]*dr[c+8*q];double z=s-rr[r+200*q];e+=z*z;n+=rr[r+200*q]*rr[r+200*q];}
  for(int j=0;j<8;j++)if(r==corners[j])for(int c=0;c<8;c++)corner=fmax(corner,fabs(v[c]-(c==j?1.:0.)));
 }
 err[r]=e;den[r]=n;ce[r]=corner;__syncthreads();
 for(int step=128;step;step/=2){if(r<step){err[r]+=err[r+step];den[r]+=den[r+step];ce[r]=fmax(ce[r],ce[r+step]);}__syncthreads();}
 if(r==0){double rigid=sqrt(err[0]/den[0]);if(!isfinite(rigid)||rigid>2e-4||ce[0]>1e-8)atomicOr(flag,8);maxPositive(checks+1,rigid);maxPositive(checks+2,ce[0]);if(atomicCAS(filled+owner,0,1)!=0)atomicOr(flag,16);}
}
__global__ void put(const float* numeric,const int* ids,const int* templates,int batch,Grid g,float* Phi,int* filled,int* flag){int b=blockIdx.x,id=ids[b],t=threadIdx.x,ix=id/g.sy,iy=id%g.sy,nf=2*redCount(ix,g.sx,g.stride)*redCount(iy,g.sy,g.stride);for(int k=t;k<nf*8;k+=blockDim.x){int r=k%nf,c=k/nf;Phi[globalDof(g,id,r)+size_t(g.nd)*c]=numeric[r+200*(c+8*templates[b])];}if(t==0&&atomicCAS(filled+id,0,1)!=0)atomicOr(flag,16);}
__global__ void coverage(const int* filled,int ns,int* flag){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<ns;i+=gridDim.x*blockDim.x)if(filled[i]!=1)atomicOr(flag,32);}
__device__ bool interfaceNode(int i,int ns,int stride){int k=i/stride,r=i%stride;return k>0&&k<ns&&(r==0||r==1);}
__global__ void load(const float* Phi,Grid g,int node,int component,double force,double* f){int t=threadIdx.x;if(t>=32)return;int c=t%8,k=t/8,x=node/(g.ny+1),y=node%(g.ny+1);bool ix=interfaceNode(x,g.sx,g.stride),iy=interfaceNode(y,g.sy,g.stride);int countx=1+ix,county=1+iy;if(k>=countx*county)return;int xx=ix?x+(k/county?1:-1):x,yy=iy?y+(k%county?1:-1):y;int owner=ownerAxis(yy,g.sy,g.stride)+g.sy*ownerAxis(xx,g.sx,g.stride);size_t d=2ULL*(yy+size_t(g.ny+1)*xx)+component;atomicAdd(f+8*owner+c,force*double(Phi[d+size_t(g.nd)*c])/(countx*county));}
__global__ void recover(const float* Phi,Grid g,const double* d,double* u){for(size_t r=size_t(blockIdx.x)*blockDim.x+threadIdx.x;r<size_t(g.nd);r+=size_t(gridDim.x)*blockDim.x){size_t node=r/2;int y=int(node%(g.ny+1)),x=int(node/(g.ny+1));int owner=ownerAxis(y,g.sy,g.stride)+g.sy*ownerAxis(x,g.sx,g.stride);double s=0;for(int c=0;c<8;c++)s+=double(Phi[r+size_t(g.nd)*c])*d[8*owner+c];u[r]=s;}}
struct Session{Grid g;Out Phi;Buf<int> filled,flag;Buf<double> checks;Session(Grid v):g(v),Phi(v.nd,8,mxSINGLE_CLASS){filled.alloc(v.ns);filled.zero(v.ns);flag.alloc(1);flag.zero(1);checks.alloc(3);double s[3]={1,0,0};ck(cudaMemcpy(checks.p,s,sizeof(s),cudaMemcpyHostToDevice));}};
std::unique_ptr<Session> session;bool registered=false;
void cleanup(){session.reset();}
void run(int nlhs,mxArray** out,int nrhs,const mxArray** in){req(nrhs>=1,"Missing command");auto cmd=str(in[0]);mxInitGPU();
 if(cmd=="begin"){req(nrhs==2&&nlhs==0,"begin(grid)");req(!session,"A basis session is already active; abort it first");session=std::make_unique<Session>(grid(in[1]));if(!registered){mexAtExit(cleanup);registered=true;}return;}
 if(cmd=="abort"){session.reset();return;}
 if(cmd=="classify"){req(nrhs==3&&nlhs==1,"flags=classify(rhoGpu,grid)");Ref rho(in[1],mxDOUBLE_CLASS);auto g=grid(in[2]);req(rho.n()==size_t(g.nx)*g.ny,"rho size mismatch");Out flags(g.ns,1,mxINT32_CLASS);classify<<<g.ns,128>>>((const double*)rho.p(),g,(int*)flags.p());ck(cudaGetLastError());out[0]=flags.give();return;}
 if(cmd=="windows"){req(nrhs==6&&nlhs==1,"X=windows(rhoGpu,idsGpu,grid,permutationGpu,[mean sd])");Ref rho(in[1],mxDOUBLE_CLASS),ids(in[2],mxINT32_CLASS),perm(in[4],mxINT32_CLASS);auto g=grid(in[3]);auto s=vec(in[5],2);int os=g.stride+1+2*g.layer;req(s[1]>0&&perm.n()==size_t(os)*os*9,"Window metadata mismatch");Out X(os*os,ids.n(),mxSINGLE_CLASS);windows<<<blocks(size_t(os)*os*ids.n()),256>>>((const double*)rho.p(),g,(const int*)ids.p(),int(ids.n()),(const int*)perm.p(),float(s[0]),float(s[1]),(float*)X.p());ck(cudaGetLastError());out[0]=X.give();return;}
 if(cmd=="load"||cmd=="recover"){req(nrhs==4&&nlhs==1,"load/recover(Phi,grid,argument)");Ref phi(in[1],mxSINGLE_CLASS);auto g=grid(in[2]);req(phi.n()==size_t(g.nd)*8,"Phi size mismatch");if(cmd=="load"){auto v=vec(in[3],3);int node=asint(v[0],0,g.nd/2-1),comp=asint(v[1],0,1);Out f(8*g.ns,1,mxDOUBLE_CLASS,true);load<<<1,32>>>((const float*)phi.p(),g,node,comp,v[2],(double*)f.p());out[0]=f.give();}else{Ref d(in[3],mxDOUBLE_CLASS);req(d.n()==8ULL*g.ns,"Coarse vector mismatch");Out u(g.nd,1,mxDOUBLE_CLASS);recover<<<blocks(g.nd),256>>>((const float*)phi.p(),g,(const double*)d.p(),(double*)u.p());out[0]=u.give();}ck(cudaGetLastError());return;}
 req(bool(session),"No active basis session");auto& s=*session;auto g=s.g;
 if(cmd=="put"){req(nrhs==4&&nlhs==0,"put(numericGpu,idsGpu,templateIndexGpu)");Ref values(in[1],mxSINGLE_CLASS),ids(in[2],mxINT32_CLASS),tpl(in[3],mxINT32_CLASS);req(ids.n()==tpl.n()&&values.n()%1600==0,"Numeric shape mismatch");if(ids.n())put<<<int(ids.n()),256>>>((const float*)values.p(),(const int*)ids.p(),(const int*)tpl.p(),int(ids.n()),g,(float*)s.Phi.p(),s.filled.p,s.flag.p);ck(cudaGetLastError());return;}
 if(cmd=="decode"){req(nrhs==11&&nlhs==0,"decode(rawGpu,idsGpu,rowMap,rowSign,colMap,colSign,rigidRows,rigidCornerRows,cornerRows,nf)");Ref raw(in[1],mxSINGLE_CLASS),ids(in[2],mxINT32_CLASS),rm(in[3],mxINT32_CLASS),rs(in[4],mxINT32_CLASS),cm(in[5],mxINT32_CLASS),cs(in[6],mxINT32_CLASS),rr(in[7],mxDOUBLE_CLASS),dr(in[8],mxDOUBLE_CLASS),co(in[9],mxINT32_CLASS);int nf=asint(mxGetScalar(in[10]),1,200),batch=int(ids.n());req(raw.n()==size_t(nf)*8*batch&&rm.n()==1800&&rs.n()==1800&&cm.n()==72&&cs.n()==72&&rr.n()==600&&dr.n()==24&&co.n()==8,"Decode metadata size mismatch");Buf<double> inverse;inverse.alloc(64ULL*batch);invert<<<batch,64>>>((const float*)raw.p(),nf,(const int*)ids.p(),batch,(const int*)co.p(),inverse.p,s.flag.p,s.checks.p);decode<<<batch,256>>>((const float*)raw.p(),nf,(const int*)ids.p(),batch,g,inverse.p,(const int*)rm.p(),(const int*)rs.p(),(const int*)cm.p(),(const int*)cs.p(),(const double*)rr.p(),(const double*)dr.p(),(const int*)co.p(),(float*)s.Phi.p(),s.filled.p,s.flag.p,s.checks.p);ck(cudaGetLastError());return;}
 if(cmd=="finish"){req(nrhs==1&&nlhs==2,"[Phi,checks]=finish()");coverage<<<blocks(g.ns),256>>>(s.filled.p,g.ns,s.flag.p);int flags=0;double checks[3];ck(cudaMemcpy(&flags,s.flag.p,sizeof(int),cudaMemcpyDeviceToHost));ck(cudaMemcpy(checks,s.checks.p,sizeof(checks),cudaMemcpyDeviceToHost));if(flags){session.reset();throw std::runtime_error("GPU basis check failed, bit flags="+std::to_string(flags));}out[0]=s.Phi.give();out[1]=mxCreateDoubleMatrix(1,3,mxREAL);std::copy(checks,checks+3,mxGetPr(out[1]));session.reset();return;}
 req(false,"Unknown basis GPU command");
}
}
#ifndef MECHANICS2D_BASIS_GPU_EMBED
void mexFunction(int nlhs,mxArray** out,int nrhs,const mxArray** in){try{basisgpu::run(nlhs,out,nrhs,in);}catch(const std::exception& e){mexErrMsgIdAndTxt("mechanics2d_basis_gpu_v1:Failure","%s",e.what());}}
#endif

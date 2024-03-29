#include "kernel.h"
#include <thrust/sort.h>

//Only include parameters file if we're not creating the shared library
#ifndef PYTHON
#include "params.h"
#endif



__device__ int smoothkernel(int n, DTYPE * x, DTYPE * y, DTYPE * w, DTYPE span, int iper, DTYPE vsmlsq, DTYPE * smo, DTYPE * acvr) 
{
      
      int i,j,jper,in,out,ibw,it; //j0,
      DTYPE xto,xti;
      DTYPE wt,fbo,fbw=0.,xm=0.,ym=0.,tmp,var=0.,cvar=0.,a,h; //,sy

      jper=abs(iper);
      ibw=0.5*span*n+0.5;
      if (ibw<2) ibw=2;
      it=2*ibw+1;

      
      
      for (i=0;i<it;i++) {
        j=i;
        if (jper==2) j=i-ibw-1;
        if (j<0) {
          j+=n;
          xti=x[j]-1.0;
        } else xti=x[j];
        wt=w[j];
        fbo=fbw;
        fbw+=wt;
        if (fbw>0) {
          xm=(fbo*xm+wt*xti)/fbw;
          ym=(fbo*ym+wt*y[j])/fbw;
        }
        if (fbo>0) {
          tmp=fbw*wt*(xti-xm)/fbo;
          var+=tmp*(xti-xm);
          cvar+=tmp*(y[j]-ym);
        }
      }

      

      
      
      for (j=0;j<n;j++) {
        out=j-ibw-1;
        in=j+ibw;

        if (jper==2 || (out>=0 && in<n)) {

          if (in>n-1) {
            in-=n;
            xti=x[in]+1.0;
          } else xti=x[in];
          if (out<0) {
            out+=n;
            xto=x[out]-1.0;
          } else xto=x[out];

          wt=w[out];
          fbo=fbw;
          fbw-=wt;
          if (fbw>0) {
            tmp=fbo*wt*(xto-xm)/fbw;
            var-=tmp*(xto-xm);
            cvar-=tmp*(y[out]-ym);
          }
          if (fbw>0) {
            xm=(fbo*xm-wt*xto)/fbw;
            ym=(fbo*ym-wt*y[out])/fbw;
          }
          wt=w[in];
          fbo=fbw;
          fbw+=wt;
          if (fbw>0) {
            xm=(fbo*xm+wt*xti)/fbw;
            ym=(fbo*ym+wt*y[in])/fbw;
          }
          if (fbo>0) {
            tmp=fbw*wt*(xti-xm)/fbo;
            var+=tmp*(xti-xm);
            cvar+=tmp*(y[in]-ym);
          }
        }

        a=0.0;
        if (var>vsmlsq) a=cvar/var;
        smo[j]=a*(x[j]-xm)+ym;

        if (iper>0) {
          h=0.0;
          if (fbw>0) h=1.0/fbw;
          if (var>vsmlsq) h+=(x[j]-xm)*(x[j]-xm)/var;
          acvr[j]=0.0;
          a=1.0-w[j]*h;
          if (a>0) acvr[j]=fabs(y[j]-smo[j])/a;
          else if (j>0) acvr[j]=acvr[j-1];
        }
      }


      
      

      //Nat: can rm -- to deal with equal time values
      // for (j=0;j<n;j++) {
      //   sy=smo[j]*w[j];
      //   fbw=w[j];

      //   j0=j;
      //   while (j<n-1 && x[j+1]<=x[j]) {
      //     j+=1;
      //     sy+=w[j]*smo[j];
      //     fbw+=w[j];
      //   }

      //   if (j>j0) {
      //     a=0.0;
      //     if (fbw>0) a=sy/fbw;
      //     for (i=j0;i<=j;i++) smo[i]=a;
      //   }
      // }
      
      return 0;
}




//Copied/pasted comments from original fortran code by Friedman
// input:
   // n : number of observations (x,y - pairs).
   // x(n) : ordered abscissa values.
   // y(n) : corresponding ordinate (response) values.
   // w(n) : weight for each (x,y) observation.
   // iper : periodic variable flag.
      // iper=1 => x is ordered interval variable.
      // iper=2 => x is a periodic variable with values
                // in the range (0.0,1.0) and period 1.0.
   // span : smoother span (fraction of observations in window).
          // span=0.0 => automatic (variable) span selection.
   // alpha : controles high frequency (small span) penality
           // used with automatic span selection (bass tone control).
           // (alpha.le.0.0 or alpha.gt.10.0 => no effect.)
// output:
  // smo(n) : smoothed ordinate (response) values.
// scratch:
  // sc(n,8) : internal working storage.

__global__ void supsmukernel(const int numThreads, const int n, const int iper, const DTYPE span, const DTYPE alpha, DTYPE * smo, 
  DTYPE * sc, DTYPE * t1_sortby_argkeys, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys)
{
    


    unsigned int tid=threadIdx.x+ (blockIdx.x*blockDim.x); 
    if (tid>=numThreads)
    {
    	return;
    }
	

  	const unsigned int dataOffset=tid*n;
  	const unsigned int scOffset=tid*n*8;


    //global memory
    // pointers to time, data, offset
    DTYPE * x=t1_sortby_argkeys+dataOffset;
    DTYPE * y=data_sortby_argkeys+dataOffset;
    DTYPE * w=weights_sortby_argkeys+dataOffset;
    DTYPE * smo_thread=smo+dataOffset;
    DTYPE * sc_thread=sc+scOffset;



    //Store thread-local variables in shared-memory
    int i,j,jper;
    DTYPE vsmlsq,sw,sy,a,scale,resmin,tmp,f;

   
    // spans to be estimated: tweeter, midrange, and woofer
    DTYPE spans[] = {0.05,0.2,0.5};
    




    if (x[n-1]<=x[0]) {
      sy=0.0;
      sw=sy;
      for (j=0;j<n;j++) {
        sy=sy+w[j]*y[j];
        sw=sw+w[j];
      }
      a=0.0;
      if (sw>0) a=sy/sw;
      for (j=0;j<n;j++) smo_thread[j] = a;
      return;
    }

    i=n/4-1;
    j=3*(i+1)-1;
    scale=x[j]-x[i];

    //Nat: rm-- should never be entered
    // while (scale<=0) {
    //   if (j<n-1) j+=1;
    //   if (i>0) i-=1;
    //   scale=x[j]-x[i];
    // }
    vsmlsq=1.e-6*scale*scale;

    jper=iper;
    if (iper==2 && (x[0]<0 || x[n-1]>1)) jper=1;
    if (jper<1 || jper>2) jper=1;
    if (span>0) {
      smoothkernel (n,x,y,w,span,jper,vsmlsq,smo_thread,sc_thread);      // fixed span
      return;
    }

    // if we made it here, the span will be estimated and variable

    for (i=0;i<3;i++) {
      smoothkernel (n,x,y,w,spans[i],jper,vsmlsq,sc_thread+2*i*n,sc_thread+6*n);
      smoothkernel (n,x,sc_thread+6*n,w,spans[1],-jper,vsmlsq,sc_thread+(2*i+1)*n,sc_thread+7*n);
    }

    for (j=0;j<n;j++) {
      resmin=1.e20;
      for (i=0;i<3;i++) {
        if (sc_thread[j+(2*i+1)*n]<resmin) {
          resmin=sc_thread[j+(2*i+1)*n];
          sc_thread[j+6*n]=spans[i];
        }
      }
      if (alpha>0 && alpha<=10 && resmin<sc_thread[j+5*n] && resmin>0) {
        tmp = resmin/sc_thread[j+5*n];
        if (tmp<1.e-7) tmp=1.e-7;
        sc_thread[j+6*n]+=(spans[2]-sc_thread[j+6*n])*pow(tmp,10.0-alpha);
      }
    }

    smoothkernel (n,x,sc_thread+6*n,w,spans[1],-jper,vsmlsq,sc_thread+n,sc_thread+7*n);

    for (j=0;j<n;j++) {
      if (sc_thread[j+n]<=spans[0]) sc_thread[j+n]=spans[0];
      if (sc_thread[j+n]>=spans[2]) sc_thread[j+n]=spans[2];
      f=sc_thread[j+n]-spans[1];
      if (f<0) {
        f/=spans[0]-spans[1];
        sc_thread[j+3*n]=(1.0-f)*sc_thread[j+2*n]+f*sc_thread[j];
      } else {
        f/=spans[2]-spans[1];
        sc_thread[j+3*n]=(1.0-f)*sc_thread[j+2*n]+f*sc_thread[j+4*n];
      }
    }
    smoothkernel (n,x,sc_thread+3*n,w,spans[0],-jper,vsmlsq,smo_thread,sc_thread+7*n);
    return;
}







//Copied/pasted comments from original fortran code by Friedman
// input:
   // n : number of observations (x,y - pairs).
   // x(n) : ordered abscissa values.
   // y(n) : corresponding ordinate (response) values.
   // w(n) : weight for each (x,y) observation.
   // iper : periodic variable flag.
      // iper=1 => x is ordered interval variable.
      // iper=2 => x is a periodic variable with values
                // in the range (0.0,1.0) and period 1.0.
   // span : smoother span (fraction of observations in window).
          // span=0.0 => automatic (variable) span selection.
   // alpha : controles high frequency (small span) penality
           // used with automatic span selection (bass tone control).
           // (alpha.le.0.0 or alpha.gt.10.0 => no effect.)
// output:
  // smo(n) : smoothed ordinate (response) values.
// scratch:
  // sc(n,7) : internal working storage.

__global__ void supsmukernelOneThread (const int n, DTYPE * x, DTYPE * y, DTYPE * w, const int iper, const DTYPE span, const DTYPE alpha, DTYPE * smo, DTYPE * sc)
{
      // sc is scratch space (8,n)
      // output is smo: smoothed version of y

      int i,j,jper;
      DTYPE vsmlsq,sw,sy,a,scale,resmin,tmp,f;
     
      // spans to be estimated: tweeter, midrange, and woofer
      DTYPE spans[] = {0.05,0.2,0.5};

      if (x[n-1]<=x[0]) {
        sy=0.0;
        sw=sy;
        for (j=0;j<n;j++) {
          sy=sy+w[j]*y[j];
          sw=sw+w[j];
        }
        a=0.0;
        if (sw>0) a=sy/sw;
        for (j=0;j<n;j++) smo[j] = a;
        return;
      }

      i=n/4-1;
      j=3*(i+1)-1;
      scale=x[j]-x[i];

      //Nat: can be removed
      // while (scale<=0) {
      //   if (j<n-1) j+=1;
      //   if (i>0) i-=1;
      //   scale=x[j]-x[i];
      // }
      vsmlsq=1.e-6*scale*scale;

      jper=iper;
      if (iper==2 && (x[0]<0 || x[n-1]>1)) jper=1;
      if (jper<1 || jper>2) jper=1;
      if (span>0) {
        smoothkernel (n,x,y,w,span,jper,vsmlsq,smo,sc);      // fixed span
        return;
      }

      // if we made it here, the span will be estimated and variable

      for (i=0;i<3;i++) {
        smoothkernel (n,x,y,w,spans[i],jper,vsmlsq,sc+2*i*n,sc+6*n);
        smoothkernel (n,x,sc+6*n,w,spans[1],-jper,vsmlsq,sc+(2*i+1)*n,sc+7*n);
      }

      for (j=0;j<n;j++) {
        resmin=1.e20;
        for (i=0;i<3;i++) {
          if (sc[j+(2*i+1)*n]<resmin) {
            resmin=sc[j+(2*i+1)*n];
            sc[j+6*n]=spans[i];
          }
        }
        if (alpha>0 && alpha<=10 && resmin<sc[j+5*n] && resmin>0) {
          tmp = resmin/sc[j+5*n];
          if (tmp<1.e-7) tmp=1.e-7;
          sc[j+6*n]+=(spans[2]-sc[j+6*n])*pow(tmp,10.0-alpha);
        }
      }

      smoothkernel (n,x,sc+6*n,w,spans[1],-jper,vsmlsq,sc+n,sc+7*n);

      for (j=0;j<n;j++) {
        if (sc[j+n]<=spans[0]) sc[j+n]=spans[0];
        if (sc[j+n]>=spans[2]) sc[j+n]=spans[2];
        f=sc[j+n]-spans[1];
        if (f<0) {
          f/=spans[0]-spans[1];
          sc[j+3*n]=(1.0-f)*sc[j+2*n]+f*sc[j];
        } else {
          f/=spans[2]-spans[1];
          sc[j+3*n]=(1.0-f)*sc[j+2*n]+f*sc[j+4*n];
        }
      }
      smoothkernel (n,x,sc+3*n,w,spans[0],-jper,vsmlsq,smo,sc+7*n);
      return;
}





//All SM synchronization must occur before and after the function
__forceinline__ __device__ void parReductionMaximumPowerinSM(DTYPE maxPowerForComputingPeriod[], unsigned int maxPowerIdxForComputingPeriod[])
{		
		int i = blockDim.x / 2;
    	while (i != 0) {
	      	if(threadIdx.x < i && maxPowerForComputingPeriod[threadIdx.x + i] > maxPowerForComputingPeriod[threadIdx.x])
	      	{
	        maxPowerForComputingPeriod[threadIdx.x] = maxPowerForComputingPeriod[threadIdx.x + i]; 
	        maxPowerIdxForComputingPeriod[threadIdx.x] = maxPowerIdxForComputingPeriod[threadIdx.x + i];
	    	  }
	    	__syncthreads();
	    	i/=2;
    	}    	
}


//one small block computes a single frequency
//DEPRECATED: This kernel design does not perform well (worse than the global memory kernel and the SM kernel one thread per freq) 
/*
__global__ void supsmukernelSMOneFreqBlock(const int n, const int iper, const DTYPE span, const DTYPE alpha, DTYPE * smo, 
  DTYPE * sc, DTYPE * t1_sortby_argkeys, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys)
{
    

      
    int i,j,jper;
    DTYPE sw,sy,a,resmin,tmp,f;

    //passed in at runtime through kernel 
    //space for x, y, and w arrays in SM
    extern __shared__ DTYPE xyw[];

    __shared__ DTYPE * x;
    __shared__ DTYPE * y;
    __shared__ DTYPE * w;
    __shared__ DTYPE * sc_thread;
    __shared__ unsigned int dataOffset;
    __shared__ DTYPE spans[3];
    __shared__ DTYPE vsmlsq;
    __shared__ DTYPE scale;

    
    
    

    DTYPE * smo_thread;

    if(threadIdx.x==0)
    {

      //Compiling with std=c++14
      //says that this causes an invalid bitcast?
      x=xyw;
      y=xyw+(n);
      w=xyw+(2*n);
     

      i=n/4-1;
      j=3*(i+1)-1;
      scale=x[j]-x[i];

      vsmlsq=1.e-6*scale*scale;
      
      dataOffset=blockIdx.x*n;
      smo_thread=smo+dataOffset;
      sc_thread=sc+blockIdx.x*n*8;
      spans[0]=0.05;
      spans[1]=0.2;
      spans[2]=0.5;
    }
    __syncthreads();

    for (int i=0; i<n && ((i+threadIdx.x)<n); i+=blockDim.x)
    {
      int idx=i+threadIdx.x;
      x[idx]=t1_sortby_argkeys[dataOffset+idx];
      y[idx]=data_sortby_argkeys[dataOffset+idx];
      w[idx]=weights_sortby_argkeys[dataOffset+idx];
    }

    __syncthreads();



    //////////////////////////////////
    //Below is the original supsmu code



      //Use one thread in the block
      if (threadIdx.x==0)
      {

      if (x[n-1]<=x[0]) {
        sy=0.0;
        sw=sy;
        for (j=0;j<n;j++) {
          sy=sy+w[j]*y[j];
          sw=sw+w[j];
        }
        a=0.0;
        if (sw>0) a=sy/sw;
        for (j=0;j<n;j++) smo_thread[j] = a;
        return;
      }


      
      //Nat: can be removed
      // while (scale<=0) {
      //   if (j<n-1) j+=1;
      //   if (i>0) i-=1;
      //   scale=x[j]-x[i];
      // }

      jper=iper;
      if (iper==2 && (x[0]<0 || x[n-1]>1)) jper=1;
      if (jper<1 || jper>2) jper=1;
      if (span>0) {
        smoothkernel (n,x,y,w,span,jper,vsmlsq,smo_thread,sc_thread);      // fixed span
        return;
      }

      } //end if threadIdx.x==0

      


      // if we made it here, the span will be estimated and variable
      __syncthreads();
      //original
      //Use one thread in the block
      if (threadIdx.x==0)
      {
      for (i=0;i<3;i++) {
        smoothkernel (n,x,y,w,spans[i],jper,vsmlsq,sc_thread+2*i*n,sc_thread+6*n);
        smoothkernel (n,x,sc_thread+6*n,w,spans[1],-jper,vsmlsq,sc_thread+(2*i+1)*n,sc_thread+7*n);
      }
      }

      
      __syncthreads();


      //original
      // for (j=0;j<n;j++) {
      //   resmin=1.e20;
      //   for (i=0;i<3;i++) {
      //     if (sc_thread[j+(2*i+1)*n]<resmin) {
      //       resmin=sc_thread[j+(2*i+1)*n];
      //       sc_thread[j+6*n]=spans[i];
      //     }
      //   }
      //   if (alpha>0 && alpha<=10 && resmin<sc_thread[j+5*n] && resmin>0) {
      //     tmp = resmin/sc_thread[j+5*n];
      //     if (tmp<1.e-7) tmp=1.e-7;
      //     sc_thread[j+6*n]+=(spans[2]-sc_thread[j+6*n])*pow(tmp,10.0-alpha);
      //   }
      // }

      //parallelized with threads
      for (j=0;j<n && ((j+threadIdx.x)<n);j+=blockDim.x) {
        int idx=j+threadIdx.x;
        resmin=1.e20;
        for (i=0;i<3;i++) {
          if (sc_thread[idx+(2*i+1)*n]<resmin) {
            resmin=sc_thread[idx+(2*i+1)*n];
            sc_thread[idx+6*n]=spans[i];
          }
        }
        if (alpha>0 && alpha<=10 && resmin<sc_thread[idx+5*n] && resmin>0) {
          tmp = resmin/sc_thread[idx+5*n];
          if (tmp<1.e-7) tmp=1.e-7;
          sc_thread[idx+6*n]+=(spans[2]-sc_thread[idx+6*n])*pow(tmp,10.0-alpha);
        }
      }
      __syncthreads();
      //one thread
      if (threadIdx.x==0)
      {
      smoothkernel (n,x,sc_thread+6*n,w,spans[1],-jper,vsmlsq,sc_thread+n,sc_thread+7*n);
      }
      __syncthreads();
      //original
      // for (j=0;j<n;j++) {
      //   if (sc_thread[j+n]<=spans[0]) sc_thread[j+n]=spans[0];
      //   if (sc_thread[j+n]>=spans[2]) sc_thread[j+n]=spans[2];
      //   f=sc_thread[j+n]-spans[1];
      //   if (f<0) {
      //     f/=spans[0]-spans[1];
      //     sc_thread[j+3*n]=(1.0-f)*sc_thread[j+2*n]+f*sc_thread[j];
      //   } else {
      //     f/=spans[2]-spans[1];
      //     sc_thread[j+3*n]=(1.0-f)*sc_thread[j+2*n]+f*sc_thread[j+4*n];
      //   }
      // }

      //parallelized with threads
      for (j=0;j<n && ((j+threadIdx.x)<n);j+=blockDim.x) {
      // for (j=0;j<n;j++) {
        int idx=j+threadIdx.x;
        if (sc_thread[idx+n]<=spans[0]) sc_thread[idx+n]=spans[0];
        if (sc_thread[idx+n]>=spans[2]) sc_thread[idx+n]=spans[2];
        f=sc_thread[idx+n]-spans[1];
        if (f<0) {
          f/=spans[0]-spans[1];
          sc_thread[idx+3*n]=(1.0-f)*sc_thread[idx+2*n]+f*sc_thread[idx];
        } else {
          f/=spans[2]-spans[1];
          sc_thread[idx+3*n]=(1.0-f)*sc_thread[idx+2*n]+f*sc_thread[idx+4*n];
        }
      }
      __syncthreads();
      //only one thread computes this
      if(threadIdx.x==0)
      {
      smoothkernel (n,x,sc_thread+3*n,w,spans[0],-jper,vsmlsq,smo_thread,sc_thread+7*n);
      }
      return;
}
*/








//kernel gets called after the main kernel
//some number of threads per frequency
__global__ void computePgramReduction(const int batchwriteoffset, const int numThreadsPerFreq, const DTYPE chi0, const int n, const int numFreq, DTYPE * smo, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys, DTYPE * pgram)
{
  int i=threadIdx.x+ (blockIdx.x*blockDim.x); 

  extern __shared__ DTYPE globalSum[];

  int freqNum=i/numThreadsPerFreq;
  int threadInFreq=i%numThreadsPerFreq;
  int freqInBlock=threadIdx.x/numThreadsPerFreq;
  DTYPE localSum=0;

  if(i<(numFreq*numThreadsPerFreq))
  {

    if (threadInFreq==0)
    {
      globalSum[freqInBlock]=0;
    }
  } //end  if(i<(numFreq*numThreadsPerFreq))
  
  __syncthreads();




  if(i<(numFreq*numThreadsPerFreq))
  {
  
    int idxmin=(freqNum*n);
    int idxmax=(idxmin+n);

    for (int k=idxmin; k<idxmax; k+=numThreadsPerFreq){
      // int idx=(freqNum*n)+(k+threadInFreq);
      int idx=k+threadInFreq;
      if (idx<idxmax)
      {
      localSum+=((data_sortby_argkeys[idx]-smo[idx])*(data_sortby_argkeys[idx]-smo[idx]))*weights_sortby_argkeys[idx];
      }
    }

  }

  if(i<(numFreq*numThreadsPerFreq))
  {

  atomicAdd(&globalSum[freqInBlock],localSum);

  }

  __syncthreads();
  
  if (threadInFreq==0 && i<(numFreq*numThreadsPerFreq))
  {
  // chi2[batchwriteoffset+freqNum]=globalSum[freqInBlock]/(n*1.0);
  // pgram[batchwriteoffset+freqNum]=(0.5*(chi0-chi2[batchwriteoffset+freqNum])*n)/chi0;
    DTYPE chi2=globalSum[freqInBlock]/(n*1.0);
    pgram[batchwriteoffset+freqNum]=(0.5*(chi0-chi2)*n)/chi0;
  }
  
}


//kernel gets called after the main kernel
//some number of threads per frequency
//data, weights reordered for coalesced memory accesses
// x[j] -> x[freqNum+(numFreq*j)]
__global__ void computePgramReductionCoalesced(const int batchwriteoffset, const int numThreadsPerFreq, const DTYPE chi0, const int n, const int numFreq, DTYPE * smo, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys, DTYPE * pgram)
{
  int i=threadIdx.x+ (blockIdx.x*blockDim.x); 


  extern __shared__ DTYPE globalSum[];

  int freqNum=i/numThreadsPerFreq;
  int threadInFreq=i%numThreadsPerFreq;
  int freqInBlock=threadIdx.x/numThreadsPerFreq;
  DTYPE localSum=0;

  if(i<(numFreq*numThreadsPerFreq))
  {

    if (threadInFreq==0)
    {
      globalSum[freqInBlock]=0;
    }
  } //end  if(i<(numFreq*numThreadsPerFreq))
  
  __syncthreads();




  if(i<(numFreq*numThreadsPerFreq))
  {
  
  int idxmin=(freqNum*n);
  int idxmax=(idxmin+n);

  for (int k=idxmin; k<idxmax; k+=numThreadsPerFreq){
    int idx=k+threadInFreq;
    int idxCoalesced=freqNum+(numFreq*(idx-idxmin)); //idx-idxmin because its one big long array, not an array with n elements


    if (idx<idxmax)
    {
      localSum+=((data_sortby_argkeys[idxCoalesced]-smo[idx])*(data_sortby_argkeys[idxCoalesced]-smo[idx]))*weights_sortby_argkeys[idxCoalesced];
    }
  }

  }

  if(i<(numFreq*numThreadsPerFreq))
  {
  atomicAdd(&globalSum[freqInBlock],localSum);
  }
  

  __syncthreads();
  

  

  if (threadInFreq==0 && i<(numFreq*numThreadsPerFreq))
  {
  // chi2[batchwriteoffset+freqNum]=globalSum[freqInBlock]/(n*1.0);
  // pgram[batchwriteoffset+freqNum]=(0.5*(chi0-chi2[batchwriteoffset+freqNum])*n)/chi0;
  DTYPE chi2=globalSum[freqInBlock]/(n*1.0);
  pgram[batchwriteoffset+freqNum]=(0.5*(chi0-chi2)*n)/chi0;  
  }
  
}




__global__ void computePeriodModFOneThreadPerUpdate(const int n, const int numFreq, const DTYPE minFreq, const uint64_t freqOffset, const DTYPE deltaf, DTYPE * t1, DTYPE * tt)
{ 
    int i=threadIdx.x+ (blockIdx.x*blockDim.x); 
    if (i>=(n*numFreq))
    {
      return;
    }

    int freqNum=i/n;
    DTYPE p=1.0/((minFreq)+(deltaf*(freqOffset+freqNum)));
    t1[i]=fmod(tt[i%n],p)/p;
}




__global__ void initializeKeyArraysOneThreadPerUpdate(const int n, const int numFreq, int * argkeys, int * freqId)
{
    int i=threadIdx.x+ (blockIdx.x*blockDim.x); 
    if (i>=(n*numFreq))
    {
      return;
    }

    int freqNum=i/n;
   
    //iota
    argkeys[i]=i%n;
    //same frequency id for the freqId array
    freqId[i]=freqNum; 
   
}



__global__ void mapUsingArgKeysOneThreadPerUpdate(const int n, const int numFreq, int * argkeys, DTYPE * data, DTYPE * weights, DTYPE * t1, DTYPE * t1_sortby_argkeys, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys)
{

    int i=threadIdx.x+ (blockIdx.x*blockDim.x); 
    if (i>=(n*numFreq))
    {
      return;
    }

    //t1 has already been sorted. Only make a copy.
    t1_sortby_argkeys[i]=t1[i];
    //map between t1 argkeys and data and weights
    data_sortby_argkeys[i]=data[argkeys[i]];
    weights_sortby_argkeys[i]=weights[argkeys[i]];




}

//used for coalesced memory mapping
__global__ void mapUsingArgKeysOneThreadPerUpdateAndReorderCoalesced(const int n, const int numFreq, int * argkeys, DTYPE * data, DTYPE * weights, DTYPE * t1, DTYPE * t1_sortby_argkeys, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys)
{

    int i=threadIdx.x+ (blockIdx.x*blockDim.x); 
    if (i>=(n*numFreq))
    {
      return;
    }

    const int idxInFreq=(i%n);
    const int freqNum=i/n;

    const int idxWrite=(idxInFreq*numFreq)+freqNum;

    //t1 has already been sorted. Only make a copy.
    // t1_sortby_argkeys[idxWrite]=t1[argkeys[i]];
    t1_sortby_argkeys[idxWrite]=t1[i];
    data_sortby_argkeys[idxWrite]=data[argkeys[i]];
    weights_sortby_argkeys[idxWrite]=weights[argkeys[i]];
}







//Uses SM but uses one thread per frequency
__global__ void supsmukernelSMOneThreadPerFreq(const int numFreq, const int n, const int iper, const DTYPE span, const DTYPE alpha, 
  DTYPE * smo, DTYPE * sc, DTYPE * t1_sortby_argkeys, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys)
{

    int tid=threadIdx.x+ (blockIdx.x*blockDim.x); 
    
    unsigned int dataOffset=tid*n;
    unsigned int scOffset=tid*n*8; 
    DTYPE * smo_thread=smo+dataOffset;
    DTYPE * sc_thread=sc+scOffset;
    
    //size passed in at runtime through kernel 
    extern __shared__ DTYPE xyw[];
    __shared__ DTYPE spans[3];

    DTYPE * x=xyw+(threadIdx.x*3*n);
    DTYPE * y=xyw+(threadIdx.x*3*n)+(n);
    DTYPE * w=xyw+(threadIdx.x*3*n)+(2*n);

    //One thread copies the spans into SM
    if(threadIdx.x==0)
    {
      spans[0]=0.05;
      spans[1]=0.2;
      spans[2]=0.5;
    }
    __syncthreads();


    if(tid<numFreq)
    {

      
      for (int i=0; i<n; i++)
      {
        x[i]=t1_sortby_argkeys[dataOffset+i];
        y[i]=data_sortby_argkeys[dataOffset+i];
        w[i]=weights_sortby_argkeys[dataOffset+i];
      }

    }
    

    
    //////////////////////////////////
    //Below is the original supsmu code


      
    int i,j,jper;
    DTYPE vsmlsq,sw,sy,a,scale,resmin,tmp,f;


    
    if(tid<numFreq)
    {  

          

          if (x[n-1]<=x[0]) {
            sy=0.0;
            sw=sy;
            for (j=0;j<n;j++) {
              sy=sy+w[j]*y[j];
              sw=sw+w[j];
            }
            a=0.0;
            if (sw>0) a=sy/sw;
            for (j=0;j<n;j++) smo_thread[j] = a;
            return;
          }

          i=n/4-1;
          j=3*(i+1)-1;
          scale=x[j]-x[i];
          
          vsmlsq=1.e-6*scale*scale;

          jper=iper;
          if (iper==2 && (x[0]<0 || x[n-1]>1)) jper=1;
          if (jper<1 || jper>2) jper=1;
          if (span>0) {
            smoothkernel (n,x,y,w,span,jper,vsmlsq,smo_thread,sc_thread);      // fixed span
            return;
          }

          

          //Nat: if we made it here, the span will be estimated and variable

          for (i=0;i<3;i++) {
            smoothkernel (n,x,y,w,spans[i],jper,vsmlsq,sc_thread+2*i*n,sc_thread+6*n);
            smoothkernel (n,x,sc_thread+6*n,w,spans[1],-jper,vsmlsq,sc_thread+(2*i+1)*n,sc_thread+7*n);
          }


          
          
            for (j=0;j<n;j++) {
              resmin=1.e20;
              for (i=0;i<3;i++) {
                if (sc_thread[j+(2*i+1)*n]<resmin) {
                  resmin=sc_thread[j+(2*i+1)*n];
                  sc_thread[j+6*n]=spans[i];
                }
              }
              if (alpha>0 && alpha<=10 && resmin<sc_thread[j+5*n] && resmin>0) {
                tmp = resmin/sc_thread[j+5*n];
                if (tmp<1.e-7) tmp=1.e-7;
                sc_thread[j+6*n]+=(spans[2]-sc_thread[j+6*n])*pow(tmp,10.0-alpha);
              }
            }
         
          
          smoothkernel (n,x,sc_thread+6*n,w,spans[1],-jper,vsmlsq,sc_thread+n,sc_thread+7*n);
          

          
            for (j=0;j<n;j++) {
              if (sc_thread[j+n]<=spans[0]) sc_thread[j+n]=spans[0];
              if (sc_thread[j+n]>=spans[2]) sc_thread[j+n]=spans[2];
              f=sc_thread[j+n]-spans[1];
              if (f<0) {
                f/=spans[0]-spans[1];
                sc_thread[j+3*n]=(1.0-f)*sc_thread[j+2*n]+f*sc_thread[j];
              } else {
                f/=spans[2]-spans[1];
                sc_thread[j+3*n]=(1.0-f)*sc_thread[j+2*n]+f*sc_thread[j+4*n];
              }
            }
          
          
          

          
          smoothkernel (n,x,sc_thread+3*n,w,spans[0],-jper,vsmlsq,smo_thread,sc_thread+7*n);
          

    
    }//end the if(tid<numFreq) around everything
      


      return;
}


//Smooth() function for single pass
//Nat's updated function
__device__ void smoothSinglePassCoalesced(const int n, const int freqNum, const int numFreq, int * ibw, DTYPE * x, DTYPE * y, DTYPE * w, const DTYPE vsmlsq, const int alpha, DTYPE * smo) 
{

    int i,j,in,out;
    DTYPE wt,xto,xti,yto,yti,ibwb,a,f,chi2,chi2m,tmp,fbo,vary=0.0;
    // DTYPE fbw[3],xm[3],ym[3],smo0[3],var[3]={0,0,0},cvar[3]={0,0,0};

    //Original: if arrays stored in registers
    DTYPE fbw[3],xm[3],ym[3],smo0[3],var[3]={0,0,0},cvar[3]={0,0,0};
    const int offset=0;
    // //end original if using arrays in registers
    //If SINGLEPASSMODE==0 or SINGLEPASSMODE==2  (if not (SINGLEPASSMODE==1), then don't need to multiply by SMALLBLOCKSIZE)

    //If using SM for these arrays
    // __shared__ DTYPE fbw[SMALLBLOCKSIZE*3];
    // __shared__ DTYPE xm[SMALLBLOCKSIZE*3];
    // __shared__ DTYPE ym[SMALLBLOCKSIZE*3];
    // __shared__ DTYPE smo0[SMALLBLOCKSIZE*3];
    // __shared__ DTYPE var[SMALLBLOCKSIZE*3];
    // __shared__ DTYPE cvar[SMALLBLOCKSIZE*3];
    // const int offset=threadIdx.x*3;
    // for (i=0; i<3; i++)
    // {
    //   var[offset+i]=0;
    //   cvar[offset+i]=0;
    // }
    

    //data access transformation for coalesced memory accesses
    // x[j] -> x[freqNum+(numFreq*j)]
    

    for (i=0;i<3;i++) {
      j=n-ibw[i]-1;
      // xm[offset+i]=x[j]-1.0;
      // ym[offset+i]=y[j];
      // fbw[offset+i]=w[j];
      xm[offset+i]=x[freqNum+(numFreq*j)]-1.0;
      ym[offset+i]=y[freqNum+(numFreq*j)];
      fbw[offset+i]=w[freqNum+(numFreq*j)];
      for (j=n-ibw[i];j<n;j++) {
        // xti=x[j]-1.0;
        // yti=y[j];
        // wt=w[j];
        xti=x[freqNum+(numFreq*j)]-1.0;
        yti=y[freqNum+(numFreq*j)];
        wt=w[freqNum+(numFreq*j)];

        fbo=fbw[offset+i];
        fbw[offset+i]+=wt;
        xm[offset+i]=(fbo*xm[offset+i]+wt*xti)/fbw[offset+i];
        ym[offset+i]=(fbo*ym[offset+i]+wt*yti)/fbw[offset+i];
        tmp=fbw[offset+i]*wt*(xti-xm[offset+i])/fbo;
        var[offset+i]+=tmp*(xti-xm[offset+i]);
        cvar[offset+i]+=tmp*(yti-ym[offset+i]);
        if (i==0) vary+=fbw[offset+0]*wt*(yti-ym[offset+0])*(yti-ym[offset+0])/fbo;
      }
      for (j=0;j<ibw[i];j++) {
        // xti=x[j];
        // yti=y[j];
        // wt=w[j];
        xti=x[freqNum+(numFreq*j)];
        yti=y[freqNum+(numFreq*j)];
        wt=w[freqNum+(numFreq*j)];
        fbo=fbw[offset+i];
        fbw[offset+i]+=wt;
        xm[offset+i]=(fbo*xm[offset+i]+wt*xti)/fbw[offset+i];
        ym[offset+i]=(fbo*ym[offset+i]+wt*yti)/fbw[offset+i];
        tmp=fbw[offset+i]*wt*(xti-xm[offset+i])/fbo;
        var[offset+i]+=tmp*(xti-xm[offset+i]);
        cvar[offset+i]+=tmp*(yti-ym[offset+i]);
        if (i==0) vary+=fbw[offset+0]*wt*(yti-ym[offset+0])*(yti-ym[offset+0])/fbo;
      }
    }  

    for (j=0;j<n;j++) {

      for (i=0;i<3;i++) {
        out=j-ibw[i]-1;
        in=j+ibw[i];

        if (in>n-1) {
          in-=n;
          // xti=x[in]+1.0;
          xti=x[freqNum+(numFreq*in)]+1.0;
        // } else xti=x[in];
        } else xti=x[freqNum+(numFreq*in)];  
        if (out<0) {
          out+=n;
          // xto=x[out]-1.0;
          xto=x[freqNum+(numFreq*out)]-1.0;
        // } else xto=x[out];
        } else xto=x[freqNum+(numFreq*out)];
        // yti=y[in];
        // yto=y[out];

        yti=y[freqNum+(numFreq*in)];
        yto=y[freqNum+(numFreq*out)];

        // wt=w[out];
        wt=w[freqNum+(numFreq*out)];
        fbo=fbw[offset+i];
        fbw[offset+i]-=wt;
        tmp=fbo*wt*(xto-xm[offset+i])/fbw[offset+i];
        var[offset+i]-=tmp*(xto-xm[offset+i]);
        cvar[offset+i]-=tmp*(yto-ym[offset+i]);
        if (i==0) vary-=fbo*wt*(yto-ym[offset+0])*(yto-ym[offset+0])/fbw[offset+0];
        xm[offset+i]=(fbo*xm[offset+i]-wt*xto)/fbw[offset+i];
        ym[offset+i]=(fbo*ym[offset+i]-wt*yto)/fbw[offset+i];

        // wt=w[in];
        wt=w[freqNum+(numFreq*in)];
        fbo=fbw[offset+i];
        fbw[offset+i]+=wt;
        xm[offset+i]=(fbo*xm[offset+i]+wt*xti)/fbw[offset+i];
        ym[offset+i]=(fbo*ym[offset+i]+wt*yti)/fbw[offset+i];
        tmp=fbw[offset+i]*wt*(xti-xm[offset+i])/fbo;
        var[offset+i]+=tmp*(xti-xm[offset+i]);
        cvar[offset+i]+=tmp*(yti-ym[offset+i]);
        if (i==0) vary+=fbw[offset+0]*wt*(yti-ym[offset+0])*(yti-ym[offset+0])/fbo;

      }

      chi2m=1.e20; ibwb=ibw[2];
      for (i=0;i<3;i++) {
        a=0.0;
        if (var[offset+i]>vsmlsq) a=cvar[offset+i]/var[offset+i];
        // smo0[offset+i]=a*(x[j]-xm[offset+i])+ym[offset+i];
        smo0[offset+i]=a*(x[freqNum+(numFreq*j)]-xm[offset+i])+ym[offset+i];
        chi2 = vary-2*a*cvar[offset+0]+a*a*var[offset+0];
        if (i>0) {
          tmp = ym[offset+i]-ym[offset+0]-a*(xm[offset+i]-xm[offset+0]);
          chi2 += tmp*tmp*fbw[offset+0];
        }
        tmp=1.0/fbw[offset+i];
        // if (var[offset+i]>vsmlsq) tmp+=(x[j]-xm[offset+i])*(x[j]-xm[offset+i])/var[offset+i];
        if (var[offset+i]>vsmlsq) tmp+=(x[freqNum+(numFreq*j)]-xm[offset+i])*(x[freqNum+(numFreq*j)]-xm[offset+i])/var[offset+i];
        // tmp = 1.0 - w[j]*tmp;
        tmp = 1.0 - w[freqNum+(numFreq*j)]*tmp;
        chi2 = fabs(chi2)/(tmp*tmp);
        if (chi2<chi2m) {
          chi2m=chi2;
          ibwb=(ibw[1]+ibw[i])/2.;
        }
      }

      tmp = sqrt(chi2m/chi2);
      if (tmp<1.e-7) tmp=1.e-7;
      ibwb+=(ibw[2]-ibwb)*pow(tmp,10.-alpha);
      f = ibwb-ibw[1];
      if (f<0) {
        f/=ibw[0]-ibw[1];
        smo[j]=(1.0-f)*smo0[offset+1]+f*smo0[offset+0];
      } else {
        f/=ibw[2]-ibw[1];
        smo[j]=(1.0-f)*smo0[offset+1]+f*smo0[offset+2];
      }
    }

}

//Smooth() function for single pass
//Nat's updated function
__device__ void smoothSinglePass(const int n, int * ibw, DTYPE * x, DTYPE * y, DTYPE * w, const DTYPE vsmlsq, const int alpha, DTYPE * smo) 
{

    int i,j,in,out;
    DTYPE wt,xto,xti,yto,yti,ibwb,a,f,chi2,chi2m,tmp,fbo,vary=0.0;
    // DTYPE fbw[3],xm[3],ym[3],smo0[3],var[3]={0,0,0},cvar[3]={0,0,0};

    //Original: if arrays stored in registers
    DTYPE fbw[3],xm[3],ym[3],smo0[3],var[3]={0,0,0},cvar[3]={0,0,0};
    const int offset=0;
    //end original if using arrays in registers
    //If SINGLEPASSMODE==2 (if not, then don't need to multiply by SMALLBLOCKSIZE)

    //If using SM for these arrays
    // __shared__ DTYPE fbw[SMALLBLOCKSIZE*3];
    // __shared__ DTYPE xm[SMALLBLOCKSIZE*3];
    // __shared__ DTYPE ym[SMALLBLOCKSIZE*3];
    // __shared__ DTYPE smo0[SMALLBLOCKSIZE*3];
    // __shared__ DTYPE var[SMALLBLOCKSIZE*3];
    // __shared__ DTYPE cvar[SMALLBLOCKSIZE*3];
    // const int offset=threadIdx.x*3;
    // for (i=0; i<3; i++)
    // {
    //   var[offset+i]=0;
    //   cvar[offset+i]=0;
    // }
    //end if using SM for these arrays



    for (i=0;i<3;i++) {
      j=n-ibw[i]-1;
      xm[offset+i]=x[j]-1.0;
      ym[offset+i]=y[j];
      fbw[offset+i]=w[j];
      for (j=n-ibw[i];j<n;j++) {
        xti=x[j]-1.0;
        yti=y[j];
        wt=w[j];
        fbo=fbw[offset+i];
        fbw[offset+i]+=wt;
        xm[offset+i]=(fbo*xm[offset+i]+wt*xti)/fbw[offset+i];
        ym[offset+i]=(fbo*ym[offset+i]+wt*yti)/fbw[offset+i];
        tmp=fbw[offset+i]*wt*(xti-xm[offset+i])/fbo;
        var[offset+i]+=tmp*(xti-xm[offset+i]);
        cvar[offset+i]+=tmp*(yti-ym[offset+i]);
        if (i==0) vary+=fbw[offset+0]*wt*(yti-ym[offset+0])*(yti-ym[offset+0])/fbo;
      }
      for (j=0;j<ibw[i];j++) {
        xti=x[j];
        yti=y[j];
        wt=w[j];
        fbo=fbw[offset+i];
        fbw[offset+i]+=wt;
        xm[offset+i]=(fbo*xm[offset+i]+wt*xti)/fbw[offset+i];
        ym[offset+i]=(fbo*ym[offset+i]+wt*yti)/fbw[offset+i];
        tmp=fbw[offset+i]*wt*(xti-xm[offset+i])/fbo;
        var[offset+i]+=tmp*(xti-xm[offset+i]);
        cvar[offset+i]+=tmp*(yti-ym[offset+i]);
        if (i==0) vary+=fbw[offset+0]*wt*(yti-ym[offset+0])*(yti-ym[offset+0])/fbo;
      }
    }  

    for (j=0;j<n;j++) {

      for (i=0;i<3;i++) {
        out=j-ibw[i]-1;
        in=j+ibw[i];

        if (in>n-1) {
          in-=n;
          xti=x[in]+1.0;
        } else xti=x[in];
        if (out<0) {
          out+=n;
          xto=x[out]-1.0;
        } else xto=x[out];
        yti=y[in];
        yto=y[out];

        wt=w[out];
        fbo=fbw[offset+i];
        fbw[offset+i]-=wt;
        tmp=fbo*wt*(xto-xm[offset+i])/fbw[offset+i];
        var[offset+i]-=tmp*(xto-xm[offset+i]);
        cvar[offset+i]-=tmp*(yto-ym[offset+i]);
        if (i==0) vary-=fbo*wt*(yto-ym[offset+0])*(yto-ym[offset+0])/fbw[offset+0];
        xm[offset+i]=(fbo*xm[offset+i]-wt*xto)/fbw[offset+i];
        ym[offset+i]=(fbo*ym[offset+i]-wt*yto)/fbw[offset+i];

        wt=w[in];
        fbo=fbw[offset+i];
        fbw[offset+i]+=wt;
        xm[offset+i]=(fbo*xm[offset+i]+wt*xti)/fbw[offset+i];
        ym[offset+i]=(fbo*ym[offset+i]+wt*yti)/fbw[offset+i];
        tmp=fbw[offset+i]*wt*(xti-xm[offset+i])/fbo;
        var[offset+i]+=tmp*(xti-xm[offset+i]);
        cvar[offset+i]+=tmp*(yti-ym[offset+i]);
        if (i==0) vary+=fbw[offset+0]*wt*(yti-ym[offset+0])*(yti-ym[offset+0])/fbo;

      }

      chi2m=1.e20; ibwb=ibw[2];
      for (i=0;i<3;i++) {
        a=0.0;
        if (var[offset+i]>vsmlsq) a=cvar[offset+i]/var[offset+i];
        smo0[offset+i]=a*(x[j]-xm[offset+i])+ym[offset+i];
        chi2 = vary-2*a*cvar[offset+0]+a*a*var[offset+0];
        if (i>0) {
          tmp = ym[offset+i]-ym[offset+0]-a*(xm[offset+i]-xm[offset+0]);
          chi2 += tmp*tmp*fbw[offset+0];
        }
        tmp=1.0/fbw[offset+i];
        if (var[offset+i]>vsmlsq) tmp+=(x[j]-xm[offset+i])*(x[j]-xm[offset+i])/var[offset+i];
        tmp = 1.0 - w[j]*tmp;
        chi2 = fabs(chi2)/(tmp*tmp);
        if (chi2<chi2m) {
          chi2m=chi2;
          ibwb=(ibw[1]+ibw[i])/2.;
        }
      }

      tmp = sqrt(chi2m/chi2);
      if (tmp<1.e-7) tmp=1.e-7;
      ibwb+=(ibw[2]-ibwb)*pow(tmp,10.-alpha);
      f = ibwb-ibw[1];
      if (f<0) {
        f/=ibw[0]-ibw[1];
        smo[j]=(1.0-f)*smo0[offset+1]+f*smo0[offset+0];
      } else {
        f/=ibw[2]-ibw[1];
        smo[j]=(1.0-f)*smo0[offset+1]+f*smo0[offset+2];
      }
    }

}



__global__ void supsmukernelSinglePassSMOneBlockPerFreq(const int n, const DTYPE inalpha, DTYPE * smo, DTYPE * t1_sortby_argkeys, 
  DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys)
{
      
    const unsigned int dataOffset=blockIdx.x*n;

    DTYPE * smo_thread=smo+dataOffset;
  

    //Shared memory
    //size passed in at runtime through kernel 
    extern __shared__ DTYPE xyw[];
    DTYPE * x=xyw;
    DTYPE * y=xyw+(n);
    DTYPE * w=xyw+(2*n);

    //SM
    __shared__ int ibw[3];
    __shared__ DTYPE spans[3];
    __shared__ DTYPE scale;
    __shared__ DTYPE vsmlsq;
    __shared__ DTYPE alpha;

    for (int i=0; i<n && ((i+threadIdx.x)<n); i++)
    {
      int idx=i+threadIdx.x;
      x[idx]=t1_sortby_argkeys[dataOffset+idx];
      y[idx]=data_sortby_argkeys[dataOffset+idx];
      w[idx]=weights_sortby_argkeys[dataOffset+idx];
    }

    __syncthreads();

    

    //one thread in the block computes everything
    if (threadIdx.x==0)
    {
    int i=n/4-1;
    int j=3*(i+1)-1;
    scale=x[j]-x[i];
    vsmlsq=1.e-6*scale*scale;
    alpha=inalpha;
    if (alpha<0) alpha=0;
    if (alpha>10) alpha=10;

    spans[0]=0.05;
    spans[1]=0.2;
    spans[2]=0.5;

    for (int i=0;i<3;i++) {
    ibw[i] = (int)( 0.5*spans[i]*n+0.5 );
    if (ibw[i]<2) ibw[i]=2;
    }
    
    smoothSinglePass(n, ibw, x, y, w, vsmlsq, alpha, smo_thread);

    }

    return;
}


__global__ void supsmukernelSinglePassSMOneThreadPerFreq(const int numFreq, const int n, const DTYPE inalpha, DTYPE * smo, 
  DTYPE * t1_sortby_argkeys, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys)
{
    


    unsigned int tid=threadIdx.x+ (blockIdx.x*blockDim.x); 
    if (tid>=numFreq)
    {
      return;
    }
  


    const unsigned int dataOffset=tid*n;

    DTYPE * smo_thread=smo+dataOffset;
  
    //Shared memory
    //size passed in at runtime through kernel 
    extern __shared__ DTYPE xyw[];
    DTYPE * x=xyw+(threadIdx.x*3*n);
    DTYPE * y=xyw+(threadIdx.x*3*n)+(n);
    DTYPE * w=xyw+(threadIdx.x*3*n)+(2*n);

    for (int i=0; i<n; i++)
    {
      x[i]=t1_sortby_argkeys[dataOffset+i];
      y[i]=data_sortby_argkeys[dataOffset+i];
      w[i]=weights_sortby_argkeys[dataOffset+i];
    }

    //use SM
    //each thread needs its own ibw
    // __shared__ int ibwSM[3*SMALLBLOCKSIZE];
    __shared__ int ibw[3*SMALLBLOCKSIZE];

    //Constant for all frequencies
    __shared__ DTYPE spans[3];
    __shared__ DTYPE alpha;

    int i=n/4-1;
    int j=3*(i+1)-1;
    DTYPE scale=x[j]-x[i];
    DTYPE vsmlsq=1.e-6*scale*scale;
    


    if (threadIdx.x==0)
    {
    //constant for all frequencies
    alpha=inalpha;
    if (alpha<0) alpha=0;
    if (alpha>10) alpha=10;

    spans[0]=0.05;
    spans[1]=0.2;
    spans[2]=0.5;
    }
    __syncthreads();
    
    //update pointer to my ibw
    // int * ibw=&ibwSM[threadIdx.x*3];
    // for (int i=0;i<3;i++) {
    // ibw[i] = (int)( 0.5*spans[i]*n+0.5 );
    // if (ibw[i]<2) ibw[i]=2;
    // }


    for (int i=0;i<3;i++) {
    ibw[threadIdx.x*3+i] = (int)( 0.5*spans[i]*n+0.5 );
    if (ibw[threadIdx.x*3+i]<2) ibw[threadIdx.x*3+i]=2;
    }

    smoothSinglePass(n, &ibw[threadIdx.x*3], x, y, w, vsmlsq, alpha, smo_thread);

    return;
}

//global memory baseline for singlepass
__global__ void supsmukernelSinglePassGlobalMemory(const int numFreq, const int n, 
   const DTYPE inalpha, DTYPE * smo, DTYPE * tt, DTYPE * t1_sortby_argkeys, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys)
{
    unsigned int tid=threadIdx.x+ (blockIdx.x*blockDim.x); 
    if (tid>=numFreq)
    {
      return;
    }



    const unsigned int dataOffset=tid*n;

    DTYPE * smo_thread=smo+dataOffset;
    // pointers to time, data, offset
    DTYPE * x=t1_sortby_argkeys+dataOffset;
    DTYPE * y=data_sortby_argkeys+dataOffset;
    DTYPE * w=weights_sortby_argkeys+dataOffset;
  
    int ibw[3];
    DTYPE spans[3] = {0.05,0.2,0.5};

    int i=n/4-1;
    int j=3*(i+1)-1;
    DTYPE scale=x[j]-x[i];
    DTYPE vsmlsq=1.e-6*scale*scale;
    DTYPE alpha=inalpha;
    if (alpha<0) alpha=0;
    if (alpha>10) alpha=10;
    
    for (int i=0;i<3;i++) {
    ibw[i] = (int)( 0.5*spans[i]*n+0.5 );
    if (ibw[i]<2) ibw[i]=2;
    }

    smoothSinglePass(n, ibw, x, y, w, vsmlsq, alpha, smo_thread);
    

    return;
}

//global memory for singlepass -- coalesced memory accesses
__global__ void supsmukernelSinglePassGlobalMemoryCoalesced(const int numFreq, const int n, 
   const DTYPE inalpha, DTYPE * smo, DTYPE * tt, DTYPE * t1_sortby_argkeys, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys)
{
    unsigned int tid=threadIdx.x+ (blockIdx.x*blockDim.x); 
    if (tid>=numFreq)
    {
      return;
    }
  
    const unsigned int dataOffset=tid*n;

    DTYPE * smo_thread=smo+dataOffset;
  
    //original in registers
    int ibw[3];
    DTYPE spans[3] = {0.05,0.2,0.5};
    //end original in registers

    //shared memory
    // __shared__ DTYPE spans[3];
    // __shared__ int ibwSM[3*SMALLBLOCKSIZE];
    // int * ibw=&ibwSM[threadIdx.x*3];
    // if (threadIdx.x==0)
    // {
    //   spans[0]=0.05;
    //   spans[1]=0.2;
    //   spans[2]=0.5;
    // }
    // __syncthreads();
    //end shared memory


    int i=n/4-1;
    int j=3*(i+1)-1;
    // DTYPE scale=x[j]-x[i];
    DTYPE scale=t1_sortby_argkeys[tid+(numFreq*j)]-t1_sortby_argkeys[tid+(numFreq*i)];
    DTYPE vsmlsq=1.e-6*scale*scale;
    DTYPE alpha=inalpha;
    if (alpha<0) alpha=0;
    if (alpha>10) alpha=10;
    
    for (int i=0;i<3;i++) {
    ibw[i] = (int)( 0.5*spans[i]*n+0.5 );
    if (ibw[i]<2) ibw[i]=2;
    }

    smoothSinglePassCoalesced(n, tid, numFreq, ibw, t1_sortby_argkeys, data_sortby_argkeys, weights_sortby_argkeys, vsmlsq, alpha, smo_thread); 
  
    return;
}



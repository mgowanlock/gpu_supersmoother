// MIT License

// Copyright (c) 2021 Mike Gowanlock and Nat Butler

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <unistd.h>
#include <istream>
#include <iostream>
#include <sstream>
#include "params.h"
#include "supsmu.h"
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <numeric>  
#include <vector>
#include <fstream>
#include <omp.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include "structs.h"
#include "main.h"
#include "kernel.h"



using namespace std;

template <typename T>
std::vector<int> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](int i1, int i2) {return v[i1] < v[i2];});

  return idx;
}


//original port from Nat's code before breaking into two separate functions
void supsmu_periodogram(int n, const DTYPE minFreq, const DTYPE maxFreq, int numFreq, DTYPE * time, DTYPE * data, DTYPE * error, DTYPE alpha, DTYPE * pgram)
{

  DTYPE deltaf=(maxFreq-minFreq)/numFreq;

  //runs supersmoother for folded lightcurves on a frequency grid
  //compute minimum time  
  DTYPE minTime=time[0];
  for (int i=0; i<n; i++)
  {
    if (time[i]<minTime)
    {
      minTime=time[i];
    }
  }

  DTYPE * tt = (DTYPE *)malloc(sizeof(DTYPE)*n);
  for (int i=0; i<n; i++)
  {
   tt[i]=time[i]-minTime;
  }  

  DTYPE * weights = (DTYPE *)malloc(sizeof(DTYPE)*n);
  for (int i=0; i<n; i++)
  {
    weights[i]=1.0/(error[i]*error[i]);
  }

  DTYPE w0=0.0;
  for (int i=0; i<n; i++)
  {
    w0+=weights[i];
  }  

  w0=w0/(n*1.0);

  DTYPE * y = (DTYPE *)malloc(sizeof(DTYPE)*n);
  std::copy(data, data+n, y);

  
  DTYPE tmp=0;
  for (int i=0; i<n; i++)
  {
    tmp+=(data[i]*weights[i]);
  }

  tmp=tmp/(n*1.0);
  DTYPE y0=tmp/w0;

  for (int i=0; i<n; i++)
  {
    y[i]=y[i]-y0;
  }

  //
  tmp=0;
  for (int i=0; i<n; i++)
  {
    tmp+=(y[i]*y[i])*weights[i];
  }

  DTYPE chi0=tmp/(n*1.0);

  DTYPE * chi2=(DTYPE *)malloc(sizeof(DTYPE)*numFreq);
  
  
  
  
  //Arrays that need to be allocated for each thread
  DTYPE * sc=(DTYPE *)malloc(sizeof(DTYPE)*n*8*NTHREADSCPU);

  DTYPE * smo=(DTYPE *)malloc(sizeof(DTYPE)*n*NTHREADSCPU); 
  DTYPE * t1=(DTYPE *)malloc(sizeof(DTYPE)*n*NTHREADSCPU); 
  int * argkeys=(int *)malloc(sizeof(int)*n*NTHREADSCPU); 
  DTYPE * t1_sortby_argkeys=(DTYPE *)malloc(sizeof(DTYPE)*n*NTHREADSCPU); 
  DTYPE * data_sortby_argkeys=(DTYPE *)malloc(sizeof(DTYPE)*n*NTHREADSCPU); 
  DTYPE * weights_sortby_argkeys=(DTYPE *)malloc(sizeof(DTYPE)*n*NTHREADSCPU); 

  
  double tstart=omp_get_wtime();

  #pragma omp parallel for num_threads(NTHREADSCPU)
  for (int i=0; i<numFreq; i++)
  {
    int tid=omp_get_thread_num();

    //Offsets into arrays for each thread
    unsigned int offset_n=tid*n;
    unsigned int offset_sc=tid*n*8;
    
    
    DTYPE p=1.0/(minFreq+(deltaf*i));
    for (int j=0; j<n; j++)
    {
      t1[offset_n+j]=fmod(tt[j],p)/p;
    }

    //Do argsort on t1
    // sortKeyValuePairsIntDouble(argkeys+offset_n, t1+offset_n, n);
    sortKeyValuePairsIntFloatDouble(argkeys+offset_n, t1+offset_n, n);
    
    //Map t1, data, and weights to the order given by argsorting t1
    mapArr(t1+offset_n, t1_sortby_argkeys+offset_n, argkeys+offset_n, n);
    mapArr(data, data_sortby_argkeys+offset_n, argkeys+offset_n, n);
    mapArr(weights, weights_sortby_argkeys+offset_n, argkeys+offset_n, n);

    chi2[i]=supsmu_chi2(n, t1_sortby_argkeys+offset_n, data_sortby_argkeys+offset_n, weights_sortby_argkeys+offset_n, smo+offset_n,  sc+offset_sc, alpha);
  }

  double tend=omp_get_wtime();
  printf("\nTime main loop: %f", tend - tstart);

  

  for (int i=0; i<numFreq; i++)
  {
    pgram[i]=(0.5*(chi0-chi2[i])*n)/chi0;
  }

}

//overloaded function for float/doubles
void mapArr(double * inArr, double * outArr, int * keys, int n)
{
  for (int i=0; i<n; i++)
  {
    outArr[i]=inArr[keys[i]];
  }
}

//overloaded function for float/doubles
void mapArr(float * inArr, float * outArr, int * keys, int n)
{
  for (int i=0; i<n; i++)
  {
    outArr[i]=inArr[keys[i]];
  }
}


//overloaded function for float/doubles
void sortKeyValuePairsIntFloatDouble(int * keys, double * values, int n)
{
  std::vector<double>val_vect(values, values+n);
  std::vector<int>keys_vect = sort_indexes(val_vect);
  std::copy(keys_vect.begin(), keys_vect.end(), keys);

}

//overloaded function for float/doubles
void sortKeyValuePairsIntFloatDouble(int * keys, float * values, int n)
{
  std::vector<float>val_vect(values, values+n);
  std::vector<int>keys_vect = sort_indexes(val_vect);
  std::copy(keys_vect.begin(), keys_vect.end(), keys);
}







DTYPE supsmu_chi2(int n, DTYPE * time, DTYPE * data, DTYPE * weights , DTYPE * smo, DTYPE * sc, DTYPE alpha)
{

    //NAT: is iper==1? [yes- means periodic]
    int iper=1;
    //NAT: is span==0.0? [yes- lets supersmoother work its magic (otherwise uses input span)]
    DTYPE span=0.0;

    supsmu(n, time, data, weights, iper, span, alpha, smo, sc);

    DTYPE tmptotal=0;
    for (int i=0; i<n; i++){
      tmptotal+=((data[i]-smo[i])*(data[i]-smo[i]))*weights[i];
    }
    return tmptotal/(n*1.0);

}


DTYPE supsmu_singlepass_chi2(int n, DTYPE * time, DTYPE * data, DTYPE * weights , DTYPE * smo, DTYPE alpha)
{

    //NAT: is iper==1? [yes- means periodic]
    int iper=1;
    //NAT: is span==0.0? [yes- lets supersmoother work its magic (otherwise uses input span)]
    DTYPE span=0.0;

    

    supsmusinglepass(n, time, data, weights, iper, span, alpha, smo);

    DTYPE tmptotal=0;
    for (int i=0; i<n; i++){
      tmptotal+=((data[i]-smo[i])*(data[i]-smo[i]))*weights[i];
    }
    return tmptotal/(n*1.0);

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

int supsmu (int n, DTYPE * x, DTYPE * y, DTYPE * w, int iper, DTYPE span, DTYPE alpha, DTYPE * smo, DTYPE * sc) {
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
        return 0;
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
        smooth (n,x,y,w,span,jper,vsmlsq,smo,sc);      // fixed span
        return 0;
      }

      // if we made it here, the span will be estimated and variable

      for (i=0;i<3;i++) {
        smooth (n,x,y,w,spans[i],jper,vsmlsq,sc+2*i*n,sc+6*n);
        smooth (n,x,sc+6*n,w,spans[1],-jper,vsmlsq,sc+(2*i+1)*n,sc+7*n);
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

      smooth (n,x,sc+6*n,w,spans[1],-jper,vsmlsq,sc+n,sc+7*n);

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
      smooth (n,x,sc+3*n,w,spans[0],-jper,vsmlsq,smo,sc+7*n);

      return 0;
}


int smooth (int n, DTYPE * x, DTYPE * y, DTYPE * w, DTYPE span, int iper, DTYPE vsmlsq, DTYPE * smo, DTYPE * acvr) {

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

      //Nat: can be removed
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



int supsmusinglepass(int n, DTYPE * x, DTYPE * y, DTYPE * w, int iper, DTYPE span, DTYPE alpha, DTYPE * smo) 
{
      
  int ibw[3];
  DTYPE vsmlsq,scale;
  DTYPE spans[] = {0.05,0.2,0.5};

  int i=n/4-1;
  int j=3*(i+1)-1;
  scale=x[j]-x[i];
  vsmlsq=1.e-6*scale*scale;

  for (i=0;i<3;i++) {
    ibw[i] = (int)( 0.5*spans[i]*n+0.5 );
    if (ibw[i]<2) ibw[i]=2;
  }

  if (alpha<0) alpha=0;
  if (alpha>10) alpha=10;

  smoothsinglepass(n, ibw, x, y, w, vsmlsq, alpha, smo);

  return 0;
}

void smoothsinglepass(int n, int *ibw, DTYPE *x, DTYPE *y, DTYPE *w, DTYPE vsmlsq, int alpha, DTYPE *smo) 
{
    int i,j,in,out;
    DTYPE wt,xto,xti,yto,yti,ibwb,smo0[3],a,f,chi2,chi2m;
    DTYPE fbo,fbw[3],xm[3],ym[3],tmp,var[3]={0,0,0},vary=0.,cvar[3]={0,0,0};

    for (i=0;i<3;i++) {
      j=n-ibw[i]-1;
      xm[i]=x[j]-1.0;
      ym[i]=y[j];
      fbw[i]=w[j];
      for (j=n-ibw[i];j<n;j++) {
        xti=x[j]-1.0;
        yti=y[j];
        wt=w[j];
        fbo=fbw[i];
        fbw[i]+=wt;
        xm[i]=(fbo*xm[i]+wt*xti)/fbw[i];
        ym[i]=(fbo*ym[i]+wt*yti)/fbw[i];
        tmp=fbw[i]*wt*(xti-xm[i])/fbo;
        var[i]+=tmp*(xti-xm[i]);
        cvar[i]+=tmp*(yti-ym[i]);
        if (i==0) vary+=fbw[0]*wt*(yti-ym[0])*(yti-ym[0])/fbo;
      }
      for (j=0;j<ibw[i];j++) {
        xti=x[j];
        yti=y[j];
        wt=w[j];
        fbo=fbw[i];
        fbw[i]+=wt;
        xm[i]=(fbo*xm[i]+wt*xti)/fbw[i];
        ym[i]=(fbo*ym[i]+wt*yti)/fbw[i];
        tmp=fbw[i]*wt*(xti-xm[i])/fbo;
        var[i]+=tmp*(xti-xm[i]);
        cvar[i]+=tmp*(yti-ym[i]);
        if (i==0) vary+=fbw[0]*wt*(yti-ym[0])*(yti-ym[0])/fbo;
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
        fbo=fbw[i];
        fbw[i]-=wt;
        tmp=fbo*wt*(xto-xm[i])/fbw[i];
        var[i]-=tmp*(xto-xm[i]);
        cvar[i]-=tmp*(yto-ym[i]);
        if (i==0) vary-=fbo*wt*(yto-ym[0])*(yto-ym[0])/fbw[0];
        xm[i]=(fbo*xm[i]-wt*xto)/fbw[i];
        ym[i]=(fbo*ym[i]-wt*yto)/fbw[i];

        wt=w[in];
        fbo=fbw[i];
        fbw[i]+=wt;
        xm[i]=(fbo*xm[i]+wt*xti)/fbw[i];
        ym[i]=(fbo*ym[i]+wt*yti)/fbw[i];
        tmp=fbw[i]*wt*(xti-xm[i])/fbo;
        var[i]+=tmp*(xti-xm[i]);
        cvar[i]+=tmp*(yti-ym[i]);
        if (i==0) vary+=fbw[0]*wt*(yti-ym[0])*(yti-ym[0])/fbo;

      }

      chi2m=1.e20; ibwb=ibw[2];
      for (i=0;i<3;i++) {
        a=0.0;
        if (var[i]>vsmlsq) a=cvar[i]/var[i];
        smo0[i]=a*(x[j]-xm[i])+ym[i];
        chi2 = vary-2*a*cvar[0]+a*a*var[0];
        if (i>0) {
          tmp = ym[i]-ym[0]-a*(xm[i]-xm[0]);
          chi2 += tmp*tmp*fbw[0];
        }
        tmp=1.0/fbw[i];
        if (var[i]>vsmlsq) tmp+=(x[j]-xm[i])*(x[j]-xm[i])/var[i];
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
        smo[j]=(1.0-f)*smo0[1]+f*smo0[0];
      } else {
        f/=ibw[2]-ibw[1];
        smo[j]=(1.0-f)*smo0[1]+f*smo0[2];
      }
    }
}


void supsmu_periodogram_innerloopcpu(int iteration, int n, DTYPE freqToTest, DTYPE * time, DTYPE * data, DTYPE * error, DTYPE alpha, DTYPE * pgram,
  DTYPE * tt, DTYPE * weights, DTYPE * chi2, DTYPE * sc, DTYPE * smo, DTYPE * t1, int * argkeys, DTYPE * t1_sortby_argkeys,
  DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys)
{

    int tid=omp_get_thread_num();

    //Offsets into arrays for each thread
    unsigned int offset_n=tid*n;
    unsigned int offset_sc=tid*n*8;
    
    DTYPE p=1.0/freqToTest;
    for (int j=0; j<n; j++)
    {
      t1[offset_n+j]=fmod(tt[j],p)/p;
    }

    sortKeyValuePairsIntFloatDouble(argkeys+offset_n, t1+offset_n, n);

    //Map t1, data, and weights to the order given by argsorting t1
    mapArr(t1+offset_n, t1_sortby_argkeys+offset_n, argkeys+offset_n, n);
    mapArr(data, data_sortby_argkeys+offset_n, argkeys+offset_n, n);
    mapArr(weights, weights_sortby_argkeys+offset_n, argkeys+offset_n, n);

    chi2[iteration]=supsmu_chi2(n, t1_sortby_argkeys+offset_n, data_sortby_argkeys+offset_n, weights_sortby_argkeys+offset_n, smo+offset_n,  sc+offset_sc, alpha);
  
}

void supsmu_singlepass_periodogram_innerloopcpu(int iteration, int n, DTYPE freqToTest, DTYPE * time, DTYPE * data, DTYPE * error, DTYPE alpha, DTYPE * pgram,
  DTYPE * tt, DTYPE * weights, DTYPE * chi2, DTYPE * smo, DTYPE * t1, int * argkeys, DTYPE * t1_sortby_argkeys,
  DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys)
{

    int tid=omp_get_thread_num();

    //Offsets into arrays for each thread
    unsigned int offset_n=tid*n;
    
    DTYPE p=1.0/freqToTest;
    for (int j=0; j<n; j++)
    {
      t1[offset_n+j]=fmod(tt[j],p)/p;
    }

    

    sortKeyValuePairsIntFloatDouble(argkeys+offset_n, t1+offset_n, n);

    //Map t1, data, and weights to the order given by argsorting t1
    mapArr(t1+offset_n, t1_sortby_argkeys+offset_n, argkeys+offset_n, n);
    mapArr(data, data_sortby_argkeys+offset_n, argkeys+offset_n, n);
    mapArr(weights, weights_sortby_argkeys+offset_n, argkeys+offset_n, n);

    chi2[iteration]=supsmu_singlepass_chi2(n, t1_sortby_argkeys+offset_n, data_sortby_argkeys+offset_n, weights_sortby_argkeys+offset_n, smo+offset_n, alpha);
  
  }





//single object processing
  //MODEFLAG- 0 default supersmoother with multiple passes
  //MODEFLAG- 1 Nat's singlepass supersmoother
void supersmoothercpusingleobject(bool MODEFLAG, DTYPE * time, DTYPE * data, DTYPE * error, const unsigned int sizeData, 
  const unsigned int numFreqs, const DTYPE minFreq, const DTYPE maxFreq, const DTYPE freqStep, DTYPE alpha, 
  DTYPE * pgram, DTYPE * foundPeriod, DTYPE * chi2, DTYPE * sc, DTYPE * smo, DTYPE * t1, int * argkeys, 
  DTYPE * t1_sortby_argkeys, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys, DTYPE * weights, DTYPE * tt)
{


      DTYPE chi0=0;
      compute_chi0_tt_weights(sizeData, time, data, error, &chi0, tt, weights);

      
      //Default supersmoother
      if(MODEFLAG==0)
      {
      
        //Single object -- parallelize over frequencies
        #pragma omp parallel for num_threads(NTHREADSCPU) schedule(static)
        for (unsigned int i=0; i<numFreqs; i++)
        {
          DTYPE freqToTest=minFreq+(freqStep*i);
          supsmu_periodogram_innerloopcpu(i, sizeData, freqToTest, time, data, error, alpha, pgram, tt, weights, chi2, sc, smo, t1, argkeys, t1_sortby_argkeys, data_sortby_argkeys, weights_sortby_argkeys);
        }
      }
      //Nat's single pass supersmoother
      else if(MODEFLAG==1)
      {
        printf("\nRunning single pass");
        //Single object -- parallelize over frequencies
        #pragma omp parallel for num_threads(NTHREADSCPU) schedule(static)
        for (unsigned int i=0; i<numFreqs; i++)
        {
          DTYPE freqToTest=minFreq+(freqStep*i);
          supsmu_singlepass_periodogram_innerloopcpu(i, sizeData, freqToTest, time, data, error, alpha, pgram, tt, weights, chi2, smo, t1, argkeys, t1_sortby_argkeys, data_sortby_argkeys, weights_sortby_argkeys);
        }
      }


      

      for (unsigned int i=0; i<numFreqs; i++)
      {
        pgram[i]=(0.5*(chi0-chi2[i])*sizeData)/chi0;
      }

      computePeriodSuperSmoother(pgram, numFreqs, minFreq, maxFreq, foundPeriod);

}

void computePeriodSuperSmoother(DTYPE * pgram, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE maxFreq, DTYPE * foundPeriod)
{
  DTYPE deltaf=(maxFreq-minFreq)/numFreqs;

  int maxPowerIdx=0;
  int maxPower=pgram[0];

  
  for (unsigned int i=0; i<numFreqs; i++)
  { 
    if (pgram[i]>maxPower)
    {
      maxPower=pgram[i];
      maxPowerIdx=i;
    }
  }

  printf("\nFreq: %f, maxpowerIdx: %d",(minFreq+(maxPowerIdx*deltaf)), maxPowerIdx);

  *foundPeriod=1.0/(minFreq+(maxPowerIdx*deltaf));
}



//MODEFLAG-0 original supsmu (multi-pass)
//MODEFLAG-1 Nat's Single-pass supsmu
void supersmootherCPUBatch(bool MODEFLAG, unsigned int * objectId, DTYPE * time,  DTYPE * data, DTYPE * error, unsigned int sizeData, const DTYPE minFreq, const DTYPE maxFreq, 
  const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE ** pgram, DTYPE * foundPeriod, DTYPE alpha, 
  DTYPE * chi2, DTYPE * sc, DTYPE * smo, DTYPE * t1, int * argkeys, 
  DTYPE * t1_sortby_argkeys, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys, DTYPE * weights, DTYPE * tt)
{
      



  //compute the object ranges in the arrays and store in struct
  //This is given by the objectId
  struct lookupObj * objectLookup=NULL;
  unsigned int numUniqueObjects;
  computeObjectRanges(objectId, &sizeData, &objectLookup, &numUniqueObjects);
  *pgram=(DTYPE *)malloc(sizeof(DTYPE)*(numFreqs)*numUniqueObjects);
  foundPeriod=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);

  const DTYPE freqStep=(maxFreq-minFreq)/(numFreqs*1.0);  

  
  //number of objects skipped because they didn't have enough observations
  unsigned int countSkippedObjectsThresh=0;


  //for each object, call the parallel cpu algorithm
  // for (unsigned int i=0; i<numUniqueObjects; i++)
  for (unsigned int i=0; i<numUniqueObjects; i++)
  {

    


    unsigned int idxMin=objectLookup[i].idxMin;
    unsigned int idxMax=objectLookup[i].idxMax;
    unsigned int sizeDataForObject=idxMax-idxMin+1;
    uint64_t pgramOffset=(uint64_t)i*(uint64_t)numFreqs;



    
    
    //make sure the object has at least OBJTHRESH observations
    if (sizeDataForObject>=OBSTHRESH)
    {
    supersmoothercpusingleobject(MODEFLAG, &time[idxMin], &data[idxMin], &error[idxMin], sizeDataForObject, 
    numFreqs, minFreq, maxFreq, freqStep, alpha, *pgram+pgramOffset, foundPeriod+i, chi2, sc, 
    smo, t1, argkeys, t1_sortby_argkeys, data_sortby_argkeys, weights_sortby_argkeys, weights, tt);     
    }
    //too few data points to compute the periods
    else
    {
      countSkippedObjectsThresh++;
      foundPeriod[i]=0.0;
    }

    printf("\nObject: %d, Period: %f",objectLookup[i].objId, foundPeriod[i]);
  }


    printf("\nNumber of objects skipped because they didn't have %d observations: %u", OBSTHRESH, countSkippedObjectsThresh);

  //Validation
  for (unsigned int i=0; i<numUniqueObjects; i++)
    {
      (*sumPeriods)+=foundPeriod[i];
    }

  

}



//Need to do back to back sorts to sort the t1 by argkeys for each frequency
//Need 3 arrays: 
//first sort the keys (argkeys) by the values (t1)
//then sort the argkeys/t1 by the freqarr
void backToBackSort(int * dev_argkeys, int * dev_freqarr, DTYPE * dev_t1, int sizeData, int numFreq, cudaStream_t stream)
{
  thrust::device_ptr<int> dev_argkeys_ptr(dev_argkeys);
  thrust::device_ptr<DTYPE> dev_t1_ptr(dev_t1);
  thrust::device_ptr<int> dev_freqarr_ptr(dev_freqarr);
  try{
  thrust::stable_sort_by_key(thrust::cuda::par.on(stream), dev_t1_ptr, dev_t1_ptr + (sizeData*numFreq), 
  thrust::make_zip_iterator(thrust::make_tuple(dev_argkeys_ptr, dev_freqarr_ptr)));
          
  thrust::stable_sort_by_key(thrust::cuda::par.on(stream), dev_freqarr_ptr, dev_freqarr_ptr + (sizeData*numFreq), 
  thrust::make_zip_iterator(thrust::make_tuple(dev_argkeys_ptr, dev_t1_ptr)));  
  }
  catch(thrust::system_error e)
  {
  std::cerr << "Error inside sort: " << e.what() << std::endl;
  exit(-1);
  } 
}





//GPU supersmoother original with multiple passes
void supsmu_original_single_object(unsigned int * objectId, unsigned int sizeData, const DTYPE minFreq, const DTYPE maxFreq, int numFreq, DTYPE * time, DTYPE * data, DTYPE * error, DTYPE alpha, DTYPE * pgram, DTYPE * foundPeriod, double underestGPUcapacityGiB)
{
  double tstartcpu=omp_get_wtime();
  
  int iper=1;
  DTYPE span=0.0;
  
  DTYPE * weights = (DTYPE *)malloc(sizeof(DTYPE)*sizeData);
  DTYPE * tt = (DTYPE *)malloc(sizeof(DTYPE)*sizeData);
  const DTYPE deltaf=(maxFreq-minFreq)/(numFreq*1.0);
  DTYPE chi0=0;

  compute_chi0_tt_weights(sizeData, time, data, error, &chi0, tt, weights);


  ////////////////////////
  //for batching the frequencies 
  //first 0-refers to using original supsmu mode
  //second 0-refers to using NUMGPUs when computing the number of batches
  unsigned int numBatches=computeNumBatches(0, sizeData, numFreq, underestGPUcapacityGiB, 0);      
  //upper limit on the number of frequencies in a batch
  int numFreqPerBatch=ceil(numFreq*1.0/numBatches*1.0);
  printf("\nNumber of batches: %d, Number of frequencies per batch: %d", numBatches, numFreqPerBatch);fflush(stdout);




  double tstartcreatestream=omp_get_wtime();
  cudaStream_t batchstreams[NUMGPU];
  createStreams(batchstreams, NUMGPU, 1);
  double tendcreatestream=omp_get_wtime();
      

  //End for batching frequencies
  ////////////////////////

  
  int * dev_freqarr[NUMGPU];
  DTYPE * dev_smo[NUMGPU];
  DTYPE * dev_t1[NUMGPU];
  int * dev_argkeys[NUMGPU];
  DTYPE * dev_t1_sortby_argkeys[NUMGPU];
  DTYPE * dev_data_sortby_argkeys[NUMGPU];
  DTYPE * dev_weights_sortby_argkeys[NUMGPU];
  DTYPE * dev_tt[NUMGPU];
  DTYPE * dev_data[NUMGPU];
  DTYPE * dev_weights[NUMGPU];
  DTYPE * dev_sc[NUMGPU];
  DTYPE * dev_pgram[NUMGPU];

  #pragma omp parallel for num_threads(NUMGPU)
  for (int i=0; i<NUMGPU; i++)
  {
  int globaltid=omp_get_thread_num();
  int tid=globaltid;  
  int gpuid=globaltid;  
  int streamnum=(gpuid)+tid;  

  cudaSetDevice(i); 
  //Those that depend on the number of frequencies (not the number per batch)
  gpuErrchk(cudaMalloc((void**)&dev_pgram[i],                     sizeof(DTYPE)*numFreq));
  
  //Arrays broken up into batches based on frequency
  gpuErrchk(cudaMalloc((void**)&dev_freqarr[i],                sizeof(int)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_argkeys[i],                sizeof(int)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_sc[i],                        sizeof(DTYPE)*(sizeData*8*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_smo[i],                    sizeof(DTYPE)*(sizeData*numFreqPerBatch))); 
  gpuErrchk(cudaMalloc((void**)&dev_t1[i],                     sizeof(DTYPE)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_t1_sortby_argkeys[i],      sizeof(DTYPE)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_data_sortby_argkeys[i],    sizeof(DTYPE)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_weights_sortby_argkeys[i], sizeof(DTYPE)*(sizeData*numFreqPerBatch)));

  //allocate on the GPU
  gpuErrchk(cudaMalloc((void**)&dev_tt[i],                    sizeof(DTYPE)*sizeData));
  gpuErrchk(cudaMalloc((void**)&dev_data[i],                    sizeof(DTYPE)*sizeData));
  gpuErrchk(cudaMalloc((void**)&dev_weights[i],                    sizeof(DTYPE)*sizeData));
  
  //copy to the GPU
  gpuErrchk(cudaMemcpyAsync( dev_tt[i],                    tt,                     sizeof(DTYPE)*sizeData, cudaMemcpyHostToDevice, batchstreams[streamnum]));
  gpuErrchk(cudaMemcpyAsync( dev_data[i],                data,                     sizeof(DTYPE)*sizeData, cudaMemcpyHostToDevice, batchstreams[streamnum]));
  gpuErrchk(cudaMemcpyAsync( dev_weights[i],          weights,                     sizeof(DTYPE)*sizeData, cudaMemcpyHostToDevice, batchstreams[streamnum]));

  }


  //Loop over batches
  #pragma omp parallel for num_threads(NUMGPU)
  for (unsigned int i=0; i<numBatches; i++)
  {
  int globaltid=omp_get_thread_num();
  //thread id for a single GPU
  int tid=globaltid;  
  int gpuid=globaltid;
  uint64_t batchWriteOffset=(uint64_t)i*(uint64_t)numFreqPerBatch;
  uint64_t offsetFreqId=(uint64_t)i*(uint64_t)numFreqPerBatch;
  int numFreqInBatch=numFreqPerBatch;
  int streamOffset=sizeData*numFreqPerBatch*tid;

  int streamnum=(gpuid)+tid;


  cudaSetDevice(gpuid); 

  //last batch has fewer frequencies
  if((numBatches!=1)&&(i==(numBatches-1)))
  {
    numFreqInBatch=min(numFreqInBatch,((int)numFreq)-((i)*numFreqPerBatch));
  }

  printf("\nglobal tid: %d, tid: %d, gpuid: %d, Stream num: %d, Batch Number: %d, number of frequencies: %d",globaltid, tid, gpuid, streamnum, i, numFreqInBatch);


  
  unsigned int NUMBLOCKSDATAFREQ=ceil((sizeData*numFreqInBatch*1.0)/LARGEBLOCKSIZE*1.0);
  computePeriodModFOneThreadPerUpdate<<<NUMBLOCKSDATAFREQ,LARGEBLOCKSIZE,0,batchstreams[streamnum]>>>(sizeData, numFreqInBatch, minFreq, offsetFreqId, deltaf, &dev_t1[gpuid][streamOffset], dev_tt[gpuid]);


  //Initialize the key arrays
  initializeKeyArraysOneThreadPerUpdate<<<NUMBLOCKSDATAFREQ,LARGEBLOCKSIZE,0,batchstreams[streamnum]>>>(sizeData, numFreqInBatch, &dev_argkeys[gpuid][streamOffset], &dev_freqarr[gpuid][streamOffset]);
      

  //Need to do back to back sorts to sort the t1 by argkeys for each frequency
  //Need 3 arrays: 
  //first sort the keys (argkeys) by the values (t1)
  //then sort the argkeys/t1 by the freqarr
  backToBackSort(&dev_argkeys[gpuid][streamOffset], &dev_freqarr[gpuid][streamOffset], &dev_t1[gpuid][streamOffset], sizeData, numFreqInBatch, batchstreams[streamnum]);
  

  //Map the keys based on argkeys
  //Separate map and transform  
  mapUsingArgKeysOneThreadPerUpdate<<<NUMBLOCKSDATAFREQ,LARGEBLOCKSIZE,0,batchstreams[streamnum]>>>(sizeData, numFreqInBatch, &dev_argkeys[gpuid][streamOffset], &dev_data[gpuid][0], &dev_weights[gpuid][0], &dev_t1[gpuid][streamOffset], &dev_t1_sortby_argkeys[gpuid][streamOffset], &dev_data_sortby_argkeys[gpuid][streamOffset], &dev_weights_sortby_argkeys[gpuid][streamOffset]);      
  

    
  ///////////////////////////////
  // Main kernels
  ///////////////////////////////  

  
  //Cascade the execution so that it is robust to running out of shared memory
  //Try executing SM kernel 1 thread per freq
  //then global memory kernel (which is guaranteed to execute)
  
  printf("\n[CASCADE] Cascade mode, launching SM one thread per frequency");
  //First, attempt 1 thread per frequency with SM
  const unsigned int numBlocks=ceil((numFreqInBatch*1.0)/(SMALLBLOCKSIZE*1.0));
  const unsigned int SMSIZEDATA=sizeof(DTYPE)*3*sizeData*SMALLBLOCKSIZE;
  supsmukernelSMOneThreadPerFreq<<<numBlocks,SMALLBLOCKSIZE,SMSIZEDATA,batchstreams[streamnum]>>>(numFreqInBatch, sizeData, iper, span, alpha, 
    &dev_smo[gpuid][streamOffset], &dev_sc[gpuid][streamOffset], &dev_t1_sortby_argkeys[gpuid][streamOffset], &dev_data_sortby_argkeys[gpuid][streamOffset], &dev_weights_sortby_argkeys[gpuid][streamOffset]);  

  

    //execute global memory kernel
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess)
    {
      printf("\n[CASCADE] Launching global memory kernel");
      const unsigned int numBlocks=ceil((numFreqInBatch*1.0)/(SMALLBLOCKSIZE*1.0));
      supsmukernel<<<numBlocks,SMALLBLOCKSIZE,0,batchstreams[streamnum]>>>(numFreqInBatch, sizeData, iper, span, alpha, &dev_smo[gpuid][streamOffset], &dev_sc[gpuid][streamOffset],
      &dev_t1_sortby_argkeys[gpuid][streamOffset], &dev_data_sortby_argkeys[gpuid][streamOffset], &dev_weights_sortby_argkeys[gpuid][streamOffset]); 
    }


  //Some number of threads per frequency
  unsigned int numThreadPerFreq2=8; //must divide evenly into the block size
  unsigned int NUMBLOCKS10=ceil((numFreqInBatch*numThreadPerFreq2*1.0)/(LARGEBLOCKSIZE*1.0));
  const unsigned int SMSIZE2=sizeof(DTYPE)*(LARGEBLOCKSIZE/numThreadPerFreq2);
  computePgramReduction<<<NUMBLOCKS10, LARGEBLOCKSIZE, SMSIZE2, batchstreams[streamnum]>>>(batchWriteOffset, numThreadPerFreq2, chi0, sizeData, numFreqInBatch, &dev_smo[gpuid][streamOffset], &dev_data_sortby_argkeys[gpuid][streamOffset], &dev_weights_sortby_argkeys[gpuid][streamOffset], &dev_pgram[gpuid][0]);

  //Copy pgram back to host
  gpuErrchk(cudaMemcpyAsync(pgram+batchWriteOffset, &dev_pgram[gpuid][batchWriteOffset], sizeof(DTYPE)*numFreqInBatch, cudaMemcpyDeviceToHost, batchstreams[streamnum]));

  
  } //end loop over batches

  
  
  
  
  computePeriodSuperSmoother(pgram, numFreq, minFreq, maxFreq, foundPeriod);  
  printf("\nFound period: %f", *foundPeriod);



  //free device data
  for (int i=0; i<NUMGPU; i++)
  {
  cudaFree(dev_sc[i]);
  cudaFree(dev_pgram[i]);                            
  cudaFree(dev_freqarr[i]);                
  cudaFree(dev_argkeys[i]);                
  cudaFree(dev_smo[i]);                    
  cudaFree(dev_t1[i]);                     
  cudaFree(dev_t1_sortby_argkeys[i]);      
  cudaFree(dev_data_sortby_argkeys[i]);    
  cudaFree(dev_weights_sortby_argkeys[i]); 
  cudaFree(dev_tt[i]);                    
  cudaFree(dev_data[i]);                    
  cudaFree(dev_weights[i]);                    
  }
  
  
  //free host data
  free(weights);
  free(tt);

}


//Only use a single GPU
void supsmu_original_single_gpu(unsigned int * objectId, unsigned int sizeData, const DTYPE minFreq, const DTYPE maxFreq, int numFreq, DTYPE * time, DTYPE * data, DTYPE * error, DTYPE alpha, DTYPE * pgram, DTYPE * foundPeriod, double underestGPUcapacityGiB, int gpuid)
{

  printf("\nObject Id: %u", *objectId);fflush(stdout);
  double tstartcpu=omp_get_wtime();
  
  int iper=1;
  DTYPE span=0.0;
  
  DTYPE * weights = (DTYPE *)malloc(sizeof(DTYPE)*sizeData);
  DTYPE * tt = (DTYPE *)malloc(sizeof(DTYPE)*sizeData);
  const DTYPE deltaf=(maxFreq-minFreq)/(numFreq*1.0);
  DTYPE chi0=0;

  compute_chi0_tt_weights(sizeData, time, data, error, &chi0, tt, weights);


  ////////////////////////
  //for batching the frequencies 
  //0-refers to using original supsmu mode
  //1- a flag referring to using a single GPU
  unsigned int numBatches=computeNumBatches(0, sizeData, numFreq, underestGPUcapacityGiB, 1);      
  //upper limit on the number of frequencies in a batch
  unsigned int numFreqPerBatch=ceil(numFreq*1.0/numBatches*1.0);
  printf("\nObject Id: %u, Number of batches: %u, Number of frequencies per batch: %u", *objectId, numBatches, numFreqPerBatch);fflush(stdout);

  

  double tstartcreatestream=omp_get_wtime();
  cudaStream_t batchstreams[1];
  createStreamsOneGPU(batchstreams, 1, gpuid);
  double tendcreatestream=omp_get_wtime();
      

  //End for batching frequencies
  ////////////////////////

  
  int * dev_freqarr[1];
  DTYPE * dev_smo[1];
  DTYPE * dev_t1[1];
  int * dev_argkeys[1];
  DTYPE * dev_t1_sortby_argkeys[1];
  DTYPE * dev_data_sortby_argkeys[1];
  DTYPE * dev_weights_sortby_argkeys[1];
  DTYPE * dev_tt[1];
  DTYPE * dev_data[1];
  DTYPE * dev_weights[1];
  DTYPE * dev_sc[1];
  DTYPE * dev_pgram[1];

  //loop used to be here

  cudaSetDevice(gpuid); 
  //Those that depend on the number of frequencies (not the number per batch)
  gpuErrchk(cudaMalloc((void**)&dev_pgram[0],                     sizeof(DTYPE)*numFreq));
  
  //Arrays broken up into batches based on frequency
  gpuErrchk(cudaMalloc((void**)&dev_freqarr[0],                sizeof(int)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_argkeys[0],                sizeof(int)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_sc[0],                        sizeof(DTYPE)*(sizeData*8*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_smo[0],                    sizeof(DTYPE)*(sizeData*numFreqPerBatch))); 
  gpuErrchk(cudaMalloc((void**)&dev_t1[0],                     sizeof(DTYPE)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_t1_sortby_argkeys[0],      sizeof(DTYPE)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_data_sortby_argkeys[0],    sizeof(DTYPE)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_weights_sortby_argkeys[0], sizeof(DTYPE)*(sizeData*numFreqPerBatch)));

  //allocate on the GPU
  gpuErrchk(cudaMalloc((void**)&dev_tt[0],                    sizeof(DTYPE)*sizeData));
  gpuErrchk(cudaMalloc((void**)&dev_data[0],                    sizeof(DTYPE)*sizeData));
  gpuErrchk(cudaMalloc((void**)&dev_weights[0],                    sizeof(DTYPE)*sizeData));
  

  //copy to the GPU
  gpuErrchk(cudaMemcpyAsync( dev_tt[0],                    tt,                     sizeof(DTYPE)*sizeData, cudaMemcpyHostToDevice, batchstreams[0]));
  gpuErrchk(cudaMemcpyAsync( dev_data[0],                data,                     sizeof(DTYPE)*sizeData, cudaMemcpyHostToDevice, batchstreams[0]));
  gpuErrchk(cudaMemcpyAsync( dev_weights[0],          weights,                     sizeof(DTYPE)*sizeData, cudaMemcpyHostToDevice, batchstreams[0]));


      

  //Loop over batches
  #pragma omp parallel for num_threads(1)
  for (unsigned int i=0; i<numBatches; i++)
  {

  cudaSetDevice(gpuid); 

  int globaltid=omp_get_thread_num();
  //thread id for a single GPU
  int tid=globaltid;  
  uint64_t batchWriteOffset=(uint64_t)i*(uint64_t)numFreqPerBatch;
  uint64_t offsetFreqId=(uint64_t)i*(uint64_t)numFreqPerBatch;
  unsigned int numFreqInBatch=numFreqPerBatch;
  unsigned int streamOffset=sizeData*numFreqPerBatch*tid;
  int streamnum=tid;

  //last batch has fewer frequencies
  if((numBatches!=1)&&(i==(numBatches-1)))
  {
    numFreqInBatch=min(numFreqInBatch,((int)numFreq)-((i)*numFreqPerBatch));
  }

  printf("\nglobal tid: %d, tid: %d, gpuid: %d, Stream num: %u, Batch Number: %u, number of frequencies: %u",globaltid, tid, gpuid, streamnum, i, numFreqInBatch);


  
  unsigned int NUMBLOCKSDATAFREQ=ceil((sizeData*numFreqInBatch*1.0)/LARGEBLOCKSIZE*1.0);
  computePeriodModFOneThreadPerUpdate<<<NUMBLOCKSDATAFREQ,LARGEBLOCKSIZE,0,batchstreams[streamnum]>>>(sizeData, numFreqInBatch, minFreq, offsetFreqId, deltaf, &dev_t1[0][streamOffset], dev_tt[0]);


  //Initialize the key arrays
  initializeKeyArraysOneThreadPerUpdate<<<NUMBLOCKSDATAFREQ,LARGEBLOCKSIZE,0,batchstreams[streamnum]>>>(sizeData, numFreqInBatch, &dev_argkeys[0][streamOffset], &dev_freqarr[0][streamOffset]);
      

  //Need to do back to back sorts to sort the t1 by argkeys for each frequency
  //Need 3 arrays: 
  //first sort the keys (argkeys) by the values (t1)
  //then sort the argkeys/t1 by the freqarr
  backToBackSort(&dev_argkeys[0][streamOffset], &dev_freqarr[0][streamOffset], &dev_t1[0][streamOffset], sizeData, numFreqInBatch, batchstreams[streamnum]);
  


  //Map the keys based on argkeys
  //Separate map and transform  
  mapUsingArgKeysOneThreadPerUpdate<<<NUMBLOCKSDATAFREQ,LARGEBLOCKSIZE,0,batchstreams[streamnum]>>>(sizeData, numFreqInBatch, &dev_argkeys[0][streamOffset], &dev_data[0][0], &dev_weights[0][0], &dev_t1[0][streamOffset], &dev_t1_sortby_argkeys[0][streamOffset], &dev_data_sortby_argkeys[0][streamOffset], &dev_weights_sortby_argkeys[0][streamOffset]);      
  
  
    
  ///////////////////////////////
  // Main kernels
  ///////////////////////////////  


  //Cascade the execution so that it is robust to running out of shared memory
  //Try executing SM kernel 1 thread per freq,
  //then global memory kernel (which is guaranteed to execute)
  
  printf("\nCascade mode");
  //First, attempt 1 thread per frequency with SM
  const unsigned int numBlocks=ceil((numFreqInBatch*1.0)/(SMALLBLOCKSIZE*1.0));
  const unsigned int SMSIZEDATA=sizeof(DTYPE)*3*sizeData*SMALLBLOCKSIZE;
  supsmukernelSMOneThreadPerFreq<<<numBlocks,SMALLBLOCKSIZE,SMSIZEDATA,batchstreams[streamnum]>>>(numFreqInBatch, sizeData, iper, span, alpha, 
    &dev_smo[0][streamOffset], &dev_sc[0][streamOffset], &dev_t1_sortby_argkeys[0][streamOffset], &dev_data_sortby_argkeys[0][streamOffset], &dev_weights_sortby_argkeys[0][streamOffset]);  

  
  
    //execute global memory kernel
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess)
    {
      // std::cout << "\nCUDA error: " << cudaGetErrorString(err2);

      printf("\n Launching global memory kernel");
      const unsigned int numBlocks=ceil((numFreqInBatch*1.0)/(SMALLBLOCKSIZE*1.0));
      supsmukernel<<<numBlocks,SMALLBLOCKSIZE,0,batchstreams[streamnum]>>>(numFreqInBatch, sizeData, iper, span, alpha, &dev_smo[0][streamOffset], &dev_sc[0][streamOffset],
      &dev_t1_sortby_argkeys[0][streamOffset], &dev_data_sortby_argkeys[0][streamOffset], &dev_weights_sortby_argkeys[0][streamOffset]); 
    }


  //Some number of threads per frequency
  unsigned int numThreadPerFreq2=8; //must divide evenly into the block size
  unsigned int NUMBLOCKS10=ceil((numFreqInBatch*numThreadPerFreq2*1.0)/(LARGEBLOCKSIZE*1.0));
  const unsigned int SMSIZE2=sizeof(DTYPE)*(LARGEBLOCKSIZE/numThreadPerFreq2);
  computePgramReduction<<<NUMBLOCKS10, LARGEBLOCKSIZE, SMSIZE2, batchstreams[streamnum]>>>(batchWriteOffset, numThreadPerFreq2, chi0, sizeData, numFreqInBatch, &dev_smo[0][streamOffset], &dev_data_sortby_argkeys[0][streamOffset], &dev_weights_sortby_argkeys[0][streamOffset], &dev_pgram[0][0]);

  //Copy pgram back to host
  gpuErrchk(cudaMemcpyAsync(pgram+batchWriteOffset, &dev_pgram[0][batchWriteOffset], sizeof(DTYPE)*numFreqInBatch, cudaMemcpyDeviceToHost, batchstreams[streamnum]));
  
  } //end loop over batches

 

  
  
  
  
  computePeriodSuperSmoother(pgram, numFreq, minFreq, maxFreq, foundPeriod);  
  printf("\nFound period: %f", *foundPeriod);



  // //free device data
  cudaFree(dev_sc[0]);
  cudaFree(dev_pgram[0]);                            
  cudaFree(dev_freqarr[0]);                
  cudaFree(dev_argkeys[0]);                
  cudaFree(dev_smo[0]);                    
  cudaFree(dev_t1[0]);                     
  cudaFree(dev_t1_sortby_argkeys[0]);      
  cudaFree(dev_data_sortby_argkeys[0]);    
  cudaFree(dev_weights_sortby_argkeys[0]); 
  cudaFree(dev_tt[0]);                    
  cudaFree(dev_data[0]);                    
  cudaFree(dev_weights[0]);                    


  destroyStreamsOneGPU(batchstreams, 1, gpuid);
  
  
  //free host data
  free(weights);
  free(tt);

}


void compute_chi0_tt_weights(unsigned int sizeData, DTYPE * time, DTYPE * data, DTYPE * error, DTYPE * chi0, DTYPE * tt, DTYPE * weights)
{
      DTYPE y=0;
      //compute minimum time  
      DTYPE minTime=time[0];
      for (unsigned int i=0; i<sizeData; i++)
      {
        if (time[i]<minTime)
        {
          minTime=time[i];
        }
      }

      DTYPE w0=0.0;
      DTYPE tmp=0;
      for (unsigned int i=0; i<sizeData; i++)
      {
       tt[i]=time[i]-minTime;
       weights[i]=1.0/(error[i]*error[i]);
       w0+=weights[i];
       tmp+=(data[i]*weights[i]);
      }  

      w0=w0/(sizeData*1.0);
      tmp=tmp/(sizeData*1.0);
      DTYPE y0=tmp/w0;

      tmp=0;
      for (unsigned int i=0; i<sizeData; i++)
      {
        y=data[i]-y0;
        tmp+=(y*y)*weights[i];
      }
      *chi0=tmp/(sizeData*1.0);
}

double getGPUCapacity()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  
  //Read the global memory capacity from the device.
  unsigned long int globalmembytes=0;
  gpuErrchk(cudaMemGetInfo(NULL,&globalmembytes));
  double totalcapacityGiB=globalmembytes*1.0/(1024*1024*1024.0);

  printf("\n[Device name: %s, Detecting GPU Global Memory Capacity] Size in GiB: %f", prop.name, totalcapacityGiB);
  double underestcapacityGiB=totalcapacityGiB*0.75;
  printf("\n[Underestimating GPU Global Memory Capacity] Size in GiB: %f", underestcapacityGiB);
  return underestcapacityGiB;
}


double computedeltaf(lookupObj * objectLookup,  DTYPE * time, unsigned int numUniqueObjects)
{
  //Find the maximum time span for all objects

  double maxTimeSpan=0;


  #pragma omp parallel for reduction(max: maxTimeSpan)
  for (unsigned int i=0; i<numUniqueObjects; i++)
  {
    unsigned int idxMin=objectLookup[i].idxMin;
    unsigned int idxMax=objectLookup[i].idxMax;

    double timeSpan=time[idxMax]-time[idxMin];

    if (maxTimeSpan<timeSpan)
    {
      maxTimeSpan=timeSpan;
    }
  }

  double df=0.1/maxTimeSpan;

  return df;
}




//mode-0 original supsmu
//mode-1 single pass supsmu
void supsmu_gpu_batch(const bool mode, unsigned int * objectId, unsigned int sizeData, const DTYPE minFreq, const DTYPE maxFreq, unsigned int numFreq, DTYPE * time, DTYPE * data, DTYPE * error, DTYPE alpha, DTYPE ** pgram, DTYPE * sumPeriods)
{
  //get the global memory capacity of the GPU and then underestimate it so that we don't have out of memory errors
  double underestGPUcapacityGiB=getGPUCapacity();
  //For doing batch processing
  struct lookupObj * objectLookup=NULL;
  unsigned int numUniqueObjects;
  computeObjectRanges(objectId, &sizeData, &objectLookup, &numUniqueObjects);  

  //utility function: compute deltaf

  // double deltaf=computedeltaf(objectLookup, time, numUniqueObjects);
  // printf("\nDelta f: %f", deltaf);

  //allocate memory for the pgram
  *pgram=(DTYPE *)malloc(sizeof(DTYPE)*(uint64_t)numFreq*(uint64_t)numUniqueObjects);

  // printf("\nPgram GiB: %f", (sizeof(DTYPE)*numFreq*numUniqueObjects*1.0)/(1024*1024*1024.0));

  // *pgram=(DTYPE *)calloc((unsigned int)numFreq*numUniqueObjects,sizeof(DTYPE));

  DTYPE * periods=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);
  // DTYPE * periods=(DTYPE *)calloc(numUniqueObjects,sizeof(DTYPE));

  //number of objects skipped because they didn't have enough observations
  unsigned int countSkippedObjectsThresh=0;

  //Computing SS is parallelized as follows:
  //1) If you are computing a single object, parallelize the object across multiple GPUs  
  //2) If you are computing a batch of objects, execute a single object per GPU (assuming you are using multiple GPUs)


  //1) single object-- parallelize single object on multiple GPUs
  if (numUniqueObjects==1)
  {
    unsigned int idxMin=objectLookup[0].idxMin;
    unsigned int idxMax=objectLookup[0].idxMax;
    unsigned int sizeDataForObject=idxMax-idxMin+1;
    uint64_t pgramOffset=0;
    DTYPE foundPeriod;

    if (sizeDataForObject>=OBSTHRESH)
    {
      //original
      if (mode==0)
      {
      supsmu_original_single_object(objectId, sizeDataForObject, minFreq, maxFreq, numFreq, &time[idxMin], &data[idxMin], &error[idxMin], alpha, *pgram+pgramOffset, &foundPeriod, underestGPUcapacityGiB);      
      }

      //single pass
      if (mode==1)
      {
      supsmu_singlepass_single_object(objectId, sizeDataForObject, minFreq, maxFreq, numFreq, &time[idxMin], &data[idxMin], &error[idxMin], alpha, *pgram+pgramOffset, &foundPeriod, underestGPUcapacityGiB);
      }
    
      periods[0]=foundPeriod;
    }
    else
    {
      periods[0]=0.0;
      countSkippedObjectsThresh++;
    }


    

    

    

  }
  //2) multiple objects -- parallelize one object per GPU
  //dynamic scheduling since time series are different lengths
  else 
  {

    #pragma omp parallel for schedule(dynamic) num_threads(NUMGPU) reduction(+:countSkippedObjectsThresh)
    for (unsigned int i=0; i<numUniqueObjects; i++)
    {

      unsigned int idxMin=objectLookup[i].idxMin;
      unsigned int idxMax=objectLookup[i].idxMax;
      unsigned int sizeDataForObject=idxMax-idxMin+1;
      uint64_t pgramOffset=(uint64_t)i*(uint64_t)numFreq;
      DTYPE foundPeriod;
      int tid=omp_get_thread_num();

      //only process objects with at least OBSTHRESH data points
      if(sizeDataForObject>=OBSTHRESH)
      {
        //original supsmu    
        if (mode==0)
        { 
          //could parallelize the batch of objectss by parallelizing each object individually
          // supsmu_original_single_object(objectId, sizeDataForObject, minFreq, maxFreq, numFreq, &time[idxMin], &data[idxMin], &error[idxMin], alpha, *pgram+pgramOffset, &foundPeriod, underestGPUcapacityGiB);        

          supsmu_original_single_gpu(&objectLookup[i].objId, sizeDataForObject, minFreq, maxFreq, numFreq, &time[idxMin], &data[idxMin], &error[idxMin], alpha, *pgram+pgramOffset, &foundPeriod, underestGPUcapacityGiB, tid);    
        }

        //single pass supsmu
        if (mode==1)
        {
          supsmu_singlepass_single_gpu(&objectLookup[i].objId, sizeDataForObject, minFreq, maxFreq, numFreq, &time[idxMin], &data[idxMin], &error[idxMin], alpha, *pgram+pgramOffset, &foundPeriod, underestGPUcapacityGiB, tid);
        }

        periods[i]=foundPeriod;
      }
      //too few data points to compute the periods
      else
      {
        countSkippedObjectsThresh++;
        periods[i]=0.0;
      }
    

    } //end parallel for loop

  } //end if statement around unique objects


  printf("\nNumber of objects skipped because they didn't have %d observations: %u", OBSTHRESH, countSkippedObjectsThresh);


  for (unsigned int i=0; i<numUniqueObjects; i++)
  {
    *sumPeriods+=periods[i];
  }

  ///////////////////////
  //Output

  
  outputPeriodsToFile(objectLookup, numUniqueObjects, periods);
  
  //Output pgram to file
  #if PRINTPGRAM==1
  outputPgramToFile(objectLookup, numUniqueObjects, numFreq, pgram);   
  #endif
    
  //End output
  ///////////////////////



  

  free(periods);
  free(objectLookup);

}


void createStreams(cudaStream_t * streams, unsigned int num_gpus, unsigned int streams_per_gpu)
{
  // #pragma omp parallel for num_threads(num_gpus)
  for (unsigned int i=0; i<num_gpus; i++)
  {
    //set device  
    cudaSetDevice(i); 

    //create stream for the device
    for (unsigned int j=0; j<streams_per_gpu; j++)
    {
    cudaStreamCreate(&streams[(i*streams_per_gpu)+j]);
    }
  }

}

void destroyStreamsOneGPU(cudaStream_t * streams, unsigned int streams_per_gpu, int gpuid)
{
  
    //set device  
    cudaSetDevice(gpuid); 

    //create stream for the device
    for (unsigned int i=0; i<streams_per_gpu; i++)
    {
    cudaStreamDestroy(streams[i]);
    }
  

}



void createStreamsOneGPU(cudaStream_t * streams, unsigned int streams_per_gpu, int gpuid)
{
  
    //set device  
    cudaSetDevice(gpuid); 

    //create stream for the device
    for (unsigned int i=0; i<streams_per_gpu; i++)
    {
    cudaStreamCreate(&streams[i]);
    }
  

}

//GPU supersmoother with single pass
//Processes a single object potentially with multiple GPUs
void supsmu_singlepass_single_gpu(unsigned int * objectId, unsigned int sizeData, const DTYPE minFreq, const DTYPE maxFreq, int numFreq, DTYPE * time, DTYPE * data, DTYPE * error, DTYPE alpha, DTYPE * pgram, DTYPE * foundPeriod, double underestGPUcapacityGiB, int gpuid)
{
  double tstartcpu=omp_get_wtime();

  //Allocate host memory 
  DTYPE * weights = (DTYPE *)malloc(sizeof(DTYPE)*sizeData);
  DTYPE * tt = (DTYPE *)malloc(sizeof(DTYPE)*sizeData);

  const DTYPE deltaf=(maxFreq-minFreq)/(numFreq*1.0);
      
  DTYPE chi0=0;

  compute_chi0_tt_weights(sizeData, time, data, error, &chi0, tt, weights);
    
  double tendcpu=omp_get_wtime();
  printf("\nCPU preamble time: %f", tendcpu - tstartcpu);


  double tstartGPUPreabble=omp_get_wtime();
  ////////////////////////
  //for batching the frequencies (not objects)
  //1- mode single pass
  //1- compute assuming a single GPU
  unsigned int numBatches=computeNumBatches(1, sizeData, numFreq, underestGPUcapacityGiB, 1);      
  //upper limit on the number of frequencies in a batch
  int numFreqPerBatch=ceil(numFreq*1.0/numBatches*1.0);
  printf("\nNumber of batches: %d, Number of frequencies per batch: %d", numBatches, numFreqPerBatch);

  cudaStream_t batchstreams[1];
  createStreamsOneGPU(batchstreams, 1, gpuid);
  

  //End for batching frequencies
  ////////////////////////

  //Device variables
  int * dev_freqarr[1];
  DTYPE * dev_smo[1];
  DTYPE * dev_t1[1];
  int * dev_argkeys[1];
  DTYPE * dev_t1_sortby_argkeys[1];
  DTYPE * dev_data_sortby_argkeys[1];
  DTYPE * dev_weights_sortby_argkeys[1];
  DTYPE * dev_tt[1];
  DTYPE * dev_data[1];
  DTYPE * dev_weights[1];
  DTYPE * dev_pgram[1]; 

  //loop used to be here


  cudaSetDevice(gpuid); 
  //Those that depend on the number of frequencies (not the number per batch)
  gpuErrchk(cudaMalloc((void**)&dev_pgram[0],                     sizeof(DTYPE)*numFreq));
  
  //Arrays broken up into batches based on frequency
  gpuErrchk(cudaMalloc((void**)&dev_freqarr[0],                sizeof(int)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_argkeys[0],                sizeof(int)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_smo[0],                    sizeof(DTYPE)*(sizeData*numFreqPerBatch))); 
  gpuErrchk(cudaMalloc((void**)&dev_t1[0],                     sizeof(DTYPE)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_t1_sortby_argkeys[0],      sizeof(DTYPE)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_data_sortby_argkeys[0],    sizeof(DTYPE)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_weights_sortby_argkeys[0], sizeof(DTYPE)*(sizeData*numFreqPerBatch)));

  //allocate on the GPU
  gpuErrchk(cudaMalloc((void**)&dev_tt[0],                    sizeof(DTYPE)*sizeData));
  gpuErrchk(cudaMalloc((void**)&dev_data[0],                    sizeof(DTYPE)*sizeData));
  gpuErrchk(cudaMalloc((void**)&dev_weights[0],                    sizeof(DTYPE)*sizeData));
  
  //copy to the GPU
  gpuErrchk(cudaMemcpyAsync( dev_tt[0],                    tt,                     sizeof(DTYPE)*sizeData, cudaMemcpyHostToDevice, batchstreams[0]));
  gpuErrchk(cudaMemcpyAsync( dev_data[0],                data,                     sizeof(DTYPE)*sizeData, cudaMemcpyHostToDevice, batchstreams[0]));
  gpuErrchk(cudaMemcpyAsync( dev_weights[0],          weights,                     sizeof(DTYPE)*sizeData, cudaMemcpyHostToDevice, batchstreams[0]));


  double tendGPUPreabble=omp_get_wtime();
  printf("\nTime GPU preamble: %f", tendGPUPreabble - tstartGPUPreabble);

  double tstartmainloop=omp_get_wtime();
  //Loop over Batches
  #pragma omp parallel for num_threads(1)
  for (unsigned int i=0; i<numBatches; i++)
  {

  cudaSetDevice(gpuid); 

  int globaltid=omp_get_thread_num();
  //thread id for a single GPU
  int tid=globaltid;  
  uint64_t batchWriteOffset=(uint64_t)i*(uint64_t)numFreqPerBatch;
  uint64_t offsetFreqId=(uint64_t)i*(uint64_t)numFreqPerBatch;
  int numFreqInBatch=numFreqPerBatch;
  int streamOffset=sizeData*numFreqPerBatch*tid;
  int streamnum=tid;

  //last batch has fewer frequencies
  if((numBatches!=1)&&(i==(numBatches-1)))
  {
    numFreqInBatch=min(numFreqInBatch,((int)numFreq)-((i)*numFreqPerBatch));
  }

  printf("\nglobal tid: %d, tid: %d, gpuid: %d, Stream num: %d, Batch Number: %d, number of frequencies: %d",globaltid, tid, gpuid, streamnum, i, numFreqInBatch);
  
  
  unsigned int NUMBLOCKSDATAFREQ=ceil((sizeData*numFreqInBatch*1.0)/LARGEBLOCKSIZE*1.0);
  computePeriodModFOneThreadPerUpdate<<<NUMBLOCKSDATAFREQ,LARGEBLOCKSIZE,0,batchstreams[streamnum]>>>(sizeData, numFreqInBatch, minFreq, offsetFreqId, deltaf, &dev_t1[0][streamOffset], dev_tt[0]);
  
  //Initialize the key arrays
  initializeKeyArraysOneThreadPerUpdate<<<NUMBLOCKSDATAFREQ,LARGEBLOCKSIZE,0,batchstreams[streamnum]>>>(sizeData, numFreqInBatch, &dev_argkeys[0][streamOffset], &dev_freqarr[0][streamOffset]);
        
  //Need to do back to back sorts to sort the t1 by argkeys for each frequency
  //Need 3 arrays: 
  //first sort the keys (argkeys) by the values (t1)
  //then sort the argkeys/t1 by the freqarr
  backToBackSort(&dev_argkeys[0][streamOffset], &dev_freqarr[0][streamOffset], &dev_t1[0][streamOffset], sizeData, numFreqInBatch, batchstreams[streamnum]);
        
  //combine map and transform for coalesced memory accesses for global memory kernel
  
  mapUsingArgKeysOneThreadPerUpdateAndReorderCoalesced<<<NUMBLOCKSDATAFREQ,LARGEBLOCKSIZE,0,batchstreams[streamnum]>>>(sizeData, numFreqInBatch, &dev_argkeys[0][streamOffset], 
    &dev_data[0][0], &dev_weights[0][0], 
    &dev_t1[0][streamOffset], &dev_t1_sortby_argkeys[0][streamOffset], &dev_data_sortby_argkeys[0][streamOffset], &dev_weights_sortby_argkeys[0][streamOffset]);      
  

  ///////////////////////////////
  // Main kernels
  ///////////////////////////////  
  
  //global memory only
  
    const unsigned int numBlocks=ceil((numFreqInBatch*1.0)/(SMALLBLOCKSIZE*1.0));

    supsmukernelSinglePassGlobalMemoryCoalesced<<<numBlocks,SMALLBLOCKSIZE, 0,batchstreams[streamnum]>>>(numFreqInBatch, sizeData, alpha, &dev_smo[0][streamOffset],
      &dev_tt[0][0], &dev_t1_sortby_argkeys[0][streamOffset], &dev_data_sortby_argkeys[0][streamOffset], &dev_weights_sortby_argkeys[0][streamOffset]); 
    
  
  //Some number of threads per frequency
  unsigned int numThreadPerFreq2=8; //must divide evenly into the block size
  unsigned int NUMBLOCKS10=ceil((numFreqInBatch*numThreadPerFreq2*1.0)/(LARGEBLOCKSIZE*1.0));
  const unsigned int SMSIZE2=sizeof(DTYPE)*(LARGEBLOCKSIZE/numThreadPerFreq2);
  
  computePgramReductionCoalesced<<<NUMBLOCKS10, LARGEBLOCKSIZE, SMSIZE2, batchstreams[streamnum]>>>(batchWriteOffset, numThreadPerFreq2, chi0, sizeData, 
    numFreqInBatch, &dev_smo[0][streamOffset], &dev_data_sortby_argkeys[0][streamOffset], &dev_weights_sortby_argkeys[0][streamOffset], &dev_pgram[0][0]);
  

    //Copy pgram back to host
    gpuErrchk(cudaMemcpyAsync(pgram+batchWriteOffset, &dev_pgram[0][batchWriteOffset], sizeof(DTYPE)*numFreqInBatch, cudaMemcpyDeviceToHost, batchstreams[streamnum]));
    
  } //end loop over batches

  double tendmainloop=omp_get_wtime();
  printf("\nTime main loop: %f",tendmainloop - tstartmainloop);
      

  ///////////////////////////////
  // End main kernels
  ///////////////////////////////  

  double tstartperiod=omp_get_wtime();  
  computePeriodSuperSmoother(pgram, numFreq, minFreq, maxFreq, foundPeriod);  
  double tendperiod=omp_get_wtime();  
  printf("\nObject id: %d, Found period: %f", *objectId, *foundPeriod);
  printf("\nTime to compute period: %f", tendperiod - tstartperiod);
  

  double tstartfree=omp_get_wtime();

  //free device data
  cudaFree(dev_pgram[0]);                              
  cudaFree(dev_freqarr[0]);                
  cudaFree(dev_argkeys[0]);                
  cudaFree(dev_smo[0]);                    
  cudaFree(dev_t1[0]);                     
  cudaFree(dev_t1_sortby_argkeys[0]);      
  cudaFree(dev_data_sortby_argkeys[0]);    
  cudaFree(dev_weights_sortby_argkeys[0]); 
  cudaFree(dev_tt[0]);                    
  cudaFree(dev_data[0]);                    
  cudaFree(dev_weights[0]);
  
  destroyStreamsOneGPU(batchstreams, 1, gpuid);

  //free host data
  free(weights);
  free(tt);

  double tendfree=omp_get_wtime();
  printf("\nTime to free: %f", tendfree - tstartfree);
  
  

}



//GPU supersmoother with single pass
//Processes a single object potentially with multiple GPUs
void supsmu_singlepass_single_object(unsigned int * objectId, unsigned int sizeData, const DTYPE minFreq, const DTYPE maxFreq, int numFreq, DTYPE * time, DTYPE * data, DTYPE * error, DTYPE alpha, DTYPE * pgram, DTYPE * foundPeriod, double underestGPUcapacityGiB)
{
  double tstartcpu=omp_get_wtime();


  //Allocate host memory 
  DTYPE * weights = (DTYPE *)malloc(sizeof(DTYPE)*sizeData);
  DTYPE * tt = (DTYPE *)malloc(sizeof(DTYPE)*sizeData);

  const DTYPE deltaf=(maxFreq-minFreq)/(numFreq*1.0);
      
  DTYPE chi0=0;

  compute_chi0_tt_weights(sizeData, time, data, error, &chi0, tt, weights);
    
  double tendcpu=omp_get_wtime();
  printf("\nCPU preamble time: %f", tendcpu - tstartcpu);


  double tstartGPUPreabble=omp_get_wtime();
  ////////////////////////
  //for batching the frequencies (not objects)
  //1- mode single pass
  //0- compute all batches assuming using all GPUs
  unsigned int numBatches=computeNumBatches(1, sizeData, numFreq, underestGPUcapacityGiB, 0);      
  //upper limit on the number of frequencies in a batch
  int numFreqPerBatch=ceil(numFreq*1.0/numBatches*1.0);
  printf("\nNumber of batches: %d, Number of frequencies per batch: %d", numBatches, numFreqPerBatch);

  
  cudaStream_t batchstreams[NUMGPU];
  createStreams(batchstreams, NUMGPU, 1);
  
  //End for batching frequencies
  ////////////////////////

  //Device variables
  int * dev_freqarr[NUMGPU];
  DTYPE * dev_smo[NUMGPU];
  DTYPE * dev_t1[NUMGPU];
  int * dev_argkeys[NUMGPU];
  DTYPE * dev_t1_sortby_argkeys[NUMGPU];
  DTYPE * dev_data_sortby_argkeys[NUMGPU];
  DTYPE * dev_weights_sortby_argkeys[NUMGPU];
  DTYPE * dev_tt[NUMGPU];
  DTYPE * dev_data[NUMGPU];
  DTYPE * dev_weights[NUMGPU];
  DTYPE * dev_pgram[NUMGPU]; 

  //Allocate memory and copy data to each GPU
  #pragma omp parallel for num_threads(NUMGPU)
  for (int i=0; i<NUMGPU; i++)
  {
  int globaltid=omp_get_thread_num();
  int tid=globaltid;  
  int gpuid=globaltid;  
  int streamnum=(gpuid)+tid;  

  cudaSetDevice(i); 
  //Those that depend on the number of frequencies (not the number per batch)
  gpuErrchk(cudaMalloc((void**)&dev_pgram[i],                     sizeof(DTYPE)*numFreq));
  
  //Arrays broken up into batches based on frequency
  gpuErrchk(cudaMalloc((void**)&dev_freqarr[i],                sizeof(int)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_argkeys[i],                sizeof(int)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_smo[i],                    sizeof(DTYPE)*(sizeData*numFreqPerBatch))); 
  gpuErrchk(cudaMalloc((void**)&dev_t1[i],                     sizeof(DTYPE)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_t1_sortby_argkeys[i],      sizeof(DTYPE)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_data_sortby_argkeys[i],    sizeof(DTYPE)*(sizeData*numFreqPerBatch)));
  gpuErrchk(cudaMalloc((void**)&dev_weights_sortby_argkeys[i], sizeof(DTYPE)*(sizeData*numFreqPerBatch)));

  //allocate on the GPU
  gpuErrchk(cudaMalloc((void**)&dev_tt[i],                    sizeof(DTYPE)*sizeData));
  gpuErrchk(cudaMalloc((void**)&dev_data[i],                    sizeof(DTYPE)*sizeData));
  gpuErrchk(cudaMalloc((void**)&dev_weights[i],                    sizeof(DTYPE)*sizeData));
  
  //copy to the GPU
  gpuErrchk(cudaMemcpyAsync( dev_tt[i],                    tt,                     sizeof(DTYPE)*sizeData, cudaMemcpyHostToDevice, batchstreams[streamnum]));
  gpuErrchk(cudaMemcpyAsync( dev_data[i],                data,                     sizeof(DTYPE)*sizeData, cudaMemcpyHostToDevice, batchstreams[streamnum]));
  gpuErrchk(cudaMemcpyAsync( dev_weights[i],          weights,                     sizeof(DTYPE)*sizeData, cudaMemcpyHostToDevice, batchstreams[streamnum]));

  }


  double tendGPUPreabble=omp_get_wtime();
  printf("\nTime GPU preamble: %f", tendGPUPreabble - tstartGPUPreabble);

  double tstartmainloop=omp_get_wtime();
  //Loop over Batches
  #pragma omp parallel for num_threads(NUMGPU)
  for (unsigned int i=0; i<numBatches; i++)
  {

  int globaltid=omp_get_thread_num();
  //thread id for a single GPU
  int tid=globaltid;  
  int gpuid=globaltid;
  uint64_t batchWriteOffset=(uint64_t)i*(uint64_t)numFreqPerBatch;
  uint64_t offsetFreqId=(uint64_t)i*(uint64_t)numFreqPerBatch;
  int numFreqInBatch=numFreqPerBatch;
  int streamOffset=sizeData*numFreqPerBatch*tid;

  int streamnum=(gpuid)+tid;

  cudaSetDevice(gpuid); 

  //last batch has fewer frequencies
  if((numBatches!=1)&&(i==(numBatches-1)))
  {
    numFreqInBatch=min(numFreqInBatch,((int)numFreq)-((i)*numFreqPerBatch));
  }

  printf("\nglobal tid: %d, tid: %d, gpuid: %d, Stream num: %d, Batch Number: %d, number of frequencies: %d",globaltid, tid, gpuid, streamnum, i, numFreqInBatch);
  
  
  unsigned int NUMBLOCKSDATAFREQ=ceil((sizeData*numFreqInBatch*1.0)/LARGEBLOCKSIZE*1.0);
  computePeriodModFOneThreadPerUpdate<<<NUMBLOCKSDATAFREQ,LARGEBLOCKSIZE,0,batchstreams[streamnum]>>>(sizeData, numFreqInBatch, minFreq, offsetFreqId, deltaf, &dev_t1[gpuid][streamOffset], dev_tt[gpuid]);
  
  //Initialize the key arrays
  initializeKeyArraysOneThreadPerUpdate<<<NUMBLOCKSDATAFREQ,LARGEBLOCKSIZE,0,batchstreams[streamnum]>>>(sizeData, numFreqInBatch, &dev_argkeys[gpuid][streamOffset], &dev_freqarr[gpuid][streamOffset]);
        
  //Need to do back to back sorts to sort the t1 by argkeys for each frequency
  //Need 3 arrays: 
  //first sort the keys (argkeys) by the values (t1)
  //then sort the argkeys/t1 by the freqarr
  backToBackSort(&dev_argkeys[gpuid][streamOffset], &dev_freqarr[gpuid][streamOffset], &dev_t1[gpuid][streamOffset], sizeData, numFreqInBatch, batchstreams[streamnum]);
        
  
  //combine map and transform for coalesced memory accesses for global memory kernel
  mapUsingArgKeysOneThreadPerUpdateAndReorderCoalesced<<<NUMBLOCKSDATAFREQ,LARGEBLOCKSIZE,0,batchstreams[streamnum]>>>(sizeData, numFreqInBatch, &dev_argkeys[gpuid][streamOffset], 
    &dev_data[gpuid][0], &dev_weights[gpuid][0], 
    &dev_t1[gpuid][streamOffset], &dev_t1_sortby_argkeys[gpuid][streamOffset], &dev_data_sortby_argkeys[gpuid][streamOffset], &dev_weights_sortby_argkeys[gpuid][streamOffset]);      

  ///////////////////////////////
  // Main kernels
  ///////////////////////////////  
  
  //global memory only
  
    const unsigned int numBlocks=ceil((numFreqInBatch*1.0)/(SMALLBLOCKSIZE*1.0));
    supsmukernelSinglePassGlobalMemoryCoalesced<<<numBlocks,SMALLBLOCKSIZE, 0,batchstreams[streamnum]>>>(numFreqInBatch, sizeData, alpha, &dev_smo[gpuid][streamOffset],
      &dev_tt[gpuid][0], &dev_t1_sortby_argkeys[gpuid][streamOffset], &dev_data_sortby_argkeys[gpuid][streamOffset], &dev_weights_sortby_argkeys[gpuid][streamOffset]); 
    
  //Some number of threads per frequency
  unsigned int numThreadPerFreq2=8; //must divide evenly into the block size
  unsigned int NUMBLOCKS10=ceil((numFreqInBatch*numThreadPerFreq2*1.0)/(LARGEBLOCKSIZE*1.0));
  const unsigned int SMSIZE2=sizeof(DTYPE)*(LARGEBLOCKSIZE/numThreadPerFreq2);
  

  
  computePgramReductionCoalesced<<<NUMBLOCKS10, LARGEBLOCKSIZE, SMSIZE2, batchstreams[streamnum]>>>(batchWriteOffset, numThreadPerFreq2, chi0, sizeData, 
    numFreqInBatch, &dev_smo[gpuid][streamOffset], &dev_data_sortby_argkeys[gpuid][streamOffset], &dev_weights_sortby_argkeys[gpuid][streamOffset], &dev_pgram[gpuid][0]);

    //Copy pgram back to host
    gpuErrchk(cudaMemcpyAsync(pgram+batchWriteOffset, &dev_pgram[gpuid][batchWriteOffset], sizeof(DTYPE)*numFreqInBatch, cudaMemcpyDeviceToHost, batchstreams[streamnum]));
    
  } //end loop over batches

  double tendmainloop=omp_get_wtime();
  printf("\nTime main loop: %f",tendmainloop - tstartmainloop);
      

  ///////////////////////////////
  // End main kernels
  ///////////////////////////////  

  double tstartperiod=omp_get_wtime();  
  computePeriodSuperSmoother(pgram, numFreq, minFreq, maxFreq, foundPeriod);  
  double tendperiod=omp_get_wtime();  
  printf("\nFound period: %f", *foundPeriod);
  printf("\nTime to compute period: %f", tendperiod - tstartperiod);
  

  double tstartfree=omp_get_wtime();

  //free device data
  #pragma omp parallel for num_threads(NUMGPU)
  for (int i=0; i<NUMGPU; i++)
  {
  cudaFree(dev_pgram[i]);                              
  cudaFree(dev_freqarr[i]);                
  cudaFree(dev_argkeys[i]);                
  cudaFree(dev_smo[i]);                    
  cudaFree(dev_t1[i]);                     
  cudaFree(dev_t1_sortby_argkeys[i]);      
  cudaFree(dev_data_sortby_argkeys[i]);    
  cudaFree(dev_weights_sortby_argkeys[i]); 
  cudaFree(dev_tt[i]);                    
  cudaFree(dev_data[i]);                    
  cudaFree(dev_weights[i]);
  }
  
  //free host data
  free(weights);
  free(tt);

  double tendfree=omp_get_wtime();
  printf("\nTime to free: %f", tendfree - tstartfree);
  
  

}





//Estimated memory footprint used to compute the number of batches
//used to compute the number of batches
//mode-0 is original
//mode-1 is single pass
//pass in the underestimated capacity 
//singlegpuflag- 0- use NUMGPU GPUs
//singlegpuflag- 1- use 1 GPU
unsigned int computeNumBatches(bool mode, unsigned int sizeData, unsigned int numFreq, double underestGPUcapacityGiB, bool singlegpuflag)
{

  printf("\n*********************");
  
  //Memory footprint assuming FP64 data
  //Single pass: sp=[1/(1024**3)]*[(8*Nf)+(3*8*Nt)+(2*4*Nf*Nt)+(5*8*Nf*Nt)+(2*3*8*nf*nt)]
  //original: sp+(8*nf*nt)

  double totalGiB=0.0;
  //pgram
  totalGiB+=sizeof(DTYPE)*numFreq/(1024*1024*1024.0);

  //tt, data, weights
  totalGiB+=3*sizeof(DTYPE)*sizeData/(1024*1024*1024.0);

  //freqArr, argkeys
  totalGiB+=2*sizeof(int)*numFreq*sizeData/(1024*1024*1024.0);

  //smo, t1, t1_sortby_argkeys, data_sortby_argkeys, weights_sortby_argkeys,
  totalGiB+=5*sizeof(DTYPE)*numFreq*sizeData/(1024*1024*1024.0);

  // sorting (out-of-place radix sorting requires an extra n storage, but overestimate 2n because back-to-back may require 2n)
  totalGiB+=2*3*sizeof(DTYPE)*numFreq*sizeData/(1024*1024*1024.0);

  //account for scratch in original algorithm
  if (mode==0)
  {
    totalGiB+=sizeof(DTYPE)*numFreq*sizeData*8/(1024*1024*1024.0);
  }
  printf("\nEstimated global memory footprint (GiB): %f", totalGiB);


  unsigned int numBatches=ceil(totalGiB/(underestGPUcapacityGiB));
  printf("\nMinimum number of batches: %u", numBatches);
  
  if (singlegpuflag==0)
  {
  numBatches=ceil((numBatches*1.0/NUMGPU))*NUMGPU;
  printf("\nNumber of batches (after ensuring batches evenly divide %d GPUs): %u", NUMGPU, numBatches);
  }
  else
  {
  printf("\nNumber of batches (after ensuring batches evenly divide 1 GPUs): %u", numBatches);  
  }

  printf("\n*********************\n");
  return numBatches;
}



void outputPgramToFile(struct lookupObj * objectLookup, unsigned int numUniqueObjects, unsigned int numFreqs, DTYPE ** pgram)
{
    char fnameoutput[]="pgram_SS.txt";
    printf("\nPrinting the pgram to file: %s", fnameoutput);
    ofstream pgramoutput;
    pgramoutput.open(fnameoutput,ios::out); 
    pgramoutput.precision(4);
    for (unsigned int i=0; i<numUniqueObjects; i++)
    {
    pgramoutput<<objectLookup[i].objId<<", ";
    for (unsigned int j=0; j<numFreqs; j++)
    {
    pgramoutput<<(*pgram)[(i*numFreqs)+j]<<", ";
    }
    pgramoutput<<endl;
    }
    pgramoutput.close();
}


void outputPeriodsToFile(struct lookupObj * objectLookup, unsigned int numUniqueObjects, DTYPE * foundPeriod)
{
  char fnamebestperiods[]="bestperiods_SS.txt";
    printf("\nPrinting the best periods to file: %s", fnamebestperiods);
  ofstream bestperiodsoutput;
  bestperiodsoutput.open(fnamebestperiods,ios::out);  
    bestperiodsoutput.precision(7);
    for (unsigned int i=0; i<numUniqueObjects; i++)
  {
    bestperiodsoutput<<objectLookup[i].objId<<", "<<foundPeriod[i]<<endl;
  }
    bestperiodsoutput.close();
}

void outputPeriodsToStdout(struct lookupObj * objectLookup, unsigned int numUniqueObjects, DTYPE * foundPeriod)
{
  for (unsigned int i=0; i<numUniqueObjects; i++)
    {
      printf("\nObject: %d Period: %f, ",objectLookup[i].objId,foundPeriod[i]);
    }
}








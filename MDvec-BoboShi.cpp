// Compile using g++ -O3 -msse4 
// Replace baseline version with "improved versions"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sys/time.h>
#include <cmath>
#include <iomanip>
using namespace std;
#include <nmmintrin.h>

#define THRESHOLD 0.0001
#define N 9000

double rtclock();
void compare(float *u1, float *u2, float r1[][3], float r2[][3]);
double clkbegin, clkend;
double t;

double rtclock()
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

int main(int argc, char* argv[]){
  
  float r[N][3], f[N][3], u[N];
  float eps = 1.0;
  float sig = 1.0;
  int i, j, k;
  double start, end;
  float rsqrt, rt[3];
  float temp;
  float q1, q2;
  q1 = 1.0;
  q2 = 1.0;
  cout << "N=" << N << endl;
  // give random r value
  for ( i = 0; i < N; i++){
    for (j = 0; j < 3; j++){
      r[i][j] = float ( rand() ) / RAND_MAX * 100;
    }
  }

  cout << " sequentail start. " << endl;
  start = rtclock();
  for ( i = 0; i < N; i++){
    for (j = 0; j < 3; j++){
      f[i][j] = 0;
    }
    u[i] = 0;
  }

  for ( i = 1; i < N; i++){
    for ( j = 0; j < i; j++){
      rsqrt = 0;
      for ( k = 0; k < 3; k++){
	rt[k] = r[i][k] - r[j][k];
	rsqrt += rt[k] * rt[k];
      }
      rsqrt = sqrt(rsqrt);
      //calculate f
      temp = q1 * q2 / pow(rsqrt, 3);
      for ( k = 0; k < 3; k++){
	f[i][k] += temp * rt[k];
	f[j][k] -= temp * rt[k];
      }
      //temp = pow(sig/rsqrt, 12) - pow(sig/rsqrt, 6);
      temp = q1 * q2 / rsqrt;
      u[i] += temp;
      u[j] += temp;
    }
  }
  
  end = rtclock();
  
  cout << " squential time: " << end - start << "seconds." << endl;
  //
  //
  // vectorization 1
  //
  
  cout << "vectorization 1 start. " << endl;
  start = rtclock();

  float r1[3][N], f1[3][N], u1[N]; // used to store the transpoed 
  float r1t[3];
  //__m128 r1v[3][N/4], f1v[3][N/4], u1v[3][N/4];
  //__m128 tempv, r1sqrtv, r1tv;
  
  start = rtclock();
  // transpose
  for ( i = 0; i < 3; i++){
    for ( j = 0; j < N; j++){
      r1[i][j] = r[j][i];
      f1[i][j] = 0.0;
    }
  }


  __m128 rsqv, rsqrtv;
  for ( i = 0; i < N; i++)
    u1[i] = 0.0;
  
  for (i = 1; i < N; i++){
    float fi[3][i];
    float ui[i];
    int nvec;
    nvec = i/4;
    __m128 r1tv[3], tempv, temp1v, temp2v;
    if (nvec > 0){
      // __m128 r1v[3][nvec], r1tv[3], f1v[3][nvec], tempv, temp1v, temp2v;
      //__m128 fiv[3][nvec], uiv[nvec], u1v[nvec];
      //__m128 *r1v, *f1v, *fiv;
      __m128 r1v[3*nvec], f1v[3*nvec], fiv[3*nvec];
      float fis[3][nvec*4], uis[nvec*4];
      //__m128 *uiv, *u1v;
      __m128 uiv[nvec], u1v[nvec];
      __m128 riv[3];
      for ( k = 0; k < 3; k++){
	riv[k] = _mm_load1_ps(r1[k]+i);
      }

      for (k = 0; k < 3; k++){
	for ( j = 0; j< nvec; j++){
	  r1v[k*nvec +j] = _mm_loadu_ps(r1[k]+j*4);
	  f1v[k*nvec +j] = _mm_loadu_ps(f1[k]+j*4);
	}
      }
      for ( j = 0; j < nvec; j++){
	u1v[j] = _mm_loadu_ps(u1+j*4);
      }
      for ( j = 0; j < nvec; j++){
	rsqv = _mm_set1_ps(0.0);
	for (k = 0; k < 3; k++){
	  r1tv[k] = _mm_sub_ps(riv[k], r1v[k*nvec+j]);
	  tempv = _mm_mul_ps(r1tv[k], r1tv[k]);
	  rsqv = _mm_add_ps(rsqv, tempv);
	}
	rsqrtv = _mm_sqrt_ps(rsqv);
	
	temp = q1 * q2;
	temp1v = _mm_set1_ps(temp);
	temp2v = _mm_div_ps(temp1v, rsqrtv);
	temp2v = _mm_div_ps(temp2v, rsqv);
	for ( k = 0; k < 3; k++){
	  fiv[k*nvec+j] = _mm_mul_ps(temp2v, r1tv[k]);
	  f1v[k*nvec+j] = _mm_sub_ps(f1v[k*nvec+j], fiv[k*nvec+j]);
	}
	uiv[j] = _mm_div_ps(temp1v, rsqrtv);
	u1v[j] = _mm_add_ps(u1v[j], uiv[j]);
      }
      // store back the temporary uiv fiv
      for (k = 0; k < 3; k++){
	for (j = 0; j < nvec; j++){
	  _mm_store_ps(fis[k]+j*4, fiv[k*nvec+j]);
	}
      }
      for (j = 0; j < nvec; j++){
	//u1v[i] = _mm_add_ps(u1v[i], uiv[j]);
	_mm_store_ps(uis+j*4, uiv[j]);
      }
      //store back
      for ( k = 0; k < 3; k++){
	for ( j = 0; j < nvec; j++){
	  _mm_store_ps(f1[k]+j*4, f1v[k*nvec+j]);
	}
      }
      for ( j = 0; j < nvec; j++){
	_mm_store_ps(u1+j*4, u1v[j]);
      }
      for ( k = 0; k < 3; k++){
	for (j = 0; j < nvec*4; j++){
	  f1[k][i] += fis[k][j];
	}
      }
      for (j = 0; j < nvec*4; j++){
	u1[i] += uis[j];
      }

    }
#pragma vector unaligned    
    for ( j = nvec * 4; j < i; j++){
      rsqrt = 0;

      r1t[0] = r1[0][i] - r1[0][j];
      r1t[1] = r1[1][i] - r1[1][j];
      r1t[2] = r1[2][i] - r1[2][j];
      
      rsqrt = r1t[0] * r1t[0] + r1t[1] * r1t[1] + r1t[2] * r1t[2];

      rsqrt = sqrt(rsqrt);
      temp = q1 * q2 / pow(rsqrt, 3);

      for ( k = 0; k < 3; k++){

	fi[k][j] = temp * r1t[k];
	f1[k][j] -= temp * r1t[k];

      }

      temp = q1 * q2 / rsqrt;
      
      //u1[i] += temp;
      ui[j] = temp;
      u1[j] += temp; 
    }
    for ( k = 0; k < 3; k++){
      for (j = nvec*4; j < i; j++){
	f1[k][i] += fi[k][j];
      }
    }
    for ( j = nvec*4; j < i; j++){
      u1[i] += ui[j];
    }
    
  }
  end = rtclock();
  cout << " vectorization 1 time: " << end - start << " seconds." << endl;

  float fout[N][3];
  for (i = 0; i< N; i++){
    for (k = 0; k < 3; k++){
      fout[i][k] = f1[k][i];
    }
  }

  compare (u, u1, f, fout);
  // vectorization 2

  __m128 r2[N], f2[N], tempt, r2t, r2t1, rev;
  float rsq, r2s[4];
  rev = _mm_set_ps(-1.0, -1.0, -1.0, 1.0);
  for ( i = 0; i < N; i++){
    f2[i] = _mm_set1_ps(0.0);
  }

  start = rtclock();
  for ( i = 1; i < N; i++){
    for ( j = 0; j < i; j++){
      rsq = 0;
      
      for ( k = 0; k < 3; k++){
	rt[k] = r[i][k] - r[j][k];
	rsq += rt[k] * rt[k];
      }
      rsqrt = sqrt(rsq);
      temp = q1 * q2 / (rsq * rsqrt);
      tempt = _mm_set1_ps(temp);
      r2t = _mm_set_ps(rt[2],rt[1],rt[0],rsq);
      tempt = _mm_mul_ps(tempt, r2t);
      f2[i] = _mm_add_ps(f2[i], tempt);
      tempt = _mm_mul_ps(tempt, rev);
      f2[j] = _mm_add_ps(f2[j], tempt);
    }
  }
  end = rtclock();
  cout << "vectorization 2 time: " << end - start << " seconds" << endl;

  float out[N][4], f2out[N][3], u2[N];
  for (i = 0; i < N; i++){
    _mm_store_ps(out[i], f2[i]);
  }
  for (i = 0; i < N; i++){
    for ( k = 1; k < 4; k++){
      f2out[i][k-1] = out[i][k];    
    }
    u2[i] = out[i][0];
  }
  compare(u, u2, f, f2out);
  
}

void compare(float *u1, float *u2, float f1[][3], float f2[][3]){
  int numdiffs, maxdiff;
  int i, j, k;
  float diff;
  numdiffs = 0;
  maxdiff = 0;
  for ( i = 0; i < N; i++){
    diff = u1[i] - u2[i];
    if (diff < 0) diff = -1.0 * diff;
    if (diff > THRESHOLD){
      numdiffs++;
      if (diff > maxdiff) maxdiff = diff;
    }
  }
  if (numdiffs > 0){
    cout << "For u, " <<  numdiffs << " Diffs found over threshold ";
    cout << THRESHOLD << ", Max Diff = " << maxdiff << endl;
  }
  else {
    cout << "No differences found between u1 and u2" << endl;
  }
  
  numdiffs = 0;
  maxdiff = 0;
  for ( i = 0; i < N; i++){
    for ( k = 0; k < 3; k++){
      diff = f1[i][k] - f2[i][k];
      if (diff < 0) diff = -1.0 * diff;
      if (diff > THRESHOLD){
	numdiffs++;
	if (diff > maxdiff) maxdiff = diff;
      }
    }
    
  }
  if ( numdiffs > 0){
    cout << "For f, " << numdiffs << " Diffs found over threshold ";
    cout << THRESHOLD << ", Max Diff = " << maxdiff << endl;
  }
  else{
    cout << "No differences found between f1 and f2" << endl;
  }
  
}

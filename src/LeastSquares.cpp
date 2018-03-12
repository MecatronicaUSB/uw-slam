/**
* This file is part of LSD-SLAM, added to UW-SLAM.
* 
* Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
* For more information see <http://vision.in.tum.de/lsdslam> 
*
* Uses the SSE implementation of LSD-SLAM to solve Normal equations.
* 
* LSD-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* UW-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with UW-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/


#include "LeastSquares.h"

namespace uw
{

inline void NormalEquationsLeastSquares::initialize(int max_num_constraints) {
    A.setZero();
    b.setZero();
    memset(SSEData,0, sizeof(float)*4*28);

    error = 0;
    this->num_constraints = 0;
};

inline void NormalEquationsLeastSquares::finishNoDivide() {
	__m128 a;

  	a = _mm_load_ps(SSEData+4*0);
  	A(0,0) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3);

  	a = _mm_load_ps(SSEData+4*1);
  	A(1,0) = (A(0,1) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3));

  	a = _mm_load_ps(SSEData+4*2);
  	A(2,0) = (A(0,2) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3));

  	a = _mm_load_ps(SSEData+4*3);
  	A(3,0) = (A(0,3) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3));

  	a = _mm_load_ps(SSEData+4*4);
  	A(4,0) = (A(0,4) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3));

  	a = _mm_load_ps(SSEData+4*5);
  	A(5,0) = (A(0,5) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3));



  	a = _mm_load_ps(SSEData+4*6);
  	A(1,1) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3);

  	a = _mm_load_ps(SSEData+4*7);
  	A(1,2) = (A(2,1) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3));

  	a = _mm_load_ps(SSEData+4*8);
  	A(1,3) = (A(3,1) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3));

  	a = _mm_load_ps(SSEData+4*9);
  	A(1,4) = (A(4,1) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3));

  	a = _mm_load_ps(SSEData+4*10);
  	A(1,5) = (A(5,1) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3));



  	a = _mm_load_ps(SSEData+4*11);
  	A(2,2) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3);

  	a = _mm_load_ps(SSEData+4*12);
  	A(2,3) = (A(3,2) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3));

  	a = _mm_load_ps(SSEData+4*13);
  	A(2,4) = (A(4,2) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3));

  	a = _mm_load_ps(SSEData+4*14);
  	A(2,5) = (A(5,2) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3));



  	a = _mm_load_ps(SSEData+4*15);
  	A(3,3) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3);

  	a = _mm_load_ps(SSEData+4*16);
  	A(3,4) = (A(4,3) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3));

  	a = _mm_load_ps(SSEData+4*17);
  	A(3,5) = (A(5,3) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3));



  	a = _mm_load_ps(SSEData+4*18);
  	A(4,4) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3);

  	a = _mm_load_ps(SSEData+4*19);
  	A(4,5) = (A(5,4) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3));



  	a = _mm_load_ps(SSEData+4*20);
  	A(5,5) += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3);



  	a = _mm_load_ps(SSEData+4*21);
  	b[0] -= SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3);

  	a = _mm_load_ps(SSEData+4*22);
  	b[1] -= SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3);

  	a = _mm_load_ps(SSEData+4*23);
  	b[2] -= SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3);

  	a = _mm_load_ps(SSEData+4*24);
  	b[3] -= SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3);

  	a = _mm_load_ps(SSEData+4*25);
  	b[4] -= SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3);

  	a = _mm_load_ps(SSEData+4*26);
  	b[5] -= SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3);



  	a = _mm_load_ps(SSEData+4*27);
  	error += SSEE(a,0) + SSEE(a,1) + SSEE(a,2) + SSEE(a,3);
};

void NormalEquationsLeastSquares::finish() {
    finishNoDivide();
    A /= (float) num_constraints;
    b /= (float) num_constraints;
    error /= (float) num_constraints;
};

inline void NormalEquationsLeastSquares::updateSSE(const __m128 &J1,const __m128 &J2,const __m128 &J3,const __m128 &J4,
  		                                            const __m128 &J5, const __m128 &J6,const __m128& res, const __m128& weight) {
    //A.noalias() += J * J.transpose() * weight;
    __m128 J1w = _mm_mul_ps(J1,weight);
    _mm_store_ps(SSEData+4*0, _mm_add_ps(_mm_load_ps(SSEData+4*0),_mm_mul_ps(J1w,J1)));
    _mm_store_ps(SSEData+4*1, _mm_add_ps(_mm_load_ps(SSEData+4*1),_mm_mul_ps(J1w,J2)));
    _mm_store_ps(SSEData+4*2, _mm_add_ps(_mm_load_ps(SSEData+4*2),_mm_mul_ps(J1w,J3)));
    _mm_store_ps(SSEData+4*3, _mm_add_ps(_mm_load_ps(SSEData+4*3),_mm_mul_ps(J1w,J4)));
    _mm_store_ps(SSEData+4*4, _mm_add_ps(_mm_load_ps(SSEData+4*4),_mm_mul_ps(J1w,J5)));
    _mm_store_ps(SSEData+4*5, _mm_add_ps(_mm_load_ps(SSEData+4*5),_mm_mul_ps(J1w,J6)));


    __m128 J2w = _mm_mul_ps(J2,weight);
    _mm_store_ps(SSEData+4*6, _mm_add_ps(_mm_load_ps(SSEData+4*6),_mm_mul_ps(J2w,J2)));
    _mm_store_ps(SSEData+4*7, _mm_add_ps(_mm_load_ps(SSEData+4*7),_mm_mul_ps(J2w,J3)));
    _mm_store_ps(SSEData+4*8, _mm_add_ps(_mm_load_ps(SSEData+4*8),_mm_mul_ps(J2w,J4)));
    _mm_store_ps(SSEData+4*9, _mm_add_ps(_mm_load_ps(SSEData+4*9),_mm_mul_ps(J2w,J5)));
    _mm_store_ps(SSEData+4*10, _mm_add_ps(_mm_load_ps(SSEData+4*10),_mm_mul_ps(J2w,J6)));


    __m128 J3w = _mm_mul_ps(J3,weight);
    _mm_store_ps(SSEData+4*11, _mm_add_ps(_mm_load_ps(SSEData+4*11),_mm_mul_ps(J3w,J3)));
    _mm_store_ps(SSEData+4*12, _mm_add_ps(_mm_load_ps(SSEData+4*12),_mm_mul_ps(J3w,J4)));
    _mm_store_ps(SSEData+4*13, _mm_add_ps(_mm_load_ps(SSEData+4*13),_mm_mul_ps(J3w,J5)));
    _mm_store_ps(SSEData+4*14, _mm_add_ps(_mm_load_ps(SSEData+4*14),_mm_mul_ps(J3w,J6)));

    __m128 J4w = _mm_mul_ps(J4,weight);
    _mm_store_ps(SSEData+4*15, _mm_add_ps(_mm_load_ps(SSEData+4*15),_mm_mul_ps(J4w,J4)));
    _mm_store_ps(SSEData+4*16, _mm_add_ps(_mm_load_ps(SSEData+4*16),_mm_mul_ps(J4w,J5)));
    _mm_store_ps(SSEData+4*17, _mm_add_ps(_mm_load_ps(SSEData+4*17),_mm_mul_ps(J4w,J6)));


    __m128 J5w = _mm_mul_ps(J5,weight);
    _mm_store_ps(SSEData+4*18, _mm_add_ps(_mm_load_ps(SSEData+4*18),_mm_mul_ps(J5w,J5)));
    _mm_store_ps(SSEData+4*19, _mm_add_ps(_mm_load_ps(SSEData+4*19),_mm_mul_ps(J5w,J6)));


    __m128 J6w = _mm_mul_ps(J6,weight);
    _mm_store_ps(SSEData+4*20, _mm_add_ps(_mm_load_ps(SSEData+4*20),_mm_mul_ps(J6w,J6)));


    //b.noalias() -= J * (res * weight);
    __m128 resw = _mm_mul_ps(res,weight);
    _mm_store_ps(SSEData+4*21, _mm_add_ps(_mm_load_ps(SSEData+4*21),_mm_mul_ps(resw,J1)));
    _mm_store_ps(SSEData+4*22, _mm_add_ps(_mm_load_ps(SSEData+4*22),_mm_mul_ps(resw,J2)));
    _mm_store_ps(SSEData+4*23, _mm_add_ps(_mm_load_ps(SSEData+4*23),_mm_mul_ps(resw,J3)));
    _mm_store_ps(SSEData+4*24, _mm_add_ps(_mm_load_ps(SSEData+4*24),_mm_mul_ps(resw,J4)));
    _mm_store_ps(SSEData+4*25, _mm_add_ps(_mm_load_ps(SSEData+4*25),_mm_mul_ps(resw,J5)));
    _mm_store_ps(SSEData+4*26, _mm_add_ps(_mm_load_ps(SSEData+4*26),_mm_mul_ps(resw,J6)));

    //error += res * res * weight;
    _mm_store_ps(SSEData+4*27, _mm_add_ps(_mm_load_ps(SSEData+4*27),_mm_mul_ps(resw,res)));

    num_constraints += 6;
};

inline void NormalEquationsLeastSquares::update(const Mat61f& J, const float& res, const float& weight) {
    A.noalias() += J * J.transpose() * weight;
    b.noalias() -= J * (res * weight);
    error += res * res * weight;
    num_constraints += 1;
};

}

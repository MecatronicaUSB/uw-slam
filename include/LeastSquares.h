/**
* This file is part of LSD-SLAM, added to UW-SLAM.
* 
* Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
* For more information see <http://vision.in.tum.de/lsdslam> 
*
* Uses the SSE implementation of LSD-SLAM to solve Normal equations.
*
* LSD-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with LSD-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once
#include <Options.h>
#include <opencv2/core.hpp>
#include <Eigen/Core>



namespace uw
{
class NormalEquationsLeastSquares
{

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  

    Mat66f A;
    Mat61f b;

    float error;
    int num_constraints;

    ~NormalEquationsLeastSquares();

    inline void initialize(int max_num_constraints);
    inline void finishNoDivide();
    void finish();
    inline void updateSSE(const __m128 &J1,const __m128 &J2,const __m128 &J3,const __m128 &J4,
  		                const __m128 &J5,const __m128 &J6,const __m128& res, const __m128& weight);
    
    inline void update(const Mat61f& J, const float& res, const float& weight);

private:
    // EIGEN_ALIGN16
    float SSEData[4*28];
};

}
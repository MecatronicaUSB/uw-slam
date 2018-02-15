/**
* This file is part of UW-SLAM.
* 
* Copyright 2018.
* Developed by Fabio Morales,
* Email: fabmoraleshidalgo@gmail.com; GitHub: @fmoralesh
*
* UW-SLAM is free software: you can redistribute it and/or modify
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

#include "Tracker.h"
#include "System.h"

namespace uw
{

Tracker::~Tracker(void) {};

// TODO(GitHub:fmoralesh, fabmoraleshidalgo@gmail.com)
//  02-13-2018 - Implement a faster way to obtain points with gradient superior of the mean threshold of 32x32 px block
void Tracker::GetCandidatePoints(Frame* frame, vector<Point2d> candidatePoints) {
    // frameXGPU.create(frame->data_.size(),CV_32FC1);
    // frameYGPU.create(frame->data_.size(),CV_32FC1);
    // Mat showGPU, frameX, frameY;
    // frameXGPU.upload(frame->data_);
    // frameYGPU.upload(frame->data_);
    // soberX_->apply(frameXGPU, frameXGPU);
    // soberY_->apply(frameYGPU, frameYGPU);
    // cuda::abs(frameXGPU, frameXGPU);
    // cuda::abs(frameYGPU, frameYGPU);
    // cuda::addWeighted(frameXGPU, 0.5, frameYGPU, 0.5, 0, frameGPU);
    
    //Mat showGPU
    double threshold;
    cuda::GpuMat frameGPU(frame->data_);
    // Applying Laplacian filter to the image
    laplacian_->apply(frameGPU, frameGPU);

    // Block size search for high gradient points
    for (int x=0; x<w_; x+=BLOCK_SIZE) {
        for (int y =0; y<h_; y+=BLOCK_SIZE) {
            Scalar mean, stdev;
            Point min_loc, max_loc;
            double min, max;
            cuda::GpuMat block(frameGPU, Rect(x,y,BLOCK_SIZE,BLOCK_SIZE));
            cuda::meanStdDev(block, mean, stdev);
            threshold = mean[0] + GRADIENT_THRESHOLD;
            cuda::minMaxLoc(block, &min, &max, &min_loc, &max_loc);

            if( max > threshold ){
                max_loc.x += x;
                max_loc.y += y;
                this->candidatePoints_.push_back(max_loc);
            }
        }
    }
    //DebugShowCandidatePoints(frame);
}

void Tracker::WarpFunction(){
    cuda::GpuMat a(1,1,CV_32F);
    Mat K = (Mat_<double>(3,3) << 1,1,1,     2,2,2,      3,3,3);
    Mat N = (Mat_<double>(3,1) << 1,2,3);
    cout << N << endl;
    cout << K << endl;
    cuda::GpuMat KGPU;
    cuda::GpuMat NGPU;
    cuda::GpuMat ResultGPU;
    Mat Result;
    KGPU.upload(K);
    NGPU.upload(N);
    cuda::gemm(KGPU, NGPU, 1.0, cuda::GpuMat(), 0.0, ResultGPU);
    //cuda::multiply(KGPU, NGPU, ResultGPU);
    ResultGPU.download(Result);
    cout << Result << endl;
}   


void Tracker::DebugShowCandidatePoints(Frame* frame){
    Mat showPoints;
    cvtColor(frame->data_, showPoints, CV_GRAY2RGB);
    
    for( int i=0; i<this->candidatePoints_.size(); i++ )
        circle(showPoints, candidatePoints_[i], 2, Scalar(255,0,0), 1, 8, 0);

    imshow("Show candidates points", showPoints);
    waitKey(0);
    this->candidatePoints_.clear();
}


}
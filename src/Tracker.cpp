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

// OJO IMPORTANTE
#include <opencv2/core/eigen.hpp>

namespace uw
{

struct CostFunctor {
  template <typename T> bool operator()(const T* const x, T* residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};

Tracker::Tracker() {
    for (Mat K: K_)
        K = Mat(3,3,CV_64FC1, Scalar(0.f));
    for (Mat invK: invK_)
        invK = Mat(3,3,CV_64FC1, Scalar(0.f));
};

Tracker::~Tracker(void) {};

// TODO(GitHub:fmoralesh, fabmoraleshidalgo@gmail.com)
// 02-13-2018 - Implement a faster way to obtain candidate points with high gradient (above of a certain threshold)
void Tracker::GetCandidatePoints(Frame* frame) {
    // frameXGPU.create(frame->data_.size(),CV_32FC1); (double) (double)
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
    cuda::GpuMat frameGPU(frame->image[0]);
    // Applying Laplacian filter to the image
    laplacian_->apply(frameGPU, frameGPU);
    int i = 0;
    // Block size search for high gradient points
    for (int x=0; x<w_[0]; x+=BLOCK_SIZE) {
        for (int y =0; y<h_[0]; y+=BLOCK_SIZE) {
            Mat point = Mat::ones(1,4,CV_64FC1);
            Scalar mean, stdev;
            Point min_loc, max_loc;
            double min, max;
            cuda::GpuMat block(frameGPU, Rect(x,y,BLOCK_SIZE,BLOCK_SIZE));
            cuda::meanStdDev(block, mean, stdev);
            threshold = mean[0] + GRADIENT_THRESHOLD;
            cuda::minMaxLoc(block, &min, &max, &min_loc, &max_loc);

            if( max > threshold ){
                point.at<double>(0,0) = x + max_loc.x;
                point.at<double>(0,1) = y + max_loc.y;
                frame->candidatePoints_.push_back(point);
            }
        }
    }
}

void Tracker::WarpFunction(Frame* previous_frame, Frame* current_frame){

    DebugShowCandidatePoints(previous_frame);


    // cout << "Prev" << endl;
    // cout << previous_frame->candidatePoints_.at<double>(previous_frame->candidatePoints_.rows-1,0) << " "
    //      << previous_frame->candidatePoints_.at<double>(previous_frame->candidatePoints_.rows-1,1) << " "
    //      << previous_frame->candidatePoints_.at<double>(previous_frame->candidatePoints_.rows-1,2) << " "
    //      << previous_frame->candidatePoints_.at<double>(previous_frame->candidatePoints_.rows-1,3) << endl;

    // cout << "Now" << endl;
    // cout << previous_frame->candidatePoints_.at<double>(0,0) << " "
    //      << previous_frame->candidatePoints_.at<double>(0,1) << " "
    //      << previous_frame->candidatePoints_.at<double>(0,2) << " "
    //      << previous_frame->candidatePoints_.at<double>(0,3) << endl;

    Mat rigid = Mat(4,4,CV_64FC1);
    eigen2cv(previous_frame->rigid_body_transformation_, rigid);

    rigid.at<double>(0,3) = 5.0;
    rigid.at<double>(1,3) = 0.0;
    rigid.at<double>(2,3) = 0.0;
    cout << rigid << endl;
    cout << previous_frame->candidatePoints_.row(0) << endl;

    double d = 0.5;

    // DSO - SLAM Warping function 

    previous_frame->candidatePoints_.col(0) *= d;
    previous_frame->candidatePoints_.col(1) *= d;
    previous_frame->candidatePoints_.col(2) *= d;
    
    current_frame->candidatePoints_ = rigid * previous_frame->candidatePoints_.t();

    current_frame->candidatePoints_.row(0) /= current_frame->candidatePoints_.row(2);
    current_frame->candidatePoints_.row(1) /= current_frame->candidatePoints_.row(2);
    current_frame->candidatePoints_.row(2) /= current_frame->candidatePoints_.row(2);
    
    // LSD - SLAM Warping function

    // previous_frame->candidatePoints_.col(0) = ((previous_frame->candidatePoints_.col(0) + cx_[0]) * invfx_[0]) * d;
    // previous_frame->candidatePoints_.col(1) = ((previous_frame->candidatePoints_.col(1) + cy_[0]) * invfy_[0]) * d;
    // previous_frame->candidatePoints_.col(2) = previous_frame->candidatePoints_.col(2) * d;

    // current_frame->candidatePoints_ = rigid * previous_frame->candidatePoints_.t();

    // current_frame->candidatePoints_.row(0) /= current_frame->candidatePoints_.row(2);
    // current_frame->candidatePoints_.row(1) /= current_frame->candidatePoints_.row(2);
    // current_frame->candidatePoints_.row(0) *= fx_[0];
    // current_frame->candidatePoints_.row(1) *= fy_[0];
    // current_frame->candidatePoints_.row(0) -= cx_[0];
    // current_frame->candidatePoints_.row(1) -= cy_[0];
    
    cout << current_frame->candidatePoints_.col(0) << endl;
    // Sophus::SE3d transformation(previous_frame->rigid_body_transformation_);
    

    //cout << previous_frame->rigid_body_transformation_ << endl;
    DebugShowCandidatePoints(current_frame);
    waitKey(0);
    //cuda::gemm(KGPU, NGPU, 1.0, cuda::GpuMat(), 0.0, ResultGPU);
    //cuda::multiply(KGPU, NGPU, ResultGPU);

    previous_frame->candidatePoints_.release();
    current_frame->candidatePoints_.release();

}   

void Tracker::EstimatePose(Frame* previous_frame, Frame* current_frame){
    double initial_x = previous_frame->id;
    double x = initial_x;

    ceres::Problem problem;
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    problem.AddResidualBlock(cost_function, NULL, &x);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    std::cout << "x : " << initial_x
                << " -> " << x << "\n";

}


void Tracker::DebugShowCandidatePoints(Frame* frame){
    Mat showPoints;
    cvtColor(frame->image[0], showPoints, CV_GRAY2RGB);
    
    for( int i=0; i<frame->candidatePoints_.rows; i++) {
        Point2d point;
        point.x = frame->candidatePoints_.at<double>(i,0);
        point.y = frame->candidatePoints_.at<double>(i,1);

        circle(showPoints, point, 2, Scalar(255,0,0), 1, 8, 0);
    }

    imshow("Show candidates points", showPoints);
}

void Tracker::InitializePyramid(int _width, int _height, Mat K) {
    w_[0] = _width;
    h_[0] = _height;
    K_[0] = K;
    invK_[0] = K.inv();

    fx_[0] = K.at<double>(0,0);
    fy_[0] = K.at<double>(1,1);
    cx_[0] = K.at<double>(0,2);
    cy_[0] = K.at<double>(1,2);
    
    invfx_[0] = 1 / fx_[0]; 
    invfy_[0] = 1 / fy_[0]; 
    invcx_[0] = 1 / cx_[0]; 
    invcy_[0] = 1 / cy_[0]; 

    for (int lvl = 1; lvl < PYRAMID_LEVELS; lvl++) {
        w_[lvl] = _width >> lvl;
        h_[lvl] = _height >> lvl;
        
        fx_[lvl] = fx_[lvl-1] * 0.5;
        fy_[lvl] = fy_[lvl-1] * 0.5;
        cx_[lvl] = (cx_[0] + 0.5) / ((int)1<<lvl) - 0.5;
        cy_[lvl] = (cy_[0] + 0.5) / ((int)1<<lvl) - 0.5;

        K_[lvl] = Mat::eye(Size(3,3), CV_64FC1);
        K_[lvl].at<double>(0,0) = fx_[lvl];
        K_[lvl].at<double>(1,1) = fy_[lvl];
        K_[lvl].at<double>(0,2) = cx_[lvl];  
        K_[lvl].at<double>(1,2) = cy_[lvl];    

        invfx_[lvl] = 1 / fx_[lvl];
        invfy_[lvl] = 1 / fy_[lvl];
        invcx_[lvl] = 1 / cx_[lvl];
        invcy_[lvl] = 1 / cy_[lvl];

        invK_[lvl] = K_[lvl].inv();
        // Need review 
        // invfx_[lvl] = invK_[lvl].at<double>(0,0);
        // invfy_[lvl] = invK_[lvl].at<double>(1,1);
        // invcx_[lvl] = invK_[lvl].at<double>(0,2);
        // invcy_[lvl] = invK_[lvl].at<double>(1,2);

    }
}

}
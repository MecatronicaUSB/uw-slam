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

#include "Mapper.h"
#include "System.h"

namespace uw
{
class Frame;

Mapper::Mapper(int _width, int _height, Mat _K) {
    w_[0] = _width;
    h_[0] = _height;
    K_[0] = _K;
    // invK_[0] = _K.inv();

    fx_[0] = _K.at<float>(0,0);
    fy_[0] = _K.at<float>(1,1);
    cx_[0] = _K.at<float>(0,2);
    cy_[0] = _K.at<float>(1,2);
    
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

        K_[lvl] = Mat::eye(Size(3,3), CV_32FC1);
        K_[lvl].at<float>(0,0) = fx_[lvl];
        K_[lvl].at<float>(1,1) = fy_[lvl];
        K_[lvl].at<float>(0,2) = cx_[lvl];  
        K_[lvl].at<float>(1,2) = cy_[lvl];    

        invfx_[lvl] = 1 / fx_[lvl];
        invfy_[lvl] = 1 / fy_[lvl];
        invcx_[lvl] = 1 / cx_[lvl];
        invcy_[lvl] = 1 / cy_[lvl];
        
        // Needs review
        // invK_[lvl] = K_[lvl].inv();
        // invfx_[lvl] = invK_[lvl].at<float>(0,0); 
        // invfy_[lvl] = invK_[lvl].at<float>(1,1);
        // invcx_[lvl] = invK_[lvl].at<float>(0,2);
        // invcy_[lvl] = invK_[lvl].at<float>(1,2);
    }
};

void Mapper::TriangulateCloudPoints(Frame* _previous_frame, Frame* _current_frame) {
    // Transform keypoints to Point2f vectors
    KeyPoint::convert(_previous_frame->keypoints_, _previous_frame->points_);
    KeyPoint::convert(_current_frame->keypoints_, _current_frame->points_);
    

    // Transform 3x4 camera matrix from eigen to Mat
    Mat T1, T2;
    SE3 Identity = SE3(SO3::exp(SE3::Point(0.0, 0.0, 0.0)), SE3::Point(0.0, 0.0, 0.0));
    eigen2cv(Identity.matrix3x4(), T1);
    eigen2cv(previous_world_pose_.matrix3x4(), T2);    
    //eigen2cv(_previous_frame->rigid_transformation_.matrix3x4(), T2);
    
    // Obtain projection matrices for the two perspectives  
    // P = K * [Rotation | translation]
    Mat P1 = K_[0] * T1;
    Mat P2 = K_[0] * T2;

    // Compute depth of 3D points using triangulation
    triangulatePoints(P1, P2, _previous_frame->points_, _current_frame->points_, _previous_frame->map_);
    
};

void Mapper::AddPointCloudFromRGBD(Frame* frame) {

    int num_cloud_points = frame->candidatePoints_[0].rows;

    recent_cloud_points_ = Mat(num_cloud_points, 3, CV_32FC1);

    for (int i=0; i<frame->candidatePoints_[0].rows; i++) {

        recent_cloud_points_.at<float>(i,0) = frame->candidatePoints_[0].at<float>(i,0);
        recent_cloud_points_.at<float>(i,1) = frame->candidatePoints_[0].at<float>(i,1);
        recent_cloud_points_.at<float>(i,2) = frame->candidatePoints_[0].at<float>(i,2);
    
    }
};


}
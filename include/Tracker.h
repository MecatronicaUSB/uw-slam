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

#pragma once
#include "Options.h"

///Basic C and C++ libraries
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <dirent.h>

/// OpenCV libraries. May need review for the final release
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/calib3d.hpp"
#include <opencv2/video.hpp>


/// CUDA specific libraries
#include <opencv2/cudafilters.hpp>
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"

// Namespaces
using namespace cv;
using namespace std;

namespace uw
{
class Frame;

class Map
{
public:
    cuda::GpuMat candidatePoints;
    cuda::GpuMat pointsDepth;    
};

class Tracker
{
public:
    Tracker();
    ~Tracker();
    void InitializePyramid(int _width, int _height, Mat K);
    void EstimatePose(Frame* previous_frame, Frame* current_frame);
    void GetCandidatePoints(Frame* frame, vector<Point2d> candidatePoints);
    void DebugShowCandidatePoints(Frame* frame);


    void WarpFunction();


    // Filters for calculating gradient in images
    Ptr<cuda::Filter> soberX_ = cuda::createSobelFilter(0, CV_32FC1, 1, 0, CV_SCHARR, 1.0, BORDER_DEFAULT);
    Ptr<cuda::Filter> soberY_ = cuda::createSobelFilter(0, CV_32FC1, 0, 1, CV_SCHARR, 1.0, BORDER_DEFAULT);
    Ptr<cuda::Filter> laplacian_ = cuda::createLaplacianFilter(0, 0, 1, 1.0);

    vector<int> w_ = vector<int>(PYRAMID_LEVELS);
    vector<int> h_ = vector<int>(PYRAMID_LEVELS);

    vector<float> fx_ = vector<float>(PYRAMID_LEVELS);
    vector<float> fy_ = vector<float>(PYRAMID_LEVELS);
    vector<float> cx_ = vector<float>(PYRAMID_LEVELS);
    vector<float> cy_ = vector<float>(PYRAMID_LEVELS);
    vector<float> invfx_ = vector<float>(PYRAMID_LEVELS);
    vector<float> invfy_ = vector<float>(PYRAMID_LEVELS);
    vector<float> invcx_ = vector<float>(PYRAMID_LEVELS);
    vector<float> invcy_ = vector<float>(PYRAMID_LEVELS);

    vector<Mat> K_ = vector<Mat>(PYRAMID_LEVELS);
    vector<Mat> invK_ = vector<Mat>(PYRAMID_LEVELS);
};



}
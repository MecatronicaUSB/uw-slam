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
#include <Options.h>
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

// Ceres library
#include "ceres/ceres.h"

// Eigen library
#include <eigen3/Eigen/Core>

// Sophus
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

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
    /**
     * @brief Tracker consturctor.
     * 
     */
    Tracker();

    /**
     * @brief Tracker destructor.
     * 
     */
    ~Tracker();

    /**
     * @brief Obtains camera instrinsic matrix and parameters for each pyramid level available.
     * 
     * @param _width    Width of images at finest level.
     * @param _height   Height of images at finest level.
     * @param K         Camera intrinsic matrix at finest level.
     */
    void InitializePyramid(int _width, int _height, Mat K);

    /**
     * @brief Computes optimal transformation given two input frames.
     * 
     * @param previous_frame 
     * @param current_frame 
     */
    void EstimatePose(Frame* previous_frame, Frame* current_frame);

    /**
     * @brief Computes gradient of a frame for each pyramid level available.
     *        Saves the result gradient images within the frame class.
     * 
     * @param frame 
     */
    void ApplyGradient(Frame* frame);

    /**
     * @brief Obtains candidate points from grandient images of a frame for each pyramid level available.
     * 
     * @param frame 
     */
    void ObtainCandidatePoints(Frame* frame);

    /**
     * @brief Computes warp projected points from one frame to another, given a rigid transformation matrix 
     *        and the depth estimation of those points. Returns matrix of warped points.
     * 
     * @param points2warp 
     * @param depth 
     * @param rigid_transformation 
     * @return Mat 
     */
    Mat WarpFunction(Mat points2warp, Mat depth, Mat44 rigid_transformation);

    /**
     * @brief Shows points in an image. Used only for debbugin.
     * 
     * @param image 
     * @param candidatePoints 
     */
    void DebugShowCandidatePoints(Mat image, Mat candidatePoints);

    // Filters for calculating gradient in images
    Ptr<cuda::Filter> soberX_ = cuda::createSobelFilter(0, CV_32FC1, 1, 0, CV_SCHARR, 1.0, BORDER_DEFAULT);
    Ptr<cuda::Filter> soberY_ = cuda::createSobelFilter(0, CV_32FC1, 0, 1, CV_SCHARR, 1.0, BORDER_DEFAULT);
    Ptr<cuda::Filter> laplacian_ = cuda::createLaplacianFilter(0, 0, 1, 1.0);

    // Width and height of images for each pyramid level available
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
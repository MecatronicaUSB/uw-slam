/**
* This file is part of UW-SLAM.
* 
* Copyright 2018.
* Developed by Fabio Morales,
* If you use this code, please cite the respective publications as
* listed on the above website.
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

class CameraModel
{
public:
    /**
     * @brief Destructor of CameraModel
     * 
     */
	~CameraModel();

	/**
	 * Creates an CameraModel by reading the distortion parameters from a file.
	 * 
	 * The file format is as follows:
	 * fx fy cx cy d1 d2 d3 d4 d5 d6
	 * inputWidth inputHeight
	 * crop / full / none
	 * outputWidth outputHeight
	 */
    void getCameraModel(string calibrationPath);
    
	/**
	 * Undistorts the given image and returns the result image.
	 */
	void undistort(const cv::Mat& image, cv::OutputArray result) const;
	
	/**
	 * Returns the intrinsic parameter matrix of the undistorted images.
	 */
	const cv::Mat& getK() const;
	
	/**
	 * Returns the intrinsic parameter matrix of the original images,
	 */
	const cv::Mat& getOriginalK() const;
	
    /**
	 * Returns the map1 computed for undistortion.
	 */
	const cv::Mat&  getMap1() const;
	
    /**
	 * Returns the map1 computed for undistortion.
	 */
	const cv::Mat&  getMap2() const;

	/**
	 * Returns the width of the undistorted images in pixels.
	 */
	int getOutputWidth() const;

	/**
	 * Returns the height of the undistorted images in pixels.
	 */
	int getOutputHeight() const;
	
	/**
	 * Returns the width of the input images in pixels.
	 */
	int getInputWidth() const;

	/**
	 * Returns the height of the input images in pixels.
	 */
	int getInputHeight() const;

	/**
	 * Returns if the undistorter was initialized successfully.
	 */
	bool isValid() const;

private:
    Mat K_;
    Mat originalK_ = cv::Mat(3, 3, CV_64F, cv::Scalar(0));

    float inputCalibration[4];
    Mat distCoeffs = cv::Mat::zeros(4, 1, CV_32F);

    int out_width, out_height;
	int in_width, in_height;
	cv::Mat map1, map2;

    bool valid;
};



}
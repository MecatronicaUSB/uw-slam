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
#include <string>
#include <dirent.h>

/// OpenCV libraries. 
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/calib3d.hpp"

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
	 * @brief Creates an CameraModel by reading the distortion parameters from a file
	 * 			Please refer to calibration.xml file to see the format
	 * @param calibrationPath String with calibration .xml file
	 */
    void GetCameraModel(string calibrationPath);
    
	/**
	 * @brief Undistorts the given image and returns the result image
	 * 
	 * @param image Image to undistorts
	 * @param result Result image
	 */
	void Undistort(const cv::Mat& image, cv::OutputArray result) const;
	
	/**
	 * @brief Returns the intrinsic parameter matrix of the undistorted images
	 * 
	 * @return const cv::Mat& getK Intrinsic parameter matrix of undistorted images
	 */
	const cv::Mat& GetK() const;
	
	/**
	 * @brief Returns the intrinsic parameter matrix of the original images
	 * 
	 * @return const cv::Mat& getOriginalK Intrinsic parameter matrix of distorted images
	 */
	const cv::Mat& GetOriginalK() const;
	
	/**
	 * @brief Returns the map1 computed for undistortion
	 * 
	 * @return const cv::Mat& getMap1 Map1 computed for undistortion
	 */
	const cv::Mat& GetMap1() const;
	
	/**
	 * @brief Returns the map2 computed for undistortion
	 * 
	 * @return const cv::Mat& getMap2 Map2 computed for undistortion
	 */
	const cv::Mat& GetMap2() const;

	/**
	 * @brief Returns the width of the undistorted images in pixels
	 * 
	 * @return int getOutputWidth 
	 */
	int GetOutputWidth() const;


	/**
	 * @brief Returns the height of the undistorted images in pixels
	 * 
	 * @return int getOutputHeight 
	 */
	int GetOutputHeight() const;
	
	/**
	 * @brief Returns the width of the input images in pixels.
	 * 
	 * @return int getInputWidth 
	 */
	int GetInputWidth() const;

	/**
	 * @brief Returns the height of the input images in pixels.
	 * 
	 * @return int getInputHeight 
	 */
	int GetInputHeight() const;

	/**
	 * @brief Returns if the undistorter was initialized successfully.
	 * 
	 * @return true 	Rectification was made
	 * @return false 	Rectification wasn't made
	 */
	bool IsValid() const;

private:
    Mat output_intrinsic_camera_;
    Mat original_intrinsic_camera_ = Mat(3, 3, CV_64FC1, Scalar(0));

    float input_calibration_[4];
    Mat dist_coeffs_ = Mat(4, 1, CV_64FC1, Scalar(0));

    int out_width_, out_height_;
	int in_width_, in_height_;
	cv::Mat map1_, map2_;

    bool valid_;
};



}
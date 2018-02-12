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

#include "CameraModel.h"

namespace uw
{

CameraModel::~CameraModel()
{
}

// Checks which Camera Model Implement (RadTan,...?)
void CameraModel::getCameraModel(string calibrationPath)
{
	valid = true;

    // Reading intrinsic parameters and distortion coefficients from file
    Mat calibration_values, distortion_values;
    FileStorage opencv_file(calibrationPath, cv::FileStorage::READ);
    if (opencv_file.isOpened()){
        cout << " ... found" << endl;
        opencv_file["in_width"] >> in_width;
        opencv_file["in_height"] >> in_height;
        opencv_file["out_width"] >> out_width;
        opencv_file["out_height"] >> out_height;
        opencv_file["calibration_values"] >> calibration_values;
        opencv_file["rectification"] >> distortion_values;
        opencv_file.release();
    }
    else{
        cout << " ... not found" << endl;
        cout << "Cannot operate without calibration" << endl;
        cout << "Exiting..." << endl;
        valid = false;
        exit(0);
    }

    // Saving parameters and distCoeffs
    for(int i = 0; i < 4; i++){
        inputCalibration[i] = calibration_values.at<double>(0,i);
		distCoeffs.at<float>(i,0) = distortion_values.at<double>(0,i);
    }

    // Checking if the intrinsic parameters needs rescaling
    if( inputCalibration[2] < 1 && inputCalibration[3] < 1){
        cout << "WARNING: cx = " << inputCalibration[2] << " < 1, which should not be the case for normal cameras" << endl;
        // Rescale. (Maybe will need -0.5 offset)      
        inputCalibration[0] = inputCalibration[0] * in_width;
        inputCalibration[1] = inputCalibration[1] * in_height;
        inputCalibration[2] = inputCalibration[2] * in_width;
        inputCalibration[3] = inputCalibration[3] * in_height;
    }

    // Saving parameters in originalK_
    originalK_.at<double>(0,0) = inputCalibration[0];
    originalK_.at<double>(1,1) = inputCalibration[1];
    originalK_.at<double>(0,2) = inputCalibration[2];
    originalK_.at<double>(1,2) = inputCalibration[3];
    originalK_.at<double>(2, 2) = 1;

    // If distCoeff are 0, dont apply rectification
    if( distCoeffs.at<float>(0,0) == 0 ){
        cout << "Distortion coefficients not found ... not rectifying" << endl;
        valid = false;
        K_ = originalK_;
    }
    if(valid){
        cout << "Distortion coefficients found ... rectifying" << endl;
        // Obtaining new Camera Matrix with outputs and inputs
        K_ = getOptimalNewCameraMatrix(originalK_, distCoeffs, cv::Size(in_width, in_height), 0, cv::Size(out_width, out_height), nullptr, false);
        initUndistortRectifyMap(originalK_, distCoeffs, cv::Mat(), K_, cv::Size(out_width, out_height), CV_16SC2, map1, map2);
        
        originalK_.at<double>(0, 0) /= in_width;
		originalK_.at<double>(0, 2) /= in_width;
		originalK_.at<double>(1, 1) /= in_height;
		originalK_.at<double>(1, 2) /= in_height;
    }
}

void CameraModel::undistort(const cv::Mat& image, cv::OutputArray result) const
{
	cv::remap(image, result, map1, map2, cv::INTER_LINEAR);
}

const cv::Mat& CameraModel::getMap1() const
{
    return map1;
}

const cv::Mat& CameraModel::getMap2() const
{
    return map2;
}

const cv::Mat& CameraModel::getK() const
{
	return K_;
}

const cv::Mat& CameraModel::getOriginalK() const
{
	return originalK_;
}

int CameraModel::getOutputWidth() const
{
	return out_width;
}

int CameraModel::getOutputHeight() const
{
	return out_height;
}
int CameraModel::getInputWidth() const
{
	return in_width;
}

int CameraModel::getInputHeight() const
{
	return in_height;
}

bool CameraModel::isValid() const
{
	return valid;
}

}
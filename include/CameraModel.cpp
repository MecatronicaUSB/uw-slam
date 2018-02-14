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

#include "CameraModel.h"

namespace uw
{

CameraModel::CameraModel() {}

CameraModel::~CameraModel() {}

// TODO(GitHub:fmoralesh, fabmoraleshidalgo@gmail.com) 02-13-2018 - Implement other camera models to UWSLAM (FOV)
void CameraModel::GetCameraModel(string calibration_path) {
    // Reading intrinsic parameters and distortion coefficients from file
    Mat calibration_values, distortion_values;
    FileStorage opencv_file(calibration_path, cv::FileStorage::READ);
    if (opencv_file.isOpened()) {
        cout << " ... found" << endl;
        opencv_file["in_width"] >> in_width_;
        opencv_file["in_height"] >> in_height_;
        opencv_file["out_width"] >> out_width_;
        opencv_file["out_height"] >> out_height_;
        opencv_file["calibration_values"] >> calibration_values;
        opencv_file["rectification"] >> distortion_values;
        opencv_file.release();
    } else {
        cout << " ... not found" << endl;
        cout << "Cannot operate without calibration" << endl;
        cout << "Exiting..." << endl;
        valid_ = false;
        exit(0);
    }

    // Saving parameters and distCoeffs
    for (int i = 0; i < 4; i++) {
        input_calibration_[i] = calibration_values.at<double>(0,i);
		dist_coeffs_.at<float>(i,0) = distortion_values.at<double>(0,i);
    }

    // Checking if the intrinsic parameters needs rescaling
    if (input_calibration_[2] < 1 && input_calibration_[3] < 1) {
        cout << "WARNING: cx = " << input_calibration_[2] << " < 1, which should not be the case for normal cameras" << endl;
        // Rescale. (Maybe will need -0.5 offset)      
        input_calibration_[0] = input_calibration_[0] * in_width_;
        input_calibration_[1] = input_calibration_[1] * in_height_;
        input_calibration_[2] = input_calibration_[2] * in_width_;
        input_calibration_[3] = input_calibration_[3] * in_height_;
    }

    // Saving parameters in original_intrinsic_camera_
    original_intrinsic_camera_.at<double>(0,0) = input_calibration_[0];
    original_intrinsic_camera_.at<double>(1,1) = input_calibration_[1];
    original_intrinsic_camera_.at<double>(0,2) = input_calibration_[2];
    original_intrinsic_camera_.at<double>(1,2) = input_calibration_[3];
    original_intrinsic_camera_.at<double>(2, 2) = 1;

    // If distCoeff are 0, dont apply rectification
    if (dist_coeffs_.at<float>(0,0) == 0) {
        cout << "Distortion coefficients not found ... not rectifying" << endl;
        valid_ = false;
        output_intrinsic_camera_ = original_intrinsic_camera_;
    }
    if (valid_) {
        cout << "Distortion coefficients found ... rectifying" << endl;
        // Obtaining new intrinsic camera matrix with undistorted images
        output_intrinsic_camera_ = getOptimalNewCameraMatrix(original_intrinsic_camera_, dist_coeffs_, cv::Size(in_width_, in_height_), 0, cv::Size(out_width_, out_height_), nullptr, false);
        initUndistortRectifyMap(original_intrinsic_camera_, dist_coeffs_, cv::Mat(), output_intrinsic_camera_, cv::Size(out_width_, out_height_), CV_16SC2, map1_, map2_);
        
        original_intrinsic_camera_.at<double>(0, 0) /= in_width_;
		original_intrinsic_camera_.at<double>(0, 2) /= in_width_;
		original_intrinsic_camera_.at<double>(1, 1) /= in_height_;
		original_intrinsic_camera_.at<double>(1, 2) /= in_height_;
    }
}

void CameraModel::Undistort(const cv::Mat& image, cv::OutputArray result) const {
	cv::remap(image, result, map1_, map2_, cv::INTER_LINEAR);
}

const cv::Mat& CameraModel::GetMap1() const {
    return map1_;
}

const cv::Mat& CameraModel::GetMap2() const {
    return map2_;
}

const cv::Mat& CameraModel::GetK() const {
	return output_intrinsic_camera_;
}

const cv::Mat& CameraModel::GetOriginalK() const {
	return original_intrinsic_camera_;
}

int CameraModel::GetOutputWidth() const {
	return out_width_;
}

int CameraModel::GetOutputHeight() const {
	return out_height_;
}

int CameraModel::GetInputWidth() const {
	return in_width_;
}

int CameraModel::GetInputHeight() const {
	return in_height_;
}

bool CameraModel::IsValid() const {
	return valid_;
}

}
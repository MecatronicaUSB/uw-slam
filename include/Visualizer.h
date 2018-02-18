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

///Basic C and C++ libraries
#include <stdlib.h>
#include <fstream> 
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

// ROS libraries
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

// Namespaces
using namespace std;

namespace uw
{


class Visualizer
{
public:
    Visualizer();
    ~Visualizer();

    void ReadGroundTruthEUROC(string groundtruth_path);
    void ReadTimeStamps();
    void SendVisualization();

    ros::NodeHandle nodehandle_camera_pose_;
    ros::Publisher publisher_camera_pose_;
    visualization_msgs::Marker camera_pose_;

    
    // image_transport::Publisher publisher_current_frame;
    vector<vector<double> > ground_truth_poses_;

};

}
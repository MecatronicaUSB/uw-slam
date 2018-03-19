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
#include <System.h>

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

class System;

class Visualizer
{
public:
    /**
     * @brief Constructor of visualizer. 
     *        Initializes all nodes, publishers and options necessary to send and visualize
     *        markers, images and points in Rviz (ROS Kinetic). Also, if given, reads 
     *        and initialize rigid transformation of ground truth data.
     *        
     * @param start_index_ 
     * @param num_images 
     * @param ground_truth_path 
     */
    Visualizer(int start_index_, int num_images, string _ground_truth_dataset, string ground_truth_path);

    /**
     * @brief Destructor of visualizer.
     * 
     */
    ~Visualizer();

    /**
     * @brief Sends messages of pose and frame to Rviz
     * 
     * @param frame 
     */
    void UpdateMessages(Frame* frame);

    /**
     * @brief Reads ground truth poses from a .csv file with the EUROC format.
     *        EUROC format of .csv file (timestamp; translation components; quaternion rotation):
     *        #timestamp [ns],  p_RS_R_x [m],  p_RS_R_y [m],  p_RS_R_z [m],  q_RS_w [m],  q_RS_x [m],  q_RS_y [m],  q_RS_z [m]
     * 
     * @param start_index 
     * @param groundtruth_path 
     */
    void ReadGroundTruthEUROC(int start_index, string groundtruth_path);

    void ReadGroundTruthTUM(int start_index, string groundtruth_path);
    

    // Ground-Truth publishers and markers
    ros::Publisher publisher_gt_pose_;
    ros::Publisher publisher_gt_trajectory_dots_;    
    ros::Publisher publisher_gt_trajectory_lines_;   
    visualization_msgs::Marker gt_pose_;
    visualization_msgs::Marker gt_trajectory_dots_;
    visualization_msgs::Marker gt_trajectory_lines_;
    
    ros::Publisher publisher_camera_pose_;
    ros::Publisher publisher_camera_trajectory_dots_;    
    ros::Publisher publisher_camera_trajectory_lines_;
    visualization_msgs::Marker camera_pose_;
    visualization_msgs::Marker camera_trajectory_dots_;
    visualization_msgs::Marker camera_trajectory_lines_;
    

    image_transport::Publisher publisher_current_frame_;

    string ground_truth_path_;
    string ground_truth_dataset_;   
    vector<vector<double> > ground_truth_poses_;
    int num_images_, num_ground_truth_poses_;
    int ground_truth_step_;
    int ground_truth_index_;

    bool use_ground_truth_;
};

}
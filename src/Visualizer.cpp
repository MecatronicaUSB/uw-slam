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

#include <Visualizer.h>

namespace uw
{

Visualizer::Visualizer(){

    ros::NodeHandle nodehandle_camera_pose;
    ros::Publisher publisher_camera_pose = nodehandle_camera_pose.advertise<visualization_msgs::Marker>("camera_pose", 100);
    visualization_msgs::Marker camera_pose;
    
    camera_pose.header.frame_id = "/camera_pose";            // Set the frame ID and timestamp. See the TF tutorials for information on these.
    camera_pose.header.stamp = ros::Time::now();
    camera_pose.ns = "uw_slam";                              // Set the namespace and id for this camera_pose.  This serves to create a unique ID. Any camera_pose. sent with the same namespace and id will overwrite the old one
    camera_pose.type = visualization_msgs::Marker::CUBE;     // Set the camera_pose. type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    camera_pose.action = visualization_msgs::Marker::ADD;    // Set the camera_pose. action.  Options are ADD, DELETE and DELETEALL
    camera_pose.pose.position.x = 0;  
    camera_pose.pose.position.y = 0;
    camera_pose.pose.position.z = 1;
    camera_pose.pose.orientation.x = 0.0;                    // Orientation of camera_pose.
    camera_pose.pose.orientation.y = 0.0;    
    camera_pose.pose.orientation.z = 0.0;
    camera_pose.pose.orientation.w = 1.0;
    camera_pose.scale.x = 0.025;                              // Set the scale of the camera_pose. -- 1x1x1 here means 1m on a side
    camera_pose.scale.y = 0.35;
    camera_pose.scale.z = 0.25;
    camera_pose.color.r = 0.12f;                             // Set the color -- set alpha to something non-zero!
    camera_pose.color.g = 0.56f;
    camera_pose.color.b = 1.0f;
    camera_pose.color.a = 1.0;
    camera_pose.lifetime = ros::Duration();                  // Life spam of camera_pose.  

    nodehandle_camera_pose_ = nodehandle_camera_pose;
    publisher_camera_pose_ = publisher_camera_pose;
    camera_pose_ = camera_pose;
    publisher_camera_pose_.publish(camera_pose_);
    ros::spinOnce();

};

void Visualizer::SendVisualization(){
    camera_pose_.pose.position.x = 0.5;
    camera_pose_.pose.position.y = 2;
    camera_pose_.pose.position.z = 1;

    publisher_camera_pose_.publish(camera_pose_);
    ros::spinOnce();

};

void Visualizer::ReadGroundTruthEUROC(string groundtruth_path) {
    string delimiter = ",";
    string line = "";
    ifstream file("/home/fabio/Documents/datasets/EUROC/V1_02_medium/mav0/state_groundtruth_estimate0/data.csv");
    if (!file.is_open()) {
        cerr << "Could not read file " << groundtruth_path << "\n";
        cerr << "Exiting.." << endl;
        return;
    }
    getline(file, line);
    while (getline(file, line)) {
        vector<double> timestamp_values;
        stringstream iss(line);
        string val;
        getline(iss, val, ',');
        for (int i=0; i<7; i++) {
            string val;
            getline(iss, val, ',');
            timestamp_values.push_back(stod(val));
        }
        ground_truth_poses_.push_back(timestamp_values);
    }
    file.close();
    int size = ground_truth_poses_.size();
}



}
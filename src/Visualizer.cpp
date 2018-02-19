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

Visualizer::Visualizer(int start_index, int num_images){
    ground_truth_index = start_index;
    ros::NodeHandle nodehandle_camera_pose;
    ros::Publisher publisher_camera_pose = nodehandle_camera_pose.advertise<visualization_msgs::Marker>("camera_pose", 2000);
    visualization_msgs::Marker camera_pose;
    
    camera_pose.header.frame_id = "/camera_pose";            // Set the frame ID and timestamp. See the TF tutorials for information on these.
    camera_pose.header.stamp = ros::Time::now();
    camera_pose.ns = "uw_slam";                              // Set the namespace and id for this camera_pose.  This serves to create a unique ID. Any camera_pose. sent with the same namespace and id will overwrite the old one
    camera_pose.type = visualization_msgs::Marker::CUBE;     // Set the camera_pose. type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    camera_pose.action = visualization_msgs::Marker::ADD;    // Set the camera_pose. action.  Options are ADD, DELETE and DELETEALL
    camera_pose.pose.position.x = 0;  
    camera_pose.pose.position.y = 0;
    camera_pose.pose.position.z = 0;
    camera_pose.pose.orientation.x = 0;                    // Orientation of camera_pose.
    camera_pose.pose.orientation.y = 0;    
    camera_pose.pose.orientation.z = 0;
    camera_pose.pose.orientation.w = 1;
    camera_pose.scale.x = 0.20;                              // Set the scale of the camera_pose. -- 1x1x1 here means 1m on a side
    camera_pose.scale.y = 0.3;
    camera_pose.scale.z = 0.025;
    camera_pose.color.r = 0.12f;                             // Set the color -- set alpha to something non-zero!
    camera_pose.color.g = 0.56f;
    camera_pose.color.b = 1.0f;
    camera_pose.color.a = 1.0;
    camera_pose.lifetime = ros::Duration();                  // Life spam of camera_pose.  

    nodehandle_camera_pose_ = nodehandle_camera_pose;
    publisher_camera_pose_ = publisher_camera_pose;
    camera_pose_ = camera_pose;
    num_images_ = num_images;
    
};

void Visualizer::SendVisualization(Mat image){
    ros::Rate r(3);
    ros::NodeHandle nh_current_frame;
    ros::NodeHandle node;
    image_transport::ImageTransport node_current_frame(nh_current_frame);
    image_transport::Publisher publisher_current_frame = node_current_frame.advertise("current_frame",1);

    sensor_msgs::ImagePtr current_frame = cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();

    camera_pose_.pose.position.x = ground_truth_poses_[ground_truth_index][0];
    camera_pose_.pose.position.y = ground_truth_poses_[ground_truth_index][1];
    camera_pose_.pose.position.z = ground_truth_poses_[ground_truth_index][2];
    camera_pose_.pose.orientation.x = ground_truth_poses_[ground_truth_index][4];           // Orientation of camera_pose.
    camera_pose_.pose.orientation.y = ground_truth_poses_[ground_truth_index][5];        
    camera_pose_.pose.orientation.z = ground_truth_poses_[ground_truth_index][6];      
    camera_pose_.pose.orientation.w = ground_truth_poses_[ground_truth_index][3]; 
    ground_truth_index += ground_truth_step;
    r.sleep();
    // Publish the marker
    while (publisher_camera_pose_.getNumSubscribers() < 1 && publisher_current_frame.getNumSubscribers() < 1) {
        if (!ros::ok()) {
            cout << "ROS core interrupted" << endl;
            cout << "Exiting..." << endl;
            exit(0); 
        }
        ROS_WARN_ONCE("Please create a subscriber to the marker/image");
        sleep(5);
    }
    r.sleep();

    publisher_camera_pose_.publish(camera_pose_);
    publisher_current_frame.publish(current_frame);
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
    num_ground_truth_poses_ = ground_truth_poses_.size();
    ground_truth_step = num_ground_truth_poses_ / num_images_ ;
    ground_truth_index *= ground_truth_step;
}



}
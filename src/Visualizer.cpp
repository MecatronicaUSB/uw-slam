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

Visualizer::Visualizer(int start_index, int num_images, string ground_truth_path){
    
    use_ground_truth = false;
    ground_truth_index = 0;
    num_images_ = num_images;

    if (not (ground_truth_path == "")){
        use_ground_truth = true;
        ground_truth_index = start_index + 120; 
        ReadGroundTruthEUROC(ground_truth_path);
    }

    ros::NodeHandle nodehandle_camera_pose;
    ros::Publisher publisher_camera_pose = nodehandle_camera_pose.advertise<visualization_msgs::Marker>("camera_pose", 50);
    visualization_msgs::Marker ground_truth_pose;
    

    ros::NodeHandle nh_current_frame;
    image_transport::ImageTransport node_current_frame(nh_current_frame);
    publisher_current_frame_ = node_current_frame.advertise("current_frame",50);

    ground_truth_pose.header.frame_id = "/camera_pose";            // Set the frame ID and timestamp. See the TF tutorials for information on these.
    ground_truth_pose.header.stamp = ros::Time::now();
    ground_truth_pose.ns = "uw_slam";                              // Set the namespace and id for this camera_pose.  This serves to create a unique ID. Any camera_pose. sent with the same namespace and id will overwrite the old one
    ground_truth_pose.type = visualization_msgs::Marker::ARROW;     // Set the camera_pose. type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    ground_truth_pose.action = visualization_msgs::Marker::ADD;    // Set the camera_pose. action.  Options are ADD, DELETE and DELETEALL
    ground_truth_pose.pose.position.x = 0;  
    ground_truth_pose.pose.position.y = 0;
    ground_truth_pose.pose.position.z = 0;
    ground_truth_pose.pose.orientation.x = 0;                    // Orientation of camera_pose.
    ground_truth_pose.pose.orientation.y = 0;    
    ground_truth_pose.pose.orientation.z = 0;
    ground_truth_pose.pose.orientation.w = 1;
    ground_truth_pose.scale.x = 0.1;                              // Set the scale of the camera_pose. -- 1x1x1 here means 1m on a side
    ground_truth_pose.scale.y = 0.15;
    ground_truth_pose.scale.z = 0.15;
    ground_truth_pose.color.r = 0.12f;                             // Set the color -- set alpha to something non-zero!
    ground_truth_pose.color.g = 0.56f;
    ground_truth_pose.color.b = 1.0f;
    ground_truth_pose.color.a = 1.0;
    ground_truth_pose.lifetime = ros::Duration();                  // Life spam of camera_pose.  


    publisher_camera_pose_ = publisher_camera_pose;
    ground_truth_pose_ = ground_truth_pose;
    
};

void Visualizer::SendVisualization(Mat image){

    ros::Rate r(60);
    sensor_msgs::ImagePtr current_frame = cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();

    ground_truth_pose_.pose.position.x = ground_truth_poses_[ground_truth_index][0];
    ground_truth_pose_.pose.position.y = ground_truth_poses_[ground_truth_index][1];
    ground_truth_pose_.pose.position.z = ground_truth_poses_[ground_truth_index][2];
    ground_truth_pose_.pose.orientation.x = ground_truth_poses_[ground_truth_index][4];           // Orientation of camera_pose.
    ground_truth_pose_.pose.orientation.y = ground_truth_poses_[ground_truth_index][5];        
    ground_truth_pose_.pose.orientation.z = ground_truth_poses_[ground_truth_index][6];      
    ground_truth_pose_.pose.orientation.w = ground_truth_poses_[ground_truth_index][3]; 
    ground_truth_index += ground_truth_step;

    // Publish the marker
    while (publisher_camera_pose_.getNumSubscribers() < 1 && publisher_current_frame_.getNumSubscribers() < 1) {
        if (!ros::ok()) {
            cout << "ROS core interrupted" << endl;
            cout << "Exiting..." << endl;
            exit(0); 
        }
        ROS_WARN_ONCE("Please create a subscriber to the marker/image");
        //sleep(5);
    }
    r.sleep();

    publisher_camera_pose_.publish(ground_truth_pose_);
    publisher_current_frame_.publish(current_frame);
    ros::spinOnce();

};

void Visualizer::ReadGroundTruthEUROC(string groundtruth_path) {
    string delimiter = ",";
    string line = "";
    ifstream file(groundtruth_path);
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
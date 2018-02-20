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
    
    use_ground_truth_ = false;
    num_images_ = num_images;

    // Camera pose marker initialization
    ros::NodeHandle nodehandle_camera_pose;
    ros::Publisher publisher_camera_pose = nodehandle_camera_pose.advertise<visualization_msgs::Marker>("camera_pose", 50);
    visualization_msgs::Marker camera_pose;

    // Camera pose marker options
    camera_pose.id = 1;
    camera_pose.header.frame_id = "/world";           
    camera_pose.header.stamp = ros::Time::now();
    camera_pose.ns = "uw_slam";                        
    camera_pose.type = visualization_msgs::Marker::ARROW;   
    camera_pose.action = visualization_msgs::Marker::ADD;
    camera_pose.lifetime = ros::Duration();
    // Dimentions of ground truth marker   
    camera_pose.scale.x = 0.1;                              
    camera_pose.scale.y = 0.15;
    camera_pose.scale.z = 0.15;
    // Color of ground truth marker
    camera_pose.color.r = 0.56f;                             
    camera_pose.color.g = 0.12f;
    camera_pose.color.b = 1.0f;
    camera_pose.color.a = 1.0;
                  
    // If ground truth is used
    if (not (ground_truth_path == "")) {
        use_ground_truth_ = true;
        ReadGroundTruthEUROC(start_index, ground_truth_path);

        // Ground truth marker initialization
        ros::NodeHandle nodehandle_ground_truth_pose;
        ros::Publisher publisher_ground_truth_pose = nodehandle_ground_truth_pose.advertise<visualization_msgs::Marker>("ground_truth", 50);
        visualization_msgs::Marker ground_truth_pose;

        // Ground truth marker options
        ground_truth_pose.id = 0;
        ground_truth_pose.header.frame_id = "/world";            
        ground_truth_pose.header.stamp = ros::Time::now();
        ground_truth_pose.ns = "uw_slam";                                                   
        ground_truth_pose.type = visualization_msgs::Marker::ARROW;                         
        ground_truth_pose.action = visualization_msgs::Marker::ADD;
        ground_truth_pose.lifetime = ros::Duration();    
        // Dimentions of ground truth marker   
        ground_truth_pose.scale.x = 0.1;                             
        ground_truth_pose.scale.y = 0.15;
        ground_truth_pose.scale.z = 0.15;
        // Color of ground truth marker
        ground_truth_pose.color.r = 0.12f;                             
        ground_truth_pose.color.g = 0.56f;
        ground_truth_pose.color.b = 1.0f;
        ground_truth_pose.color.a = 1.0;
            
        // Initialize the camera pose marker in the same place and orientation as the ground truth marker
        // This initialization point depends of the start_index argument in SLAM process

        // Initial position and orientation of camera pose marker   
        camera_pose.pose.position.x = ground_truth_poses_[ground_truth_index_][0];  
        camera_pose.pose.position.y = ground_truth_poses_[ground_truth_index_][1];
        camera_pose.pose.position.z = ground_truth_poses_[ground_truth_index_][2];
        camera_pose.pose.orientation.x = ground_truth_poses_[ground_truth_index_][4];
        camera_pose.pose.orientation.y = ground_truth_poses_[ground_truth_index_][5];    
        camera_pose.pose.orientation.z = ground_truth_poses_[ground_truth_index_][6]; 
        camera_pose.pose.orientation.w = ground_truth_poses_[ground_truth_index_][3]; 

        // Initial position and orientation of ground truth marker   
        ground_truth_pose.pose.position.x = ground_truth_poses_[ground_truth_index_][0];    
        ground_truth_pose.pose.position.y = ground_truth_poses_[ground_truth_index_][1];
        ground_truth_pose.pose.position.z = ground_truth_poses_[ground_truth_index_][2];
        ground_truth_pose.pose.orientation.x = ground_truth_poses_[ground_truth_index_][4]; 
        ground_truth_pose.pose.orientation.y = ground_truth_poses_[ground_truth_index_][5];    
        ground_truth_pose.pose.orientation.z = ground_truth_poses_[ground_truth_index_][6]; 
        ground_truth_pose.pose.orientation.w = ground_truth_poses_[ground_truth_index_][3]; 

        // Saving ground truth marker configuration
        publisher_ground_truth_pose_ = publisher_ground_truth_pose;
        ground_truth_pose_ = ground_truth_pose;

    } else {
        // Initial position and orientation of camera pose marker (origin)
        camera_pose.pose.position.x = 0;  
        camera_pose.pose.position.y = 0;
        camera_pose.pose.position.z = 0;
        camera_pose.pose.orientation.x = 0;
        camera_pose.pose.orientation.y = 0;    
        camera_pose.pose.orientation.z = 0; 
        camera_pose.pose.orientation.w = 1; 
    }

    // Initializing image marker for image visualization
    ros::NodeHandle nh_current_frame;
    image_transport::ImageTransport node_current_frame(nh_current_frame);
    publisher_current_frame_ = node_current_frame.advertise("current_frame",50);
    
    // Saving camera pose marker configuration
    publisher_camera_pose_ = publisher_camera_pose;
    camera_pose_ = camera_pose;
};

void Visualizer::UpdateMessages(Mat image){
    // Rate (Hz) of publishing messages
    ros::Rate r(40);

    // Update image message
    sensor_msgs::ImagePtr current_frame = cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();

    // Update ground truth marker position
    if (use_ground_truth_) {
        ground_truth_pose_.pose.position.x = ground_truth_poses_[ground_truth_index_][0];
        ground_truth_pose_.pose.position.y = ground_truth_poses_[ground_truth_index_][1];
        ground_truth_pose_.pose.position.z = ground_truth_poses_[ground_truth_index_][2];
        ground_truth_pose_.pose.orientation.x = ground_truth_poses_[ground_truth_index_][4];           
        ground_truth_pose_.pose.orientation.y = ground_truth_poses_[ground_truth_index_][5];        
        ground_truth_pose_.pose.orientation.z = ground_truth_poses_[ground_truth_index_][6];      
        ground_truth_pose_.pose.orientation.w = ground_truth_poses_[ground_truth_index_][3]; 
        ground_truth_index_ += ground_truth_step_;
    }

    // Wait for Rviz to start sending messages
    while (publisher_camera_pose_.getNumSubscribers() < 1 && publisher_current_frame_.getNumSubscribers() < 1) {
        if (!ros::ok()) {
            cout << "ROS core interrupted" << endl;
            cout << "Exiting..." << endl;
            exit(0); 
        }
        ROS_WARN_ONCE("Please create a subscriber to the marker/image");
    }
    r.sleep();

    // Send messages
    publisher_current_frame_.publish(current_frame);
    publisher_camera_pose_.publish(camera_pose_);
    if (use_ground_truth_)
        publisher_ground_truth_pose_.publish(ground_truth_pose_);

    ros::spinOnce();

};

void Visualizer::ReadGroundTruthEUROC(int start_index, string groundtruth_path) {
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
    ground_truth_step_ = num_ground_truth_poses_ / num_images_ ;
    ground_truth_index_ = start_index * ground_truth_step_ + 600;  // + 600 is a temporarly fix for syncronous video and pose of ground truth (for EUROC V1_02_medium)
};
                                               
}
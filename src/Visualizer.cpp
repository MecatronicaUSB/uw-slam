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

Visualizer::Visualizer(int start_index, int num_images, string _ground_truth_dataset, string ground_truth_path) {
    
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
    
    // Publishers of Ground Truth Trajectory markers (dots or continuous line)
    ros::NodeHandle n_camera_dots;
    ros::NodeHandle n_camera_lines;    
    ros::Publisher publisher_camera_trajectory_dots = n_camera_dots.advertise<visualization_msgs::Marker>("camera_trajectory_dots", 50);
    ros::Publisher publisher_camera_trajectory_lines = n_camera_lines.advertise<visualization_msgs::Marker>("camera_trajectory_lines", 50);

    // Initializing Ground Truth trajectory markers
    visualization_msgs::Marker camera_trajectory_dots, camera_trajectory_lines;
    camera_trajectory_dots.header.frame_id = camera_trajectory_lines.header.frame_id  = "/world";
    camera_trajectory_dots.header.stamp = camera_trajectory_lines.header.stamp = ros::Time::now();
    camera_trajectory_dots.ns = camera_trajectory_lines.ns = "points_and_lines";
    camera_trajectory_dots.action = camera_trajectory_lines.action = visualization_msgs::Marker::ADD;
    camera_trajectory_dots.pose.orientation.w = camera_trajectory_lines.pose.orientation.w = 1.0;

    camera_trajectory_dots.id = 2;
    camera_trajectory_lines.id = 3;

    camera_trajectory_dots.type = visualization_msgs::Marker::SPHERE_LIST;
    camera_trajectory_lines.type = visualization_msgs::Marker::LINE_STRIP;

    camera_trajectory_dots.scale.x = 0.01;
    camera_trajectory_dots.scale.y = 0.01;

    camera_trajectory_lines.scale.x = 0.01;

    // Dots and lines are green
    camera_trajectory_dots.color.r = 0.40f;    
    camera_trajectory_dots.color.g = 0.08f;
    camera_trajectory_dots.color.b = 0.9f;    
    camera_trajectory_dots.color.a = 1.0;
    camera_trajectory_lines.color.r = 0.40f;    
    camera_trajectory_lines.color.g = 0.08f;
    camera_trajectory_lines.color.b = 0.9f;    
    camera_trajectory_lines.color.a = 1.0;

    // If ground truth is used
    if (not (ground_truth_path == "")) {
        use_ground_truth_ = true;
        ground_truth_dataset_ = _ground_truth_dataset;

        // Ground truth marker initialization
        ros::NodeHandle nodehandle_gt_pose;
        ros::Publisher publisher_gt_pose = nodehandle_gt_pose.advertise<visualization_msgs::Marker>("gt_pose", 50);
        visualization_msgs::Marker gt_pose;
        
        // Ground truth marker options
        gt_pose.id = 0;
        gt_pose.header.frame_id = "/world";            
        gt_pose.header.stamp = ros::Time::now();
        gt_pose.ns = "uw_slam";                                                   
        gt_pose.action = visualization_msgs::Marker::ADD;
        gt_pose.lifetime = ros::Duration();    

        // Color of ground truth marker
        gt_pose.color.r = 0.12f;                             
        gt_pose.color.g = 0.56f;
        gt_pose.color.b = 1.0f;
        gt_pose.color.a = 1.0;
            
        // Publishers of Ground Truth Trajectory markers (dots or continuous line)
        ros::NodeHandle n_dots;
        ros::NodeHandle n_lines;    
        ros::Publisher publisher_gt_trajectory_dots = n_dots.advertise<visualization_msgs::Marker>("gt_trajectory_dots", 50);
        ros::Publisher publisher_gt_trajectory_lines = n_lines.advertise<visualization_msgs::Marker>("gt_trajectory_lines", 50);

        // Initializing Ground Truth trajectory markers
        visualization_msgs::Marker gt_trajectory_dots, gt_trajectory_lines;
        gt_trajectory_dots.header.frame_id = gt_trajectory_lines.header.frame_id  = "/world";
        gt_trajectory_dots.header.stamp = gt_trajectory_lines.header.stamp = ros::Time::now();
        gt_trajectory_dots.ns = gt_trajectory_lines.ns = "points_and_lines";
        gt_trajectory_dots.action = gt_trajectory_lines.action = visualization_msgs::Marker::ADD;
        gt_trajectory_dots.pose.orientation.w = gt_trajectory_lines.pose.orientation.w = 1.0;

        gt_trajectory_dots.id = 4;
        gt_trajectory_lines.id = 5;

        gt_trajectory_dots.type = visualization_msgs::Marker::SPHERE_LIST;
        gt_trajectory_lines.type = visualization_msgs::Marker::LINE_STRIP;

        gt_trajectory_dots.scale.x = 0.01;
        gt_trajectory_dots.scale.y = 0.01;

        gt_trajectory_lines.scale.x = 0.01;

        // Dots and lines are green
        gt_trajectory_dots.color.g = 1.0f;
        gt_trajectory_dots.color.a = 1.0;
        gt_trajectory_lines.color.g = 1.0f;
        gt_trajectory_lines.color.a = 1.0;

        if (ground_truth_dataset_ == "EUROC") {
            ReadGroundTruthEUROC(start_index, ground_truth_path);

            gt_pose.type = visualization_msgs::Marker::ARROW;                         

            // Dimentions of ground truth marker   
            gt_pose.scale.x = 0.1;                             
            gt_pose.scale.y = 0.15;
            gt_pose.scale.z = 0.15;
            // EUROC Convention of quaternion: qw, qx, qy, qz
            // Initial position and orientation of ground truth marker   
            gt_pose.pose.position.x = ground_truth_poses_[ground_truth_index_][0];    
            gt_pose.pose.position.y = ground_truth_poses_[ground_truth_index_][1];
            gt_pose.pose.position.z = ground_truth_poses_[ground_truth_index_][2];
            gt_pose.pose.orientation.x = ground_truth_poses_[ground_truth_index_][4]; 
            gt_pose.pose.orientation.y = ground_truth_poses_[ground_truth_index_][5];    
            gt_pose.pose.orientation.z = ground_truth_poses_[ground_truth_index_][6]; 
            gt_pose.pose.orientation.w = ground_truth_poses_[ground_truth_index_][3]; 

            // Initialize the camera pose marker in the same place and orientation as the ground truth marker
            // This initialization point depends of the start_index argument in SLAM process  
            camera_pose.pose.position.x = ground_truth_poses_[ground_truth_index_][0];  
            camera_pose.pose.position.y = ground_truth_poses_[ground_truth_index_][1];
            camera_pose.pose.position.z = ground_truth_poses_[ground_truth_index_][2];
            camera_pose.pose.orientation.x = ground_truth_poses_[ground_truth_index_][4];
            camera_pose.pose.orientation.y = ground_truth_poses_[ground_truth_index_][5];    
            camera_pose.pose.orientation.z = ground_truth_poses_[ground_truth_index_][6]; 
            camera_pose.pose.orientation.w = ground_truth_poses_[ground_truth_index_][3]; 
        }

        if (ground_truth_dataset_ == "TUM") {
            ReadGroundTruthTUM(start_index, ground_truth_path);
            // Camera model changed for TUM dataset
            camera_pose.type = visualization_msgs::Marker::CUBE;  
            camera_pose.scale.x = 0.35;                              
            camera_pose.scale.y = 0.2;
            camera_pose.scale.z = 0.025;
            // Dimentions of ground truth marker               
            gt_pose.type = visualization_msgs::Marker::CUBE;                         
            gt_pose.scale.x = 0.35;                             
            gt_pose.scale.y = 0.2;
            gt_pose.scale.z = 0.025;
            // TUM Convention of quaternion: qx, qy, qz, qw
            gt_pose.pose.position.x = ground_truth_poses_[ground_truth_index_][0];    
            gt_pose.pose.position.y = ground_truth_poses_[ground_truth_index_][1];
            gt_pose.pose.position.z = ground_truth_poses_[ground_truth_index_][2];
            gt_pose.pose.orientation.x = ground_truth_poses_[ground_truth_index_][3]; 
            gt_pose.pose.orientation.y = ground_truth_poses_[ground_truth_index_][4];    
            gt_pose.pose.orientation.z = ground_truth_poses_[ground_truth_index_][5]; 
            gt_pose.pose.orientation.w = ground_truth_poses_[ground_truth_index_][6]; 

            // Initialize the camera pose marker in the same place and orientation as the ground truth marker
            // This initialization point depends of the start_index argument in SLAM process  
            camera_pose.pose.position.x = ground_truth_poses_[ground_truth_index_][0];  
            camera_pose.pose.position.y = ground_truth_poses_[ground_truth_index_][1];
            camera_pose.pose.position.z = ground_truth_poses_[ground_truth_index_][2];
            camera_pose.pose.orientation.x = ground_truth_poses_[ground_truth_index_][3];
            camera_pose.pose.orientation.y = ground_truth_poses_[ground_truth_index_][4];    
            camera_pose.pose.orientation.z = ground_truth_poses_[ground_truth_index_][5]; 
            camera_pose.pose.orientation.w = ground_truth_poses_[ground_truth_index_][6]; 
        }

        // Saving ground truth markers configuration
        publisher_gt_pose_ = publisher_gt_pose;
        gt_pose_ = gt_pose;

        publisher_gt_trajectory_dots_ = publisher_gt_trajectory_dots;
        publisher_gt_trajectory_lines_ = publisher_gt_trajectory_lines;

        gt_trajectory_dots_  = gt_trajectory_dots;
        gt_trajectory_lines_ = gt_trajectory_lines;

    } else {
        // Initialize the camera pose marker in the same place and orientation as the ground truth marker
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

    publisher_camera_trajectory_dots_ = publisher_camera_trajectory_dots;
    publisher_camera_trajectory_lines_ = publisher_camera_trajectory_lines;

    camera_trajectory_dots_  = camera_trajectory_dots;
    camera_trajectory_lines_ = camera_trajectory_lines;

};

Visualizer::~Visualizer() {};

void Visualizer::UpdateMessages(Frame* frame){
    // Rate (Hz) of publishing messages
    ros::Rate r(200);

    // Update image message
    sensor_msgs::ImagePtr current_frame = cv_bridge::CvImage(std_msgs::Header(), "mono8", frame->images_[0]).toImageMsg();

    SE3 previous_pose;
    SE3 current_pose = frame->rigid_transformation_;
    Mat31f t = 100* current_pose.translation();
    Quaternion2 quaternion = current_pose.unit_quaternion();
    cout << t << endl << endl;
    camera_pose_.pose.position.x += t(0);  
    camera_pose_.pose.position.y += t(1);
    camera_pose_.pose.position.z += t(2);
    camera_pose_.pose.orientation.x += 0;
    camera_pose_.pose.orientation.y += 0;    
    camera_pose_.pose.orientation.z += 0; 
    camera_pose_.pose.orientation.w += 0;

    geometry_msgs::Point p;
    p.x = camera_pose_.pose.position.x;
    p.y = camera_pose_.pose.position.y;
    p.z = camera_pose_.pose.position.z;
    camera_trajectory_dots_.points.push_back(p);        
    camera_trajectory_lines_.points.push_back(p);

    // Update ground truth marker position
    if (use_ground_truth_) {
        // EUROC Conventiobn: x, y, z, qw, qx, qy, qz
        if (ground_truth_dataset_ == "EUROC") {
            gt_pose_.pose.position.x = ground_truth_poses_[ground_truth_index_][0];
            gt_pose_.pose.position.y = ground_truth_poses_[ground_truth_index_][1];
            gt_pose_.pose.position.z = ground_truth_poses_[ground_truth_index_][2];
            gt_pose_.pose.orientation.x = ground_truth_poses_[ground_truth_index_][4];           
            gt_pose_.pose.orientation.y = ground_truth_poses_[ground_truth_index_][5];        
            gt_pose_.pose.orientation.z = ground_truth_poses_[ground_truth_index_][6];      
            gt_pose_.pose.orientation.w = ground_truth_poses_[ground_truth_index_][3];
        }
        // TUM Convention: x, y, z, qx, qy, qz, qw (changed translation)
        if (ground_truth_dataset_ == "TUM") {
            gt_pose_.pose.position.x = ground_truth_poses_[ground_truth_index_][0];
            gt_pose_.pose.position.y = ground_truth_poses_[ground_truth_index_][1];
            gt_pose_.pose.position.z = ground_truth_poses_[ground_truth_index_][2];
            gt_pose_.pose.orientation.x = ground_truth_poses_[ground_truth_index_][3];           
            gt_pose_.pose.orientation.y = ground_truth_poses_[ground_truth_index_][4];        
            gt_pose_.pose.orientation.z = ground_truth_poses_[ground_truth_index_][5];      
            gt_pose_.pose.orientation.w = ground_truth_poses_[ground_truth_index_][6];
        }
        
        geometry_msgs::Point gt_p;
        gt_p.x = gt_pose_.pose.position.x;
        gt_p.y = gt_pose_.pose.position.y;
        gt_p.z = gt_pose_.pose.position.z;
        gt_trajectory_dots_.points.push_back(gt_p);        
        gt_trajectory_lines_.points.push_back(gt_p);
        ground_truth_index_ += ground_truth_step_;
    }

    // Wait for Rviz to start sending messages
    while (publisher_camera_pose_.getNumSubscribers() < 1 && publisher_current_frame_.getNumSubscribers() < 1) {
        if (!ros::ok()) {
            cout << "ROS core interrupted" << endl;
            cout << "Exiting..." << endl;
            exit(0); 
        }
        ROS_WARN_ONCE("Please open RVIZ to continue...");
        sleep(1);
    }

    // Send messages
    publisher_current_frame_.publish(current_frame);
    publisher_camera_pose_.publish(camera_pose_);
    publisher_camera_trajectory_dots_.publish(camera_trajectory_dots_);
    publisher_camera_trajectory_lines_.publish(camera_trajectory_lines_);

    if (use_ground_truth_) {
        publisher_gt_pose_.publish(gt_pose_);
        publisher_gt_trajectory_dots_.publish(gt_trajectory_dots_);
        publisher_gt_trajectory_lines_.publish(gt_trajectory_lines_);
    }
    r.sleep();

    ros::spinOnce();

};

void Visualizer::ReadGroundTruthTUM(int start_index, string groundtruth_path) {
    string delimiter = ",";
    string line = "";
    ifstream file(groundtruth_path);
    if (!file.is_open()) {
        cerr << "Could not read file " << groundtruth_path << "\n";
        cerr << "Exiting.." << endl;
        return;
    }
    getline(file, line);
    getline(file, line);    
    getline(file, line);        
    while (getline(file, line)) {
        vector<double> timestamp_values;
        stringstream iss(line);
        string val;
        getline(iss, val, ' ');
        for (int i=0; i<7; i++) {
            string val;
            getline(iss, val, ' ');       
            timestamp_values.push_back(stod(val));
        }
        ground_truth_poses_.push_back(timestamp_values);
    }
    file.close();
    num_ground_truth_poses_ = ground_truth_poses_.size();
    ground_truth_step_ = num_ground_truth_poses_ / num_images_ ;
    ground_truth_index_ = start_index * ground_truth_step_;  // + 600 is a temporarly fix for syncronous video and pose of ground truth (for EUROC V1_02_medium)
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
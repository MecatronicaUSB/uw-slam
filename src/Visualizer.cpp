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

Visualizer::Visualizer(int start_index, int num_images, Mat K, string _ground_truth_dataset, string ground_truth_path) {

    use_ground_truth_ = false;
    num_images_ = num_images;

    // Saves intrinsic paramters
    fx_ = K.at<float>(0,0);
    fy_ = K.at<float>(1,1);
    cx_ = K.at<float>(0,2);
    cy_ = K.at<float>(1,2);
    invfx_ = 1 / fx_;
    invfy_ = 1 / fy_; 
    
    // Camera pose marker initialization
    ros::NodeHandle nodehandle_camera_pose;
    ros::Publisher publisher_camera_pose = nodehandle_camera_pose.advertise<visualization_msgs::Marker>("camera_pose", 50);
    visualization_msgs::Marker camera_pose;

    // Choose starting point of SLAM system
    init_x_ = 0;
    init_y_ = 0;
    init_z_ = 1;
    init_gt_qx_ = 0;
    init_gt_qy_ = 0;
    init_gt_qz_ = 0;
    init_gt_qw_ = 1;
    
    // Camera pose marker options
    camera_pose.id = 1;
    camera_pose.header.frame_id = "world";           
    camera_pose.header.stamp = ros::Time::now();
    camera_pose.ns = "uw_slam";                        
    camera_pose.type = visualization_msgs::Marker::CUBE;   
    camera_pose.action = visualization_msgs::Marker::ADD;
    camera_pose.lifetime = ros::Duration();
    // Dimentions of camera pose marker   
    camera_pose.scale.x = 0.1;                              
    camera_pose.scale.y = 0.15;
    camera_pose.scale.z = 0.15;
    // Color of camera pose marker
    camera_pose.color.b = 1.0f;
    camera_pose.color.a = 1.0;
    
    // Publishers of Camera Pose Trajectory markers (dots or continuous line)
    ros::NodeHandle n_camera_dots;
    ros::NodeHandle n_camera_lines;    
    ros::Publisher publisher_camera_trajectory_dots = n_camera_dots.advertise<visualization_msgs::Marker>("camera_trajectory_dots", 50);
    ros::Publisher publisher_camera_trajectory_lines = n_camera_lines.advertise<visualization_msgs::Marker>("camera_trajectory_lines", 50);

    // Initializing Camera Pose trajectory markers
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

    camera_trajectory_dots.scale.x = 0.005;
    camera_trajectory_dots.scale.y = 0.005;

    camera_trajectory_lines.scale.x = 0.005;

    // Dots and lines are green
    camera_trajectory_dots.color.b = 1.0f;    
    camera_trajectory_dots.color.a = 1.0;
    camera_trajectory_lines.color.b = 1.0f;    
    camera_trajectory_lines.color.a = 1.0;

    // Point-Cloud initialization
    ros::NodeHandle nodehandle_point_cloud;
    ros::Publisher publisher_point_cloud = nodehandle_point_cloud.advertise<visualization_msgs::Marker>("point_cloud", 50);
    visualization_msgs::Marker point_cloud;

    point_cloud.header.frame_id = "/world";
    point_cloud.header.stamp = ros::Time::now();
    point_cloud.ns = "point_cloud";
    point_cloud.action = visualization_msgs::Marker::ADD;
    point_cloud.pose.orientation.w = 1.0;
    point_cloud.id = 4;

    point_cloud.type = visualization_msgs::Marker::SPHERE_LIST;

    point_cloud.scale.x = 0.01;
    point_cloud.scale.y = 0.01;

    // Point-Cloud color
    point_cloud.color.b = 0.8f;    
    point_cloud.color.g = 0.8f;    
    point_cloud.color.r = 0.8f;    
    point_cloud.color.a = 0.5f;


    // If ground truth is used
    if (not (ground_truth_path == "")) {
        use_ground_truth_ = true;
        ground_truth_dataset_ = _ground_truth_dataset;

        // Ground truth marker initialization
        ros::NodeHandle nodehandle_gt_pose;
        ros::Publisher publisher_gt_pose = nodehandle_gt_pose.advertise<visualization_msgs::Marker>("gt_pose", 50);
        visualization_msgs::Marker gt_pose;
        
        // Ground truth marker options
        gt_pose.id = 5;
        gt_pose.header.frame_id = "/world";            
        gt_pose.header.stamp = ros::Time::now();
        gt_pose.ns = "uw_slam";                                                   
        gt_pose.action = visualization_msgs::Marker::ADD;
        gt_pose.lifetime = ros::Duration();    

        // Color of ground truth marker                 
        gt_pose.color.g = 1.0f;
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

        gt_trajectory_dots.id = 6;
        gt_trajectory_lines.id = 7;

        gt_trajectory_dots.type = visualization_msgs::Marker::SPHERE_LIST;
        gt_trajectory_lines.type = visualization_msgs::Marker::LINE_STRIP;

        gt_trajectory_dots.scale.x = 0.005;
        gt_trajectory_dots.scale.y = 0.005;

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
            camera_pose.scale.y = 0.025;
            camera_pose.scale.z = 0.2;
            // Dimentions of ground truth marker               
            gt_pose.type = visualization_msgs::Marker::CUBE;                         
            gt_pose.scale.x = 0.35;                             
            gt_pose.scale.y = 0.2;
            gt_pose.scale.z = 0.025;
            // TUM Convention of quaternion: qx, qy, qz, qw
            off_gt_x_ = ground_truth_poses_[ground_truth_index_][0];   
            off_gt_y_ = ground_truth_poses_[ground_truth_index_][1];   
            off_gt_z_ = ground_truth_poses_[ground_truth_index_][2];   
            
            off_gt_qx_ = ground_truth_poses_[ground_truth_index_][3]; 
            off_gt_qy_ = ground_truth_poses_[ground_truth_index_][4];    
            off_gt_qz_ = ground_truth_poses_[ground_truth_index_][5]; 
            off_gt_qw_ = ground_truth_poses_[ground_truth_index_][6]; 

            // Initialize the camera pose marker in the same place and orientation as the ground truth marker
            // This initialization point depends of the start_index argument in SLAM process 
            double tx = ground_truth_poses_[ground_truth_index_][0];  
            double ty = ground_truth_poses_[ground_truth_index_][1];
            double tz = ground_truth_poses_[ground_truth_index_][2];
            double qx = ground_truth_poses_[ground_truth_index_][3];
            double qy = ground_truth_poses_[ground_truth_index_][4];    
            double qz = ground_truth_poses_[ground_truth_index_][5]; 
            double qw = ground_truth_poses_[ground_truth_index_][6];

            // Initialization with Ground Truth            
            camera_pose.pose.position.x = 0; 
            camera_pose.pose.position.y = 0;
            camera_pose.pose.position.z = 0;
            camera_pose.pose.orientation.x = 0;
            camera_pose.pose.orientation.y = 0;    
            camera_pose.pose.orientation.z = 0;
            camera_pose.pose.orientation.w = 1;


        }

        // Saving ground truth markers configuration
        publisher_gt_pose_ = publisher_gt_pose;
        gt_pose_ = gt_pose;

        publisher_gt_trajectory_dots_ = publisher_gt_trajectory_dots;
        publisher_gt_trajectory_lines_ = publisher_gt_trajectory_lines;

        gt_trajectory_dots_  = gt_trajectory_dots;
        gt_trajectory_lines_ = gt_trajectory_lines;

    } else {
        // Identity initialization
        previous_world_pose_ = SE3(SO3::exp(SE3::Point(0.0, 0.0, 0.0)), SE3::Point(0.0, 0.0, 0.0));

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

    point_cloud_ = point_cloud;
    publisher_point_cloud_ = publisher_point_cloud;

    graph_position_ = vector<vector<float> >(6);
    

};

Visualizer::~Visualizer() {};

void Visualizer::UpdateMessages(Frame* _previous_frame){
    // Rate (Hz) of publishing messages
    ros::Rate r(2000);

    // Update image message
    sensor_msgs::ImagePtr current_frame = cv_bridge::CvImage(std_msgs::Header(), "bgr8", _previous_frame->image_to_send).toImageMsg();

    //
    //SE3 final_pose = previous_world_pose_ * _previous_frame->rigid_transformation_;
    
    Mat31f t = previous_world_pose_.translation();
    Quaternion quaternion = previous_world_pose_.unit_quaternion();

    float x = t(0) + init_x_;
    float y = t(1) + init_y_;
    float z = t(2) + init_z_;
    float qx = quaternion.x();
    float qy = quaternion.y();
    float qz = quaternion.z();
    float qw = quaternion.w();
    
    camera_pose_.pose.position.x = - t(0);  
    camera_pose_.pose.position.y = - t(2);
    camera_pose_.pose.position.z = - t(1);
    camera_pose_.pose.orientation.x = quaternion.x();
    camera_pose_.pose.orientation.y = quaternion.y();    
    camera_pose_.pose.orientation.z = quaternion.z(); 
    camera_pose_.pose.orientation.w = quaternion.w();

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(x,y,z));
    tf::Quaternion q;
    q.setX(qx);
    q.setY(qy);
    q.setZ(qz);
    q.setW(qw);
    transform.setRotation(q);
    
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "camera"));

    //previous_world_pose_ = final_pose;
    

    geometry_msgs::Point p;
    p.x = x;
    p.y = y;
    p.z = z;
    camera_trajectory_dots_.points.push_back(p);        
    camera_trajectory_lines_.points.push_back(p);

    // Add Point-Cloud
    //AddPointCloudFromRGBD(_previous_frame);
    AddPointCloud(_previous_frame);



    // Update ground truth marker position
    if (use_ground_truth_ && ground_truth_index_ < num_ground_truth_poses_) {
        float x_gt, y_gt, z_gt;
        float qx_gt, qy_gt, qz_gt, qw_gt;
  
        // EUROC Convention: x, y, z, qw, qx, qy, qz (changed quaternion order)
        if (ground_truth_dataset_ == "EUROC") {
            x_gt = -off_gt_x_ + ground_truth_poses_[ground_truth_index_][0] + init_x_;
            y_gt = -off_gt_y_ + ground_truth_poses_[ground_truth_index_][1] + init_y_;
            z_gt = -off_gt_z_ + ground_truth_poses_[ground_truth_index_][2] + init_z_;
            qx_gt = -off_gt_qx_ + ground_truth_poses_[ground_truth_index_][4];
            qy_gt = -off_gt_qy_ + ground_truth_poses_[ground_truth_index_][5];
            qz_gt = -off_gt_qz_ + ground_truth_poses_[ground_truth_index_][6];
            qw_gt = -off_gt_qw_ + ground_truth_poses_[ground_truth_index_][3];
            
        }
        // TUM Convention: x, y, z, qx, qy, qz, qw 
        if (ground_truth_dataset_ == "TUM") {
            x_gt = off_gt_y_ - ground_truth_poses_[ground_truth_index_][1] + init_x_;
            y_gt = off_gt_x_ - ground_truth_poses_[ground_truth_index_][0] + init_y_;
            z_gt = -off_gt_z_ + ground_truth_poses_[ground_truth_index_][2] + init_z_;
            // qx_gt = -off_gt_qx_ + ground_truth_poses_[ground_truth_index_][3];
            // qy_gt = -off_gt_qy_ + ground_truth_poses_[ground_truth_index_][4];
            // qz_gt = -off_gt_qz_ + ground_truth_poses_[ground_truth_index_][5];
            // qw_gt = -off_gt_qw_ + ground_truth_poses_[ground_truth_index_][6];
            qx_gt = 0;
            qy_gt = 0;
            qz_gt = 0;
            qw_gt = 1;
        }
        // Send gt position
        gt_pose_.pose.position.x = x_gt;
        gt_pose_.pose.position.y = y_gt;
        gt_pose_.pose.position.z = z_gt;
        gt_pose_.pose.orientation.x = qx_gt;     
        gt_pose_.pose.orientation.y = qy_gt;       
        gt_pose_.pose.orientation.z = qz_gt; 
        gt_pose_.pose.orientation.w = qw_gt;

        tf::Transform transform_gt;
        transform_gt.setOrigin(tf::Vector3(x_gt,y_gt,z_gt));
        tf::Quaternion q_gt;
        q_gt.setX(qx_gt);
        q_gt.setY(qy_gt);
        q_gt.setZ(qz_gt);
        q_gt.setW(qw_gt);
        transform_gt.setRotation(q_gt);
        br.sendTransform(tf::StampedTransform(transform_gt, ros::Time::now(), "world", "gt"));
        
        //
        geometry_msgs::Point gt_p;
        gt_p.x = x_gt;
        gt_p.y = y_gt;
        gt_p.z = z_gt;
        gt_trajectory_dots_.points.push_back(gt_p);        
        gt_trajectory_lines_.points.push_back(gt_p);

        ground_truth_index_ += ground_truth_step_;

        graph_position_[0].push_back(x);
        graph_position_[1].push_back(x_gt);
        graph_position_[2].push_back(y);
        graph_position_[3].push_back(y_gt);        
        graph_position_[4].push_back(z);
        graph_position_[5].push_back(z_gt);

        if(graph_position_[0].size() == 1000) {
            SaveGraph(graph_position_[0], graph_position_[1], "X.txt");
            //GraphXYZ(graph_position_);
        }

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
    publisher_point_cloud_.publish(point_cloud_);
    point_cloud_.points.clear();

    if (use_ground_truth_) {
        publisher_gt_pose_.publish(gt_pose_);
        publisher_gt_trajectory_dots_.publish(gt_trajectory_dots_);
        publisher_gt_trajectory_lines_.publish(gt_trajectory_lines_);
    }
    r.sleep();

    ros::spinOnce();

};

void Visualizer::AddPointCloud(Frame* frame) {
    int num_cloud_points = frame->map_.cols;
    Mat points_3D = frame->map_.clone();

    Mat31f t = previous_world_pose_.translation();

    for (int i=0; i< points_3D.cols; i++) {
        geometry_msgs::Point p3D;

        p3D.x = -points_3D.at<float>(0,i) / points_3D.at<float>(3,i) - t(0);
        p3D.y = -points_3D.at<float>(2,i) / points_3D.at<float>(3,i) - t(2);
        p3D.z = -points_3D.at<float>(1,i) / points_3D.at<float>(3,i) - t(1);

        point_cloud_.points.push_back(p3D);    
    }
};

void Visualizer::AddPointCloudFromRGBD(Frame* frame) {
    int num_cloud_points = frame->candidatePoints_[0].rows;
    Mat points_3D = frame->candidatePoints_[0].clone();

    Mat31f t = previous_world_pose_.translation();

    // 2D -> 3D
    // X  = (x - cx) * Z / fx 
    points_3D.col(0) = ((points_3D.col(0) - cx_) * invfx_);
    points_3D.col(0) = points_3D.col(0).mul(points_3D.col(2));

    // Y  = (y - cy) * Z / fy    
    points_3D.col(1) = ((points_3D.col(1) - cy_) * invfy_);
    points_3D.col(1) = points_3D.col(1).mul(points_3D.col(2));


    for (int i=0; i< points_3D.rows; i++) {
        geometry_msgs::Point p3D;

        p3D.x = -points_3D.at<float>(i,0) - t(0);
        p3D.y =  points_3D.at<float>(i,2) + t(2);
        p3D.z = -points_3D.at<float>(i,1) - t(1);

        point_cloud_.points.push_back(p3D);    
    }
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
    ground_truth_index_ = start_index * ground_truth_step_ + 100;  // + 100 is a temporarly fix for syncronous video and pose of ground truth
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

template <typename T>
string to_string_with_precision(const T a_value, const int n = 6) {
    std::ostringstream out;
    out << std::setprecision(n) << a_value;
    return out.str();
};

void Visualizer::SaveGraph(vector<float> estimated_values, vector<float> gt_values, string filename) {
    int num_values = estimated_values.size();

    std::ofstream output(filename);

    for (int i=0; i<num_values; i++) {
        float e_v = estimated_values[i];
        float gt_v = gt_values[i];
        output << i << " ";
        output << fixed << setprecision(4) << e_v << " ";
        output << fixed << setprecision(4) << gt_v << endl;
    }

    output.close();
};

void Visualizer::GraphXYZ(vector<vector<float> > graph_values_) {
    string graph_names[3];
    graph_names[0] = "Gráfica X vs GT X.jpg";
    graph_names[1] = "Gráfica Y vs GT Y.jpg";
    graph_names[2] = "Gráfica Z vs GT Z.jpg";
    
    // Obtain graph of X, Y and Z movement compared to ground truth position
    for (int j=0; j<6; j+=2) {
        // Graph X - Ground Truth X
        Mat graphX(1480,1580, CV_8UC3, cv::Scalar(255,255,255));

        // Obtain max and min values for both x and gt_x
        auto max_x1 = *max_element(graph_values_[j].begin(), graph_values_[j].end());
        auto min_x1 = *min_element(graph_values_[j].begin(), graph_values_[j].end());
        auto max_x2 = *max_element(graph_values_[j+1].begin(), graph_values_[j+1].end());
        auto min_x2 = *min_element(graph_values_[j+1].begin(), graph_values_[j+1].end());

        float max, min;
        int num_values = graph_values_[j].size();

        if (max_x1 > max_x2) {
            max = max_x1;
        } else {
            max = max_x2;
        }
        if (min_x1 < min_x2) {
            min = min_x1;
        } else {
            min = min_x2;
        }
        
        if (abs(max)<0.000001) {
            max = 0.0;
        }
        if (abs(min)<0.000001) {
            min = 0.0;
        }

        float label_step_y = abs((max - min) / 4);
        int label_step_x = round(num_values / 4);

        float step_x = (float)(1430-145) / (num_values-1);
        float step_y = abs(max - min) / 1280;

        circle(graphX, Point(1430,1380), 5, Scalar(0,255,0), -1, 6, 0);
        
        Point prev_pt, curr_pt;
        for (int i=0; i<graph_values_[j].size(); i++) {
            float x = graph_values_[j][i];
            curr_pt.x = round(145 + step_x * i);

            float x_aux = max;
            for (int j=0; j<1280; j++) {
                curr_pt.y = 100 + j;
                if (x_aux < x)
                    break;
                x_aux -= step_y;
            }

            circle(graphX, curr_pt, 2, Scalar(255,0,0), -1, 6, 0);
            if (i!=0)
                line(graphX, prev_pt, curr_pt, Scalar(255,0,0), 1, 8, 0);
            prev_pt = curr_pt;
        }

        for (int i=0; i<graph_values_[j+1].size(); i++) {
            float x = graph_values_[j+1][i];
            curr_pt.x = round(145 + step_x * i);

            float x_aux = max;
            for (int j=0; j<1280; j++) {
                curr_pt.y = 100 + j;
                if (x_aux < x)
                    break;
                x_aux -= step_y;
            }

            circle(graphX, curr_pt, 2, Scalar(0,255,0), -1, 6, 0);
            if (i!=0)
                line(graphX, prev_pt, curr_pt, Scalar(0,255,0), 1, 8, 0);
            prev_pt = curr_pt;
        }
        // y-axis labels
        cv::rectangle(graphX,cv::Point(130,1400),cv::Point(1450,80),cvScalar(0,0,0),1);
        cv::putText(graphX, to_string_with_precision(min + 4*label_step_y,2), cv::Point(10,100), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
        cv::putText(graphX, to_string_with_precision(min + 3*label_step_y,2), cv::Point(10,420), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
        cv::putText(graphX, to_string_with_precision(min + 2*label_step_y,2), cv::Point(10,740), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
        cv::putText(graphX, to_string_with_precision(min + label_step_y,2), cv::Point(10,1060), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
        cv::putText(graphX, to_string_with_precision(min,2), cv::Point(10,1380), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
        // x-axis labels
        cv::putText(graphX, std::to_string(0), cv::Point(152-7*1,1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
        cv::putText(graphX, std::to_string(label_step_x), cv::Point(467-7*2,1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
        cv::putText(graphX, std::to_string(2*label_step_x), cv::Point(787-7*3,1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
        cv::putText(graphX, std::to_string(3*label_step_x), cv::Point(1107-7*3,1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
        cv::putText(graphX, std::to_string(4*label_step_x), cv::Point(1427-7*3,1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);

        Mat show;
        resize(graphX, show, Size(), 0.5, 0.5);
        // imshow(graph_names[j/2], show);
        // waitKey(0);
        imwrite(graph_names[j/2], show);
    }
};



}
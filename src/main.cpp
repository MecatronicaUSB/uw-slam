/********************************************
 * FILE NAME:   feature_detector.cpp        *
 * DESCRIPTION: Homogenuos detection of     *
 *              features                    *
 *                                          *
 * AUTHOR:      Fabio Morales               *
 ********************************************/

#include "feature_detector.h"

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

int main( int argc, char** argv ){
    cv::Mat img_1, img_2;
    cv::Mat R_f, t_f;

    char dataSetPath[300];
    char filename1[300];
    char filename2[300];

    // Change the file path according to where your dataset is saved before running
    // Get the file names of the first two images
    std::sprintf(filename1, "/home/fabio/catkin_ws/src/uw_slam/src/data/kitti/odometry/00/image_2/%06d.png", 60);
    std::sprintf(filename2, "/home/fabio/catkin_ws/src/uw_slam/src/data/kitti/odometry/00/image_2/%06d.png", 61);
    
    // Read the first two images from the dataset
    // cv::Mat img_1_color = cv::imread("/home/fabio/Documents/Tesis/SfM-Toy-Library-master/dataset/crazyhorse/P1000968.JPG");
    // cv::Mat img_2_color = cv::imread("/home/fabio/Documents/Tesis/SfM-Toy-Library-master/dataset/crazyhorse/P1000971.JPG");
    cv::Mat img_1_color = cv::imread(filename1);
    cv::Mat img_2_color = cv::imread(filename2);

    // Check for errors
    if (!img_1_color.data || !img_2_color.data){
        std::cout << "(!) Error reading images " << std::endl; return -1;
    }

    // Resize the images
    // resize(img_1_color, img_1_color, cv::Size(1240,376), 0, 0, cv::INTER_CUBIC);
    // resize(img_2_color, img_2_color, cv::Size(1240,376), 0, 0, cv::INTER_CUBIC);

    // Convert the two images to grayscale
    cv::cvtColor(img_1_color, img_1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_2_color, img_2, cv::COLOR_BGR2GRAY);

    // Start clock variable for time measurement
    // int start_s = clock();  

    // Initialize stats of feature detector
    struct Stats stats = _initStats("SHI_T");

    // Feature detection and tracking
    vector<Point2f> points1, points2;
    vector<uchar> status;
    //getCornerEdges(img_1, img_2, points1, points2);
    featureDetectionTracking(img_1,img_2, points1, points2, status, "FAST");
    // Stop clock variable
    // int stop_s = clock();
    // stats.exec_time = calculateTime(start_s, stop_s);

    // Obtain Essential Matrix with the Five-Point Algorithm (David Nister, 2004)
    Mat EssentialMat, mask;
    double focal = 7.188560000000e+02; 
    Point2d pp = Point2d(6.071928000000e+02, 1.852157000000e+02);
    Mat K = (Mat_<double>(3,3) << focal, 0, 6.071928000000e+02, 0, focal,  1.852157000000e+02, 0, 0, 1);

    EssentialMat = findEssentialMat(points1, points2, focal, pp, cv::RANSAC, 0.999, 3.0, mask);

    // TODO
    // Show number of inliers/outliers found by findEssentialMat()
    // printLiers(mask);

    // Obtain Rotation matrix and translation vector
    Mat R, t;
    int inliers2;
    inliers2 = recoverPose(EssentialMat, points1, points2, R, t, focal, pp, mask);
    // cout << "ROTATION MATRIX" << endl;
    // cout << R << endl;
    // cout << "TRANSLATION VECTOR" << endl;
    // cout << t << endl;
    // cout << "Inliers: " << inliers2 << endl;

    // Obtain projection matrices
    Mat I3 = Mat::eye(3, 3, CV_64F);
    Mat v0 = Mat(3,1,CV_64F, double(0));
    Mat P1, P2;

    hconcat(I3, v0, P1);
    hconcat(R, t, P2);

    P1 = K * P1;
    P2 = K * P2;
    // P1 = K * [I3 | 0]
    // P2 = K2 * [R | t]

    // Compute depth of 3D points using triangulation
    Mat mapPoints;
    triangulatePoints(P1, P2, points1, points2, mapPoints);

    // Show features detected/tracked and stats information
    Mat imgShow = img_1_color.clone();
    showFeatures(imgShow, points1, points2);
    //imshow("Output",imgShow);

    ros::init(argc, argv, "uw_slam");
    ros::NodeHandle n;
    ros::Rate r(1);
    ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 100);

    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher img_pub = it.advertise("camera/image",1);
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", imgShow).toImageMsg();

    // Set our initial shape type to be a cube
    uint32_t shape = visualization_msgs::Marker::CUBE;

    visualization_msgs::Marker marker;

    // Set the frame ID and timestamp.  See the TF tutorials for information on these.
    marker.header.frame_id = "/main_uw";
    marker.header.stamp = ros::Time::now();

    // Set the namespace and id for this marker.  This serves to create a unique ID
    // Any marker sent with the same namespace and id will overwrite the old one
    marker.ns = "uw_slam";

    // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    marker.type = shape;

    // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    marker.action = visualization_msgs::Marker::ADD;

    // Set the scale of the marker -- 1x1x1 here means 1m on a side
    marker.scale.x = 0.10;
    marker.scale.y = 0.10;
    marker.scale.z = 0.10;

    // Set the color -- be sure to set alpha to something non-zero!
    marker.color.r = 0.12f;
    marker.color.g = 0.56f;
    marker.color.b = 1.0f;
    marker.color.a = 1.0;

    marker.lifetime = ros::Duration();
    //while (ros::ok()){
        for(int i=0; i < mapPoints.cols; i++){
                        
            // Set the frame ID and timestamp.  See the TF tutorials for information on these.
            marker.header.frame_id = "/main_uw";
            marker.header.stamp = ros::Time::now();

            // Set the namespace and id for this marker.  This serves to create a unique ID
            // Any marker sent with the same namespace and id will overwrite the old one
            marker.ns = "uw_slam";
            marker.id = i;

            // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
            marker.type = shape;

            // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
            marker.action = visualization_msgs::Marker::ADD;

            // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
            marker.pose.position.x = -10 + (mapPoints.at<float>(2,i)/mapPoints.at<float>(3,i));
            marker.pose.position.y = 0 + (mapPoints.at<float>(0,i)/mapPoints.at<float>(3,i));
            marker.pose.position.z = 2 -(mapPoints.at<float>(1,i)/mapPoints.at<float>(3,i));
            marker.pose.position.x /= 2;
            marker.pose.position.y /= 2;
            marker.pose.position.z /= 2;

            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;

            // Set the scale of the marker -- 1x1x1 here means 1m on a side
            marker.scale.x = 0.10;
            marker.scale.y = 0.10;
            marker.scale.z = 0.10;

            // Set the color -- be sure to set alpha to something non-zero!
            marker.color.r = 0.12f;
            marker.color.g = 0.56f;
            marker.color.b = 1.0f;
            marker.color.a = 1.0;
            
            marker.lifetime = ros::Duration();

            // Publish the marker
            while (marker_pub.getNumSubscribers() < 1 && img_pub.getNumSubscribers() < 1)
            {
                if (!ros::ok())
                {
                return 0;
                }
                ROS_WARN_ONCE("Please create a subscriber to the marker/image");
                sleep(1);
            }

            img_pub.publish(msg);
            marker_pub.publish(marker);
            ros::spinOnce();

            //r.sleep();
        }
    //}
    return 0;
    
}





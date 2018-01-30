#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

void obtainFeaturesMarkers(const visualization_msgs::Marker& msg){

}


void main (int argc, char** argv){
    ros::init(argc, argv, "map_node");
    
    ros::NodeHandle n;

    ros::Subscriber sub_features = n.subscribe("feature_markers", 1000, obtainFeaturesMarkers);

    ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 2);


}
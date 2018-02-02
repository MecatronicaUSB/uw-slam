// OpenCV libraries. May need review for the final release
#include "opencv2/core/core.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/calib3d/calib3d.hpp"

// Basic C and C++ libraries
#include <stdio.h>
#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <ctime> // for time calculations
#include <dirent.h>

// Include args library
#include "args.hxx"

// CUDA libraries
#include <opencv2/cudafilters.hpp>
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"

// ROS libraries
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

// Namespace
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace cv::cuda;

// Args declarations
args::ArgumentParser parser("Feature Detection Module.", "Author: Fabio Morales.");
args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

args::ValueFlagList<std::string> inputImages(parser, "NameImages", "Names of input images. Must specify two file names (one per flag)", {'i'});
args::ValueFlag<std::string> inputVideo(parser, "NameVideo", "Video input name", {'v'});
args::ValueFlag<std::string> dir_dataset(parser, "directory", "Directory of dataset files", {'d'});
args::ValueFlag<std::string> parse_calibration(parser, "calibration", "Name of input XML calibration file", {"calibration"});
args::ValueFlag<int> feature_detector(parser, "feature detector", "Choose feature detector to use. SURF = 0, ORB = 1. (Default = SURF)",{'f'});
args::Flag feature_stats(parser, "show detector stats", "Show results of feature detector algorithm", {'o'});
args::ValueFlag<std::string> output(parser, "OutPut", "Output", {'d'});

// Function declarations
vector<string> read_filenames(string dir_ent);

Mat readCameraMat(std::string dir_calibrarionFile);

// Print the features tracked in an image
void showMovementFeatures(Mat imgShow, vector<Point2f>& points1, vector<Point2f>& points2);

// TODO
// Print the number of inliers and outliers found by findEssentialMat()
void printLiers(Mat mask);

// Print the stats of a feature detector and descriptor
void showFeatureStats(std::string detectorName, int nfeatures[2], int nmatches);

// Calculates the time in (ms) using the output of two clock() variables
int calculateTime(int start, int stop);

vector<DMatch> getGoodMatches(vector< vector< DMatch> > matches);

array<vector<KeyPoint>,2> getGoodKeypoints(vector<DMatch> goodMatches, array< vector< KeyPoint>, 2 > keypoints);

vector<DMatch> gridFiltering(vector<DMatch> goodMatches, vector<KeyPoint> keypoints);

Mat getProjectionMat(Mat cameraMat, Mat rotationMat, Mat translationMat);
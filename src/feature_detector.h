#include "opencv2/core/core.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/calib3d/calib3d.hpp"

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

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

// Structure for feature detector/tracking statistics
struct Stats{
    std::string descriptor_name;
    double exec_time;
    int n_features;
    int ok_features;
};

// Initialization of structure Stats
struct Stats _initStats(std::string descriptor){
    struct Stats stats;
    stats.descriptor_name = descriptor;
    stats.exec_time = 0;
    stats.n_features = 0;
    stats.ok_features = 0;
    return stats;
}

int getDistance(Point2f a, Point2f b){
    return sqrt(pow(b.y-a.y,2)+pow(b.x-a.x,2));
}

// Function to get Corners and Edges homogeneously in the images
void getCornerEdges(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2){
    // Patameters for Shi-Tomasi algorithm
    double qualityLevel = 0.98;
    double minDistance = 30;
    int maxCorners = 1;
    int blockSize = 20;
    bool useHarrisDetector = false;
    double k = 0.04;
    
    // Patch Size of each cell of the image
    int pS = 32;
    for(int x = 0; x < img_1.size().width - pS; x = x + pS){
        for(int y = 0; y < img_1.size().height- pS; y = y + pS){
            vector<Point2f> points_aux;
            Rect ROI(x,y,pS,pS);
            Mat img_aux = img_1(ROI);
            // imshow("HEY", img_aux);
            // waitKey();
            goodFeaturesToTrack(img_aux, points_aux, maxCorners, qualityLevel,  minDistance, Mat(), blockSize, useHarrisDetector, k);

            if(points_aux.size()==0){
                
            }else{
                points_aux[0].x += x;
                points_aux[0].y += y; 
                points1.push_back(points_aux[0]);
            }
        }
    }
    // Feature Tracking
    vector<float> err;
    vector<uchar> status;
    Size winSize = Size(21,21);
    TermCriteria termcrit = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    // Deleting points that the tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    int dThreshold = 60; 
    for(int i=0; i<status.size(); i++){
        Point2f pt1 = points1.at(i - indexCorrection);
        Point2f pt2 = points2.at(i - indexCorrection);
        int d = getDistance(pt1,pt2);
        if((status.at(i) == 0)||(pt2.x<0)||(pt2.y<0)||(d > dThreshold)){
            status.at(i) = 0;
            points1.erase (points1.begin() + (i - indexCorrection));
            points2.erase (points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}

// Function to detect and track features on two images (FAST | ORB | SURF | AKAZE) 
void featureDetectionTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status, std::string descriptor){
    if(descriptor=="FAST"){
        vector<KeyPoint> keypoints_1;
        int fast_threshold = 20;
        bool nonmaxSuppression = true;

        // Feature Detection
        FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
        KeyPoint::convert(keypoints_1, points1, vector<int>());

        // Saving the number of features detected
        //stats->n_features = points1.size();

        // Feature Tracking
        vector<float> err;
        Size winSize = Size(21,21);
        TermCriteria termcrit = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
        calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

        // Deleting points that the tracking failed or those who have gone outside the frame
        int indexCorrection = 0;
        for(int i=0; i<status.size(); i++){
            Point2f pt = points2.at(i - indexCorrection);
            if((status.at(i) == 0)||(pt.x<0)||(pt.y<0)){
                if((pt.x<0)||(pt.y<0)){
                    status.at(i) = 0;
                }
                points1.erase (points1.begin() + (i - indexCorrection));
                points2.erase (points2.begin() + (i - indexCorrection));
                indexCorrection++;
            }
        }
        // Saving the number of features tracked succesfully
        //stats->ok_features = points1.size();
    }

    if(descriptor=="ORB"){
        Mat descriptors_1, descriptors_2;
        vector<KeyPoint> keypoints_1, keypoints_2;
        vector<DMatch> matches, ok_matches;
        
        Ptr<FeatureDetector> orb_detector = ORB::create(4000);
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
        
        // Feature Detection
        orb_detector->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);
        orb_detector->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);
        
        // Feature Tracking
        matcher->match (descriptors_1, descriptors_2, matches);

        // Deleting points that are too seperated
        double min_dist = 1000, max_dist = 0;
        for(int i=0; i<descriptors_1.rows; i++){
            double dist = matches[i].distance;
            if(dist < min_dist) min_dist = dist;
            if(dist > max_dist) max_dist = dist;
        }
        for(int i=0; i < descriptors_1.rows; i++){
            if(matches[i].distance <= max(2*min_dist, 30.0)){
                ok_matches.push_back(matches[i]);
            }
        }

        // Obtaining the coordinates of the ok_points
        int key1_index;
        int key2_index;
        for(int i=0; i<ok_matches.size(); i++){
            key1_index = ok_matches[i].queryIdx;
            key2_index = ok_matches[i].trainIdx;
            points1.push_back(keypoints_1[key1_index].pt);
            points2.push_back(keypoints_2[key2_index].pt);
        }

        // Uncomment to show an alternative image of the ok_points found between the two frames
        // Mat img_match, img_okmatch;
        // drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
        // drawMatches(img_1, keypoints_1, img_2, keypoints_2, ok_matches, img_okmatch);  
        // imshow("Matches", img_match);
        // imshow("OK Matches", img_okmatch);
        // waitKey(0);
        
        // Saving the number of features and ok_features detected
        //stats->n_features = matches.size();
        //stats->ok_features = ok_matches.size();
    }

    if(descriptor == "SURF"){
        int minHessian = 400;
        Mat descriptors_1, descriptors_2;
        vector<KeyPoint> keypoints_1, keypoints_2;
        vector<DMatch> matches, ok_matches;
        
        Ptr<SURF> surf_detector = SURF::create();
        surf_detector->setHessianThreshold(minHessian);

        Ptr<SurfDescriptorExtractor> surf_descriptor = SURF::create();
        FlannBasedMatcher matcher;

        // Feature Detection
        surf_detector->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);
        surf_detector->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);
        
        // Feature Tracking
        matcher.match (descriptors_1, descriptors_2, matches);

        // Deleting points that are too seperated
        double min_dist = 100, max_dist = 0;
        for(int i=0; i<descriptors_1.rows; i++){
            double dist = matches[i].distance;
            if(dist < min_dist) min_dist = dist;
            if(dist > max_dist) max_dist = dist;
        }
        for(int i=0; i < descriptors_1.rows; i++){
            if(matches[i].distance <= max(2*min_dist, 0.08)){
                ok_matches.push_back(matches[i]);
            }
        }

        // Obtaining the coordinates of the ok_points
        int key1_index;
        int key2_index;
        for(int i=0; i<ok_matches.size(); i++){
            key1_index = ok_matches[i].queryIdx;
            key2_index = ok_matches[i].trainIdx;
            points1.push_back(keypoints_1[key1_index].pt);
            points2.push_back(keypoints_2[key2_index].pt);
        }
        
        // Saving the number of features and ok_features detected
        //stats->n_features = matches.size();
        //stats->ok_features = ok_matches.size();
    }

    if(descriptor == "AKAZE"){
        Mat descriptors_1, descriptors_2;
        vector<KeyPoint> matched1, matched2, keypoints_1, keypoints_2, inliers1, inliers2;
        vector< vector<DMatch> > matches;
        vector<DMatch> ok_matches;
        
        Ptr<AKAZE> akaze = AKAZE::create();
        BFMatcher matcher(NORM_HAMMING);
        
        // Feature Detection
        akaze->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);
        akaze->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);

        // Feature Tracking
        matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);

        // Deleting points that are too seperated
        const float inlier_threshold = 120.0f;
        const float nn_match_ratio = 0.8f;
        for(size_t i = 0; i < matches.size(); i++) {
            DMatch first = matches[i][0];
            float dist1 = matches[i][0].distance;
            float dist2 = matches[i][1].distance;

            if(dist1 < nn_match_ratio * dist2) {
                matched1.push_back(keypoints_1[first.queryIdx]);
                matched2.push_back(keypoints_2[first.trainIdx]);
            }
        }
        for(unsigned i = 0; i < matched1.size(); i++) {
            Mat col = Mat::ones(3, 1, CV_64F);
            col.at<double>(0) = matched1[i].pt.x;
            col.at<double>(1) = matched1[i].pt.y;

            col /= col.at<double>(2);
            double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
                                pow(col.at<double>(1) - matched2[i].pt.y, 2));
            if(dist < inlier_threshold) {
                int new_i = static_cast<int>(inliers1.size());
                inliers1.push_back(matched1[i]);
                inliers2.push_back(matched2[i]);
                ok_matches.push_back(DMatch(new_i, new_i, 0));
            }
        }   

        // Obtaining the coordinates of the ok_points
        int key1_index;
        int key2_index;
        for(int i=0; i<ok_matches.size(); i++){
            key1_index = ok_matches[i].queryIdx;
            key2_index = ok_matches[i].trainIdx;
            points1.push_back(inliers1[key1_index].pt);
            points2.push_back(inliers2[key2_index].pt);
        }

        // Uncomment to show an alternative image of the ok_points found between the two frames
        // Mat img_match, img_okmatch;
        // drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
        // drawMatches(img_1, keypoints_1, img_2, keypoints_2, ok_matches, img_okmatch);  
        // imshow("Matches", img_match);
        // imshow("OK Matches", img_okmatch);
        // waitKey(0);
        
        // Saving the number of features and ok_features detected
        //stats->n_features = matches.size();
        //stats->ok_features = ok_matches.size();
    }
}

// Print the features tracked in an image
void showFeatures(Mat imgShow, vector<Point2f>& points1, vector<Point2f>& points2){
    for(int i=0; i<points1.size(); i++){
        Point2f a = points1.at(i);
        Point2f b = points2.at(i);
        cv::circle(imgShow, a, 3, Scalar(0,255,0), -1, 8 , 0);
        cv::line(imgShow , a, b, (0,0,255),1);
    }
}

// TODO
// Print the number of inliers and outliers found by findEssentialMat()
void printLiers(Mat mask){
    int outliers = 0, inliers =0;
    // for(int i=0; i<mask.size(); i++){
    //     if(mask.at(i) == 0){
    //         outliers += 1;
    //     }else{
    //         inliers += 1;
    //     }
    // }
    cout << "Intliers: "<< inliers << endl;
    cout << "Outliers: "<< outliers << endl;
    cout << "Total: "<< inliers + outliers << endl;
}

// Print the stats of a feature detector and descriptor
void printStats(struct Stats stats){
    std::cout << "Stats of feature detection" << std::endl;
    std::cout << "Name: " << stats.descriptor_name << std::endl;
    std::cout << "Features found: " << stats.n_features << std::endl;
    std::cout << "Features tracked: " << stats.ok_features << std::endl;
    std::cout << "Execution time (ms): " << stats.exec_time << std::endl;
}

// Calculates the time in (ms) using the output of two clock() variables
int calculateTime(int start, int stop){
    return (stop - start)/double(CLOCKS_PER_SEC)*1000;
}


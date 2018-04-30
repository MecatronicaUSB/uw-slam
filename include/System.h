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

// TODO(GitHub:fmoralesh, fabmoraleshidalgo@gmail.com) 02-13-2018 - Clean unused libraries.
#pragma once
#include "Options.h"
#include "CameraModel.h"
#include "Tracker.h"
#include "Visualizer.h"
#include "Mapper.h"

///Basic C and C++ libraries
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <dirent.h>

/// OpenCV libraries. May need review for the final release
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/calib3d.hpp"

/// CUDA specific libraries
#include <opencv2/cudafilters.hpp>
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"

// Namespaces
using namespace cv;
using namespace std;

namespace uw
{

class CameraModel;
class Tracker;
class Map;
class Visualizer;

class Frame {
public:
    /**
     * @brief Constructor of Frame.
     * 
     */
    Frame();

    /**
     * @brief Destructor of Frame
     * 
     */
    ~Frame();


    int id;
    Mat image_to_send;
    vector<Mat> images_    = vector<Mat>(PYRAMID_LEVELS);
    vector<Mat> depths_    = vector<Mat>(PYRAMID_LEVELS);    
    vector<Mat> gradientX_ = vector<Mat>(PYRAMID_LEVELS);
    vector<Mat> gradientY_ = vector<Mat>(PYRAMID_LEVELS);
    vector<Mat> gradient_  = vector<Mat>(PYRAMID_LEVELS);
    
    vector<Mat> candidatePoints_   = vector<Mat>(PYRAMID_LEVELS);
    vector<Mat> informationPoints_ = vector<Mat>(PYRAMID_LEVELS);

    vector<KeyPoint> keypoints_;
    vector<Point2f> points_;
    
    int idFrame_;
    int n_matches_;
    vector<int> valid_points_ = vector<int>(4);
    Frame* previous_frame_;
    Frame* next_frame_;    
    Mat map_;
    SE3 rigid_transformation_;
    
    bool obtained_gradients_;
    bool obtained_candidatePoints_; 
    bool depth_available_;           
    bool isKeyFrame_;
};

class System {
public:
    /**
     * @brief Constructor of System. Contains args from parser for ROS initialization.
     * 
     * @param argc 
     * @param argv 
     * @param _start_index 
     */
    System(int argc, char *argv[], int _start_index);

    /**
     * @brief Destructor of System.
     * 
     */
    ~System();

    /**
     * @brief Configures new Intrinsic Parameters Camera Matrix with the parameters from
     *        the calibration .xml file. Refer to calibration/calibration.xml for file structure.
     *        Camera Models supported: Pinhole, RadTan / Equidistant.
     * 
     * @param _calibration_path 
     */
    void Calibration(string _calibration_path);

    /**
     * @brief Calculates ROI of images (that are inside of frame after undistortion)
     *        A list of images must exist before executing this function.
     *        Assumes that every image in the dataset have same width and height
     */
    void CalculateROI();

    /**
     * @brief Initializes necessary variables to start SLAM system.
     *        Call after Calibration() but before adding the first frame to the system.
     */
    void InitializeSystem(string _images_path, string _ground_truth_dataset, string _ground_truth_path, string _depth_path);

    void UpdateWorldPose(SE3& _previous_world_pose, SE3 _current_pose);

    /**
     * @brief Starts tracking thread of the next frame. 
     *        Computes image alingment and optimization of camera poses given two frames and
     *        their inverse depth map.
     */
    void Tracking();

    /**
     * @brief Starts mapping thread of the next frame.
     * 
     */
    void Mapping();

    void printHistogram(int histogram[256], std::string filename, cv::Scalar color);
    
    void getHistogram(Mat img, int *histogram);
    
    void imgChannelStretch(Mat imgOriginal, Mat imgStretched, int lowerPercentile, int higherPercentile);
    
    void Visualize();
    
    /**
     * @brief Adds the frame corresponding on the id position from all the dataset.
     * 
     * @param _id 
     */
    void AddFrame(int _id);

    /**
     * @brief Adds num_images frames to the system, starting from the id position from
     *        all the dataset. Only used for debuggin purposes.
     * 
     * @param _id 
     * @param _num_images 
     */
    void AddFramesGroup(int _id, int _num_images);

    /**
     * @brief Adds the keyframe corresponding on the id position from all the dataset.
     * 
     * @param _id 
     */
    void AddKeyFrame(int _id);

    /**
     * @brief Adds a list of images path to the system, for future reading of the frames.
     *        Propagates ground_truth_path to later use of Visualizer (optional).
     * 
     * @param _path 
     * @param _depth_path    
     */
    void AddLists(string _path, string _depth_path);

    /**
     * @brief Fast function to show an id frame. Only used for debuggin purposes.
     * 
     * @param _id 
     */
    void ShowFrame(int _id);

    /**
     * @brief Deletes oldest frame of list to mantain memory consumption
     * 
     */
    void FreeFrames();

    CameraModel* camera_model_;
    Tracker* tracker_;
    Mapper* mapper_;
    Visualizer* visualizer_;

    int start_index_;
    int num_valid_images_;
    int num_frames_;
    int num_keyframes_;
    int w_, h_, w_input_, h_input_;
    float fx_, fy_, cx_, cy_;
    
    Frame* current_frame_;
    Frame* previous_frame_;
    Frame* current_keyframe_;
    vector<Frame*> frames_;
    vector<Frame*> keyframes_;

    vector<string> images_list_;
    vector<string> depth_list_;
    
    string ground_truth_dataset_;    
    string ground_truth_path_;

    SE3 temp_previous_world_pose_;    
    SE3 previous_world_pose_;

    Mat K_;
    Mat map1_, map2_;
    Rect ROI;
    
    bool initialized_;
    bool distortion_valid_;
    bool depth_available_;
};

}
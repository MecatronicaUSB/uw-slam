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
    vector<Mat> image = vector<Mat>(PYRAMID_LEVELS);
    vector<Point2d> candidatePoints_;
    vector<float> map_;
    Mat rigid_body_transformation_;
    int idFrame_;
    bool isKeyFrame_;
};

class System {
public:
    /**
     * @brief Constructor of System. Contains args from parser for ROS initialization.
     * 
     */
    System(int argc, char *argv[]);

    /**
     * @brief Destructor of System.
     * 
     */
    ~System();

    /**
     * @brief Configures new Intrinsic Parameters Camera Matrix with the parameters from
     *        the calibration .xml file. Refer to sample/calibration.xml for file structure.
     *        Camera Models supported: Pinhole, RadTan / Equidistant.
     * 
     * @param calibration_path 
     */
    void Calibration(string calibration_path);

    /**
     * @brief Initializes necessary variables to start SLAM system.
     *        Call after Calibration() but before adding the first frame to the system.
     */
    void InitializeSystem();

    /**
     * @brief Starts tracking thread of the next frame. 
     *        Computes image alingment and optimization of camera poses given two frames and
     *        their inverse depth map.
     */
    void Tracking();

    /**
     * @brief Adds the frame corresponding on the id position from all the dataset.
     * 
     * @param id 
     */
    void AddFrame(int id);

    /**
     * @brief Adds num_images frames to the system, starting from the id position from
     *        all the dataset. Only used for debuggin purposes.
     * 
     * @param nImages 
     */
    void AddFramesGroup(int id, int num_images);

    /**
     * @brief Adds the keyframe corresponding on the id position from all the dataset.
     * 
     * @param id 
     */
    void AddKeyFrame(int id);

    /**
     * @brief Adds a list of images path to the system, for future reading of the frames.
     * 
     * @param path 
     */
    void AddListImages(string path);

    /**
     * @brief Fast function to show an id frame. Only used for debuggin purposes.
     * 
     * @param id 
     */
    void ShowFrame(int id);


    CameraModel* camera_model_;
    Tracker* tracker_;
    Visualizer* visualizer_;

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
    
    Mat K_;
    Mat map1_, map2_;

    bool initialized_;
    bool rectification_valid_;
    
};

}
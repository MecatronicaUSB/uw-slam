/**
* This file is part of UW-SLAM.
* 
* Copyright 2018.
* Developed by Fabio Morales,
* If you use this code, please cite the respective publications as
* listed on the above website.
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
#pragma once
#include "Options.h"
#include "CameraModel.h"

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

class Frame{
public:
    Frame();

    int idFrame;
    Mat data;

    Frame* nextFrame;
    Frame* prevFrame;
};


class System{
public:
    System();
    ~System();

    void addFrame(int id);
    void addFrameGroup(int nImages);
    void addListImages(string path);
    void Calibration(string calibrationPath);
    void showFrame(int id);

    Mat applyGradient(int id);

    vector<Frame*> frames;

    int w, h;
    int nFrames;
    vector<string> imagesList;
    Mat K;
    Mat map1, map2;
    CameraModel* cameraModel;
    
    bool rectificationValid;
    // Filters for calculating gradient in images
    Ptr<cuda::Filter> soberX = cuda::createSobelFilter(0, 0, 1, 0, CV_SCHARR, 1.0, BORDER_DEFAULT);
    Ptr<cuda::Filter> soberY = cuda::createSobelFilter(0, 0, 0, 1, CV_SCHARR, 1.0, BORDER_DEFAULT);
    
};




}
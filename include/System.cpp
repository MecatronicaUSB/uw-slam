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

#include "System.h"

namespace uw
{

Frame::Frame(){
    idFrame  = 0;

    nextFrame = 0;
    prevFrame = 0;
}

System::System(){
    firstFrame  = 0;
    lastFrame   = 0;
    cameraMat   = 0;
    nFrames     = 0;
}

void System::addFrame(int id){
    Frame* newFrame = new Frame();
    newFrame->data = imread(imagesList[id], CV_LOAD_IMAGE_GRAYSCALE);

    if( firstFrame == 0){
        firstFrame = newFrame;
    }
    else{
        lastFrame->nextFrame = newFrame;
        newFrame->prevFrame  = lastFrame;
    }

    lastFrame = newFrame;
    nFrames++;
}


void System::addFrameGroup(int nImages){
    for(int i = nFrames; i < nImages; i++){
        System::addFrame(i);
    }
}


void System::addListImages(string path){
    vector<string> file_names;
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir(path.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            file_names.push_back(path + string(ent->d_name));
        }
        closedir (dir);
    }else{
        // If the directory could not be opened
        std::cout << "Directory could not be opened" <<endl;
    }
    // Sorting the vector of strings so it is alphabetically ordered
    sort(file_names.begin(), file_names.end());
    file_names.erase(file_names.begin(), file_names.begin()+2);

    file_names.size();
    imagesList = file_names;
}

void System::addCalibrationMat(string calibrationPath){
    Mat cameraMatrix;
    FileStorage opencv_file(calibrationPath, cv::FileStorage::READ);
    if (opencv_file.isOpened()){
        opencv_file["cameraMatrix"] >> cameraMat;
        opencv_file.release();
    }

    cameraMat = cameraMatrix;
}



}
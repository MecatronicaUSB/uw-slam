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
#include "CameraModel.h"
#include "Tracker.h"

namespace uw
{

System::System(void){
    initialized = false;
    rectificationValid = false;
    nFrames     = 0;
    nKeyFrames  = 0;
}

System::~System(void){
    cout << "System deleted" << endl;

}

Frame::Frame(void){
    idFrame = 0;
    isKeyFrame = false;
}

// Adding frame to system. Assuming same dimentions for all images
void System::addFrame(int id){
    Frame* newFrame   = new Frame();
    newFrame->data    = imread(imagesList[id], CV_LOAD_IMAGE_GRAYSCALE);

    if(rectificationValid){
        remap(newFrame->data, newFrame->data, this->map1, this->map2, INTER_LINEAR);
    }
    newFrame->idFrame = id;
    newFrame->isKeyFrame = false;
    this->nFrames++;
    this->currentFrame = newFrame;
    frames.push_back(newFrame);
}

void System::addKeyFrame(int id){
    if(id > this->nFrames){
        cout << "Can't add keyframe because frame " << id << " is not part of the systems frames" << endl;
        cout << "Exiting ..." << endl;
        exit(0);
    }
    
    Frame* newKeyFrame   = new Frame();
    newKeyFrame = frames[id];
    currentKeyFrame = newKeyFrame;

    this->frames[id]->isKeyFrame = true;
    this->nKeyFrames++;
    keyFrames.push_back(newKeyFrame);
}

void System::showFrame(int id){
    imshow("Show last frame", frames[id]->data);
    waitKey(0);
}

void System::addFramesGroup(int nImages){
    for(int i = nFrames; i < nImages; i++)
        System::addFrame(i);
}


void System::addListImages(string path){
    
    vector<string> file_names;
    DIR *dir;
    struct dirent *ent;

    cout << "Searching images files in directory ... ";
    if ((dir = opendir(path.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            file_names.push_back(path + string(ent->d_name));
        }
        closedir (dir);
    }else{
        // If the directory could not be opened
        cout << "can not find directory" << endl;
        cout << "Exiting..." << endl;
        exit(0);
    }
    // Sorting the vector of strings so it is alphabetically ordered
    sort(file_names.begin(), file_names.end());
    file_names.erase(file_names.begin(), file_names.begin()+2);

    if(file_names.size() < 15){
        cout << "\nInsufficient number of images found. Please use a larger dataset" << endl;
        cout << "Exiting..." << endl;
        exit(0);
    }
    cout << file_names.size() << " found"  << endl;
    imagesList = file_names;
}

void System::Calibration(string calibrationPath){
    cout << "Reading calibration xml file";
    cameraModel = new CameraModel();
    cameraModel->getCameraModel(calibrationPath);

    if(w%16!=0 || h%16!=0){
		cout << "Output image dimensions must be multiples of 32. Choose another output dimentions" << endl;
        cout << "Exiting..." << endl;
		exit(0);
	}
}

void System::initializeSystem(){
    
    this->K = cameraModel->getK();
    this->w_inp = cameraModel->getInputHeight();
    this->h_inp = cameraModel->getInputWidth();
    this->w = cameraModel->getOutputWidth();
    this->h = cameraModel->getOutputHeight();
    this->map1 = cameraModel->getMap1();
    this->map2 = cameraModel->getMap2();
    this->fx = cameraModel->getK().at<double>(0,0);
    this->fy = cameraModel->getK().at<double>(1,1);
    this->cx = cameraModel->getK().at<double>(0,2);
    this->cy = cameraModel->getK().at<double>(1,2);
    this->rectificationValid = cameraModel->isValid();

    tracker = new Tracker();
    tracker->w = this->w;
    tracker->h = this->h;

    this->initialized = true;       
}

void System::Tracking(){

    tracker->getCandidatePoints(this->currentFrame, tracker->candidatePoints);

}


}
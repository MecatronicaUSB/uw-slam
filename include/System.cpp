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
    initialized_ = false;
    rectification_valid_ = false;
    num_frames_     = 0;
    num_keyframes_  = 0;
}

System::~System(void){
    cout << "System deleted" << endl;
}

void System::Calibration(string calibration_path) {
    cout << "Reading calibration xml file";
    camera_model_ = new CameraModel();
    camera_model_->GetCameraModel(calibration_path);

    if (w_%16!=0 || h_%16!=0) {
		cout << "Output image dimensions must be multiples of 32. Choose another output dimentions" << endl;
        cout << "Exiting..." << endl;
		exit(0);
	}
}

void System::InitializeSystem() {
    this->intrinsic_camera_ = camera_model_->GetK();
    this->w_input_ = camera_model_->GetInputHeight();
    this->h_input_ = camera_model_->GetInputWidth();
    this->w_ = camera_model_->GetOutputWidth();
    this->h_ = camera_model_->GetOutputHeight();
    this->map1_ = camera_model_->GetMap1();
    this->map2_ = camera_model_->GetMap2();
    this->fx_ = camera_model_->GetK().at<double>(0,0);
    this->fy_ = camera_model_->GetK().at<double>(1,1);
    this->cx_ = camera_model_->GetK().at<double>(0,2);
    this->cy_ = camera_model_->GetK().at<double>(1,2);
    this->rectification_valid_ = camera_model_->IsValid();

    tracker_ = new Tracker();
    tracker_->w_ = this->w_;
    tracker_->h_ = this->h_;

    this->initialized_ = true;       
}

void System::Tracking() {
    tracker_->GetCandidatePoints(this->current_frame_, tracker_->candidatePoints_);
    // tracker->warpFunction();
}

Frame::Frame(void){
    idFrame_    = 0;
    isKeyFrame_ = false;
}

// Adding frame to system. Assuming same dimentions for all images
void System::AddFrame(int id) {
    Frame* newFrame   = new Frame();
    newFrame->data_   = imread(images_list_[id], CV_LOAD_IMAGE_GRAYSCALE);

    if (rectification_valid_) {
        remap(newFrame->data_, newFrame->data_, this->map1_, this->map2_, INTER_LINEAR);
    }
    newFrame->idFrame_ = id;
    newFrame->isKeyFrame_ = false;
    this->num_frames_++;
    this->current_frame_ = newFrame;
    frames_.push_back(newFrame);
}

void System::AddKeyFrame(int id) {
    if(id > this->current_frame_->idFrame_){
        cout << "Can not add keyframe because frame " << id << " is not part of the systems frames" << endl;
        cout << "Exiting ..." << endl;
        exit(0);
    }
    
    Frame* newKeyFrame   = new Frame();
    newKeyFrame = frames_[this->num_frames_ - 1];
    current_keyframe_ = newKeyFrame;

    this->frames_[this->num_frames_ - 1]->isKeyFrame_ = true;
    this->num_keyframes_++;
    keyframes_.push_back(newKeyFrame);
}

void System::ShowFrame(int id) {
    imshow("Show last frame", frames_[id]->data_);
    waitKey(0);
}

void System::AddFramesGroup(int id, int num_images) {
    for (int i = id; i < num_images; i++)
        System::AddFrame(i);
}

void System::AddListImages(string path) {
    vector<string> file_names;
    DIR *dir;
    struct dirent *ent;

    cout << "Searching images files in directory ... ";
    if ((dir = opendir(path.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            file_names.push_back(path + string(ent->d_name));
        }
        closedir (dir);
    } else {
        // If the directory could not be opened
        cout << "can not find directory" << endl;
        cout << "Exiting..." << endl;
        exit(0);
    }
    // Sorting the vector of strings so it is alphabetically ordered
    sort(file_names.begin(), file_names.end());
    file_names.erase(file_names.begin(), file_names.begin()+2);

    if (file_names.size() < 15) {
        cout << "\nInsufficient number of images found. Please use a larger dataset" << endl;
        cout << "Exiting..." << endl;
        exit(0);
    }
    cout << file_names.size() << " found"  << endl;
    images_list_ = file_names;
}





}

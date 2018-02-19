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

#include "System.h"
#include "CameraModel.h"
#include "Tracker.h"

namespace uw
{

System::System(int argc, char *argv[], int start_index) {
    ros::init(argc, argv, "uw_slam");
    start_index_ = start_index;
    initialized_ = false;
    rectification_valid_ = false;
    num_frames_     = 0;
    num_keyframes_  = 0;
}

System::~System(void) {
    cout << "System deleted" << endl;
}

Frame::Frame(void) {
    idFrame_    = 0;
    isKeyFrame_ = false;
}

Frame::~Frame(void) {}


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
    K_ = camera_model_->GetK();
    w_input_ = camera_model_->GetInputHeight();
    h_input_ = camera_model_->GetInputWidth();
    w_ = camera_model_->GetOutputWidth();
    h_ = camera_model_->GetOutputHeight();
    map1_ = camera_model_->GetMap1();
    map2_ = camera_model_->GetMap2();
    fx_ = camera_model_->GetK().at<double>(0,0);
    fy_ = camera_model_->GetK().at<double>(1,1);
    cx_ = camera_model_->GetK().at<double>(0,2);
    cy_ = camera_model_->GetK().at<double>(1,2);
    rectification_valid_ = camera_model_->IsValid();

    tracker_ = new Tracker();
    tracker_->InitializePyramid(w_, h_, K_);

    ros::NodeHandle nodehandle_camera_pose;
    visualizer_ = new Visualizer(start_index_, images_list_.size());
    visualizer_->ReadGroundTruthEUROC("/home/fabio/Documents/datasets/EUROC/V1_02_medium/mav0/state_groundtruth_estimate/data.csv");

    cout << "Initializing system ... done" << endl;
    initialized_ = true;
}

void System::Tracking() {
    // tracker_->EstimatePose(previous_frame_, current_frame_);
    // tracker_->GetCandidatePoints(current_frame_, current_frame_->candidatePoints_);
    // tracker->warpFunction();
}

void System::AddFrame(int id) {
    Frame* newFrame   = new Frame();
    newFrame->idFrame_ = id;
    newFrame->image[0]   = imread(images_list_[id], CV_LOAD_IMAGE_GRAYSCALE);

    if (rectification_valid_)
        remap(newFrame->image[0], newFrame->image[0], map1_, map2_, INTER_LINEAR);

    for (int i=1; i<PYRAMID_LEVELS; i++)
        resize(newFrame->image[i-1], newFrame->image[i], Size(), 0.5, 0.5);

    // for (int i=0; i<PYRAMID_LEVELS; i++){
    //     imshow("", newFrame->image[i]);
    //     waitKey(0);
    // }
    if (num_frames_ == 0) {
        current_frame_ = newFrame;
    } else {
        previous_frame_ = current_frame_;
        current_frame_ = newFrame;
    }
    frames_.push_back(newFrame);
    num_frames_++;
}

void System::AddKeyFrame(int id) {
    if(id > current_frame_->idFrame_){
        cout << "Can not add keyframe because frame " << id << " is not part of the systems frames" << endl;
        cout << "Exiting ..." << endl;
        exit(0);
    }

    Frame* newKeyFrame   = new Frame();
    newKeyFrame = frames_[num_frames_ - 1];
    current_keyframe_ = newKeyFrame;

    frames_[num_frames_ - 1]->isKeyFrame_ = true;
    num_keyframes_++;
    keyframes_.push_back(newKeyFrame);
}

void System::ShowFrame(int id) {
    imshow("Show last frame", frames_[id]->image[0]);
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

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

namespace uw
{

System::System(){
    nFrames  = 0;
}

System::~System(){

}

Frame::Frame(){
    idFrame  = 0;
}

void System::addFrame(int id){
    Frame* newFrame   = new Frame();
    newFrame->data    = imread(imagesList[id], CV_LOAD_IMAGE_GRAYSCALE);
    newFrame->idFrame = id;

    frames.push_back(newFrame);

}

void System::showFrame(int id){
    imshow(to_string(id), frames[id]->data);
    waitKey(0);
}

void System::addFrameGroup(int nImages){
    for(int i = nFrames; i < nImages; i++)
        System::addFrame(i);
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

void System::Calibration(string calibrationPath){
    cameraModel = new CameraModel();
    cameraModel->getCameraModel(calibrationPath);
    this->K = cameraModel->getK();
    this->w = cameraModel->getOutputWidth();
    this->h = cameraModel->getOutputHeight();
    this->map1 = cameraModel->getMap1();
    this->map2 = cameraModel->getMap2();
    this->rectificationValid = cameraModel->isValid();
}

Mat System::applyGradient(int id){
    Mat gradientImage;
    cuda::GpuMat frameX, frameY, frame;

    frameX.upload(frames[id]->data);
    frameY.upload(frames[id]->data);

    soberX->apply(frameX, frameX);
    soberX->apply(frameY, frameY);

    cuda::addWeighted(frameX, 0.5, frameY, 0.5, 0, frame);

    Mat example;
    for(int x = 0; x < TARGET_WIDTH; x+= BLOCK_SIZE ){
        for(int y = 0; y < TARGET_HEIGHT; y += BLOCK_SIZE){
            cuda::GpuMat block = cuda::GpuMat(frame, Rect(x,y,BLOCK_SIZE,BLOCK_SIZE));
            block.download(example);
        }
    }
    
    frame.download(gradientImage);
    imshow("0", gradientImage);     
    waitKey(0);
    return gradientImage;
}

}
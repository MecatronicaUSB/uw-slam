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
#include "Tracker.h"
#include "System.h"

namespace uw
{

Tracker::~Tracker(void){};

void Tracker::getCandidatePoints(Frame* frame, vector<Point2d> candidatePoints){
    
    cuda::GpuMat frameX, frameY, frameGPU;

    frameX.upload(frame->data);
    frameY.upload(frame->data);

    soberX->apply(frameX, frameX);
    soberX->apply(frameY, frameY);

    cuda::addWeighted(frameX, 0.5, frameY, 0.5, 0, frameGPU);

    for( int x = 0; x < w; x += BLOCK_SIZE ){
        for( int y = 0; y < h; y += BLOCK_SIZE ){
            Scalar mean, stdev;
            Point min_loc, max_loc;
            double min, max;
            double threshold = GTH;
            cuda::GpuMat block = cuda::GpuMat(frameGPU, Rect(x,y,BLOCK_SIZE,BLOCK_SIZE));
            cuda::meanStdDev(block, mean, stdev);
            threshold += mean[0];
            cuda::minMaxLoc(block, &min, &max, &min_loc, &max_loc);
            if( max > threshold ){
                max_loc.x += x;
                max_loc.y += y;
                this->candidatePoints.push_back(max_loc);
            }
        }
    }

    debugShowCandidatePoints(frame);
}

void Tracker::debugShowCandidatePoints(Frame* frame){
    Mat showPoints;
    cvtColor(frame->data, showPoints, CV_GRAY2RGB);
    
    for( int i=0; i<this->candidatePoints.size(); i++ )
        circle(showPoints, candidatePoints[i], 2, Scalar(255,0,0), 1, 8, 0);

    imshow("Show candidates points", showPoints);
    waitKey(0);
    this->candidatePoints.clear();
}


}
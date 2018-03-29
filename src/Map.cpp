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

#include "Map.h"
#include "System.h"

namespace uw
{
class Frame;

Map::Map() {
 
};

void Map::AddPointCloudFromRGBD(Frame* frame) {
    int num_cloud_points = frame->candidatePoints_[0].rows;

    recent_cloud_points_ = Mat(num_cloud_points, 3, CV_32FC1);

    for (int i=0; i<frame->candidatePoints_[0].rows; i++) {

        recent_cloud_points_.at<float>(i,0) = frame->candidatePoints_[0].at<float>(i,0);
        recent_cloud_points_.at<float>(i,1) = frame->candidatePoints_[0].at<float>(i,1);
        recent_cloud_points_.at<float>(i,2) = frame->candidatePoints_[0].at<float>(i,2);
    
    }
};


}
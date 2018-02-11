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

#include "CameraModel.h"

namespace uw
{

CameraModel::CameraModel(){
    bool valid = false;
}

CameraModel::~CameraModel()
{
	if(remapX != 0) delete[] remapX;
	if(remapY != 0) delete[] remapY;
}

// Checks which Camera Model Implement (FOV,...?)
void CameraModel::getCameraModel(int in_width, int in_height, int  out_width, int out_height, 
                                                        Mat calibration_values, Mat rectification)
{
    wOrg = in_width;
    hOrg = in_height;
    w = out_width;
    h = out_height;
    
    // FOV Camera Model
    for(int i = 0; i < 5; i++){
        intrinsicParam.push_back(calibration_values.at<double>(0,i));
        distCoeff.push_back(rectification.at<double>(0,i));
    }

    // TODO - So what's the point of having rectification?
    if( distCoeff[0] != 0 ){
        // Rescale and substract 0.5 offset
        K.at<float>(0,0) = distCoeff[0] * w;
        K.at<float>(1,1) = distCoeff[1] * h;
        K.at<float>(0,2) = distCoeff[2] * w - 0.5;
        K.at<float>(1,2) = distCoeff[3] * h - 0.5;

    }else{
        if( intrinsicParam[2] < 1 && intrinsicParam[3] < 1){
            // Rescale and substract 0.5 offset        
            K.at<float>(0,0) = intrinsicParam[0] * in_width;
            K.at<float>(1,1) = intrinsicParam[1] * in_height;
            K.at<float>(0,2) = intrinsicParam[2] * in_width - 0.5;
            K.at<float>(1,2) = intrinsicParam[3] * in_height - 0.5;
        }
        else{
            // No need to rescale and substract 0.5 offset    
            K.at<float>(0,0) = intrinsicParam[0];
            K.at<float>(1,1) = intrinsicParam[1];
            K.at<float>(0,2) = intrinsicParam[2];
            K.at<float>(1,2) = intrinsicParam[3];
        }
    }

    remapX = new float[w*h];
    remapY = new float[w*h];

	for(int y=0;y<h;y++)
		for(int x=0;x<w;x++){
			remapX[x+y*w] = x;
			remapY[x+y*w] = y;
		}

    // FOV
    distortCordinatesFOV(remapX, remapY, remapX, remapY, w*h);
	for(int y=0;y<h;y++)
		for(int x=0;x<w;x++){
			// Rounding resistant
			float ix = remapX[x+y*w];
			float iy = remapY[x+y*w];

			if(ix == 0) ix = 0.001;
			if(iy == 0) iy = 0.001;
			if(ix == in_width-1) ix = in_width-1.001;
			if(iy == in_height-1) ix = in_height-1.001;

			if(ix > 0 && iy > 0 && ix < in_width-1 &&  iy < in_height-1)
			{
				remapX[x+y*w] = ix;
				remapY[x+y*w] = iy;
			}
			else
			{
				remapX[x+y*w] = -1;
				remapY[x+y*w] = -1;
			}
		}
}

void CameraModel::distortCordinatesFOV(float* in_x, float* in_y, float* out_x, float* out_y, int n)
{
	float dist = intrinsicParam[4];
	float d2t = 2.0f * tan(dist / 2.0f);

	// Current Camera Parameters
    float fx = intrinsicParam[0];
    float fy = intrinsicParam[1];
    float cx = intrinsicParam[2];
    float cy = intrinsicParam[3];
    
    // Output Camera Intrinsic Parameters
	float ofx = K.at<float>(0,0);
	float ofy = K.at<float>(1,1);
	float ocx = K.at<float>(0,2);
	float ocy = K.at<float>(1,2);

	for(int i=0;i<n;i++){
		float x = in_x[i];
		float y = in_y[i];
		float ix = (x - ocx) / ofx;
		float iy = (y - ocy) / ofy;

		float r = sqrtf(ix*ix + iy*iy);
		float fac = (r==0 || dist==0) ? 1 : atanf(r * d2t)/(dist*r);

		ix = fx*fac*ix+cx;
		iy = fy*fac*iy+cy;

		out_x[i] = ix;
		out_y[i] = iy;
	}
}


}
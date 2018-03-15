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

#include "LeastSquares.h"
#include "Tracker.h"
#include "System.h"


#include <opencv2/core/eigen.hpp>

namespace uw
{

class LS;
class ResidualIntensity;
class LocalParameterizationSE3;

Tracker::Tracker(bool _depth_available) {
    depth_available_ = _depth_available;
    for (Mat K: K_)
        K = Mat(3,3,CV_32FC1, Scalar(0.f));
};

Tracker::~Tracker() {
    w_.clear();
    h_.clear();
    fx_.clear();
    fy_.clear();
    cx_.clear();
    cy_.clear();
    invfx_.clear();
    invfy_.clear();
    invcx_.clear();
    invcy_.clear();

    K_.clear();
};

// TODO(GitHub:fmoralesh, fabmoraleshidalgo@gmail.com)
// 02-20-2018 - Review: Obtaining the precomputed invK (for each pyramid level) is necessary?
void Tracker::InitializePyramid(int _width, int _height, Mat _K) {
    w_[0] = _width;
    h_[0] = _height;
    K_[0] = _K;
    // invK_[0] = _K.inv();

    fx_[0] = _K.at<float>(0,0);
    fy_[0] = _K.at<float>(1,1);
    cx_[0] = _K.at<float>(0,2);
    cy_[0] = _K.at<float>(1,2);
    
    invfx_[0] = 1 / fx_[0]; 
    invfy_[0] = 1 / fy_[0]; 
    invcx_[0] = 1 / cx_[0]; 
    invcy_[0] = 1 / cy_[0]; 

    for (int lvl = 1; lvl < PYRAMID_LEVELS; lvl++) {
        w_[lvl] = _width >> lvl;
        h_[lvl] = _height >> lvl;
        
        fx_[lvl] = fx_[lvl-1] * 0.5;
        fy_[lvl] = fy_[lvl-1] * 0.5;
        cx_[lvl] = (cx_[0] + 0.5) / ((int)1<<lvl) - 0.5;
        cy_[lvl] = (cy_[0] + 0.5) / ((int)1<<lvl) - 0.5;

        K_[lvl] = Mat::eye(Size(3,3), CV_32FC1);
        K_[lvl].at<float>(0,0) = fx_[lvl];
        K_[lvl].at<float>(1,1) = fy_[lvl];
        K_[lvl].at<float>(0,2) = cx_[lvl];  
        K_[lvl].at<float>(1,2) = cy_[lvl];    

        invfx_[lvl] = 1 / fx_[lvl];
        invfy_[lvl] = 1 / fy_[lvl];
        invcx_[lvl] = 1 / cx_[lvl];
        invcy_[lvl] = 1 / cy_[lvl];
        
        // Needs review
        // invK_[lvl] = K_[lvl].inv();
        // invfx_[lvl] = invK_[lvl].at<float>(0,0); 
        // invfy_[lvl] = invK_[lvl].at<float>(1,1);
        // invcx_[lvl] = invK_[lvl].at<float>(0,2);
        // invcy_[lvl] = invK_[lvl].at<float>(1,2);
    }
}


// Gauss-Newton using Foward Compositional Algorithm
void Tracker::EstimatePose(Frame* _previous_frame, Frame* _current_frame) {
    // Gauss-Newton Options
    int max_iterations = 6;
    float error_threshold = 0.005;

    // Variables initialization
    float error = 0.0;
    float last_error = 0.0;

    // Initial pose and deltapose (assumes little movement between frames)
    Mat deltaMat = Mat::zeros(6,1,CV_32FC1);
    Sophus::Vector<float, SE3::DoF> deltaVector;
    deltaVector(0) = 0;
    deltaVector(1) = 0;
    deltaVector(2) = 0;
    deltaVector(3) = 0;
    deltaVector(4) = 0;
    deltaVector(5) = 0;
    SE3 current_pose = SE3(SO3::exp(SE3::Point(0.0, 0.0, 0.0)), SE3::Point(-0.0004, 0.0, 0.0003));

    // Sparse to Fine iteration
    // Create for()
    int lvl = 1;
    // Initialize error   
    error = 0.0;
    last_error = 50000.0;
    
    // Obtain points and depth of initial frame 
    Mat candidatePoints      = _previous_frame->candidatePoints_[lvl].clone();
    Mat candidatePointsDepth = _previous_frame->candidatePointsDepth_[lvl].clone();

    // Obtain gradients
    Mat image1 = _previous_frame->images_[lvl].clone();
    Mat image2 = _current_frame->images_[lvl].clone();    
    Mat gradient1 = _previous_frame->gradient_[lvl].clone();
    Mat gradient2 = _current_frame->gradient_[lvl].clone();    
    Mat gradientX1 = _previous_frame->gradientX_[lvl].clone();
    Mat gradientY1 = _previous_frame->gradientY_[lvl].clone();

    // Obtain intrinsic parameters 
    Mat K = K_[lvl];

    // Computation of Jw (constant in Compositional Algorithm)
    vector<Mat> Jws;
    for (int i=0; i<candidatePoints.rows; i++) {

        Mat Jw = Mat(2,6,CV_32FC1);

        // Points of frame 1
        float x1 = candidatePoints.at<float>(i,0);
        float y1 = candidatePoints.at<float>(i,1);
        float z1 = candidatePoints.at<float>(i,2);

        float inv_z1 = 1 / z1;

        Jw.at<float>(0,0) = fx_[lvl] * inv_z1;
        Jw.at<float>(0,1) = 0.0;
        Jw.at<float>(0,2) = -(fx_[lvl] * x1 * inv_z1 * inv_z1);
        Jw.at<float>(0,3) = -(fx_[lvl] * x1 * y1 * inv_z1 * inv_z1);
        Jw.at<float>(0,4) = (fx_[lvl] * (1 + x1 * x1 * inv_z1 * inv_z1));   
        Jw.at<float>(0,5) = - fx_[lvl] * y1 * inv_z1;

        Jw.at<float>(1,0) = 0.0;
        Jw.at<float>(1,1) = fy_[lvl] * inv_z1;
        Jw.at<float>(1,2) = -(fy_[lvl] * y1 * inv_z1 * inv_z1);
        Jw.at<float>(1,3) = -(fy_[lvl] * (1 + y1 * y1 * inv_z1 * inv_z1));
        Jw.at<float>(1,4) = fy_[lvl] * x1 * y1 * inv_z1 * inv_z1;
        Jw.at<float>(1,5) = fy_[lvl] * x1 * inv_z1;

        Jws.push_back(Jw);
    }
    
    // Optimization iteration
    for (int k=0; k<max_iterations; k++) {

        // Warp points with current pose and delta pose (from previous iteration)
        SE3 deltaSE3;
        Mat warpedPoints = Mat(candidatePoints.size(), CV_32FC1);
        warpedPoints = WarpFunction(candidatePoints, SE3::exp(deltaVector) * current_pose, lvl);

        Mat imageWarped = Mat::zeros(image1.size(), CV_8UC1);
        ObtainImageTransformed(image1, candidatePoints, warpedPoints, imageWarped);
        imshow("Image warped",imageWarped);
        waitKey(0);
        //DebugShowWarpedPerspective(gradient1, gradient2, candidatePoints, warpedPoints, lvl);

        Mat gradientX2 = Mat(imageWarped.size(), CV_32FC1);
        Mat gradientY2 = Mat(imageWarped.size(), CV_32FC1);
        ObtainGradientXY(imageWarped, gradientX2, gradientY2);

        // Computation of Jacobian and Residuals
        vector<Mat> Jacobian = vector<Mat>(candidatePoints.rows);    
        vector<uchar> intensities1;  
        vector<uchar> intensities2;
        for (int i=0; i<candidatePoints.rows; i++) {
            
            Mat Jl = Mat(1,2,CV_32FC1);
            // Point in frame 1            
            float x1 = candidatePoints.at<float>(i,0);
            float y1 = candidatePoints.at<float>(i,1);
            float z1 = candidatePoints.at<float>(i,2);
            // Warped point in frame 2
            int x2 = round(warpedPoints.at<float>(i,0));
            int y2 = round(warpedPoints.at<float>(i,1));
            int z2 = round(warpedPoints.at<float>(i,2));

            int intensity1 = image1.at<uchar>(y1,x1);
            int intensity2;
            if (x2>=0 && x2<image2.cols && y2>=0 && y2<image2.rows) {
                intensity2 = image2.at<uchar>(y2,x2);
            } else {
                intensity2 = 0;   
            }

            intensities1.push_back(intensity1);                
            intensities2.push_back(intensity2);

            Jl.at<float>(0,0) = gradientX2.at<int>(y1,x1);
            Jl.at<float>(0,1) = gradientY2.at<int>(y1,x1);

            Jacobian[i] = Mat(1,6, CV_32FC1);
            Jacobian[i] = Jl * Jws[i];
        }

        DebugShowJacobians(Jacobian, imageWarped);

        // Computation of Residuals
        // Workaround to work with float numbers (intensities are in CV_8UC1)  
        Mat Residuals = Mat(intensities2.size(),1,CV_8UC1);        
        Mat I1 = Mat(intensities1.size(),1, CV_32FC1);
        Mat I2 = Mat(intensities2.size(),1, CV_32FC1);

        Mat aux1 = Mat(intensities1);
        Mat aux2 = Mat(intensities2);
        aux1.convertTo(I1, CV_32FC1);
        aux2.convertTo(I2, CV_32FC1);

        Residuals = I2 - I1;

        // Computation of Weights (Identity or Tukey function)
        Mat W = IdentityWeights(Residuals.rows);
        //Mat W = TukeyFunctionWeights(Residuals);

        // Computation of error
        float inv_num_residuals = 1.0 / Residuals.rows;
        Mat ResidualsW = Residuals.mul(W);
        Mat errorMat =  inv_num_residuals * Residuals.t() * ResidualsW;
        error = errorMat.at<float>(0,0);

        // Break if error increases
        if (error > last_error) {
            deltaMat = Mat::zeros(6,1,CV_32FC1);
            deltaVector(0) = 0;
            deltaVector(1) = 0;
            deltaVector(2) = 0;
            deltaVector(3) = 0;
            deltaVector(4) = 0;
            deltaVector(5) = 0;

            // Show results of optimization at lvl 0
            if (lvl == 0){
                // DebugShowResidual(gradient1, gradient2, candidatePoints, warpedPoints, lvl);
                DebugShowWarpedPerspective(gradient1, gradient2, candidatePoints, warpedPoints, lvl);
            }
            break;
        }

        cout << "Error: " << error << endl;
        last_error = error;

        // Update new pose with delta
        current_pose = SE3::exp(deltaVector) * current_pose;

        //DebugShowWarpedPerspective(gradient1, gradient2, candidatePoints, warpedPoints, lvl);   

        // Computation of new delta
        LS ls;
        ls.initialize(Residuals.rows);
        // for (int i=0; i<Residuals.rows; i++) {
        //     Mat61f jacobian;
        //     cv2eigen(Jacobian.row(i), jacobian);
        //     ls.update(jacobian, Residuals.at<float>(i,0), W.at<float>(i,0));
        // }
        ls.finish();
        // Solve LS system
        float LM_lambda = 0.2;
        Mat61f b = -ls.b;
        Mat66f A = ls.A;
        cout << A << endl;
        cout << b << endl;
        deltaVector = A.ldlt().solve(b);
        cout << deltaVector << endl;
        // deltaMat = -1 * ((Jacobian.t() * JacobianW).inv() * Jacobian.t() * ResidualsW);

        // Apply update
        // deltaVector(0) = deltaMat.at<float>(0,0);
        // deltaVector(1) = deltaMat.at<float>(1,0);
        // deltaVector(2) = deltaMat.at<float>(2,0);
        // deltaVector(3) = deltaMat.at<float>(3,0);
        // deltaVector(4) = deltaMat.at<float>(4,0);
        // deltaVector(5) = deltaMat.at<float>(5,0);
    }
    

    // Mat candidatePoints      = _previous_frame->candidatePoints_[0].clone();
    // Mat candidatePointsDepth = _previous_frame->candidatePointsDepth_[0].clone();
    // Mat warpedPoints = Mat(candidatePoints.size(), CV_32FC1);
    // warpedPoints = WarpFunction(candidatePoints, candidatePointsDepth, current_pose, 0);
    // DebugShowWarpedPoints(_previous_frame->gradient_[0], _current_frame->gradient_[0], candidatePoints, warpedPoints);
    _previous_frame->rigid_transformation_ = current_pose;

}

// TODO(GitHub:fmoralesh, fabmoraleshidalgo@gmail.com)
// 02-20-2018 - Consider other methods to obtain gradient from an image (Sober, Laplacian, ...) 
//            - Calculate gradient for each pyramid image or scale the finest?
void Tracker::ApplyGradient(Frame* _frame) {
    Mat gradient;
    Mat gradientX; 
    Mat gradientY;

    // Ptr<cuda::Filter> soberX_ = cuda::createDerivFilter(0, CV_16S, 1, 0, 3, 0,BORDER_DEFAULT,BORDER_DEFAULT);
    // Ptr<cuda::Filter> soberY_ = cuda::createDerivFilter(0, CV_16S, 0, 1, 3, 0,BORDER_DEFAULT,BORDER_DEFAULT);    
    Ptr<cuda::Filter> soberX_ = cuda::createSobelFilter(0, CV_16S, 1, 0, 3, 1, BORDER_DEFAULT, BORDER_DEFAULT);
    Ptr<cuda::Filter> soberY_ = cuda::createSobelFilter(0, CV_16S, 0, 1, 3, 1, BORDER_DEFAULT, BORDER_DEFAULT);

    for (int lvl=PYRAMID_LEVELS-1; lvl>=0; lvl--) {
        // Filters for calculating gradient in images
        cuda::GpuMat frameGPU = cuda::GpuMat(_frame->images_[lvl]);
        // Apply gradient in x and y
        cuda::GpuMat frameXGPU, frameYGPU;

        cuda::GpuMat absX, absY, out;
        soberX_->apply(frameGPU, frameXGPU);
        soberY_->apply(frameGPU, frameYGPU);
        cuda::abs(frameXGPU, frameXGPU);
        cuda::abs(frameYGPU, frameYGPU);
        frameXGPU.convertTo(absX, CV_8UC1);
        frameYGPU.convertTo(absY, CV_8UC1);
        
        cuda::addWeighted(absX, 0.5, absY, 0.5, 0, out);

        absX.download(gradientX);
        absY.download(gradientY);
        out.download(gradient);

        _frame->gradient_[lvl] = gradient.clone();
        _frame->gradientX_[lvl] = gradientX.clone();
        _frame->gradientY_[lvl] = gradientY.clone();

    }

    _frame->obtained_gradients_ = true;
    
}


void Tracker::ObtainAllPoints(Frame* _frame) {
    // Factor of TUM depth images
    float factor = 0.0002;
    for (int lvl=0; lvl< PYRAMID_LEVELS; lvl++) {
        _frame->candidatePoints_[lvl] = Mat::ones(w_[lvl] * h_[lvl], 4, CV_32FC1);
        for (int x=0; x<w_[lvl]; x++) {
            for (int y =0; y<h_[lvl]; y++) {
                Point3f point;
                point.x = x;
                point.y = y;
                point.z = 1;
                _frame->framePoints_[lvl].push_back(point);
                _frame->candidatePoints_[lvl].at<float>(y+h_[lvl]*x,0) = x;
                _frame->candidatePoints_[lvl].at<float>(y+h_[lvl]*x,1) = y;
                if (_frame->depth_available_){
                    _frame->candidatePoints_[lvl].at<float>(y+h_[lvl]*x,2) = _frame->depths_[lvl].at<uchar>(y,x) * factor;
                } else {
                    _frame->candidatePoints_[lvl].at<float>(y+h_[lvl]*x,2) = 1;
                }
            }
        } 
    }
    _frame->obtained_candidatePoints_ = true;
}

// TODO(GitHub:fmoralesh, fabmoraleshidalgo@gmail.com)
// 02-13-2018 - Implement a faster way to obtain candidate points with high gradient value in patches (above of a certain threshold)
void Tracker::ObtainCandidatePoints(Frame* _frame) {
    // Block size search for high gradient points in image 
    for (int i = 0; i < PYRAMID_LEVELS - 1; i++){
        int block_size = BLOCK_SIZE - i * 5;
        cuda::GpuMat frameGPU(_frame->gradient_[i]);
        for (int x=0; x<w_[i]-block_size; x+=block_size) {
            for (int y =0; y<h_[i]-block_size; y+=block_size) {
                Mat point = Mat::ones(1,4,CV_32FC1);
                Mat depth = Mat::ones(1,1,CV_32FC1);                
                Scalar mean, stdev;
                Point min_loc, max_loc;
                double min, max;
                cuda::GpuMat block(frameGPU, Rect(x,y,block_size,block_size));
                block.convertTo(block, CV_8UC1);
                cuda::meanStdDev(block, mean, stdev);
                //cuda::minMaxLoc(block, &min, &max, &min_loc, &max_loc);
                
                if (max > mean[0] + GRADIENT_THRESHOLD) {
                    point.at<float>(0,0) = (float) (x + max_loc.x);
                    point.at<float>(0,1) = (float) (y + max_loc.y);

                    _frame->candidatePoints_[i].push_back(point);
                    _frame->candidatePointsDepth_[i].push_back(depth);
                }
            }
        }
    }

    for (int lvl = 1; lvl < PYRAMID_LEVELS; lvl++) {
        // frame->candidatePoints_[lvl] = frame->candidatePoints_[lvl-1] * 0.5;
        // DebugShowCandidatePoints(frame->gradient_[lvl-1], frame->candidatePoints_[lvl-1]);
    }
    _frame->obtained_candidatePoints_ = true;
}

Mat Tracker::WarpFunction(Mat _points2warp, SE3 _rigid_transformation, int _lvl) {
    int lvl = _lvl;
    Mat33f R = _rigid_transformation.rotationMatrix();
    Mat31f t = _rigid_transformation.translation();
    Quaternion2 quaternion = _rigid_transformation.unit_quaternion();

    Mat projected_points = Mat(_points2warp.size(), CV_64FC1);
    projected_points = _points2warp.clone();

    Mat44f rigidEigen = _rigid_transformation.matrix();
    Mat rigid = Mat(4,4,CV_32FC1);
    eigen2cv(rigidEigen, rigid);
    //cout << rigid << endl;

    float fx = fx_[lvl];
    float fy = fy_[lvl];
    float invfx = invfx_[lvl];
    float invfy = invfy_[lvl];
    float cx = cx_[lvl];
    float cy = cy_[lvl];
    // cout << "fx: " << fx << endl;
    // cout << "fy: " << fy << endl;
    // cout << "cx: " << cx << endl;
    // cout << "cy: " << cy << endl;
    
    // 2D -> 3D
    // X  = (x - cx) * Z / fx
    projected_points.col(0) = ((projected_points.col(0) - cx) * invfx);
    cout << projected_points.row(0) << endl;
    projected_points.col(0) = projected_points.col(0).mul(projected_points.col(2));
    cout << projected_points.row(0) << endl;

    // Y  = (y - cy) * Z / fy    
    projected_points.col(1) = ((projected_points.col(1) - cy) * invfy);
    projected_points.col(1) = projected_points.col(1).mul(projected_points.col(2));

    // Z = Z (depth)

    // Transformation of a point rigid body motion
    projected_points = rigid * projected_points.t();

    // 3D -> 2D
    // x = (X * fx / Z) + cx
    projected_points.row(0) /= projected_points.row(2);
    projected_points.row(0) *= fx;
    projected_points.row(0) += cx;
    
    // x = (Y * fy / Z) + cy    
    projected_points.row(1) /= projected_points.row(2);
    projected_points.row(1) *= fy;
    projected_points.row(1) += cy;
    
    // Transposing the points due transformation multiplication
    return projected_points.t();
}

void Tracker::ObtainImageTransformed(Mat _originalImage, Mat _candidatePoints, Mat _warpedPoints, Mat _outputImage) {

    // Obtaining warped image from warpedpoints
    Mat validPixel = Mat::zeros(_outputImage.size(), CV_8UC1);
    for (int i=0; i<_warpedPoints.rows; i++) {
        int x1 = _candidatePoints.at<float>(i,0);
        int y1 = _candidatePoints.at<float>(i,1);
        int x2 = round(_warpedPoints.at<float>(i,0));
        int y2 = round(_warpedPoints.at<float>(i,1));

        if (y2>0 && y2<_originalImage.rows && x2>0 && x2<_originalImage.cols){
            _outputImage.at<uchar>(y2,x2) = _originalImage.at<uchar>(y1,x1);
            validPixel.at<uchar>(y2,x2) = 1;
        }
    } 
    // Applying bilinear interpolation of resulting waped image
    for (int x=0; x<_outputImage.cols; x++) {
        for (int y=0; y<_outputImage.rows; y++) {
            
            if (_outputImage.at<uchar>(y,x) == 0) {                
                int x1 = x - 1;
                int x2 = x + 1;
                int y1 = y - 1;
                int y2 = y + 1;
                if (x1 < 0) x1 = 0;
                if (y1 < 0) y1 = 0;
                if (x2 == _outputImage.cols) x2 = x2-1;   
                if (y2 == _outputImage.rows) y2 = y2-1;
                if (validPixel.at<uchar>(y1,x1) == 1 || validPixel.at<uchar>(y1,x) == 1 || validPixel.at<uchar>(y1,x2) == 1 ||
                    validPixel.at<uchar>(y,x1)  == 1 || validPixel.at<uchar>(y,x)  == 1 || validPixel.at<uchar>(y,x2)  == 1 ||
                    validPixel.at<uchar>(y2,x1) == 1 || validPixel.at<uchar>(y2,x) == 1 || validPixel.at<uchar>(y2,x2) == 1    ) {

                    int Q11 = _outputImage.at<uchar>(y2,x1);
                    int Q21 = _outputImage.at<uchar>(y2,x2);
                    int Q12 = _outputImage.at<uchar>(y1,x1);
                    int Q22 = _outputImage.at<uchar>(y1,x2);

                    if (Q12 == 0) Q12 = Q22;
                    if (Q22 == 0) Q22 = Q12;
                    if (Q11 == 0) Q11 = Q21;
                    if (Q21 == 0) Q21 = Q11;
                    
                    int f_y1 = (Q12 * 0.5) + (Q22 * 0.5);
                    int f_y2 = (Q11 * 0.5) + (Q21 * 0.5);
                    
                    _outputImage.at<uchar>(y,x) = (f_y1 * 0.5) + (f_y2 * 0.5);
                }
            }
        }
    }
}

bool Tracker::PixelIsBackground(Mat _inputImage, int y, int x) {
    if (_inputImage.at<uchar>(y-1,x-1) == 1) return true;
    if (_inputImage.at<uchar>(y-1,x+1) == 1) return true;
    if (_inputImage.at<uchar>(y+1,x-1) == 1) return true;
    if (_inputImage.at<uchar>(y+1,x+1) == 1) return true;

    if (_inputImage.at<uchar>(y,x+1) == 1) return true;
    if (_inputImage.at<uchar>(y,x-1) == 1) return true;
    if (_inputImage.at<uchar>(y-1,x) == 1) return true;
    if (_inputImage.at<uchar>(y+1,x) == 1) return true;
    
    return false;    
}

void Tracker::ObtainGradientXY(Mat _inputImage, Mat& _gradientX, Mat& _gradientY) {
    // // Filters for calculating gradient in images
    // cuda::GpuMat frameGPU = cuda::GpuMat(_inputImage);
    // Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(frameGPU.type(), frameGPU.type(), Size(3,3), 0);

    // // Apply gradient in x and y
    // cuda::GpuMat frameXGPU, frameYGPU;   
    // Ptr<cuda::Filter> soberX_ = cuda::createSobelFilter(0, CV_16S, 1, 0, 3, 1, BORDER_DEFAULT, BORDER_DEFAULT);
    // Ptr<cuda::Filter> soberY_ = cuda::createSobelFilter(0, CV_16S, 0, 1, 3, 1, BORDER_DEFAULT, BORDER_DEFAULT);

    // cuda::GpuMat absX, absY, out;
    // soberX_->apply(frameGPU, frameXGPU);
    // soberY_->apply(frameGPU, frameYGPU);

    // frameXGPU.download(_gradientX);
    // frameYGPU.download(_gradientY);

    // Non CUDA Implementation
    Scharr(_inputImage, _gradientX, CV_32FC1, 1, 0, 1, 0, BORDER_REFLECT);
    Scharr(_inputImage, _gradientY, CV_32FC1, 0, 1, 1, 0, BORDER_REFLECT);

}

void Tracker::DebugShowJacobians(vector<Mat> Jacobians, Mat original) {
    vector<Mat> image_jacobians = vector<Mat>(6);
    for (int i=0; i<6; i++) 
        image_jacobians[i] = Mat::zeros(original.size(), CV_8UC1);

    int index = 0;
    for (int x=0; x<original.cols; x++) {
        for (int y=0; y<original.rows; y++) {
            for (int i=0; i<6; i++){
                if (Jacobians[index].at<float>(0,i) < -10){
                    image_jacobians[i].at<uchar>(y,x) = 90;
                }
                if (Jacobians[index].at<float>(0,i) > 10) {             
                    image_jacobians[i].at<uchar>(y,x) = 255;
                }
            }
            index++;
        }
    }

    cout << Jacobians.size() << endl;
    cout << index << endl;
    imshow("Jacobian for v1", image_jacobians[0]);
    imshow("Jacobian for v2", image_jacobians[1]);
    imshow("Jacobian for v3", image_jacobians[2]);
    imshow("Jacobian for w1", image_jacobians[3]);
    imshow("Jacobian for w2", image_jacobians[4]);
    imshow("Jacobian for w3", image_jacobians[5]);
    
    waitKey(0);
}

float Tracker::MedianMat(Mat _input) {
    Mat channel = Mat(_input.rows, _input.cols, CV_8UC1);
    _input.convertTo(channel, CV_8UC1);

    float m = (channel.rows*channel.cols) / 2;
    int bin = 0;
    float med = -1.0;

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;
    Mat hist;
    calcHist(&channel, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    for (int i = 0; i < histSize && med < 0.0; ++i) {
        bin += cvRound(hist.at<float>(i));
        if (bin > m && med < 0.0)
            med = i;
    }

    return med;
}

SE3 Tracker::Mat2SE3(Mat _input) {
    float w1 = _input.at<float>(0,0);
    float w2 = _input.at<float>(1,0);
    float w3 = _input.at<float>(2,0);
    float x1 = _input.at<float>(3,0);
    float x2 = _input.at<float>(4,0);
    float x3 = _input.at<float>(5,0);
    
    return SE3(SO3::exp(SE3::Point(w1, w2, w3)), SE3::Point(x1, x2, x3));
}

float Tracker::MedianAbsoluteDeviation(float _c, Mat _input) {

    Mat deviation = Mat(_input.rows, _input.cols, CV_32FC1);
    float median = MedianMat(_input);
    // Absolute Deviation from the _input's median
    deviation = abs(_input - median);

    // Median of deviation
    float MAD = MedianMat(deviation);

    return _c * MAD;
}

Mat Tracker::IdentityWeights(int _num_residuals) {
    Mat W = Mat::ones(_num_residuals,1,CV_32FC1);    
    return W;
}

Mat Tracker::TukeyFunctionWeights(Mat _input) {
    int num_residuals = _input.rows;
    float c = 1.4826;
    float b = 4.6851; // Achieve 95% efficiency if assumed Gaussian distribution for outliers
    Mat W = Mat(num_residuals,1,CV_32FC1);    
    float MAD = c * MedianAbsoluteDeviation(c, _input);
    if (MAD == 0) {
        cout << "Median Absolute Deviation (MAD) is 0" << endl;
        cout << "Exiting ..." << endl;
        exit(0);
    }

    float inv_MAD = 1.0 / MAD;
    float inv_b2 = 1.0 / (b * b);

    for (int i=0; i<num_residuals; i++) {
        float prueba = _input.at<float>(i,0);
        float x = _input.at<float>(i,0) * inv_MAD;

        if (abs(x) <= b) {
            float tukey = (1.0 - (x * x) * inv_b2);
            W.at<float>(i,0) = tukey * tukey;
        } else {
            W.at<float>(i,0) = 0.0;
        }
    }

    return W;
}


void Tracker::DebugShowCandidatePoints(Mat _image, Mat _candidatePoints){
    Mat showPoints;
    cvtColor(_image, showPoints, CV_GRAY2RGB);

    for( int i=0; i<_candidatePoints.rows; i++) {
        Point2d point;
        point.x = _candidatePoints.at<float>(i,0);
        point.y = _candidatePoints.at<float>(i,1);

        circle(showPoints, point, 2, Scalar(255,0,0), 1, 8, 0);
    }
    imshow("Show candidates points", showPoints);
    waitKey(0);
}

// TODO Implement
void Tracker::DebugShowResidual(Mat _image1, Mat _image2, Mat _candidatePoints, Mat _warpedPoints, int _lvl) {
    Mat showResidual = _image2.clone();
    
    for (int i=0; i<_candidatePoints.rows; i++) {
        int x1 = _candidatePoints.at<float>(i,0);
        int y1 = _candidatePoints.at<float>(i,1);
        int x2 = _warpedPoints.at<float>(i,0);
        int y2 = _warpedPoints.at<float>(i,1);

        int intensity1 = _image1.at<uchar>(y1,x1);
        int intensity2 = _image2.at<uchar>(y2,x2);
        
        if (x2<_image2.cols && x2>0 && y2<_image2.rows && y2>0) {
            showResidual.at<int>(y2,x2) = abs(intensity2 - intensity1);
        }
    }

    imshow("", showResidual);
    waitKey(0);
}

void Tracker::DebugShowWarpedPerspective(Mat _image1, Mat _image2, Mat _candidatePoints, Mat _warped, int _lvl) {
    int lvl = _lvl + 1;
    Mat warpImage = Mat::zeros(_image1.size(), CV_8UC1);
    Mat noalign = Mat::zeros(_image2.size(), CV_8UC1);
    Mat showPoints1, showPoints2;
    Mat substraction;

    for (int i=0; i<_candidatePoints.rows; i++) {
        int x1 = _candidatePoints.at<float>(i,0);
        int y1 = _candidatePoints.at<float>(i,1);
        int x2 = round(_warped.at<float>(i,0));
        int y2 = round(_warped.at<float>(i,1));

        if (x2<_image2.cols && x2>=0 && y2<_image2.rows && y2>=0) {
            warpImage.at<uchar>(y2,x2) = _image1.at<uchar>(y1,x1);
            int a = warpImage.at<uchar>(y2,x2);
        }
    }

    addWeighted(_image1, 0.5, _image2, 0.5, 1.0, noalign);
    
    //cv::subtract(conv2, conv1, substraction);
    cvtColor(warpImage, showPoints1, CV_GRAY2RGB);
    cvtColor(_image2, showPoints2, CV_GRAY2RGB);
    cvtColor(noalign, noalign, CV_GRAY2RGB);
    
    addWeighted(showPoints1, 0.5, showPoints2, 0.5, 1.0, substraction);

    Point2d p1, p2, p3 ,p4;
    p1.x = _warped.at<float>(0,0);
    p1.y = _warped.at<float>(0,1);
    
    p2.x = _warped.at<float>(_image2.rows-1,0);
    p2.y = _warped.at<float>(_image2.rows-1,1);

    p3.x = _warped.at<float>(_image2.rows*(_image2.cols-1),0);
    p3.y = _warped.at<float>(_image2.rows*(_image2.cols-1),1);

    p4.x = _warped.at<float>(_warped.rows-1,0);
    p4.y = _warped.at<float>(_warped.rows-1,1);

    line(showPoints1,p1,p2,Scalar(255,0,0), 1, 8, 0);
    line(showPoints1,p2,p4,Scalar(255,0,0), 1, 8, 0);
    line(showPoints1,p4,p3,Scalar(255,0,0), 1, 8, 0);
    line(showPoints1,p3,p1,Scalar(255,0,0), 1, 8, 0);

    line(showPoints1,p1,p4,Scalar(255,0,0), 1, 8, 0);
    line(showPoints1,p2,p3,Scalar(255,0,0), 1, 8, 0);

    Mat imShow1, imShow2, imShow;
    hconcat(showPoints1, showPoints2, imShow1);
    hconcat(substraction, noalign, imShow2);
    vconcat(imShow1, imShow2, imShow);

    while (_lvl > 1) {
        resize(imShow, imShow, Size(), 2.0, 2.0);
        _lvl--;        
    }

    imshow("Result", imShow);
    waitKey(0);

}


}

    // CERES CODE -- FOR REVIEW
    // float initial_x = _previous_frame->id;
    // float x = initial_x;
    // int num_residuals = _previous_frame->images_[2].rows * _previous_frame->images_[2].cols;
    // cout << num_residuals << endl;
    // _previous_frame->rigid_transformation_ = SE3(Sophus::SO3d::exp(SE3::Point(0.2, 0.5, 0.0)), SE3::Point(10, 0, 0));
    // Optimization parameter
    // Sophus::Vector<float, 7> pose = _previous_frame->rigid_transformation_.params();
    // Sophus::Vector<float, 6> lie_algebra = _previous_frame->rigid_transformation_.log();
    
    // Build problem
    // ceres::Problem problem;

    // problem.AddParameterBlock(_previous_frame->rigid_transformation_.data(), SE3::num_parameters,
    //                             new LocalParameterizationSE3);

    // ceres::CostFunction* cost_function = new ResidualIntensity<22080, SE3::num_parameters>(_previous_frame->images_[2],
    //                                                             _current_frame->images_[2],
    //                                                             2,
    //                                                             fx_[2],
    //                                                             fy_[2],
    //                                                             cx_[2],
    //                                                             cy_[2],
    //                                                             invfx_[2],
    //                                                             invfy_[2]);
                                                                
    // cost_function->num_residuals(num_residuals);
    // ceres::CostFunction* cost_function = new 
    //     new ceres::NumericDiffCostFunction<ResidualIntensity, ceres::CENTRAL, ceres::DYNAMIC, SE3::num_parameters>
    //         (photometric, ceres::TAKE_OWNERSHIP, num_residuals);
    
    // cout << _previous_frame->rigid_transformation_.matrix() << endl;
    // problem.AddResidualBlock(cost_function, NULL, _previous_frame->rigid_transformation_.data());
    
    // ceres::Solver::Options options;
    // options.gradient_tolerance = 0.01 * Sophus::Constants<float>::epsilon();
    // options.function_tolerance = 0.01 * Sophus::Constants<float>::epsilon();
    // options.linear_solver_type = ceres::DENSE_QR;

    // Solve
    // ceres::Solver::Summary summary;
    // Solve(options, &problem, &summary);
    // cout << summary.BriefReport() << endl;
    // // Create and add cost function. Derivaties will be evaluate via automatic differentiation
    // PhotometricErrorOptimization* c = new PhotometricErrorOptimization(
    //                                     _previous_frame->images_[0],
    //                                     _current_frame->images_[0],
    //                                     0,
    //                                     fx_[0], fy_[0], cx_[0], cy_[0], invfx_[0], invfy_[0]);

    // ceres::AutoDiffCostFunction<PhotometricErrorOptimization, ceres::DYNAMIC, SE3::num_parameters>* cost_function = 
    //                                 new PhotometricErrorOptimization
    


    // problem.AddResidualBlock(cost_function, NULL, body_transformation.data());


    // ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<PhotometricError, 1, 1>(new PhotometricError);
    // problem.AddResidualBlock(cost_function, NULL, &x);

    // ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_QR;
    // options.minimizer_progress_to_stdout - true;
    // ceres::Solver::Summary summary;
    // Solve(options, &problem, &summary);

    // std::cout << summary.BriefReport() << "\n";
    // std::cout << "x : " << initial_x
    //             << " -> " << x << "\n";

    // Sophus::SE3d transformation(_previous_frame->rigid_transformation_);

    // Sophus::SE3d::QuaternionType p;
    // p = transformation.unit_quaternion();
    // cout << "Quaternion: " << p.coeffs() << endl;
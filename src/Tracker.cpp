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

#include "Tracker.h"
#include "System.h"

#include <opencv2/core/eigen.hpp>

namespace uw
{

class ResidualIntensity;
class LocalParameterizationSE3;

Tracker::Tracker() {
    for (Mat K: K_)
        K = Mat(3,3,CV_64FC1, Scalar(0.f));
    for (Mat invK: invK_)
        invK = Mat(3,3,CV_64FC1, Scalar(0.f));
};

Tracker::~Tracker(void) {};

// TODO(GitHub:fmoralesh, fabmoraleshidalgo@gmail.com)
// 02-20-2018 - Review: Obtaining the precomputed invK (for each pyramid level) is necessary?
void Tracker::InitializePyramid(int _width, int _height, Mat K) {
    w_[0] = _width;
    h_[0] = _height;
    K_[0] = K;
    invK_[0] = K.inv();

    fx_[0] = K.at<double>(0,0);
    fy_[0] = K.at<double>(1,1);
    cx_[0] = K.at<double>(0,2);
    cy_[0] = K.at<double>(1,2);
    
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

        K_[lvl] = Mat::eye(Size(3,3), CV_64FC1);
        K_[lvl].at<double>(0,0) = fx_[lvl];
        K_[lvl].at<double>(1,1) = fy_[lvl];
        K_[lvl].at<double>(0,2) = cx_[lvl];  
        K_[lvl].at<double>(1,2) = cy_[lvl];    

        invfx_[lvl] = 1 / fx_[lvl];
        invfy_[lvl] = 1 / fy_[lvl];
        invcx_[lvl] = 1 / cx_[lvl];
        invcy_[lvl] = 1 / cy_[lvl];

        invK_[lvl] = K_[lvl].inv();
        // Review 
        // invfx_[lvl] = invK_[lvl].at<double>(0,0);
        // invfy_[lvl] = invK_[lvl].at<double>(1,1);
        // invcx_[lvl] = invK_[lvl].at<double>(0,2);
        // invcy_[lvl] = invK_[lvl].at<double>(1,2);
    }
}

void Tracker::DebugVariationIntensity(Frame* previous_frame, Frame* current_frame) {
    // Changing parameter
    int num_steps = 100;
    double step = 0.2;
    int lvl = 0;
    int range = 10;

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w1 = 0.0; 
    double w2 = 0.0; 
    double w3 = 0.0; 
    
    double acumulated_error = 0.0;
    // Obtain points and depth of initial frame
    Mat candidatePoints      = previous_frame->candidatePoints_[lvl].clone();
    Mat candidatePointsDepth = previous_frame->candidatePointsDepth_[lvl].clone();

    // Obtain gradients at current lvl
    Mat gradient1 = previous_frame->gradient_[lvl].clone();
    Mat gradient2 = current_frame->gradient_[lvl].clone();
    Mat gradientX1 = previous_frame->gradientX_[lvl].clone();
    Mat gradientY1 = previous_frame->gradientY_[lvl].clone();
    Mat gradientX2 = current_frame->gradientX_[lvl].clone();   
    Mat gradientY2 = current_frame->gradientY_[lvl].clone();

    int counter = 0;
    for (double i =-range; i<range; i+=step) {
        counter++;
        vector<Mat> Jws;    
        vector<Mat> Jls;
        vector<uchar> intensities1;            
        vector<uchar> intensities2;
        int num_invalid_pixels = 0;

        // Warp points with current pose
        SE3 current_pose = SE3(SO3::exp(SE3::Point(0.0, 0.0, 0.0)), SE3::Point(0.0, i, 0.0));
        Mat warpedPoints = Mat(candidatePoints.size(), CV_64FC1);            
        warpedPoints = WarpFunction(candidatePoints, candidatePointsDepth, current_pose, lvl);


        if (counter % 11 == 0 || i == 0) 
            DebugShowWarpedPoints(previous_frame->gradient_[lvl], current_frame->gradient_[lvl], candidatePoints, warpedPoints);

        // Computation of intensities of frame 1 and 2
        // Computation of Jws and Jls
        for (int i=0; i<candidatePoints.rows; i++) {

            uchar intensity1; 
            uchar intensity2;

            // Points of frame 1
            double x1 = candidatePoints.at<double>(i,0);
            double y1 = candidatePoints.at<double>(i,1);
            double z1 = candidatePoints.at<double>(i,2);

            // Points of frame 2
            double x2 = round(warpedPoints.at<double>(i,0));
            double y2 = round(warpedPoints.at<double>(i,1));
            double z2 = round(warpedPoints.at<double>(i,2));

            // If points 1 and points 2 are inside frame 1 and 2, 
            // Compute intensity, Jw and Jl of that point
            if (x1<gradient1.cols && x1>0 && y1<gradient1.rows && y1>0 &&
                x2<gradient2.cols && x2>0 && y2<gradient2.rows && y2>0) {

                intensity1 = gradient1.at<uchar>(y1,x1);
                intensity2 = gradient2.at<uchar>(y2,x2);

                intensities1.push_back(intensity1);
                intensities2.push_back(intensity2);
                
            } else {
                num_invalid_pixels++; // Num of pixels out of range
            }


        }

        // Computation of Residuals
        Mat Residuals = Mat(intensities2.size(),1,CV_8UC1);   
        // Workaround to work with double numbers (intensities are in CV_8UC1)         
        Mat I1 = Mat(intensities1.size(),1, CV_64FC1);
        Mat I2 = Mat(intensities2.size(),1, CV_64FC1);
        Mat aux1 = Mat(intensities1);
        Mat aux2 = Mat(intensities2);
        aux1.convertTo(I1, CV_64FC1);
        aux2.convertTo(I2, CV_64FC1);
    
        Residuals = abs(I2 - I1);

        // Computation of Weights
        // Identity Weigths
        // Mat W = IdentityWeights(Residuals);
        // Tukey function
        Mat W = TukeyFunctionWeights(Residuals);

        // Computation of error
        double inv_num_residuals = 1.0 / Residuals.rows;
        Mat ResidualsW = Residuals.mul(W);

        Mat error =  inv_num_residuals * Residuals.t() * ResidualsW;
        acumulated_error = error.at<double>(0,0);

        cout << acumulated_error << endl;
    }
    
}

// Gauss-Newton using Foward Additive Algorithm
void Tracker::EstimatePose(Frame* previous_frame, Frame* current_frame) {
    // Gauss-Newton Options
    int max_iterations = 50;
    double error_threshold = 0.005;

    // Variables initialization
    double final_error = 0.0; 
    double acumulated_error = 0.0;
    double last_acumulated_error = 250.0;

    // Initial pose and deltapose (assumes little movement between frames)

    Mat deltaMat = Mat::zeros(6,1,CV_64FC1);
    SE3 current_pose = SE3(SO3::exp(SE3::Point(0.0, 0.0, 0.0)), SE3::Point(0.0, 0.0, 0.0));

    cout << endl;
    cout << "------------------------------------------" << endl;
    // Sparse to Fine iteration
    for (int lvl=PYRAMID_LEVELS-2; lvl>=0; lvl--) {

        cout << "----------- Iteration level: " << lvl << " -----------" << endl;
        // Initialize error for current pyramid lvl        
        acumulated_error = 50000.0;
        last_acumulated_error = 0.0;
        
        // Obtain points and depth of initial frame
        Mat candidatePoints      = previous_frame->candidatePoints_[lvl].clone();
        Mat candidatePointsDepth = previous_frame->candidatePointsDepth_[lvl].clone();

        // Obtain gradients at current lvl
        Mat gradient1 = previous_frame->gradient_[lvl].clone();
        Mat gradient2 = current_frame->gradient_[lvl].clone();
        Mat gradientX1 = previous_frame->gradientX_[lvl].clone();
        Mat gradientY1 = previous_frame->gradientY_[lvl].clone();
        Mat gradientX2 = current_frame->gradientX_[lvl].clone();   
        Mat gradientY2 = current_frame->gradientY_[lvl].clone();

        // Optimization iteration
        for (int k=0; k<max_iterations; k++) {
            vector<Mat> Jws;    
            vector<Mat> Jls;
            vector<uchar> intensities1;            
            vector<uchar> intensities2;
            int num_invalid_pixels = 0;

            // Apply update of pose (pose = deltapose ° pose)
            // Left or right multiplication?
            // SE3 deltaSE3 = Mat2SE3(deltaMat);  
            // current_pose = deltaSE3 * current_pose; 
            
            // Warp points with current pose
            Mat warpedPoints = Mat(candidatePoints.size(), CV_64FC1);            
            warpedPoints = WarpFunction(candidatePoints, candidatePointsDepth, current_pose, lvl);
            // Optional - show change of frame with the deltapose update
            // DebugShowWarpedPoints(gradient1, gradient2, candidatePoints, warpedPoints);

            // Computation of intensities of frame 1 and 2
            // Computation of Jws and Jls
            for (int i=0; i<candidatePoints.rows; i++) {

                uchar intensity1; 
                uchar intensity2;
                Mat Jw = Mat(2,6,CV_64FC1);
                Mat Jl = Mat(1,2,CV_64FC1);

                // Points of frame 1
                double x1 = candidatePoints.at<double>(i,0);
                double y1 = candidatePoints.at<double>(i,1);
                double z1 = candidatePoints.at<double>(i,2);

                // Points of frame 2
                double x2 = round(warpedPoints.at<double>(i,0));
                double y2 = round(warpedPoints.at<double>(i,1));
                double z2 = round(warpedPoints.at<double>(i,2));

                // If points 1 and points 2 are inside frame 1 and 2, 
                // Compute intensity, Jw and Jl of that point
                if (x1<gradient1.cols && x1>0 && y1<gradient1.rows && y1>0 &&
                    x2<gradient2.cols && x2>0 && y2<gradient2.rows && y2>0) {

                    intensity1 = gradient1.at<uchar>(y1,x1);
                    intensity2 = gradient2.at<uchar>(y2,x2);

                    Jl.at<double>(0,0) = gradientX1.at<uchar>(y1,x1);
                    Jl.at<double>(0,1) = gradientY1.at<uchar>(y1,x1);
                    
                    double inv_z2 = 1 / z2;

                    Jw.at<double>(0,0) = fx_[lvl] * inv_z2;
                    Jw.at<double>(0,1) = 0.0;
                    Jw.at<double>(0,2) = -(fx_[lvl] * x2 * inv_z2 * inv_z2);
                    Jw.at<double>(0,3) = -(fx_[lvl] * x2 * y2 * inv_z2 * inv_z2);
                    Jw.at<double>(0,4) =   fx_[lvl] * (1 + (x2 * x2 * inv_z2 * inv_z2));
                    Jw.at<double>(0,5) = - fx_[lvl] * y2 * inv_z2;

                    Jw.at<double>(1,0) = 0.0;
                    Jw.at<double>(1,1) = fy_[lvl] * inv_z2;
                    Jw.at<double>(1,2) = -(fy_[lvl] * y2 * inv_z2 * inv_z2);
                    Jw.at<double>(1,3) = -(fy_[lvl] * (1 + y2 * y2 * inv_z2 * inv_z2));
                    Jw.at<double>(1,4) = fy_[lvl] * x2 * y2 * inv_z2 * inv_z2;
                    Jw.at<double>(1,5) = fy_[lvl] * x2 * inv_z2;

                    Jws.push_back(Jw);
                    Jls.push_back(Jl);

                    intensities1.push_back(intensity1);
                    intensities2.push_back(intensity2);
                    
                } else {
                    num_invalid_pixels++; // Num of pixels out of range
                }
            }

            // Computation of Residuals
            Mat Residuals = Mat(intensities2.size(),1,CV_8UC1);   
            // Workaround to work with double numbers (intensities are in CV_8UC1)         
            Mat I1 = Mat(intensities1.size(),1, CV_64FC1);
            Mat I2 = Mat(intensities2.size(),1, CV_64FC1);
            Mat aux1 = Mat(intensities1);
            Mat aux2 = Mat(intensities2);
            aux1.convertTo(I1, CV_64FC1);
            aux2.convertTo(I2, CV_64FC1);
        
            Residuals = abs(I2 - I1);

            // Computation of Weights
            // Identity Weigths
            // Mat W = IdentityWeights(Residuals);
            // Tukey function
            Mat W = TukeyFunctionWeights(Residuals);

            // Computation of Jacobian
            Mat Jacobian = Mat(Residuals.rows, 6, CV_64FC1);
            for (int i=0; i<Jacobian.rows; i++) {
                Mat Jacobian_row = Mat(1,6,CV_64FC1);
                Jacobian_row = Jls[i] * Jws[i];

                Jacobian_row.copyTo(Jacobian.row(i));
                
            }

            // Computation of error
            double inv_num_residuals = 1.0 / Residuals.rows;
            Mat ResidualsW = Residuals.mul(W);
            
            Mat error =  inv_num_residuals * Residuals.t() * ResidualsW;
            last_acumulated_error = acumulated_error;
            acumulated_error = error.at<double>(0,0);

            // Break if error increases
            if (acumulated_error > last_acumulated_error) {
                break;
            }
            cout << "Error: " << acumulated_error << endl;
            
            // Computation of delta update
            Mat JacobianW = Jacobian.clone();
            for (int i=0; i<JacobianW.rows; i++) {
                JacobianW.row(i) =  W.at<double>(i,0) * Jacobian.row(i);                
            }
            deltaMat = -1 * ((Jacobian.t() * JacobianW).inv() * Jacobian.t() * ResidualsW);

            // Apply update
            Sophus::Vector<double, SE3::DoF> deltaVector;

            deltaVector(3) = deltaMat.at<double>(0,0);
            deltaVector(4) = deltaMat.at<double>(1,0);
            deltaVector(5) = deltaMat.at<double>(2,0);
            deltaVector(0) = deltaMat.at<double>(3,0);
            deltaVector(1) = deltaMat.at<double>(4,0);
            deltaVector(2) = deltaMat.at<double>(5,0);

            // Make exp multiplication
            SE3 deltaSE3;
            current_pose = deltaSE3.exp(deltaVector) * current_pose;
            
        }
    }

    // Mat candidatePoints      = previous_frame->candidatePoints_[0].clone();
    // Mat candidatePointsDepth = previous_frame->candidatePointsDepth_[0].clone();
    // Mat warpedPoints = Mat(candidatePoints.size(), CV_64FC1);
    // warpedPoints = WarpFunction(candidatePoints, candidatePointsDepth, current_pose, 0);
    // DebugShowWarpedPoints(previous_frame->gradient_[0], current_frame->gradient_[0], candidatePoints, warpedPoints);
    previous_frame->rigid_transformation_ = current_pose;



    

    // double initial_x = previous_frame->id;
    // double x = initial_x;
    // int num_residuals = previous_frame->image_[2].rows * previous_frame->image_[2].cols;
    // cout << num_residuals << endl;
    // previous_frame->rigid_transformation_ = SE3(Sophus::SO3d::exp(SE3::Point(0.2, 0.5, 0.0)), SE3::Point(10, 0, 0));
    // Optimization parameter
    // Sophus::Vector<double, 7> pose = previous_frame->rigid_transformation_.params();
    // Sophus::Vector<double, 6> lie_algebra = previous_frame->rigid_transformation_.log();
    
    // Build problem
    // ceres::Problem problem;

    // problem.AddParameterBlock(previous_frame->rigid_transformation_.data(), SE3::num_parameters,
    //                             new LocalParameterizationSE3);

    // ceres::CostFunction* cost_function = new ResidualIntensity<22080, SE3::num_parameters>(previous_frame->image_[2],
    //                                                             current_frame->image_[2],
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
    
    // cout << previous_frame->rigid_transformation_.matrix() << endl;
    // problem.AddResidualBlock(cost_function, NULL, previous_frame->rigid_transformation_.data());
    
    // ceres::Solver::Options options;
    // options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
    // options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
    // options.linear_solver_type = ceres::DENSE_QR;

    // Solve
    // ceres::Solver::Summary summary;
    // Solve(options, &problem, &summary);
    // cout << summary.BriefReport() << endl;
    // // Create and add cost function. Derivaties will be evaluate via automatic differentiation
    // PhotometricErrorOptimization* c = new PhotometricErrorOptimization(
    //                                     previous_frame->image_[0],
    //                                     current_frame->image_[0],
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

    // Sophus::SE3d transformation(previous_frame->rigid_transformation_);

    // Sophus::SE3d::QuaternionType p;
    // p = transformation.unit_quaternion();
    // cout << "Quaternion: " << p.coeffs() << endl;


}

// TODO(GitHub:fmoralesh, fabmoraleshidalgo@gmail.com)
// 02-20-2018 - Consider other methods to obtain gradient from an image (Sober, Laplacian, ...) 
//            - Calculate gradient for each pyramid image or scale the finest?
void Tracker::ApplyGradient(Frame* frame) {
    Mat gradient;
    Mat gradientX; 
    Mat gradientY;
    
    // Filters for calculating gradient in images
    cuda::GpuMat frameGPU = cuda::GpuMat(frame->image_[0]);
    Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(frameGPU.type(), frameGPU.type(), Size(3,3), 0);

    // Apply gradient in x and y
    cuda::GpuMat frameXGPU, frameYGPU;
    // Ptr<cuda::Filter> soberX_ = cuda::createDerivFilter(0, CV_16S, 1, 0, 3, 0,BORDER_DEFAULT,BORDER_DEFAULT);
    // Ptr<cuda::Filter> soberY_ = cuda::createDerivFilter(0, CV_16S, 0, 1, 3, 0,BORDER_DEFAULT,BORDER_DEFAULT);    
    Ptr<cuda::Filter> soberX_ = cuda::createSobelFilter(0, CV_16S, 1, 0, 3, 1, BORDER_DEFAULT,BORDER_DEFAULT);
    Ptr<cuda::Filter> soberY_ = cuda::createSobelFilter(0, CV_16S, 0, 1, 3, 1, BORDER_DEFAULT,BORDER_DEFAULT);

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

    frame->gradient_[0] = gradient.clone();
    frame->gradientX_[0] = gradientX.clone();
    frame->gradientY_[0] = gradientY.clone();

    for (int i=1; i<PYRAMID_LEVELS; i++){
        resize(frame->gradient_[i-1], frame->gradient_[i], Size(), 0.5, 0.5);
        
        resize(frame->gradientX_[i-1], frame->gradientX_[i], Size(), 0.5, 0.5);
        resize(frame->gradientY_[i-1], frame->gradientY_[i], Size(), 0.5, 0.5);
    }

    frame->obtained_gradients_ = true;
    
}


void Tracker::ObtainAllPoints(Frame* frame) {
    for (int lvl=0; lvl< PYRAMID_LEVELS-1; lvl++) {
        frame->candidatePoints_[lvl] = Mat::ones(w_[lvl] * h_[lvl], 4, CV_64FC1);
        frame->candidatePointsDepth_[lvl] = 250 * Mat::ones(w_[lvl] * h_[lvl], 1, CV_64FC1);
        for (int x=0; x<w_[lvl]; x++) {
            for (int y =0; y<h_[lvl]; y++) {
                frame->candidatePoints_[lvl].at<double>(y+h_[lvl]*x,0) = x;
                frame->candidatePoints_[lvl].at<double>(y+h_[lvl]*x,1) = y;
                
            }
        }
    }


    frame->obtained_candidatePoints_ = true;
}

// TODO(GitHub:fmoralesh, fabmoraleshidalgo@gmail.com)
// 02-13-2018 - Implement a faster way to obtain candidate points with high gradient in patches (above of a certain threshold)
void Tracker::ObtainCandidatePoints(Frame* frame) {
    // Block size search for high gradient points in image 
    // (Very slow, must have gradients precomputed *see ApplyGradient)
    for (int i = 0; i < PYRAMID_LEVELS - 1; i++){
        int block_size = BLOCK_SIZE - i * 5;
        cuda::GpuMat frameGPU(frame->gradient_[i]);
        for (int x=0; x<w_[i]-block_size; x+=block_size) {
            for (int y =0; y<h_[i]-block_size; y+=block_size) {
                Mat point = Mat::ones(1,4,CV_64FC1);
                Mat depth = Mat::ones(1,1,CV_64FC1);                
                Scalar mean, stdev;
                Point min_loc, max_loc;
                double min, max;
                cuda::GpuMat block(frameGPU, Rect(x,y,block_size,block_size));
                block.convertTo(block, CV_8UC1);
                cuda::meanStdDev(block, mean, stdev);
                cuda::minMaxLoc(block, &min, &max, &min_loc, &max_loc);
                
                if (max > mean[0] + GRADIENT_THRESHOLD) {
                    point.at<double>(0,0) = (double) (x + max_loc.x);
                    point.at<double>(0,1) = (double) (y + max_loc.y);

                    frame->candidatePoints_[i].push_back(point);
                    frame->candidatePointsDepth_[i].push_back(depth);
                }
            }
        }
    }

    for (int lvl = 1; lvl < PYRAMID_LEVELS; lvl++) {
        //frame->candidatePoints_[lvl] = frame->candidatePoints_[lvl-1] * 0.5;
        // DebugShowCandidatePoints(frame->gradient_[lvl-1], frame->candidatePoints_[lvl-1]);
    }
    frame->obtained_candidatePoints_ = true;
}



// TODO(GitHub:fmoralesh, fabmoraleshidalgo@gmail.com)
// 02-20-2018 - Consider which of both warping functions proposals is the correct one (DSO or LSD SLAM).
Mat Tracker::WarpFunction(Mat points2warp, Mat depth, SE3 rigid_transformation, int lvl) {

    Mat33 R = rigid_transformation.rotationMatrix();
    Mat31 t = rigid_transformation.translation();
    Quaternion2 quaternion = rigid_transformation.unit_quaternion();

    Mat projected_points = Mat(points2warp.size(), CV_64FC1);
    projected_points = points2warp.clone();

    Mat44 rigidEigen = rigid_transformation.matrix();
    Mat rigid = Mat(4,4,CV_64FC1);
    eigen2cv(rigidEigen, rigid);
    
    // DSO-SLAM Warping function 
    // {
    // projected_points.col(0) = projected_points.col(0).mul(depth);
    // projected_points.col(1) = projected_points.col(1).mul(depth);
    // projected_points.col(2) = projected_points.col(2).mul(depth);
    
    // projected_points = rigid * projected_points.t();

    // projected_points.row(0) /= projected_points.row(2);
    // projected_points.row(1) /= projected_points.row(2);
    // projected_points.row(2) /= projected_points.row(2);
    // }

    // LSD-SLAM Warping function
    {
    double fx = fx_[lvl];
    double fy = fy_[lvl];
    double cx = cx_[lvl];
    double cy = cy_[lvl];
    
    projected_points.col(0) = ((projected_points.col(0) - cx_[lvl]) * invfx_[lvl]);
    projected_points.col(0) = projected_points.col(0).mul(depth);
    projected_points.col(1) = ((projected_points.col(1) - cy_[lvl]) * invfy_[lvl]);
    projected_points.col(1) = projected_points.col(1).mul(depth);
    projected_points.col(2) = projected_points.col(2).mul(depth);
    
    projected_points = rigid * projected_points.t();

    projected_points.row(0) /= projected_points.row(2);
    projected_points.row(1) /= projected_points.row(2);
    projected_points.row(0) *= fx_[lvl];
    projected_points.row(1) *= fy_[lvl];
    projected_points.row(0) += cx_[lvl];
    projected_points.row(1) += cy_[lvl];
    
    }

    // Check projected_points arrangement
    return projected_points.t();
}   

void Tracker::DebugShowCandidatePoints(Mat image, Mat candidatePoints){
    Mat showPoints;
    cvtColor(image, showPoints, CV_GRAY2RGB);

    for( int i=0; i<candidatePoints.rows; i++) {
        Point2d point;
        point.x = candidatePoints.at<double>(i,0);
        point.y = candidatePoints.at<double>(i,1);

        circle(showPoints, point, 2, Scalar(255,0,0), 1, 8, 0);
    }
    imshow("Show candidates points", showPoints);
    waitKey(0);
}

void Tracker::DebugShowWarpedPoints(Mat image1, Mat image2, Mat candidatePoints, Mat warped){
    Mat showPoints1, showPoints2;
    Mat substraction;
    Mat transformed_image = Mat(image2.size(), CV_8UC1);

    for (int i=0; i<candidatePoints.rows; i++) {
        int x1 = candidatePoints.at<double>(i,0);
        int y1 = candidatePoints.at<double>(i,1);
        int x2 = warped.at<double>(i,0);
        int y2 = warped.at<double>(i,1);

        int intensity = image1.at<uchar>(y1,x1);

        if (x2<image2.cols && x2>0 && y2<image2.rows && y2>0)
            transformed_image.at<uchar>(y2,x2) = intensity;
    }

    //cv::subtract(conv2, conv1, substraction);
    cvtColor(image1, showPoints1, CV_GRAY2RGB);
    cvtColor(image2, showPoints2, CV_GRAY2RGB);

    Point2d p1, p2, p3 ,p4;
    p1.x = warped.at<double>(0,0);
    p1.y = warped.at<double>(0,1);
    
    p2.x = warped.at<double>(image2.rows-1,0);
    p2.y = warped.at<double>(image2.rows-1,1);

    p3.x = warped.at<double>(image2.rows*(image2.cols-1),0);
    p3.y = warped.at<double>(image2.rows*(image2.cols-1),1);

    p4.x = warped.at<double>(warped.rows-1,0);
    p4.y = warped.at<double>(warped.rows-1,1);
    
    line(showPoints1,p1,p2,Scalar(255,0,0), 1, 8, 0);
    line(showPoints1,p2,p4,Scalar(255,0,0), 1, 8, 0);
    line(showPoints1,p4,p3,Scalar(255,0,0), 1, 8, 0);
    line(showPoints1,p3,p1,Scalar(255,0,0), 1, 8, 0);

    line(showPoints1,p1,p4,Scalar(255,0,0), 1, 8, 0);
    line(showPoints1,p2,p3,Scalar(255,0,0), 1, 8, 0);

    imshow("First image", showPoints1);
    imshow("Second image", showPoints2);
    imshow("Transformed", transformed_image);
    
    waitKey(0);
    
    
    // for( int i=0; i<warped.rows; i+=150) {
    //     Point2d point;
    //     point.x = warped.at<double>(i,0);
    //     point.y = warped.at<double>(i,1);

    //     circle(showPoints, point, 5, Scalar(0,255,0), 1, 8, 0);
    // }

}

double Tracker::MedianMat(Mat input) {
    Mat channel = Mat(input.rows,input.cols,CV_8UC1);
    input.convertTo(channel, CV_8UC1);

    double m = (channel.rows*channel.cols) / 2;
    int bin = 0;
    double med = -1.0;

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

SE3 Tracker::Mat2SE3(Mat input) {
    double w1 = input.at<double>(0,0);
    double w2 = input.at<double>(1,0);
    double w3 = input.at<double>(2,0);
    double x1 = input.at<double>(3,0);
    double x2 = input.at<double>(4,0);
    double x3 = input.at<double>(5,0);
    
    return SE3(SO3::exp(SE3::Point(w1, w2, w3)), SE3::Point(x1, x2, x3));
}

double Tracker::MedianAbsoluteDeviation(double c, Mat input) {

    Mat deviation = Mat(input.rows, input.cols, CV_64FC1);
    double median = MedianMat(input);
    // Absolute Deviation from the input's median
    deviation = abs(input - median);

    // Median of deviation
    double MAD = MedianMat(deviation);

    return c * MAD;
}

Mat Tracker::IdentityWeights(Mat input) {
    int num_residuals = input.rows;
    Mat W = Mat::ones(num_residuals,1,CV_64FC1);    
    return W;
}

Mat Tracker::TukeyFunctionWeights(Mat input) {
    int num_residuals = input.rows;
    double c = 1.4826;
    double b = 4.6851; // Achieve 95% efficiency if assumed Gaussian distribution for outliers
    Mat W = Mat(num_residuals,1,CV_64FC1);    
    double MAD = c * MedianAbsoluteDeviation(c, input);
    if (MAD == 0) {
        cout << "Median Absolute Deviation (MAD) is 0" << endl;
        cout << "Exiting ..." << endl;
        exit(0);
    }

    double inv_MAD = 1.0 / MAD;
    double inv_b2 = 1.0 / (b * b);

    for (int i=0; i<num_residuals; i++) {
        double prueba = input.at<double>(i,0);
        double x = input.at<double>(i,0) * inv_MAD;

        if (abs(x) <= b) {
            double tukey = (1.0 - (x * x) * inv_b2);
            W.at<double>(i,0) = tukey * tukey;
        } else {
            W.at<double>(i,0) = 0.0;
        }
    }

    return W;
}


}
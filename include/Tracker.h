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

#pragma once
#include <Options.h>
///Basic C and C++ libraries
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <dirent.h>

/// OpenCV libraries. May need review for the final release
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/calib3d.hpp"
#include <opencv2/video.hpp>

// Ceres library
#include "ceres/ceres.h"

// Eigen library
#include <eigen3/Eigen/Core>

// Sophus
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

/// CUDA specific libraries
#include <opencv2/cudafilters.hpp>
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"

// Namespaces
using namespace cv;
using namespace std;


namespace uw
{
class Frame;

class Tracker
{
public:
    /**
     * @brief Tracker consturctor.
     * 
     */
    Tracker();

    /**
     * @brief Tracker destructor.
     * 
     */
    ~Tracker();

    /**
     * @brief Obtains camera instrinsic matrix and parameters for each pyramid level available.
     * 
     * @param _width    Width of images at finest level.
     * @param _height   Height of images at finest level.
     * @param K         Camera intrinsic matrix at finest level.
     */
    void InitializePyramid(int _width, int _height, Mat K);
    
    /**
     * @brief Computes optimal transformation given two input frames.
     * 
     * @param previous_frame 
     * @param current_frame 
     */
    void EstimatePose(Frame* previous_frame, Frame* current_frame);

    /**
     * @brief Computes gradient of a frame for each pyramid level available.
     *        Saves the result gradient images within the frame class.
     * 
     * @param frame 
     */
    void ApplyGradient(Frame* frame);

    /**
     * @brief Obtains candidate points from grandient images of a frame for each pyramid level available.
     * 
     * @param frame 
     */
    void ObtainCandidatePoints(Frame* frame);

    void ObtainAllPoints(Frame* frame);
    /**
     * @brief Computes warp projected points from one frame to another, given a rigid transformation matrix 
     *        and the depth estimation of those points. Returns matrix of warped points.
     * 
     * @param points2warp 
     * @param depth 
     * @param rigid_transformation 
     * @return Mat 
     */
    Mat WarpFunction(Mat points2warp, Mat depth, SE3 rigid_transformation, int lvl);

    Mat WarpFunction2(vector<Point3f> originalPoints, vector<Point2f> warpedPoitns, Mat K, SE3 rigid_transformation);
    /**
     * @brief Shows points in an image. Used only for debbugin.
     * 
     * @param image 
     * @param candidatePoints 
     */
    void DebugShowCandidatePoints(Mat image, Mat candidatePoints);

    SE3 Mat2SE3(Mat input);

    void ObtainImageTransformed(Mat originalImage, Mat candidatePoints, Mat warpedPoints, Mat outputImage);
    
    void ObtainGradientXY(Mat inputImage, Mat gradientX, Mat gradientY);
    
    void DebugShowResidual(Mat image1, Mat image2, Mat candidatePoints, Mat warped, int _lvl);

    void DebugShowWarpedPerspective(Mat image1, Mat image2, Mat candidatePoints, Mat warped, int _lvl);

    double MedianMat(Mat input);

    double MedianAbsoluteDeviation(double c, Mat input);

    Mat TukeyFunctionWeights(Mat input);

    Mat IdentityWeights(Mat input);

    template<int kNumResiduals, int N1 = 0>
    class ResidualIntensity : public ceres::SizedCostFunction<kNumResiduals, N1>
    {
    public:
        ResidualIntensity(Mat _image1, Mat _image2, int lvl, double fx, double fy, 
                                    double cx, double cy, double invfx, double invfy) {
            image1_ = _image1; image2_ = _image2;
            lvl_ = lvl;
            fx_ = fx; fy_ = fy; cx_ = cx; cy_ = cy;
            invfx_ = invfx; invfy_ = invfy;
        }

        // Calculates Cost Function of Photometric Error between two images
        virtual bool Evaluate(double const* const* pose_in,
                                double* residuals,
                                double** jacobians) const {
            
            // Initialize residuals with an initial value
            for (int i=0; i<image1_.rows; i++) {
                for (int j=0; j<image1_.cols; j++){
                    residuals[j+i*j] = 0.0;
                }
            }

            // Eigen Map to manipulate T class with Sophus
            double qx = pose_in[0][0];
            double qy = pose_in[0][1];
            double qz = pose_in[0][2];
            double qw = pose_in[0][3];
            double tx = pose_in[0][4];
            double ty = pose_in[0][5];
            double tz = pose_in[0][6];
            
            SE3 pose1 = SE3(Eigen::Quaternion<double>(qw,qx,qy,qz), SE3::Point(tx, ty, tz));
            Mat44 G = pose1.matrix();

            double point3D[4];
            double point3DTransformed[3];
            double point2DTransformed[2];
            double intensity_pixel1, intensity_pixel2;
            
            // Iteration for each pixel of the image
            for (int i=0; i<image1_.rows; i++){
                for (int j=0; j<image1_.cols; j++){
                    //double d = depth_.at<double>(i,j);
                    double d = 1.0;
                    if (d>0) {  // For valid depth
                        double inv_depth_ = 1.0 / d;
                        // Obtain local 3D coordinates of pixel (i,j) of image 1
                        point3D[2] = inv_depth_;
                        point3D[0] = (i-cx_) * inv_depth_ * invfx_;
                        point3D[1] = (j-cy_) * inv_depth_ * invfy_;
                        point3D[3] = 1.0;
                        
                        // Transform 3D point using Rotation and translation input
                        point3DTransformed[0] = G(0,0)*point3D[0]+G(0,1)*point3D[1]+G(0,2)*point3D[2]+G(0,3)*point3D[3];
                        point3DTransformed[1] = G(1,0)*point3D[0]+G(1,1)*point3D[1]+G(1,2)*point3D[2]+G(1,3)*point3D[3];
                        point3DTransformed[2] = G(2,0)*point3D[0]+G(2,1)*point3D[1]+G(2,2)*point3D[2]+G(2,3)*point3D[3];
                        // point3DTransformed[3] = G[3][0]*point3D[0]+G[3][1]*point3D[1]+G[3][2]*point3D[2]+G[3][3]*point3D[3];

                        // Project 3D point to 2D plane of image 2
                        inv_depth_ = 1.0 / point3DTransformed[2];
                        point2DTransformed[0] = (point3DTransformed[0] * fx_ * inv_depth_) + cx_;
                        point2DTransformed[1] = (point3DTransformed[1] * fx_ * inv_depth_) + cy_;

                        // Check if 2D transformed point went out of the visual field of the image 2
                        if ((point2DTransformed[0]>=0.0 && point2DTransformed[0]< image2_.rows) &&
                            (point2DTransformed[1]>=0.0 && point2DTransformed[1]< image2_.cols)) {
                            // Compute the proyected coordinates of the transformed 3D point
               
                            
                            int i_projected = (int)(point2DTransformed[0]);
                            int j_projected = (int)(point2DTransformed[1]);

                            // Obtain intensities from images
                            intensity_pixel1 = image1_.at<double>(i,j);
                            intensity_pixel2 = image2_.at<double>(i_projected,j_projected);
                            
                            // Compute residual
                            residuals[j+i*j] = intensity_pixel1 - intensity_pixel2;
                            jacobians[j+i*j][0] = i_projected - 1.0;
                            jacobians[j+i*j][1] = i_projected *1.0;
                            jacobians[j+i*j][2] = j_projected + 1.0;
                            jacobians[j+i*j][3] = i_projected * 1.0;
                            jacobians[j+i*j][4] = i_projected * 1.0;
                            jacobians[j+i*j][5] = i_projected* 1.0;

                        }
                    }
                }
            }

            //SE3 pose(quaternion, translation);
            
            //Eigen::Map<SE3 const> const pose(pose_raw);
            //residuals[0] = 0.0;
            // // Saving each 
            // T G[4][4];
            // for (int i=0; i<4; i++) {
            //     for (int j=0; j<4; j++) {
            //         G[i][j] = pose.matrix()(i,j);
            //     }
            // }
            // T fx = T(fx_);
            // T fy = T(fy_);
            // T cx = T(cx_);
            // T cy = T(cy_);
            // T invfx = T(invfx_);
            // T invfy = T(invfy_);

            // T point3D[4];
            // T point3DTransformed[4];
            // T point2DTransformed[2];
            // T intensity_pixel1;
            // T intensity_pixel2;
        
            // // Initialize residuals with an initial value
            // for (int i=0; i<image1_.rows; i++) {
            //     for (int j=0; j<image1_.cols; j++){
            //         residuals[j+i*j] = T(0.0);
            //     }
            // }

            // // Iteration for each pixel of the image
            // for (int i=0; i<image1_.rows; i++){
            //     for (int j=0; j<image1_.cols; j++){
            //         //double d = depth_.at<double>(i,j);
            //         double d = 1.0;
            //         if (d>0) {  // For valid depth
            //             T inv_depth = T(1.0) / T(d);
            //             // Obtain local 3D coordinates of pixel (i,j) of image 1
            //             point3D[2] = inv_depth;
            //             point3D[0] = (T(i)-cx) * inv_depth * invfx;
            //             point3D[1] = (T(j)-cy) * inv_depth * invfy;
            //             point3D[3] = T(1.0);
                        
            //             // Transform 3D point using Rotation and translation input
            //             point3DTransformed[0] = G[0][0]*point3D[0]+G[0][1]*point3D[1]+G[0][2]*point3D[2]+G[0][3]*point3D[3];
            //             point3DTransformed[1] = G[1][0]*point3D[0]+G[1][1]*point3D[1]+G[1][2]*point3D[2]+G[1][3]*point3D[3];
            //             point3DTransformed[2] = G[2][0]*point3D[0]+G[2][1]*point3D[1]+G[2][2]*point3D[2]+G[2][3]*point3D[3];
            //             // point3DTransformed[3] = G[3][0]*point3D[0]+G[3][1]*point3D[1]+G[3][2]*point3D[2]+G[3][3]*point3D[3];

            //             // Project 3D point to 2D plane of image 2
            //             inv_depth = T(1.0) / point3DTransformed[2];
            //             point2DTransformed[0] = (point3DTransformed[0] * fx * inv_depth) + cx;
            //             point2DTransformed[1] = (point3DTransformed[1] * fx * inv_depth) + cy;

            //             // Check if 2D transformed point went out of the visual field of the image 2
            //             if ((point2DTransformed[0]>=T(0.0) && point2DTransformed[0]< T(image2_.rows)) &&
            //                 (point2DTransformed[1]>=T(0.0) && point2DTransformed[1]< T(image2_.cols))) {
            //                 //Compute the proyected coordinates of the transformed 3D point
            //                 double x = point2DTransformed[0][0];
                            
            //                 // int i_projected = static_cast<int>(point2DTransformed[0].a());
            //                 //int j_projected = static_cast<int>(ceres::Jet<T,1>::GetScalar(point2DTransformed[1]));

            //                 // Obtain intensities from images
            //                 intensity_pixel1 = T(image1_.at<double>(i,j));
            //                 //intensity_pixel2 = T(image2.at<double>(i_projected,j_projected));

            //                 // Compute residual
            //                 residuals[j+i*j] = intensity_pixel1 - intensity_pixel2;
            //             }
            //         }
            //     }
            // }
        }

        // Variables
        SE3 pose_;
        int lvl_;
        Mat image1_, image2_;
        Mat depth_;
        double fx_, fy_, cx_, cy_, invfx_, invfy_;
    };

    class LocalParameterizationSE3 : public ceres::LocalParameterization {
    public:
        virtual ~LocalParameterizationSE3() {}

        // SE3 plus operation for Ceres
        //
        //  T * exp(x)
        //
        virtual bool Plus(double const* T_raw, double const* delta_raw,
                        double* T_plus_delta_raw) const {
            Eigen::Map<SE3 const> const T(T_raw);
            Eigen::Map<Mat61 const> const delta(delta_raw);
            Eigen::Map<SE3> T_plus_delta(T_plus_delta_raw);
            T_plus_delta = T * SE3::exp(delta);
            return true;
        }

        // Jacobian of SE3 plus operation for Ceres
        //
        // Dx T * exp(x)  with  x=0
        //
        virtual bool ComputeJacobian(double const* T_raw,
                                    double* jacobian_raw) const {
            Eigen::Map<SE3 const> T(T_raw);
            Eigen::Map<Eigen::Matrix<double, 6, 7> > jacobian(jacobian_raw);
            jacobian = T.Dx_this_mul_exp_x_at_0();
            return true;
        }

        virtual int GlobalSize() const { return SE3::num_parameters; }

        virtual int LocalSize() const { return SE3::DoF; }
    };

    // Width and height of images for each pyramid level available
    vector<int> w_ = vector<int>(PYRAMID_LEVELS);
    vector<int> h_ = vector<int>(PYRAMID_LEVELS);

    vector<double> fx_ = vector<double>(PYRAMID_LEVELS);
    vector<double> fy_ = vector<double>(PYRAMID_LEVELS);
    vector<double> cx_ = vector<double>(PYRAMID_LEVELS);
    vector<double> cy_ = vector<double>(PYRAMID_LEVELS);
    vector<double> invfx_ = vector<double>(PYRAMID_LEVELS);
    vector<double> invfy_ = vector<double>(PYRAMID_LEVELS);
    vector<double> invcx_ = vector<double>(PYRAMID_LEVELS);
    vector<double> invcy_ = vector<double>(PYRAMID_LEVELS);

    vector<Mat> K_ = vector<Mat>(PYRAMID_LEVELS);
    vector<Mat> invK_ = vector<Mat>(PYRAMID_LEVELS);
};


}
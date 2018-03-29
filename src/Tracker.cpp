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

class LocalParameterizationSE3;

class RobustMatcher;

RobustMatcher::RobustMatcher() {
    isSURF_ = true;
    isORB_ = false;
};

// Clear matches for which NN ratio is > than threshold
// return the number of removed points
// (corresponding entries being cleared,
// i.e. size will be 0)
int RobustMatcher::ratioTest(vector<vector<DMatch> > &matches) {
    int removed=0;
    // for all matches
    for (std::vector<std::vector<cv::DMatch> >::iterator
        matchIterator= matches.begin();
        matchIterator!= matches.end(); ++matchIterator) {
        // if 2 NN has been identified
        if (matchIterator->size() > 1) {
            // check distance ratio
            if ((*matchIterator)[0].distance/(*matchIterator)[1].distance > ratio_) {
                    matchIterator->clear(); // remove match
                    removed++;
            }
        } else { // does not have 2 neighbours
            matchIterator->clear(); // remove match
            removed++;
        }
    }
    return removed;
}

// Insert symmetrical matches in symMatches vector
void RobustMatcher::symmetryTest(const vector<vector<DMatch> >& matches1, const vector<vector<DMatch> >& matches2, vector<DMatch>& symMatches) {
    // for all matches image 1 -> image 2
    for (vector<vector<DMatch> >::
    const_iterator matchIterator1= matches1.begin();
    matchIterator1!= matches1.end(); ++matchIterator1) {
        // ignore deleted matches
        if (matchIterator1->size() < 2)
        continue;
        // for all matches image 2 -> image 1
        for (std::vector<std::vector<cv::DMatch> >::
        const_iterator matchIterator2= matches2.begin();
        matchIterator2!= matches2.end();
        ++matchIterator2) {
            // ignore deleted matches
            if (matchIterator2->size() < 2)
            continue;
            // Match symmetry test
            if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx && 
                (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) {
                // add symmetrical match
                symMatches.push_back(DMatch((*matchIterator1)[0].queryIdx,
                            (*matchIterator1)[0].trainIdx,
                            (*matchIterator1)[0].distance));
                break; // next match in image 1 -> image 2
            }
        }
    }
}

// Identify good matches using RANSAC
// Return fundemental matrix
Mat RobustMatcher::ransacTest(const vector<DMatch>& matches, const vector<KeyPoint>& keypoints1, 
                    const vector<KeyPoint>& keypoints2, vector<DMatch>& outMatches) {

    // Convert keypoints into Point2f
    vector<Point2f> points1, points2;
    Mat fundemental;
    for (vector<DMatch>::const_iterator it= matches.begin(); it!= matches.end(); ++it) {
        // Get the position of left keypoints
        float x= keypoints1[it->queryIdx].pt.x;
        float y= keypoints1[it->queryIdx].pt.y;
        points1.push_back(cv::Point2f(x,y));
        // Get the position of right keypoints
        x= keypoints2[it->trainIdx].pt.x;
        y= keypoints2[it->trainIdx].pt.y;
        points2.push_back(cv::Point2f(x,y));
    }
    // Compute F matrix using RANSAC
    std::vector<uchar> inliers(points1.size(),0);
    if (points1.size()>0&&points2.size()>0){
        cv::Mat fundemental= findFundamentalMat(
            cv::Mat(points1),cv::Mat(points2), // matching points
            inliers,       // match status (inlier or outlier)
            CV_FM_RANSAC, // RANSAC method
            distance_,      // distance to epipolar line
            confidence_); // confidence probability
        // extract the surviving (inliers) matches
        std::vector<uchar>::const_iterator
                            itIn= inliers.begin();
        std::vector<cv::DMatch>::const_iterator
                            itM= matches.begin();
        // for all matches
        for ( ;itIn!= inliers.end(); ++itIn, ++itM) {
            if (*itIn) { // it is a valid match
                outMatches.push_back(*itM);
            }
        }
        if (refineF_) {
        // The F matrix will be recomputed with
        // all accepted matches
            // Convert keypoints into Point2f
            // for final F computation
            points1.clear();
            points2.clear();
            for (std::vector<cv::DMatch>::
                    const_iterator it= outMatches.begin();
                it!= outMatches.end(); ++it) {
                // Get the position of left keypoints
                float x= keypoints1[it->queryIdx].pt.x;
                float y= keypoints1[it->queryIdx].pt.y;
                points1.push_back(cv::Point2f(x,y));
                // Get the position of right keypoints
                x= keypoints2[it->trainIdx].pt.x;
                y= keypoints2[it->trainIdx].pt.y;
                points2.push_back(cv::Point2f(x,y));
            }
            // Compute 8-point F from all accepted matches
            if (points1.size()>0&&points2.size()>0){
                fundemental= cv::findFundamentalMat(
                cv::Mat(points1),cv::Mat(points2), // matches
                CV_FM_8POINT); // 8-point method
            }
        }
    }
    return fundemental;
}

void RobustMatcher::TrackFeatures(Frame* _previous_frame, Frame* _current_frame) {
    cuda::GpuMat previous_frameGPU, current_frameGPU;
    cuda::GpuMat keypointsGPU[2];
    cuda::GpuMat descriptorsGPU[2];
    array<vector<KeyPoint>,2> keypoints;
    array<vector<float>,2> descriptors;
    
    // Upload images to GPU
    previous_frameGPU.upload(_previous_frame->images_[0]);
    current_frameGPU.upload(_current_frame->images_[0]);

    // Matching descriptors
    vector< vector< DMatch> > matches1;
    vector< vector< DMatch> > matches2;

    // SURF as feature detector
    if(isSURF_) {
        cuda::SURF_CUDA surf;        
        Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher();

        // Loading previous keypoints found
        surf.uploadKeypoints(_previous_frame->keypoints_, keypointsGPU[0]);
        
        // Detecting keypoints and computing descriptors
        surf(previous_frameGPU, cuda::GpuMat(), keypointsGPU[0], descriptorsGPU[0], true);
        surf(current_frameGPU, cuda::GpuMat(), keypointsGPU[1], descriptorsGPU[1]);

        // Matching descriptors
        matcher->knnMatch(descriptorsGPU[0], descriptorsGPU[1], matches1, 2);
        matcher->knnMatch(descriptorsGPU[1], descriptorsGPU[0], matches2, 2);
        
        // Downloading results
        surf.downloadKeypoints(keypointsGPU[0], keypoints[0]);
        surf.downloadKeypoints(keypointsGPU[1], keypoints[1]);
        surf.downloadDescriptors(descriptorsGPU[0], descriptors[0]);
        surf.downloadDescriptors(descriptorsGPU[1], descriptors[1]);

    }

    // ORB as feature detector
    if(isORB_)  {
        Ptr<cuda::ORB> orb = cv::cuda::ORB::create();
        Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
        orb->detectAndCompute(previous_frameGPU, cuda::GpuMat(), keypoints[0], descriptorsGPU[0]);
        orb->detectAndCompute(current_frameGPU, cuda::GpuMat(), keypoints[1], descriptorsGPU[1]);
 
        matcher->knnMatch(descriptorsGPU[0], descriptorsGPU[1], matches1, 2);
        matcher->knnMatch(descriptorsGPU[1], descriptorsGPU[0], matches2, 2);
    }

    // Remove matches for which NN ratio is > than threshold
    // clean image 1 -> image 2 matches
    int removed = ratioTest(matches1);
    // clean image 2 -> image 1 matches
    removed = ratioTest(matches2);

    // Remove non-symmetrical matches
    vector<DMatch> symMatches;
    symmetryTest(matches1, matches2, symMatches);

    // Validate matches using RANSAC
    vector< DMatch> goodMatches;    
    Mat fundamental = ransacTest(symMatches, keypoints[0], keypoints[1], goodMatches);

    // Obtain good keypoints from goodMatches
    array<vector<KeyPoint>,2> goodKeypoints;
    goodKeypoints = getGoodKeypoints(goodMatches, keypoints);

    _previous_frame->n_matches_ = goodMatches.size();
    _current_frame->n_matches_ = goodMatches.size();
    
    _previous_frame->keypoints_ = goodKeypoints[0];    
    _current_frame->keypoints_  = goodKeypoints[1];

    // Show results
    Mat img_matches;
    drawMatches(Mat(previous_frameGPU), keypoints[0], Mat(current_frameGPU), keypoints[1], 
                goodMatches, img_matches, Scalar::all(-1), Scalar::all(-1), 
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("ORB", img_matches);
    
    waitKey(0);
}

void RobustMatcher::DetectAndTrackFeatures(Frame* _previous_frame, Frame* _current_frame) {
    cuda::GpuMat previous_frameGPU, current_frameGPU;
    cuda::GpuMat keypointsGPU[2];
    cuda::GpuMat descriptorsGPU[2];
    array<vector<KeyPoint>,2> keypoints;
    array<vector<float>,2> descriptors;
    
    // Upload images to GPU
    previous_frameGPU.upload(_previous_frame->images_[0]);
    current_frameGPU.upload(_current_frame->images_[0]);

    // Matching descriptors
    vector< vector< DMatch> > matches1;
    vector< vector< DMatch> > matches2;

    // SURF as feature detector
    if(isSURF_) {
        cuda::SURF_CUDA surf;        
        Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher();

        // Detecting keypoints and computing descriptors
        surf(previous_frameGPU, cuda::GpuMat(), keypointsGPU[0], descriptorsGPU[0]);
        surf(current_frameGPU, cuda::GpuMat(), keypointsGPU[1], descriptorsGPU[1]);

        // Matching descriptors
        matcher->knnMatch(descriptorsGPU[0], descriptorsGPU[1], matches1, 2);
        matcher->knnMatch(descriptorsGPU[1], descriptorsGPU[0], matches2, 2);
        
        // Downloading results
        surf.downloadKeypoints(keypointsGPU[0], keypoints[0]);
        surf.downloadKeypoints(keypointsGPU[1], keypoints[1]);
        surf.downloadDescriptors(descriptorsGPU[0], descriptors[0]);
        surf.downloadDescriptors(descriptorsGPU[1], descriptors[1]);

    }

    // ORB as feature detector
    if(isORB_)  {
        Ptr<cuda::ORB> orb = cv::cuda::ORB::create();
        Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
        orb->detectAndCompute(previous_frameGPU, cuda::GpuMat(), keypoints[0], descriptorsGPU[0]);
        orb->detectAndCompute(current_frameGPU, cuda::GpuMat(), keypoints[1], descriptorsGPU[1]);
 
        matcher->knnMatch(descriptorsGPU[0], descriptorsGPU[1], matches1, 2);
        matcher->knnMatch(descriptorsGPU[1], descriptorsGPU[0], matches2, 2);
    }

    // Remove matches for which NN ratio is > than threshold
    // clean image 1 -> image 2 matches
    int removed = ratioTest(matches1);
    // clean image 2 -> image 1 matches
    removed = ratioTest(matches2);

    // Remove non-symmetrical matches
    vector<DMatch> symMatches;
    symmetryTest(matches1, matches2, symMatches);

    // Validate matches using RANSAC
    vector< DMatch> goodMatches;    
    Mat fundamental = ransacTest(symMatches, keypoints[0], keypoints[1], goodMatches);

    // Obtain good keypoints from goodMatches
    array<vector<KeyPoint>,2> goodKeypoints;
    goodKeypoints = getGoodKeypoints(goodMatches, keypoints);

    _previous_frame->n_matches_ = goodMatches.size();
    _current_frame->n_matches_ = goodMatches.size();
    
    _previous_frame->keypoints_ = goodKeypoints[0];    
    _current_frame->keypoints_  = goodKeypoints[1];

    // Show results
    Mat img_matches;
    drawMatches(Mat(previous_frameGPU), keypoints[0], Mat(current_frameGPU), keypoints[1], 
                goodMatches, img_matches, Scalar::all(-1), Scalar::all(-1), 
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("ORB", img_matches);
    
    waitKey(0);     

}

array<vector<KeyPoint>,2> RobustMatcher::getGoodKeypoints(vector<DMatch> goodMatches, array< vector< KeyPoint>, 2 > keypoints) {
    array<vector<KeyPoint>,2> goodKeypoints;
    int key1_index, key2_index;
    for(int i=0; i < goodMatches.size(); i++){
        key1_index = goodMatches[i].queryIdx;
        key2_index = goodMatches[i].trainIdx;
        goodKeypoints[0].push_back(keypoints[0][key1_index]);
        goodKeypoints[1].push_back(keypoints[1][key2_index]);
    }
    return goodKeypoints;
}

Tracker::Tracker(bool _depth_available) {
    robust_matcher_ = new RobustMatcher();
    patch_size_ = 5;
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

void Tracker::InitializeMasks() {

    num_grids_ = (w_[0]/grid_size_) * (h_[0]/grid_size_);

    for (int x1=0, x2=grid_size_; x2<=w_[0]; x1+=grid_size_, x2+=grid_size_) {
        for (int y1=0, y2=grid_size_; y2<=h_[0]; y1+=grid_size_, y2+=grid_size_) {
            Mat mask = Mat::zeros(h_[0], w_[0], CV_8UC1);
            for (int x=0; x<w_[0]; x++) {
                for (int y=0; y<h_[0]; y++) {
                    if (x>=x1 && x<x2 && y>=y1 && y<y2) {
                        mask.at<uchar>(y,x) = 1;
                    }
                }
            }
            masks_.push_back(mask);
        }
    }
}

// Gauss-Newton using Foward Compositional Algorithm
void Tracker::EstimatePose(Frame* _previous_frame, Frame* _current_frame) {
    // Gauss-Newton Optimization Options
    float epsilon = 0.001;
    float intial_factor = 10;
    int max_iterations = 50;
    float error_threshold = 0.005;
    int first_pyramid_lvl = PYRAMID_LEVELS-1;
    int last_pyramid_lvl = 1;
    
    float z_factor = 1;
    float angle_factor = 1;
    
    // Variables initialization
    float error         = 0.0;
    float initial_error = 0.0;    
    float last_error    = 0.0;

    // Initial pose and deltapose (assumes little movement between frames)
    Mat deltaMat = Mat::zeros(6,1,CV_32FC1);
    Sophus::Vector<float, SE3::DoF> deltaVector;
    for (int i=0; i<6; i++)
        deltaVector(i) = 0;

    SE3 current_pose = SE3(SO3::exp(SE3::Point(0.0, 0.0, 0.0)), SE3::Point(0.0, 0.0, 0.0));

    // Sparse to Fine iteration
    // Create for() WORKED WITH LVL 2
    for (int lvl = first_pyramid_lvl; lvl>=last_pyramid_lvl; lvl--) {
        // lvl = 0;
        // Initialize error   
        error = 0.0;
        last_error = 50000.0;
        float factor = intial_factor * (lvl + 1);

        // Obtain image 1 and 2
        Mat image1 = _previous_frame->images_[lvl].clone();
        Mat image2 = _current_frame->images_[lvl].clone();

        // Obtain points and depth of initial frame 
        Mat candidatePoints1  = _previous_frame->candidatePoints_[lvl].clone();
        Mat candidatePoints2  = _current_frame->candidatePoints_[lvl].clone();    
        
        // Obtain gradients           
        Mat gradientX1 = Mat::zeros(image1.size(), CV_16SC1);
        Mat gradientY1 = Mat::zeros(image1.size(), CV_16SC1);
        gradientX1 = _previous_frame->gradientX_[lvl].clone();
        gradientY1 = _previous_frame->gradientY_[lvl].clone();

        // Obtain intrinsic parameters 
        Mat K = K_[lvl];

        // Optimization iteration
        for (int k=0; k<max_iterations; k++) {
            
            // Warp points with current pose and delta pose (from previous iteration)
            SE3 deltaSE3;
            Mat warpedPoints = Mat(candidatePoints1.size(), CV_32FC1);


            //warpedPoints = WarpFunctionOpenCV(candidatePoints1, current_pose, lvl);
            warpedPoints = WarpFunction(candidatePoints1, current_pose, lvl);
            // Mat imageWarped = Mat::zeros(image1.size(), CV_8UC1);
            // ObtainImageTransformed(image1, candidatePoints1, warpedPoints, imageWarped);    
            // imshow("warped", imageWarped);
            // waitKey(0);
            // Computation of Jacobian and Residuals
            Mat Jacobians;
            Mat Residuals;      

            int num_valid = 0;
            for (int i=0; i<candidatePoints1.rows; i++) {
                Mat Residual = Mat(1,1,CV_32FC1);        
                Mat Jacobian_row = Mat::zeros(1,6,CV_32FC1);
                Mat Jw = Mat::zeros(2,6,CV_32FC1);
                Mat Jl = Mat::zeros(1,2,CV_32FC1);
                
                // Point in frame 1            
                float x1 = candidatePoints1.at<float>(i,0);
                float y1 = candidatePoints1.at<float>(i,1);
                float z1 = candidatePoints1.at<float>(i,2);
                // Points of warped frame
                float x2 = warpedPoints.at<float>(i,0);
                float y2 = warpedPoints.at<float>(i,1);
                float z2 = warpedPoints.at<float>(i,2);

                float inv_z2 = 1 / z2;

                // Check if warpedPoints are out of boundaries
                if (y2>0 && y2<image2.rows && x2>0 && x2<image2.cols) {
                    if (z2!=0) {
                        if (inv_z2<0) 
                            inv_z2 = 0;
                        num_valid++;
                        Jw.at<float>(0,0) = fx_[lvl] * inv_z2;
                        Jw.at<float>(0,1) = 0.0;
                        Jw.at<float>(0,2) = -(fx_[lvl] * x2 * inv_z2 * inv_z2) * z_factor;
                        Jw.at<float>(0,3) = -(fx_[lvl] * x2 * y2 * inv_z2 * inv_z2) * angle_factor;
                        Jw.at<float>(0,4) = (fx_[lvl] * (1 + x2 * x2 * inv_z2 * inv_z2)) * angle_factor;   
                        Jw.at<float>(0,5) = - fx_[lvl] * y2 * inv_z2 * angle_factor;

                        Jw.at<float>(1,0) = 0.0;
                        Jw.at<float>(1,1) = fy_[lvl] * inv_z2;
                        Jw.at<float>(1,2) = -(fy_[lvl] * y2 * inv_z2 * inv_z2) * z_factor;
                        Jw.at<float>(1,3) = -(fy_[lvl] * (1 + y2 * y2 * inv_z2 * inv_z2)) * angle_factor;
                        Jw.at<float>(1,4) = fy_[lvl] * x2 * y2 * inv_z2 * inv_z2 * angle_factor;
                        Jw.at<float>(1,5) = fy_[lvl] * x2 * inv_z2 * angle_factor;

                    
                        // Intensities
                        int intensity1 = image1.at<uchar>(y1,x1);
                        int intensity2 = image2.at<uchar>(round(y2),round(x2));

                        Residual.at<float>(0,0) = intensity2 - intensity1;
                        
                        Jl.at<float>(0,0) = gradientX1.at<short>(y1,x1);
                        Jl.at<float>(0,1) = gradientY1.at<short>(y1,x1);

                        Jacobian_row = Jl * Jw;
                        // cout << "Residual: " << Residual.at<float>(0,0) << endl;
                        // cout << "Jl: " << Jl << endl;                
                        // cout << "Jw: " << Jw << endl;                                
                        // cout << "Jacobian: " << Jacobian_row << endl;
                        // cout << endl;
                        
                        Jacobians.push_back(Jacobian_row);
                        Residuals.push_back(Residual);
                    }
                }
            }
            // cout << "Valid points found: " << num_valid << endl;
            // DebugShowJacobians(Jacobians, warpedPoints, w_[lvl], h_[lvl]);

            // Computation of Weights (Identity or Tukey function)
            Mat W = IdentityWeights(Residuals.rows);
            //Mat W = TukeyFunctionWeights(Residuals);

            // Computation of error
            float inv_num_residuals = 1.0 / Residuals.rows;
            Mat ResidualsW = Residuals.mul(W);
            Mat errorMat =  inv_num_residuals * Residuals.t() * ResidualsW;
            error = errorMat.at<float>(0,0);
   
            if (k==0)
                initial_error = error;

            // Break if error increases
            if (error >= last_error || k == max_iterations-1 || abs(error - last_error) < epsilon) {
                // cout << "Pyramid level: " << lvl << endl;
                // cout << "Number of iterations: " << k << endl;
                // cout << "Initial-Final Error: " << initial_error << " - " << last_error << endl << endl;

                // if (lvl == last_pyramid_lvl) {
                //     //DebugShowJacobians(Jacobians, warpedPoints, w_[lvl], h_[lvl]);
                //     Mat imageWarped = Mat::zeros(image1.size(), CV_8UC1);
                //     ObtainImageTransformed(image1, candidatePoints1, warpedPoints, imageWarped);             
                //     DebugShowWarpedPerspective(image1, image2, imageWarped, lvl);
                // }

                // Reset delta
                deltaMat = Mat::zeros(6,1,CV_32FC1);
           
                for (int i=0; i<6; i++)
                    deltaVector(i) = 0;

                break;
            }

            last_error = error;

            // Checking dimentions of matrices
            // cout << "Jacobians dimentions: " << Jacobians.size() << endl;
            // cout << "Weights dimentions: " << W.size() << endl;
            // cout << "Residuals dimentions: " << Residuals.size() << endl;
            
            // Computation of new delta (DSO-way)
            // LS ls;
            // ls.initialize(Residuals.rows);
            // for (int i=0; i<Residuals.rows; i++) {
            //     Mat61f jacobian;
            //     cv2eigen(Jacobians.row(i), jacobian);
                
            //     ls.update(jacobian, Residuals.at<float>(i,0), W.at<float>(i,0));
            // }
            // ls.finish();
            // // Solve LS system
            // float LM_lambda = 0.2;
            // Mat61f b = -ls.b;
            // Mat66f A = ls.A;
            // deltaVector = A.ldlt().solve(b);

            // Computation of new delta (Kerl-way)            
            // Multiplication of W to Jacobian
            for (int i=0; i<Jacobians.rows; i++) {
                float wi = W.at<float>(i,0);
                Jacobians.row(i) = wi * Jacobians.row(i);
            }

            Residuals = Residuals.mul(50);  // Workaround to make delta updates larger
            Mat A = Jacobians.t() * Jacobians;                    
            Mat b = -Jacobians.t() * Residuals.mul(W);

            //cout << b << endl;
            deltaMat = A.inv() * b;
            //cout << A.inv() << endl;
            //cout << A << endl;
            

            // Convert info from eigen to cv
            for (int i=0; i<6; i++)
                deltaVector(i) = deltaMat.at<float>(i,0);

            // Update new pose with computed delta
            current_pose = current_pose * SE3::exp(deltaVector);
            //cout << current_pose.matrix() << endl;
            
        }

        // Scale current_pose estimation to next lvl
        if (lvl !=0) {
            Mat31f t = current_pose.translation();
        
            Quaternion quaternion = current_pose.unit_quaternion();

            quaternion.x() = quaternion.x() * 2;
            quaternion.y() = quaternion.y() * 2;
            quaternion.z() = quaternion.z() * 2;
            
            current_pose = SE3(quaternion, t);
        }
        
        //current_pose = SE3(current_pose.unit_quaternion() * 2, current_pose.translation() * 2);
    }

    _previous_frame->rigid_transformation_ = current_pose;
    cout << current_pose.matrix() << endl;
}

Mat Tracker::AddPatchPointsFeatures(Mat candidatePoints, int lvl) {

    Mat newCandidatePoints = candidatePoints.clone();
    int start_point = (patch_size_ - 1) / 2;
    int n = 0;

    for (int index=0; index<candidatePoints.rows; index++) {

        float x = round(candidatePoints.at<float>(index,0));
        float y = round(candidatePoints.at<float>(index,1));
        float z = candidatePoints.at<float>(index,2);
        
        for (int i=x-start_point; i<=x+start_point; i++) {
            for (int j=y-start_point; j<=y+start_point; j++) {
                if (i>0 && i<w_[lvl] && j>0 && j<h_[lvl] && !(i==x && j==y)) {
                    Mat pointMat = Mat::ones(1, 4, CV_32FC1);                
                    pointMat.at<float>(0,0) = i;
                    pointMat.at<float>(0,1) = j;
                    pointMat.at<float>(0,2) = z;
                    
                    newCandidatePoints.push_back(pointMat);
                } else {
                    n++;
                }
            }
        }

    }

    return newCandidatePoints; 
}

// Gauss-Newton using Foward Compositional Algorithm - Using features
void Tracker::EstimatePoseFeatures(Frame* _previous_frame, Frame* _current_frame) {
    // Gauss-Newton Optimization Options
    float epsilon = 0.001;
    float intial_factor = 10;
    int max_iterations = 10;
    float error_threshold = 0.005;
    int first_pyramid_lvl = 0;
    int last_pyramid_lvl = 0;
    float z_factor = 0.002;

    // Variables initialization
    float error         = 0.0;
    float initial_error = 0.0;    
    float last_error    = 0.0;

    // Initial pose and deltapose (assumes little movement between frames)
    Mat deltaMat = Mat::zeros(6,1,CV_32FC1);
    Sophus::Vector<float, SE3::DoF> deltaVector;
    for (int i=0; i<6; i++)
        deltaVector(i) = 0;

    SE3 current_pose = SE3(SO3::exp(SE3::Point(0.0, 0.0, 0.0)), SE3::Point(0.0, 0.0, 0.0));

    // Sparse to Fine iteration
    // Create for() WORKED WITH LVL 2
    for (int lvl = first_pyramid_lvl; lvl>=last_pyramid_lvl; lvl--) {
        // lvl = 0;
        // Initialize error   
        error = 0.0;
        last_error = 50000.0;
        float factor = intial_factor * (lvl + 1);

        // Obtain image 1 and 2
        Mat image1 = _previous_frame->images_[lvl].clone();
        Mat image2 = _current_frame->images_[lvl].clone();

        // Obtain points and depth of initial frame 
        Mat candidatePoints1  = _previous_frame->candidatePoints_[lvl].clone();
        Mat candidatePoints2  = _current_frame->candidatePoints_[lvl].clone();    
        
        //candidatePoints1 = AddPatchPointsFeatures(candidatePoints1, lvl);
        // cout << candidatePoints1.rows << endl;
        // Mat imageWarped = Mat::zeros(image1.size(), CV_8UC1);
        // ObtainImageTransformed(image1, candidatePoints1, candidatePoints1, imageWarped);             
        // DebugShowWarpedPerspective(image1, image2, imageWarped, lvl);

        // Obtain gradients           
        Mat gradientX1 = Mat::zeros(image1.size(), CV_16SC1);
        Mat gradientY1 = Mat::zeros(image1.size(), CV_16SC1);
        gradientX1 = _previous_frame->gradientX_[lvl].clone();
        gradientY1 = _previous_frame->gradientY_[lvl].clone();

        // Obtain intrinsic parameters 
        Mat K = K_[lvl];

        // Optimization iteration
        for (int k=0; k<max_iterations; k++) {
            
            // Warp points with current pose and delta pose (from previous iteration)
            SE3 deltaSE3;
            Mat warpedPoints = Mat(candidatePoints1.size(), CV_32FC1);


            //warpedPoints = WarpFunctionOpenCV(candidatePoints1, current_pose, lvl);
            warpedPoints = WarpFunction(candidatePoints1, current_pose, lvl);
            // Mat imageWarped = Mat::zeros(image1.size(), CV_8UC1);
            // ObtainImageTransformed(image1, candidatePoints1, warpedPoints, imageWarped);    
            // imshow("warped", imageWarped);
            // waitKey(0);
            // Computation of Jacobian and Residuals
            Mat Jacobians;
            Mat Residuals;      

            int num_valid = 0;
            for (int i=0; i<candidatePoints1.rows; i++) {
                Mat Residual = Mat(1,1,CV_32FC1);        
                Mat Jacobian_row = Mat::zeros(1,6,CV_32FC1);
                Mat Jw = Mat::zeros(2,6,CV_32FC1);
                Mat Jl = Mat::zeros(1,2,CV_32FC1);
                
                // Point in frame 1            
                float x1 = candidatePoints1.at<float>(i,0);
                float y1 = candidatePoints1.at<float>(i,1);
                float z1 = candidatePoints1.at<float>(i,2);
                // Points of warped frame
                float x2 = warpedPoints.at<float>(i,0);
                float y2 = warpedPoints.at<float>(i,1);
                float z2 = warpedPoints.at<float>(i,2);

                float inv_z2 = 1 / z2;

                // Check if warpedPoints are out of boundaries
                if (y2>0 && y2<image2.rows && x2>0 && x2<image2.cols) {
                    if (z2!=0) {
                        if (inv_z2<0) 
                            inv_z2 = 0;
                        num_valid++;
                        Jw.at<float>(0,0) = fx_[lvl] * inv_z2;
                        Jw.at<float>(0,1) = 0.0;
                        Jw.at<float>(0,2) = -(fx_[lvl] * x2 * inv_z2 * inv_z2) * z_factor;
                        Jw.at<float>(0,3) = -(fx_[lvl] * x2 * y2 * inv_z2 * inv_z2);
                        Jw.at<float>(0,4) = (fx_[lvl] * (1 + x2 * x2 * inv_z2 * inv_z2));   
                        Jw.at<float>(0,5) = - fx_[lvl] * y2 * inv_z2;

                        Jw.at<float>(1,0) = 0.0;
                        Jw.at<float>(1,1) = fy_[lvl] * inv_z2;
                        Jw.at<float>(1,2) = -(fy_[lvl] * y2 * inv_z2 * inv_z2) * z_factor;
                        Jw.at<float>(1,3) = -(fy_[lvl] * (1 + y2 * y2 * inv_z2 * inv_z2));
                        Jw.at<float>(1,4) = fy_[lvl] * x2 * y2 * inv_z2 * inv_z2;
                        Jw.at<float>(1,5) = fy_[lvl] * x2 * inv_z2;

                    
                        // Intensities
                        int intensity1 = image1.at<uchar>(y1,x1);
                        int intensity2 = image2.at<uchar>(round(y2),round(x2));

                        Residual.at<float>(0,0) = intensity2 - intensity1;
                        
                        Jl.at<float>(0,0) = gradientX1.at<short>(y1,x1);
                        Jl.at<float>(0,1) = gradientY1.at<short>(y1,x1);

                        Jacobian_row = Jl * Jw;
                        // cout << "Residual: " << Residual.at<float>(0,0) << endl;
                        // cout << "Jl: " << Jl << endl;                
                        // cout << "Jw: " << Jw << endl;                                
                        // cout << "Jacobian: " << Jacobian_row << endl;
                        // cout << endl;
                        
                        Jacobians.push_back(Jacobian_row);
                        Residuals.push_back(Residual);
                    }
                }
            }
            // cout << "Valid points found: " << num_valid << endl;
            // DebugShowJacobians(Jacobians, warpedPoints, w_[lvl], h_[lvl]);

            // Computation of Weights (Identity or Tukey function)
            Mat W = IdentityWeights(Residuals.rows);
            //Mat W = TukeyFunctionWeights(Residuals);

            // Computation of error
            float inv_num_residuals = 1.0 / Residuals.rows;
            Mat ResidualsW = Residuals.mul(W);
            Mat errorMat =  inv_num_residuals * Residuals.t() * ResidualsW;
            error = errorMat.at<float>(0,0);
   
            if (k==0)
                initial_error = error;

            // Break if error increases
            if (error >= last_error || k == max_iterations-1 || abs(error - last_error) < epsilon) {
                // cout << "Pyramid level: " << lvl << endl;
                // cout << "Number of iterations: " << k << endl;
                // cout << "Initial-Final Error: " << initial_error << " - " << last_error << endl << endl;

                // if (lvl == last_pyramid_lvl) {
                //     DebugShowJacobians(Jacobians, warpedPoints, w_[lvl], h_[lvl]);
                //     Mat imageWarped = Mat::zeros(image1.size(), CV_8UC1);
                //     ObtainImageTransformed(image1, candidatePoints1, warpedPoints, imageWarped);             
                //     DebugShowWarpedPerspective(image1, image2, imageWarped, lvl);
                // }

                // Reset delta
                deltaMat = Mat::zeros(6,1,CV_32FC1);
           
                for (int i=0; i<6; i++)
                    deltaVector(i) = 0;

                break;
            }

            last_error = error;

            // Checking dimentions of matrices
            // cout << "Jacobians dimentions: " << Jacobians.size() << endl;
            // cout << "Weights dimentions: " << W.size() << endl;
            // cout << "Residuals dimentions: " << Residuals.size() << endl;
            
            // Computation of new delta (DSO-way)
            // LS ls;
            // ls.initialize(Residuals.rows);
            // for (int i=0; i<Residuals.rows; i++) {
            //     Mat61f jacobian;
            //     cv2eigen(Jacobians.row(i), jacobian);
                
            //     ls.update(jacobian, Residuals.at<float>(i,0), W.at<float>(i,0));
            // }
            // ls.finish();
            // // Solve LS system
            // float LM_lambda = 0.2;
            // Mat61f b = -ls.b;
            // Mat66f A = ls.A;
            // deltaVector = A.ldlt().solve(b);

            // Computation of new delta (Kerl-way)            
            // Multiplication of W to Jacobian
            for (int i=0; i<Jacobians.rows; i++) {
                float wi = W.at<float>(i,0);
                Jacobians.row(i) = wi * Jacobians.row(i);
            }

            Residuals = Residuals.mul(1);  // Workaround to make delta updates larger
            Mat A = Jacobians.t() * Jacobians;                    
            Mat b = -Jacobians.t() * Residuals.mul(W);

            //cout << b << endl;
            deltaMat = A.inv() * b;
            //cout << A.inv() << endl;
            //cout << A << endl;
            

            // Convert info from eigen to cv
            for (int i=0; i<6; i++)
                deltaVector(i) = deltaMat.at<float>(i,0);

            // Update new pose with computed delta
            current_pose = current_pose * SE3::exp(deltaVector);
            //cout << current_pose.matrix() << endl;
            
        }

        // Scale current_pose estimation to next lvl
        if (lvl !=0) {
            Mat31f t = 2 * current_pose.translation();

            Quaternion quaternion = current_pose.unit_quaternion();

            quaternion.x() = quaternion.x() * 2;
            quaternion.y() = quaternion.y() * 2;
            quaternion.z() = quaternion.z() * 2;
            
            current_pose = SE3(quaternion, t);
        }
        
        //current_pose = SE3(current_pose.unit_quaternion() * 2, current_pose.translation() * 2);
    }

    _previous_frame->rigid_transformation_ = current_pose;

}

// Gauss-Newton using Foward Compositional Algorithm "Fast"
void Tracker::FastEstimatePose(Frame* _previous_frame, Frame* _current_frame) {
    // Gauss-Newton Optimization Options
    float epsilon = 0.001;
    float intial_factor = 10;
    int max_iterations = 50;
    float error_threshold = 0.005;
    int first_pyramid_lvl = PYRAMID_LEVELS-1;
    int last_pyramid_lvl = 0;
    
    float z_factor = 0.01;
    float angle_factor = 1;
    
    // Variables initialization
    float error         = 0.0;
    float initial_error = 0.0;    
    float last_error    = 0.0;

    // Initial pose and deltapose (assumes little movement between frames)
    Mat deltaMat = Mat::zeros(6,1,CV_32FC1);
    Sophus::Vector<float, SE3::DoF> deltaVector;
    for (int i=0; i<6; i++)
        deltaVector(i) = 0;

    SE3 current_pose = SE3(SO3::exp(SE3::Point(0.0, 0.0, 0.0)), SE3::Point(0.0, 0.0, 0.0));

    // Sparse to Fine iteration
    // Create for() WORKED WITH LVL 2
    for (int lvl = first_pyramid_lvl; lvl>=last_pyramid_lvl; lvl--) {
        //lvl = 0;
        // Initialize error   
        error = 0.0;
        last_error = 50000.0;
        float factor = intial_factor * (lvl + 1);

        // Obtain image 1 and 2
        Mat image1 = _previous_frame->images_[lvl].clone();
        Mat image2 = _current_frame->images_[lvl].clone();
        
        // Obtain points and depth of initial frame 
        Mat candidatePoints1  = _previous_frame->candidatePoints_[lvl].clone();
        Mat informationPoints1 = _previous_frame->informationPoints_[lvl].clone();

        // Obtain intrinsic parameters 
        Mat K = K_[lvl];

        // Optimization iteration
        for (int k=0; k<max_iterations; k++) {
            
            // Warp points with current pose and delta pose (from previous iteration)
            SE3 deltaSE3;
            Mat warpedPoints = Mat(candidatePoints1.size(), CV_32FC1);

            Mat Residuals = Mat(candidatePoints1.size(), CV_32FC1);
            warpedPoints = WarpFunction(candidatePoints1, current_pose, lvl);
            Mat imageWarped = Mat::zeros(image1.size(), CV_8UC1);
            Mat validPixels = ObtainImageTransformed(image1, candidatePoints1, warpedPoints, imageWarped);
            Mat image2_filtered = image2.mul(validPixels);

            vector<uchar> image2_aux;
            image2_aux.assign((uchar*)image2.datastart, (uchar*)image2.dataend);
            Mat intensities2_uchar = Mat(image2.size(), CV_32FC1);
            intensities2_uchar = Mat(image2_aux);
            
            Mat intensities2;
            intensities2_uchar.convertTo(intensities2, CV_32FC1, 1, 0);

            Mat intensities1 = Mat(image2.size(), CV_32FC1);
            intensities1 = informationPoints1.col(0);

            Residuals = intensities2 - intensities1;

            // Computation of Jacobian and Residuals
            Mat Jacobians;
            Mat Jw1 = Mat::zeros(warpedPoints.rows, 6, CV_32FC1);
            Mat Jw2 = Mat::zeros(warpedPoints.rows, 6, CV_32FC1);
    
            Mat valid_values = Mat(warpedPoints.rows, 1, CV_32FC1);
            Mat bellowZeroThreshold = Mat(warpedPoints.size(), CV_32FC1);
            Mat aboveHeight = Mat(warpedPoints.rows, 1, CV_32FC1);
            Mat aboveWeight = Mat(warpedPoints.rows, 1, CV_32FC1);
            
            // Putting to zero all negative values
            threshold(warpedPoints, bellowZeroThreshold, 0, 1, THRESH_BINARY);
            // Putting to zero all values above the weight and height of the image            
            threshold(warpedPoints.col(0), aboveWeight, w_[lvl]-1, 1, THRESH_BINARY_INV);
            threshold(warpedPoints.col(1), aboveHeight, h_[lvl]-1, 1, THRESH_BINARY_INV);
            // Concatenatiing threshold results on valid_values
            valid_values = bellowZeroThreshold.col(0).mul(bellowZeroThreshold.col(1).mul(bellowZeroThreshold.col(3)));
            valid_values = valid_values.mul(aboveWeight.mul(aboveHeight));

            Mat inv_z2 = 1 / warpedPoints.col(2);
            Mat inv2_z2 = inv_z2.mul(inv_z2);

            // Jw1
            Jw1.col(0) = fx_[lvl] * inv_z2;
            
            Jw1.col(1) = 0.0;
            
            Jw1.col(2) = -fx_[lvl];
            Jw1.col(2) = Jw1.col(2).mul(warpedPoints.col(0).mul(inv2_z2));

            Jw1.col(3) = -fx_[lvl];
            Jw1.col(3) = Jw1.col(3).mul(warpedPoints.col(0).mul(warpedPoints.col(1).mul(inv2_z2)));

            Jw1.col(4) = 1 + warpedPoints.col(0).mul(warpedPoints.col(0).mul(inv2_z2));
            Jw1.col(4) *= fx_[lvl];

            Jw1.col(5) = -fx_[lvl];
            Jw1.col(5) = Jw1.col(5).mul(warpedPoints.col(1).mul(inv_z2));

            // Jw2
            Jw1.col(0) = 0.0;
            
            Jw2.col(1) = fy_[lvl] * inv_z2;

            Jw2.col(2) = -fy_[lvl];
            Jw2.col(2) = Jw2.col(2).mul(warpedPoints.col(1).mul(inv2_z2));

            Jw2.col(3) = 1 + warpedPoints.col(1).mul(warpedPoints.col(1).mul(inv2_z2));
            Jw2.col(3) *= -fy_[lvl];

            Jw2.col(4) = fy_[lvl];
            Jw2.col(4) = Jw2.col(4).mul(warpedPoints.col(0).mul(warpedPoints.col(1).mul(inv2_z2)));

            Jw2.col(5) = fy_[lvl];
            Jw2.col(5) = Jw2.col(5).mul(warpedPoints.col(0).mul(inv_z2));

            Mat gradientX_Nx6 = Mat(warpedPoints.rows, 6, CV_32FC1);
            Mat gradientY_Nx6 = Mat(warpedPoints.rows, 6, CV_32FC1);
            Mat valid_Jacobians = Mat(warpedPoints.rows, 6, CV_32FC1);
            for (int i=0; i<6; i++) {
                gradientX_Nx6.col(i) = informationPoints1.col(1).mul(1);
                gradientY_Nx6.col(i) = informationPoints1.col(2).mul(1);
                valid_Jacobians.col(i) = valid_values.mul(1);
            }

            Jacobians = Jw1.mul(gradientX_Nx6) + Jw2.mul(gradientY_Nx6);
            Jacobians = Jacobians.mul(valid_Jacobians);

            Residuals = Residuals.mul(valid_values);
            

            // Computation of Weights (Identity or Tukey function)
            Mat W = IdentityWeights(Residuals.rows);
            //Mat W = TukeyFunctionWeights(Residuals);

            // Computation of error
            float inv_num_residuals = 1.0 / Residuals.rows;
            Mat ResidualsW = Residuals.mul(W);
            Mat errorMat =  inv_num_residuals * Residuals.t() * ResidualsW;
            error = errorMat.at<float>(0,0);
   
            if (k==0)
                initial_error = error;

            // Break if error increases
            if (error >= last_error || k == max_iterations-1 || abs(error - last_error) < epsilon) {
                cout << "Pyramid level: " << lvl << endl;
                cout << "Number of iterations: " << k << endl;
                cout << "Initial-Final Error: " << initial_error << " - " << last_error << endl << endl;

                if (lvl == last_pyramid_lvl) {
                    //DebugShowJacobians(Jacobians, warpedPoints, w_[lvl], h_[lvl]);
                    Mat imageWarped = Mat::zeros(image1.size(), CV_8UC1);
                    ObtainImageTransformed(image1, candidatePoints1, warpedPoints, imageWarped);             
                    DebugShowWarpedPerspective(image1, image2, imageWarped, lvl);
                }

                // Reset delta
                deltaMat = Mat::zeros(6,1,CV_32FC1);
           
                for (int i=0; i<6; i++)
                    deltaVector(i) = 0;

                break;
            }

            last_error = error;

            // Checking dimentions of matrices
            // cout << "Jacobians dimentions: " << Jacobians.size() << endl;
            // cout << "Weights dimentions: " << W.size() << endl;
            // cout << "Residuals dimentions: " << Residuals.size() << endl;
            
            // Computation of new delta (DSO-way)
            // LS ls;
            // ls.initialize(Residuals.rows);
            // for (int i=0; i<Residuals.rows; i++) {
            //     Mat61f jacobian;
            //     cv2eigen(Jacobians.row(i), jacobian);
                
            //     ls.update(jacobian, Residuals.at<float>(i,0), W.at<float>(i,0));
            // }
            // ls.finish();
            // // Solve LS system
            // float LM_lambda = 0.2;
            // Mat61f b = -ls.b;
            // Mat66f A = ls.A;
            // deltaVector = A.ldlt().solve(b);

            // Computation of new delta (Kerl-way)            
            // Multiplication of W to Jacobian
            for (int i=0; i<Jacobians.rows; i++) {
                float wi = W.at<float>(i,0);
                Jacobians.row(i) = wi * Jacobians.row(i);
            }

            Residuals = Residuals.mul(50);  // Workaround to make delta updates larger
            Mat A = Jacobians.t() * Jacobians;                    
            Mat b = -Jacobians.t() * Residuals.mul(W);
            cout << A << endl;
            cout << b << endl;
            
            //cout << b << endl;
            deltaMat = A.inv() * b;
            //cout << A.inv() << endl;
            //cout << A << endl;
            

            // Convert info from eigen to cv
            for (int i=0; i<6; i++)
                deltaVector(i) = deltaMat.at<float>(i,0);

            // Update new pose with computed delta
            current_pose = current_pose * SE3::exp(deltaVector);
            //cout << current_pose.matrix() << endl;
            
        }

        // Scale current_pose estimation to next lvl
        if (lvl !=0) {
            Mat31f t = current_pose.translation();
        
            Quaternion quaternion = current_pose.unit_quaternion();

            quaternion.x() = quaternion.x() * 2;
            quaternion.y() = quaternion.y() * 2;
            quaternion.z() = quaternion.z() * 2;
            
            current_pose = SE3(quaternion, t);
        }
        
        //current_pose = SE3(current_pose.unit_quaternion() * 2, current_pose.translation() * 2);
    }

    _previous_frame->rigid_transformation_ = current_pose;

}

// TODO(GitHub:fmoralesh, fabmoraleshidalgo@gmail.com)
// 02-20-2018 - Consider other methods to obtain gradient from an image (Sober, Laplacian, ...) 
//            - Calculate gradient for each pyramid image or scale the finest?
void Tracker::ApplyGradient(Frame* _frame) {

    for (int lvl = 0; lvl<PYRAMID_LEVELS; lvl++) {
        Mat gradientX = Mat(_frame->images_[lvl].size(), CV_16SC1); 
        Mat gradientY = Mat(_frame->images_[lvl].size(), CV_16SC1);

        Scharr(_frame->images_[lvl], _frame->gradientX_[lvl], CV_16S, 1, 0, 3, 0, BORDER_DEFAULT);
        Scharr(_frame->images_[lvl], _frame->gradientY_[lvl], CV_16S, 0, 1, 3, 0, BORDER_DEFAULT);
        
        Scharr(_frame->images_[lvl], gradientX, CV_16S, 1, 0, 3, 0, BORDER_DEFAULT);
        Scharr(_frame->images_[lvl], gradientY, CV_16S, 0, 1, 3, 0, BORDER_DEFAULT);

        convertScaleAbs(gradientX, gradientX, 1.0, 0.0);
        convertScaleAbs(gradientY, gradientY, 1.0, 0.0);
        
        addWeighted(gradientX, 0.5, gradientY, 0.5, 0, _frame->gradient_[lvl]);
    }

    // Ptr<cuda::Filter> soberX_ = cuda::createDerivFilter(0, CV_16S, 1, 0, 3, 0,BORDER_DEFAULT,BORDER_DEFAULT);
    // Ptr<cuda::Filter> soberY_ = cuda::createDerivFilter(0, CV_16S, 0, 1, 3, 0,BORDER_DEFAULT,BORDER_DEFAULT);    
    // Ptr<cuda::Filter> soberX_ = cuda::createSobelFilter(0, CV_16S, 1, 0, 3, 1, BORDER_DEFAULT, BORDER_DEFAULT);
    // Ptr<cuda::Filter> soberY_ = cuda::createSobelFilter(0, CV_16S, 0, 1, 3, 1, BORDER_DEFAULT, BORDER_DEFAULT);

    // for (int lvl=PYRAMID_LEVELS-1; lvl>=0; lvl--) {
    //     // Filters for calculating gradient in images
    //     cuda::GpuMat frameGPU = cuda::GpuMat(_frame->images_[lvl]);
    //     // Apply gradient in x and y
    //     cuda::GpuMat frameXGPU, frameYGPU;
    //     cuda::GpuMat absX, absY, out;
    //     soberX_->apply(frameGPU, frameXGPU);
    //     soberY_->apply(frameGPU, frameYGPU);
    //     cuda::abs(frameXGPU, frameXGPU);
    //     cuda::abs(frameYGPU, frameYGPU);
    //     frameXGPU.convertTo(absX, CV_8UC1);
    //     frameYGPU.convertTo(absY, CV_8UC1);
        
    //     cuda::addWeighted(absX, 0.5, absY, 0.5, 0, out);

    //     absX.download(gradientX);
    //     absY.download(gradientY);
    //     out.download(gradient);

    //     _frame->gradient_[lvl] = gradient.clone();
    //     _frame->gradientX_[lvl] = gradientX.clone();
    //     _frame->gradientY_[lvl] = gradientY.clone();

    // }

    _frame->obtained_gradients_ = true;   
}

void Tracker::ObtainPatchesPoints(Frame* _previous_frame) {
    vector<KeyPoint> goodKeypoints;
    
    goodKeypoints = _previous_frame->keypoints_;
    int num_max_keypoints = goodKeypoints.size();

    // Saves features found
    float factor_depth = 0.0002, factor_lvl;
    float depth_initialization = 1;    

    int lvl = 0;
    factor_lvl = 1 / pow(2, lvl);
    int start_point = patch_size_ - 1 / 2;

    for (int i=0; i< min(num_max_keypoints, 200); i++) {
        if (_previous_frame->depth_available_) {

            float x = goodKeypoints[i].pt.x;
            float y = goodKeypoints[i].pt.y;

            circle(_previous_frame->image_to_send, Point(x,y), 2, Scalar(0,255,0), -1, 6, 0);

            if (_previous_frame->depths_[0].at<short>(y,x) != 0) {

                float z = _previous_frame->depths_[0].at<short>(y,x) * factor_depth * factor_lvl;

                for (int i=x-start_point; i<=x+start_point; i++) {
                    for (int j=y-start_point; j<=y+start_point; j++) {
                        if (i>0 && i<w_[lvl] && j>0 && j<h_[lvl]) {
                            Mat pointMat_patch = Mat::ones(1, 4, CV_32FC1);                
                            pointMat_patch.at<float>(0,0) = i;
                            pointMat_patch.at<float>(0,1) = j;
                            pointMat_patch.at<float>(0,2) = z;

                            _previous_frame->candidatePoints_[lvl].push_back(pointMat_patch);
                        }
                    }
                }
            }
        }
    }
    // Show points saved in candidatePoints
    // Mat showPoints;
    // cvtColor(_previous_frame->images_[lvl], showPoints, CV_GRAY2RGB);

    // for (int i=0; i<_previous_frame->candidatePoints_[lvl].rows; i++) {
    //     Point2f p;
    //     p.x = _previous_frame->candidatePoints_[lvl].at<float>(i,0);
    //     p.y = _previous_frame->candidatePoints_[lvl].at<float>(i,1);

    //     circle(showPoints, p, 2, Scalar(0,255,0), 1, 8, 0);
    // }

    // imshow("debug", showPoints);
    // waitKey(0);
}

void Tracker::ObtainAllPoints(Frame* _frame) {
    // Factor of TUM depth images
    float factor = 0.0002, factor_lvl;
    float depth_initialization = 1;

    for (int lvl=0; lvl< PYRAMID_LEVELS; lvl++) {

        factor_lvl = factor / pow(2, lvl);

        for (int y=0; y<h_[lvl]; y++) {
            for (int x=0; x<w_[lvl]; x++) {

                if (_frame->depth_available_) {
                    if (_frame->depths_[lvl].at<short>(y,x) > 0) {
                        Mat pointMat = Mat::ones(1, 4, CV_32FC1);
                        Mat informationPoint = Mat::ones(1, 4, CV_32FC1);                
                                        
                        pointMat.at<float>(0,0) = x;
                        pointMat.at<float>(0,1) = y;
                        pointMat.at<float>(0,2) = _frame->depths_[lvl].at<short>(y,x) * factor_lvl;

                        informationPoint.at<float>(0,0) = _frame->images_[lvl].at<uchar>(y,x);
                        informationPoint.at<float>(0,1) = _frame->gradientX_[lvl].at<short>(y,x);
                        informationPoint.at<float>(0,2) = _frame->gradientY_[lvl].at<short>(y,x);
                        informationPoint.at<float>(0,3) = 1; // Valid point

                        _frame->candidatePoints_[lvl].push_back(pointMat);                       
                        _frame->informationPoints_[lvl].push_back(informationPoint);

                    } else {
                        // Not valid point
                        Mat pointMat = Mat::zeros(1, 4, CV_32FC1);
                        pointMat.at<float>(0,2) = 1;                     
                        Mat informationPoint = Mat::zeros(1, 4, CV_32FC1);

                        _frame->candidatePoints_[lvl].push_back(pointMat);                       
                        _frame->informationPoints_[lvl].push_back(informationPoint);  
                    }
                // Needs review
                } else {
                    Mat pointMat = Mat::ones(1, 4, CV_32FC1);                
                    pointMat.at<float>(0,0) = x;
                    pointMat.at<float>(0,1) = y;
                    pointMat.at<float>(0,2) = depth_initialization;
                    _frame->candidatePoints_[lvl].push_back(pointMat);
                }

            }
        } 
    }
    _frame->obtained_candidatePoints_ = true;
}

// TODO(GitHub:fmoralesh, fabmoraleshidalgo@gmail.com)
// 02-13-2018 - Implement a faster way to obtain candidate points with high gradient value in patches (above of a certain threshold)
void Tracker::ObtainCandidatePoints(Frame* _frame) {
    // Factor of TUM depth images
    float factor = 0.0002;
    float depth_initialization = 1;

    for (int lvl = 0; lvl < PYRAMID_LEVELS; lvl++){
        Scalar mean, stdev;
        float thres = 0.0;
        cuda::GpuMat filteredGPU;

        cuda::GpuMat frameGPU(_frame->gradient_[lvl]);
        cuda::meanStdDev(frameGPU, mean, stdev);
        
        thres = mean[0] + GRADIENT_THRESHOLD;

        cuda::threshold(frameGPU, filteredGPU, thres, 255, THRESH_BINARY);

        Mat filtered = Mat(_frame->gradient_[lvl].size(), CV_8UC1);
        filteredGPU.download(filtered);

        for (int x=0; x<w_[lvl]; x++) {
            for (int y =0; y<h_[lvl]; y++) {
                Mat point = Mat::ones(1,4,CV_32FC1);
                Mat depth = Mat::ones(1,1,CV_32FC1);                
                if (_frame->depth_available_) {
                    if (_frame->depths_[lvl].at<uchar>(y,x) != 0 && filtered.at<uchar>(y,x) != 0) {
                        
                        Mat pointMat = Mat::ones(1, 4, CV_32FC1);                
                        pointMat.at<float>(0,0) = x;
                        pointMat.at<float>(0,1) = y;
                        pointMat.at<float>(0,2) = _frame->depths_[lvl].at<uchar>(y,x) * factor;
                        _frame->candidatePoints_[lvl].push_back(pointMat);

                    }
                } else {
                    if (filtered.at<uchar>(y,x) != 0) {

                        Mat pointMat = Mat::ones(1, 4, CV_32FC1);                
                        pointMat.at<float>(0,0) = x;
                        pointMat.at<float>(0,1) = y;
                        pointMat.at<float>(0,2) = depth_initialization;
                        _frame->candidatePoints_[lvl].push_back(pointMat);
                    }
                }
            }
        }


    }

    // Making block-size. Pretty slow 
    // for (int lvl = 0; lvl < PYRAMID_LEVELS; lvl++){
    //     int block_size = BLOCK_SIZE - lvl * 5;
    //     cuda::GpuMat frameGPU(_frame->gradient_[lvl]);
    //     for (int x=0; x<w_[lvl]-block_size; x+=block_size) {
    //         for (int y =0; y<h_[lvl]-block_size; y+=block_size) {
    //             Mat point = Mat::ones(1,4,CV_32FC1);
    //             Mat depth = Mat::ones(1,1,CV_32FC1);                
    //             Scalar mean, stdev;
    //             Point min_loc, max_loc;
    //             double min, max;
    //             cuda::GpuMat block(frameGPU, Rect(x,y,block_size,block_size));
    //             block.convertTo(block, CV_8UC1);
    //             cuda::meanStdDev(block, mean, stdev);
    //             cuda::minMaxLoc(block, &min, &max, &min_loc, &max_loc);
                
    //             if (max > mean[0] + GRADIENT_THRESHOLD) {
                    
    //                 point.at<float>(0,0) = (float) (x + max_loc.x);
    //                 point.at<float>(0,1) = (float) (y + max_loc.y);

    //                 _frame->candidatePoints_[lvl].push_back(point);
    //                 _frame->candidatePointsDepth_[lvl].push_back(depth);
    //             }
    //         }
    //     }
    // }

    
    for (int lvl = 0; lvl < PYRAMID_LEVELS; lvl++) {
        // frame->candidatePoints_[lvl] = frame->candidatePoints_[lvl-1] * 0.5;
        // DebugShowCandidatePoints(_frame->gradient_[lvl], _frame->candidatePoints_[lvl]);
    }
    _frame->obtained_candidatePoints_ = true;
}

Mat Tracker::WarpFunctionOpenCV(Mat _points2warp, SE3 _rigid_transformation, int _lvl) {

    Mat original_points = Mat::ones(3, 1, CV_32FC1);
    original_points.at<float>(0,0) = _points2warp.at<float>(0,0);
    original_points.at<float>(1,0) = _points2warp.at<float>(0,1);
    original_points.at<float>(2,0) = _points2warp.at<float>(0,2);

    Mat44f rigidEigen = _rigid_transformation.matrix();
    Mat rigid = Mat(4,4,CV_32FC1);
    eigen2cv(rigidEigen, rigid);

    Mat K = K_[_lvl];

    Mat world_coordinates = K.inv() * original_points;

}

Mat Tracker::WarpFunction(Mat _points2warp, SE3 _rigid_transformation, int _lvl) {
    int lvl = _lvl;

    Mat projected_points = Mat(_points2warp.size(), CV_32FC1);
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

    // 2D -> 3D

    // X  = (x - cx) * Z / fx 
    projected_points.col(0) = ((projected_points.col(0) - cx) * invfx);
    projected_points.col(0) = projected_points.col(0).mul(projected_points.col(2));

    // Y  = (y - cy) * Z / fy    
    projected_points.col(1) = ((projected_points.col(1) - cy) * invfy);
    projected_points.col(1) = projected_points.col(1).mul(projected_points.col(2));
    //cout << projected_points.row(projected_points.rows-1) << endl;    

    // Z = Z

    // Transformation of a point rigid body motion
    projected_points = rigid * projected_points.t();

    // 3D -> 2D
    // x = (X * fx / Z) + cx
    projected_points.row(0) *= fx;    
    projected_points.row(0) /= projected_points.row(2);
    projected_points.row(0) += cx;
    
    // x = (Y * fy / Z) + cy    
    projected_points.row(1) *= fy;    
    projected_points.row(1) /= projected_points.row(2);
    projected_points.row(1) += cy;

    //cout << projected_points.col(0) << endl;
    
    // Cleaning invalid points
    projected_points.row(0) = projected_points.row(0).mul(projected_points.row(3));
    projected_points.row(1) = projected_points.row(1).mul(projected_points.row(3));

    // Transposing the points due transformation multiplication
    return projected_points.t();
}

Mat Tracker::ObtainImageTransformed(Mat _originalImage, Mat _candidatePoints, Mat _warpedPoints, Mat _outputImage) {

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
    // for (int x=0; x<_outputImage.cols; x++) {
    //     for (int y=0; y<_outputImage.rows; y++) {
            
    //         if (_outputImage.at<uchar>(y,x) == 0) {                
    //             int x1 = x - 1;
    //             int x2 = x + 1;
    //             int y1 = y - 1;
    //             int y2 = y + 1;
    //             if (x1 < 0) x1 = 0;
    //             if (y1 < 0) y1 = 0;
    //             if (x2 == _outputImage.cols) x2 = x2-1;   
    //             if (y2 == _outputImage.rows) y2 = y2-1;
    //             if (validPixel.at<uchar>(y1,x1) == 1 || validPixel.at<uchar>(y1,x) == 1 || validPixel.at<uchar>(y1,x2) == 1 ||
    //                 validPixel.at<uchar>(y,x1)  == 1 || validPixel.at<uchar>(y,x)  == 1 || validPixel.at<uchar>(y,x2)  == 1 ||
    //                 validPixel.at<uchar>(y2,x1) == 1 || validPixel.at<uchar>(y2,x) == 1 || validPixel.at<uchar>(y2,x2) == 1    ) {

    //                 int Q11 = _outputImage.at<uchar>(y2,x1);
    //                 int Q21 = _outputImage.at<uchar>(y2,x2);
    //                 int Q12 = _outputImage.at<uchar>(y1,x1);
    //                 int Q22 = _outputImage.at<uchar>(y1,x2);

    //                 if (Q12 == 0) Q12 = Q22;
    //                 if (Q22 == 0) Q22 = Q12;
    //                 if (Q11 == 0) Q11 = Q21;
    //                 if (Q21 == 0) Q21 = Q11;
                    
    //                 int f_y1 = (Q12 * 0.5) + (Q22 * 0.5);
    //                 int f_y2 = (Q11 * 0.5) + (Q21 * 0.5);
                    
    //                 _outputImage.at<uchar>(y,x) = (f_y1 * 0.5) + (f_y2 * 0.5);
    //             }
    //         }
    //     }
    // }

    return validPixel;
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

void Tracker::DebugShowJacobians(Mat Jacobians, Mat points, int width, int height) {
    vector<Mat> image_jacobians = vector<Mat>(6);
    for (int i=0; i<6; i++) 
        image_jacobians[i] = Mat::zeros(height, width, CV_8UC1);

    for (int index=0; index<Jacobians.rows; index++) {
        for (int i=0; i<6; i++) {
            float x = round(points.row(index).at<float>(0,0));
            float y = round(points.row(index).at<float>(0,1));
            if (x>0 && x<image_jacobians[i].cols && y>0 && y<image_jacobians[i].rows) {
                if (Jacobians.row(index).at<float>(0,i) < -10){
                    image_jacobians[i].at<uchar>(y,x) = 90;
                }
                if (Jacobians.row(index).at<float>(0,i) > 10) {             
                    image_jacobians[i].at<uchar>(y,x) = 200;
                }
            }
        }
    }

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

float Tracker::MedianAbsoluteDeviation(Mat _input) {
    float c = 1.4826;

    Mat deviation = Mat(_input.rows, _input.cols, CV_32FC1);
    float median = MedianMat(_input);
    // Absolute Deviation from the _input's median
    deviation = abs(_input - median);

    // Median of deviation
    float MAD = MedianMat(deviation);

    return c * MAD;
}

Mat Tracker::IdentityWeights(int _num_residuals) {
    Mat W = Mat::ones(_num_residuals,1,CV_32FC1);    
    return W;
}

Mat Tracker::TukeyFunctionWeights(Mat _input) {
    int num_residuals = _input.rows;
    float b = 4.6851; // Achieve 95% efficiency if assumed Gaussian distribution for outliers
    Mat W = Mat(num_residuals,1,CV_32FC1);    

    // Computation of scale (Median Absolute Deviation)
    float MAD = MedianAbsoluteDeviation(_input);       

    if (MAD == 0) {
        cout << "Warning: MAD = 0." << endl;
        MAD = 1;
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

void Tracker::DebugShowWarpedPerspective(Mat _image1, Mat _image2, Mat _imageWarped, int _lvl) {
    int lvl = _lvl + 1;
    Mat noalign = Mat::zeros(_image2.size(), CV_8UC1);
    Mat showPoints1, showPoints2;
    Mat substraction;

    addWeighted(_image1, 0.5, _image2, 0.5, 1.0, noalign);    
    addWeighted(_imageWarped, 0.5, _image2, 0.5, 1.0, substraction);

    // Point2d p1, p2, p3 ,p4;
    // p1.x = _warped.at<float>(0,0);
    // p1.y = _warped.at<float>(0,1);
    
    // p2.x = _warped.at<float>(_image2.rows-1,0);
    // p2.y = _warped.at<float>(_image2.rows-1,1);

    // p3.x = _warped.at<float>(_image2.rows*(_image2.cols-1),0);
    // p3.y = _warped.at<float>(_image2.rows*(_image2.cols-1),1);

    // p4.x = _warped.at<float>(_warped.rows-1,0);
    // p4.y = _warped.at<float>(_warped.rows-1,1);

    // line(showPoints1,p1,p2,Scalar(255,0,0), 1, 8, 0);
    // line(showPoints1,p2,p4,Scalar(255,0,0), 1, 8, 0);
    // line(showPoints1,p4,p3,Scalar(255,0,0), 1, 8, 0);
    // line(showPoints1,p3,p1,Scalar(255,0,0), 1, 8, 0);

    // line(showPoints1,p1,p4,Scalar(255,0,0), 1, 8, 0);
    // line(showPoints1,p2,p3,Scalar(255,0,0), 1, 8, 0);

    Mat imShow1, imShow2, imShow;
    hconcat(_image1, _image2, imShow1);
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
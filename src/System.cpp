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
#include "Options.h"
#include "System.h"
#include "CameraModel.h"
#include "Tracker.h"

namespace uw
{

System::System(int argc, char *argv[], int _start_index) {
    ros::init(argc, argv, "uw_slam");  // Initialize ROS
    start_index_ = _start_index;
    initialized_ = false;
    distortion_valid_ = false;
    depth_available_ = false;
    num_frames_     = 0;
    num_keyframes_  = 0;
    num_valid_images_ = 0;
    mean_n_keypoints_ = 0;

    previous_world_pose_ = SE3(SO3::exp(SE3::Point(0.0, 0.0, 0.0)), SE3::Point(0.0, 0.0, 0.0));
}

System::~System() {
    cout << "SLAM System shutdown ..." << endl;
    frames_.clear();
    keyframes_.clear();
    camera_model_->~CameraModel();
    tracker_->~Tracker();
    visualizer_->~Visualizer();

    delete camera_model_;
    delete tracker_;
    delete visualizer_;
}

Frame::Frame(void) {
    
    rigid_transformation_ = SE3();
    idFrame_    = 0;
    n_matches_  = 0;
    depth_available_ = false;
    
    isKeyFrame_ = false;
}

Frame::~Frame(void) {

    images_.clear();
    gradientX_.clear();
    gradientY_.clear();
    gradient_.clear();
    candidatePoints_.clear();

}


void System::Calibration(string _calibration_path) {
    cout << "Reading calibration xml file";
    camera_model_ = new CameraModel();
    camera_model_->GetCameraModel(_calibration_path);
    w_ = camera_model_->GetOutputWidth();
    h_ = camera_model_->GetOutputHeight();

    if (w_%2!=0 || h_%2!=0) {
		cout << "Output image dimensions must be multiples of 32. Choose another output dimentions" << endl;
        cout << "Exiting..." << endl;
		exit(0);
	}
}

void System::InitializeSystem(string _images_path, string _ground_truth_dataset, string _ground_truth_path, string _depth_path) {
    // Check if depth images are available
    if (_depth_path != "")
        depth_available_ = true;

    // Add list of the dataset images names
    AddLists(_images_path, _depth_path);\
    
    if (start_index_>images_list_.size()) {
        cout << "The image " << start_index_ << " doesn't exist." << endl;
        cout << "Exiting..." << endl;
        exit(0);
    }
    // Obtain parameters of camera_model
    K_ = camera_model_->GetK();
    w_input_ = camera_model_->GetInputHeight();
    h_input_ = camera_model_->GetInputWidth();
    map1_ = camera_model_->GetMap1();
    map2_ = camera_model_->GetMap2();
    fx_ = camera_model_->GetK().at<float>(0,0);
    fy_ = camera_model_->GetK().at<float>(1,1);
    cx_ = camera_model_->GetK().at<float>(0,2);
    cy_ = camera_model_->GetK().at<float>(1,2);
    distortion_valid_ = camera_model_->IsValid();

    // Obtain ROI for distorted images
    if (distortion_valid_)
        CalculateROI();

    // Initialize tracker system
    tracker_ = new Tracker(depth_available_, w_, h_, K_);
    tracker_->InitializeMasks();

    // Initialize map
    mapper_ = new Mapper(w_, h_, K_);

    // Initialize output visualizer
    ground_truth_path_    = _ground_truth_path;
    ground_truth_dataset_ = _ground_truth_dataset;
    visualizer_ = new Visualizer(start_index_, images_list_.size(), K_, _ground_truth_dataset, _ground_truth_path);

    // Cheking if the number of depth images are greater or lower than the actual number of images
    if (depth_available_) {
        if (images_list_.size() > depth_list_.size())
            num_valid_images_ = depth_list_.size();
        if (images_list_.size() <= depth_list_.size())
            num_valid_images_ = images_list_.size();
    } else {
        num_valid_images_ = images_list_.size();
    }

    initialized_ = true;
    cout << "Initializing system ... done" << endl << endl;
}

void System::printHistogram(int histogram[256], std::string filename, cv::Scalar color) {
	// Finding the maximum value of the histogram. It will be used to scale the
	// histogram to fit the image.
	int max = 0, i;
	for (i = 0; i < 256; i++)
	{
		if (histogram[i] > max)
			max = histogram[i];
	}
	// Creating an image from the histogram.
	cv::Mat imgHist(1480, 1580, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Point pt1, pt2;
	pt1.y = 1380;
	for (i = 0; i < 256; i++)
	{
		pt1.x = 150 + 5 * i + 1;
		pt2.x = 150 + 5 * i + 3;
		pt2.y = 1380 - 1280 * histogram[i] / max;
		cv::rectangle(imgHist, pt1, pt2, color, CV_FILLED);
	}
	// y-axis labels
	cv::rectangle(imgHist, cv::Point(130, 1400), cv::Point(1450, 80), cvScalar(0, 0, 0), 1);
	cv::putText(imgHist, std::to_string(max), cv::Point(10, 100), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(max * 3 / 4), cv::Point(10, 420), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(max / 2), cv::Point(10, 740), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(max / 4), cv::Point(10, 1060), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(0), cv::Point(10, 1380), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	// x-axis labels
	cv::putText(imgHist, std::to_string(0), cv::Point(152 - 7 * 1, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(63), cv::Point(467 - 7 * 2, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(127), cv::Point(787 - 7 * 3, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(191), cv::Point(1107 - 7 * 3, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(255), cv::Point(1427 - 7 * 3, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);

	// Saving the image
    // cv::imwrite(filename, imgHist);
    // Showing the image (too big for screen)
    // imshow("Histogram", imgHist);
    // waitKey(0);
	
}

void System::getHistogram(cv::Mat img, int *histogram) {
	int i = 0, j = 0;
	// Initializing the histogram. TODO: Check if there is a faster way
	for (i = 0; i < 256; i++) {
		histogram[i] = 0;
	}
	// by using aux variables, we decrease overhead create by multiple calls to cvMat image methods to retrieve its size
	// TODO: is it possible to measure the impact?
	int width, height;
	width = img.size().width;
	height = img.size().height;

	// Computing the histogram as a cumulative of each integer value. WARNING: this will fail for any non-integer image matrix
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
				histogram[img.at<unsigned char>(i, j)]++;
}

void System::imgChannelStretch(cv::Mat imgOriginal, cv::Mat imgStretched, int lowerPercentile, int higherPercentile) {
	// Computing the histograms
	int histogram[256];

	getHistogram(imgOriginal, histogram);

    // Show histogram (debug purposes)
    //printHistogram(histogram, "Original.jpg", Scalar(255,0,0));
	// Computing the percentiles. We force invalid values as initial values (just in case)
	int channelLowerPercentile = -1, channelHigherPercentile = -1;
	int height = imgOriginal.size().height;
	int width = imgOriginal.size().width;
	// Channel percentiles
	int i = 0;
	float sum = 0;

	float normImgSize = height * width / 100.0;
	// while we don't reach the highPercentile threshold...
	// This is some fashion of CFD: cumulative function distribution
	while (sum < higherPercentile * normImgSize)
	{
		if (sum < lowerPercentile * normImgSize)
			channelLowerPercentile++;
		channelHigherPercentile++;
		sum += histogram[i++];
	}
	int j;
	float m;
	cv::Scalar b;
	m = 255 / (channelHigherPercentile - channelLowerPercentile);
	b = channelLowerPercentile;
	imgStretched -= b;
	imgStretched *= m;


    //int histogramFinal[256];
	//getHistogram(imgStretched, histogramFinal);
    //printHistogram(histogramFinal, "Final.jpg", Scalar(255,0,0));
    
}

void System::CalculateROI() {
    // Load first image
    Mat distorted, undistorted;
    distorted = imread(images_list_[0], CV_LOAD_IMAGE_GRAYSCALE);
    remap(distorted, undistorted, map1_, map2_, INTER_LINEAR);

    // Find middle x and y of image (supposing a symmetrical distortion)
    int x_middle = (undistorted.cols - 1) * 0.5;
    int y_middle = (undistorted.rows - 1) * 0.5;
    
    Point p1, p2;    
    p1.x = 0;
    p1.y = 0;
    p2.x = undistorted.cols - 1;
    p2.y = undistorted.rows - 1;

    // Search x1_ROI distance to crop
    while (undistorted.at<uchar>(y_middle, p1.x) == 0)
        p1.x++;

    // Search x2_ROI distance to crop
    while (undistorted.at<uchar>(y_middle, p2.x) == 0)
        p2.x--;

    // Search y1_ROI distance to crop
    while (undistorted.at<uchar>(p1.y, x_middle) == 0)
        p1.y++;

    // Search y2_ROI distance to crop
    while (undistorted.at<uchar>(p2.y, x_middle) == 0)
        p2.y--;

    // Considering an error margin
    p1.x += 5;
    p2.x -= 5;
    p1.y += 5;
    p2.y -= 5;

    ROI = Rect(p1,p2);
    
    // Update w_ and h_ with ROI dimentions
    w_ = p2.x - p1.x;
    h_ = p2.y - p1.y;
}

void System::UpdateWorldPose(SE3& _previous_world_pose, SE3& _current_pose) {
    // Scaled movement of camera
    Mat31f t_;
    // t_(0) = 1 * - _current_pose.translation().x();
    // t_(1) = 1 * - _current_pose.translation().z();
    // t_(2) = 1 *   _current_pose.translation().y();

    // _previous_world_pose = _previous_world_pose * SE3(_current_pose.unit_quaternion(), (t_));
    _previous_world_pose = _previous_world_pose * _current_pose;    
}

void System::Tracking() {

    mean_n_keypoints_ += previous_frame_->n_matches_;

    bool usekeypoints = true;

    tracker_->ApplyGradient(previous_frame_);
    
    if (previous_frame_->n_matches_ <= 200)
        usekeypoints = false;
    

    tracker_->robust_matcher_->DetectAndTrackFeatures(previous_frame_, current_frame_, usekeypoints);  

    if (previous_frame_->n_matches_ > 20){      
        tracker_->ObtainPatchesPoints(previous_frame_);

        //tracker_->ObtainAllPoints(previous_frame_);

        //tracker_->ObtainCandidatePoints(current_frame_);

        //tracker_->EstimatePose(previous_frame_, current_frame_);
        tracker_->FastEstimatePose(previous_frame_, current_frame_);
    }

    // cout << previous_world_pose_.matrix3x4() << endl;
    // Change coordinates to correspond RViz coordinates
    // UpdateWorldPose(previous_world_pose_, previous_frame_->rigid_transformation_);
    
    // cout << previous_world_pose_.matrix3x4() << endl;
    
}

void System::Mapping() {
    // Update world pose of the camera
    mapper_->previous_world_pose_ = previous_world_pose_;
    
    if (previous_frame_->n_matches_ > 10){  
        mapper_->TriangulateCloudPoints(previous_frame_, current_frame_);
    }
}

void System::Visualize() {
    // Update world pose of the camera
    visualizer_->previous_world_pose_ = previous_world_pose_;

    visualizer_->UpdateMessages(previous_frame_);

    UpdateWorldPose(previous_world_pose_, previous_frame_->rigid_transformation_);
    
}

void System::AddFrame(int _id) {
    Frame* newFrame   = new Frame();
    newFrame->idFrame_ = _id;

    newFrame->images_[0] = imread(images_list_[_id], CV_LOAD_IMAGE_GRAYSCALE);
    // Resizing for ScottReef Dataset
    // Mat image_aux;
    // image_aux = imread(images_list_[_id], CV_LOAD_IMAGE_GRAYSCALE);
    // resize(image_aux, newFrame->images_[0], Size(), 0.5, 0.5);
    
    cvtColor(newFrame->images_[0], newFrame->image_to_send, COLOR_GRAY2BGR);
    
    if (distortion_valid_) {
        Mat distortion;
        remap(newFrame->images_[0], distortion, map1_, map2_, INTER_LINEAR);
        newFrame->images_[0] = distortion(ROI);
        // imshow("Undistorted", distortion);
        // imshow("Croped", newFrame->images_[0]);
        // waitKey(0);
    }

    if (depth_available_) {
        newFrame->depth_available_ = true;
        newFrame->depths_[0] = imread(depth_list_[_id], -1);
    }

    // Stretch image histogram with 1% and 99% percentile
    // imshow("Sin stretch", newFrame->images_[0]);
    imgChannelStretch(newFrame->images_[0], newFrame->images_[0], 1, 99);
    // imshow("Con stretch", newFrame->images_[0]);
    // waitKey(0);

    for (int i=1; i<PYRAMID_LEVELS; i++) {
        resize(newFrame->images_[i-1], newFrame->images_[i], Size(), 0.5, 0.5);
        if (depth_available_) {
            resize(newFrame->depths_[i-1], newFrame->depths_[i], Size(), 0.5, 0.5);    
        }
    }

    if (num_frames_ == 0) {
        previous_frame_ = newFrame;        
        current_frame_ = newFrame;
    } else {
        previous_frame_ = current_frame_;
        current_frame_ = newFrame;
    }
    frames_.push_back(newFrame);
    num_frames_++;
}

void System::AddKeyFrame(int _id) {
    if(_id > current_frame_->idFrame_){
        cout << "Can not add keyframe because frame " << _id << " is not part of the systems frames" << endl;
        cout << "Exiting ..." << endl;
        exit(0);
    }

    Frame* newKeyFrame   = new Frame();
    newKeyFrame = frames_[num_frames_ - 1];
    current_keyframe_ = newKeyFrame;

    frames_[num_frames_-1]->isKeyFrame_ = true;
    num_keyframes_++;
    keyframes_.push_back(newKeyFrame);
}

void System::ShowFrame(int _id) {
    imshow("Show last frame", frames_[_id]->images_[0]);
    waitKey(0);
}

void System::AddFramesGroup(int _id, int _num_images) {
    for (int i = _id; i < _num_images; i++)
        System::AddFrame(i);
}

void System::AddLists(string _path, string _depth_path) {
    vector<string> file_names;  
    DIR *dir;
    struct dirent *ent;
    
    cout << "Searching images files in directory ... ";
    if ((dir = opendir(_path.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            file_names.push_back(_path + string(ent->d_name));
        }
        closedir (dir);
    } else {
        // If the directory could not be opened
        cout << "Could not open directory of images: " << _path << endl;
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

    if (depth_available_) {
        vector<string> depth_names;  
        DIR *dir_depth;
        struct dirent *ent_depth;

        cout << "Searching for depth images (TUM dataset) ... ";
        if ((dir_depth = opendir(_depth_path.c_str())) != NULL) {
            while ((ent_depth = readdir (dir_depth)) != NULL) {
                depth_names.push_back(_depth_path + string(ent_depth->d_name));
            }
            closedir (dir_depth);
        } else {
            // If the directory could not be opened
            cout << "Could not open directory of depth images: " << _depth_path << endl;
            cout << "Exiting..." << endl;
            exit(0);
        }
        // Sorting the vector of strings so it is alphabetically ordered
        sort(depth_names.begin(), depth_names.end());
        depth_names.erase(depth_names.begin(), depth_names.begin()+2);

        if (depth_names.size() < 15) {
            cout << "\nInsufficient number of depth images found. Consider not to use -p flag." << endl;
            cout << "Exiting..." << endl;
            exit(0);
        }
        cout << depth_names.size() << " found"  << endl;

        depth_list_ = depth_names;
    }
}

void System::FreeFrames() {
    frames_[0]->~Frame();
    frames_.erase(frames_.begin());
}

}

/********************************************
 * FILE NAME:   feature_detector.cpp        *
 * DESCRIPTION: Homogenuos detection of     *
 *              features                    *
 *                                          *
 * AUTHOR:      Fabio Morales               *
 ********************************************/

#include "feature_detector.h"

#define DIMENSION_PATCH         10
#define DIMENSION_WIDTH         640
#define DIMENSION_HEIGHT        480
#define NEAREST_NEIGHBOR_RATIO  0.6

int main( int argc, char** argv ){
    // CAMBIAR DE POSICION. ID DE CADA UNO DE LOS MARKERS DE ROS
    int id = 0;

    // Detecting CUDA Device
    int nCuda = cuda::getCudaEnabledDeviceCount();
    cuda::DeviceInfo deviceInfo;
    if (nCuda > 0){
        std::cout << "CUDA enabled devices detected: " << deviceInfo.name() << endl;
        cuda::setDevice(0);
    }
    else {
        std::cout << "No CUDA device detected" << endl;
        return -1;
    }
    std::cout << "***************************************" << endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // Parser Section
    std::vector<std::string> image_names;
    std::string videoPath;
    std::string dir_calibration;
    std::string detectorName;
    int detector = -1;

    try{
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help){
        std::cout << parser;
        return 0;
    }
    catch (args::ParseError e){
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    catch (args::ValidationError e){
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    if (!inputImages && !inputVideo && !dir_dataset) {
        std::cout<< "Insuficient input data"<< endl;
        std::cerr << "Use -h, --help command to see usage" << std::endl;
        return -1;
    }
    if (inputImages && !inputVideo && !dir_dataset) {
        for (const auto ch: args::get(inputImages)){
            image_names.push_back(ch);
            std::cout << "Input image: " << ch << endl;
        }
    }
    else if (inputVideo && !inputImages && !dir_dataset){
        videoPath = args::get(inputVideo);
        std::cout << "Input video: " << videoPath << endl;
    }
    else if (dir_dataset && !inputImages && !inputVideo){
        image_names = read_filenames(args::get(dir_dataset));
        std::cout << "Directory of images: " << args::get(dir_dataset) << endl;
    }else{
        std::cout<< "Only one method of input argument is permited"<< endl;
        std::cerr << "Use -h, --help command to see usage" << std::endl;
        return -1;
    }
    if(feature_detector){
        detector = args::get(feature_detector);
    }
    if(parse_calibration){
        dir_calibration = args::get(parse_calibration);
        std::cout << "Directory of calibration xml file: " << args::get(parse_calibration) << endl;
    }else{
        dir_calibration = "./include/calibration.xml";
    }
    std::cout << endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // Initialization section

    // Creating indexes for current and next image
    int currImg = 0, nextImg = 1;

    // Loading first image
    GpuMat currImgGPU;
    Mat currImgRes;
    currImgRes = imread(image_names[currImg]);

    // Checking for error
    if (!currImgRes.data){
        std::cout << "Error loading first image." << std::endl; return -1;
    }

    // Resizing current image to 640 x 480
    resize(currImgRes, currImgRes, Size(DIMENSION_WIDTH, DIMENSION_HEIGHT), 0 ,0, CV_INTER_LINEAR);

    // Converting next image to GRAY
    cvtColor(currImgRes,currImgRes,COLOR_BGR2GRAY);

    // Uploading next image to GpuMat
    currImgGPU.upload(currImgRes);  

    // Creating current and next Rotation and Translation Matrices
    Mat currRotationMat     = Mat::eye(3, 3, CV_64F);
    Mat currTranslationMat  = Mat(3, 1, CV_64F, double(0));
    Mat nextRotationMat, nextTranslationMat;
    double x_pos = 0, y_pos = 0, z_pos = 0;

    // Obtaining camera matrix from calibration XML file - (Parameters: focal and principal point)
    // (Default: /include/calibration.xml. Must be called cameraMatrix within the file
    Mat cameraMat = readCameraMat(dir_calibration);
    if(cameraMat.empty()){
        std::cout << "Error reading calibration xml file." << endl;
        return -1;
    }
    double focal = cameraMat.at<double>(0,0);
    Point2d pp = Point2d(cameraMat.at<double>(0,2), cameraMat.at<double>(1,2));

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // ROS Kinetic inicialization
    ros::init(argc, argv, "uw_slam");
    ros::NodeHandle n;
    ros::Rate r(3);
    ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 100);

    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher img_pub = it.advertise("camera/image",1);
    
    // Initialize parameters of marker
    visualization_msgs::Marker marker;
    marker.header.frame_id = "/main_uw";                // Set the frame ID and timestamp. See the TF tutorials for information on these.
    marker.header.stamp = ros::Time::now();
    marker.ns = "uw_slam";                              // Set the namespace and id for this marker.  This serves to create a unique ID. Any marker sent with the same namespace and id will overwrite the old one
    marker.type = visualization_msgs::Marker::CUBE;     // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    marker.action = visualization_msgs::Marker::ADD;    // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    marker.pose.orientation.x = 0.0;                    // Orientation of marker
    marker.pose.orientation.y = 0.0;    
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.10;                              // Set the scale of the marker -- 1x1x1 here means 1m on a side
    marker.scale.y = 0.10;
    marker.scale.z = 0.10;
    marker.color.r = 0.12f;                             // Set the color -- set alpha to something non-zero!
    marker.color.g = 0.56f;
    marker.color.b = 1.0f;
    marker.color.a = 1.0;
    marker.lifetime = ros::Duration();                  // Life spam of marker  

    // Loop while ROS core is active
    while(ros::ok()){
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        // Read next image
        Mat nextImgRes;
        nextImgRes = imread(image_names[nextImg]);
        
        // Checking end of dataset
        if (!nextImgRes.data){
            std::cout << "***************************************" << endl;
            std::cout << "Finished." << std::endl; return -1;
        }
        // Update next image to ROS message
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", nextImgRes).toImageMsg();

        // Resizing next image to 640 x 480
        resize(nextImgRes, nextImgRes, Size(DIMENSION_WIDTH , DIMENSION_HEIGHT), 0 ,0, CV_INTER_LINEAR);

        // Converting next image to GRAY
        cvtColor(nextImgRes,nextImgRes,COLOR_BGR2GRAY);

        // Uploading next image to GpuMat
        GpuMat nextImgGPU;
        nextImgGPU.upload(nextImgRes);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        // Feature detection section
        GpuMat keypointsGPU[2];
        GpuMat descriptorsGPU[2];
        GpuMat matchesGPU;
        vector< vector< DMatch> > matches;
        array<vector<KeyPoint>,2> keypoints;
        array<vector<float>,2> descriptors;
        int nfeatures[2], nmatches;

        // SURF as feature detector (Default detector)
        if(detector == 0 || detector == -1){
            detectorName = "SURF Detector";
            
            Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher();
            cv::cuda::SURF_CUDA surf;
            Ptr<cv::cuda::Feature2DAsync> detector;
            // Detecting keypoints and computing descriptors
            surf(currImgGPU, GpuMat(), keypointsGPU[0], descriptorsGPU[0]);
            surf(nextImgGPU, GpuMat(), keypointsGPU[1], descriptorsGPU[1]);
            // Matching descriptors
            matcher->knnMatch(descriptorsGPU[0], descriptorsGPU[1], matches, 2);
            // Downloading results
            surf.downloadKeypoints(keypointsGPU[0], keypoints[0]);
            surf.downloadKeypoints(keypointsGPU[1], keypoints[1]);
            surf.downloadDescriptors(descriptorsGPU[0], descriptors[0]); //REVISAR SU FUTURA UTILIDAD
            surf.downloadDescriptors(descriptorsGPU[1], descriptors[1]); //REVISAR SU FUTURA UTILIDAD
        }
        // ORB as feature detector
        if(detector == 1){
            detectorName = "ORB Detector";

            // Detecting kypoints and computing descriptors
            Ptr<cuda::ORB> orb = cv::cuda::ORB::create();
            Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
            orb->detectAndCompute(currImgGPU, noArray(), keypoints[0], descriptorsGPU[0]);
            orb->detectAndCompute(nextImgGPU, noArray(), keypoints[1], descriptorsGPU[1]);
            // Matching descriptors
            matcher->knnMatch(descriptorsGPU[0], descriptorsGPU[1], matches, 2);
        }
        
        // Obtain good matches (delete outliers)
        vector<DMatch> goodMatches;
        goodMatches = getGoodMatches(matches);

        // Apply grid filtering of matches found (for homogenous sparse of features)
        //goodMatches = gridFiltering(goodMatches, keypoints[0]);

        // Obtain good keypoints from goodMatches
        array<vector<KeyPoint>,2> goodKeypoints;
        goodKeypoints = getGoodKeypoints(goodMatches, keypoints);

        // Show results of feature detection (optional)
        if(feature_stats){
            Mat img_matches;
            drawMatches(Mat(currImgGPU), keypoints[0], Mat(nextImgGPU), keypoints[1], 
                            goodMatches, img_matches,Scalar::all(-1), Scalar::all(-1), 
                            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            nfeatures[0] = keypoints[0].size();
            nfeatures[1] = keypoints[1].size();
            nmatches = goodMatches.size();
            showFeatureStats(detectorName,nfeatures,nmatches);
            imshow(detectorName, img_matches);
            waitKey(0);     
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pose Estimation Section
        // Transform keypoints to Point2f vectors of coordinates 
        array<vector<Point2f>,2> points;
        KeyPoint::convert(goodKeypoints[0], points[0], vector<int>());
        KeyPoint::convert(goodKeypoints[1], points[1], vector<int>());

        // Obtain Essential Matrix with the Five-Point Algorithm (David Nister, 2004)
        Mat EssentialMat, mask;
        EssentialMat = findEssentialMat(points[0], points[1], focal, pp, cv::RANSAC, 0.999, 3.0, mask);
        // TODO print inliers and outliers 

        // Obtain Pose from Essential Matrix (Rotation Matrix and Translation Vector)
        int inliers2 = recoverPose(EssentialMat, points[0], points[1], nextRotationMat, 
                                    nextTranslationMat, focal, pp, mask);

        // Compute current Rotation and Translation
        nextTranslationMat.at<double>(0,2) = abs(nextTranslationMat.at<double>(0,2));
        currTranslationMat = currTranslationMat + (currRotationMat * nextTranslationMat);

        // Obtain projection matrices for the two perspectives
        Mat P1 = getProjectionMat(cameraMat, currRotationMat, currTranslationMat);
        Mat P2 = getProjectionMat(cameraMat, nextRotationMat, nextTranslationMat);

        // Compute depth of 3D points using triangulation
        Mat mapPoints;
        triangulatePoints(P1, P2, points[0], points[1], mapPoints);

        marker.id = id;

        // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
        cout << currTranslationMat << endl;
        marker.pose.position.x += currTranslationMat.at<double>(2,0);
        marker.pose.position.y += currTranslationMat.at<double>(0,0);
        marker.pose.position.z += currTranslationMat.at<double>(1,0);

        marker.pose.position.x /= 10;
        marker.pose.position.y /= 10;
        marker.pose.position.z /= 10;


        while (marker_pub.getNumSubscribers() < 1 && img_pub.getNumSubscribers() < 1){
            if (!ros::ok()){
                return 0;
            }
            ROS_WARN_ONCE("Please create a subscriber to the marker/image");
            sleep(1);
        }

        // Publish the marker
        marker_pub.publish(marker);
        img_pub.publish(msg);
        ros::spinOnce();
        id += 1;
        r.sleep();

        // Updating values for next frame
        currImg += 1;
        nextImg += 1;
        currImgRes = nextImgRes.clone();
        currImgGPU = nextImgGPU.clone();
        currRotationMat    = nextRotationMat.clone();
        currTranslationMat = nextTranslationMat.clone();

        // // Release memory
        // img[currImg].release();
        // keypointsGPU[0].release(), keypointsGPU[1].release();
        // currImgRes.release(), nextImgRes.release();
        // matchesGPU.release();
        // keypoints[0].~vector(), keypoints[1].~vector();
        // descriptors[0].~vector(), descriptors[1].~vector(); 
        // goodMatches.~vector();
        // goodKeypoints[0].~vector(), goodKeypoints[1].~vector();
        
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////
    // // ROS Kinetic section
    // ros::init(argc, argv, "uw_slam");
    // ros::NodeHandle n;
    // ros::Rate r(1);
    // ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 100);

    // ros::NodeHandle nh;
    // image_transport::ImageTransport it(nh);
    // image_transport::Publisher img_pub = it.advertise("camera/image",1);
    // sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", currImgRes).toImageMsg();

    // // Set our initial shape type to be a cube
    // uint32_t shape = visualization_msgs::Marker::CUBE;

    // visualization_msgs::Marker marker;

    // // Set the frame ID and timestamp.  See the TF tutorials for information on these.
    // marker.header.frame_id = "/main_uw";
    // marker.header.stamp = ros::Time::now();

    // // Set the namespace and id for this marker.  This serves to create a unique ID
    // // Any marker sent with the same namespace and id will overwrite the old one
    // marker.ns = "uw_slam";

    // // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    // marker.type = shape;

    // // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    // marker.action = visualization_msgs::Marker::ADD;

    // // Set the scale of the marker -- 1x1x1 here means 1m on a side
    // marker.scale.x = 0.10;
    // marker.scale.y = 0.10;
    // marker.scale.z = 0.10;

    // // Set the color -- be sure to set alpha to something non-zero!
    // marker.color.r = 0.12f;
    // marker.color.g = 0.56f;
    // marker.color.b = 1.0f;
    // marker.color.a = 1.0;

    // marker.lifetime = ros::Duration();
    // //while (ros::ok()){
    //     for(int i=0; i < mapPoints.cols; i++){
                        
    //         // Set the frame ID and timestamp.  See the TF tutorials for information on these.
    //         marker.header.frame_id = "/main_uw";
    //         marker.header.stamp = ros::Time::now();

    //         // Set the namespace and id for this marker.  This serves to create a unique ID
    //         // Any marker sent with the same namespace and id will overwrite the old one
    //         marker.ns = "uw_slam";
    //         marker.id = i;

    //         // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    //         marker.type = shape;

    //         // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    //         marker.action = visualization_msgs::Marker::ADD;

    //         // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
    //         marker.pose.position.x = -10 + (mapPoints.at<float>(2,i)/mapPoints.at<float>(3,i));
    //         marker.pose.position.y = 0 + (mapPoints.at<float>(0,i)/mapPoints.at<float>(3,i));
    //         marker.pose.position.z = 2 -(mapPoints.at<float>(1,i)/mapPoints.at<float>(3,i));
    //         marker.pose.position.x /= 2;
    //         marker.pose.position.y /= 2;
    //         marker.pose.position.z /= 2;

    //         marker.pose.orientation.x = 0.0;
    //         marker.pose.orientation.y = 0.0;
    //         marker.pose.orientation.z = 0.0;
    //         marker.pose.orientation.w = 1.0;

    //         // Set the scale of the marker -- 1x1x1 here means 1m on a side
    //         marker.scale.x = 0.10;
    //         marker.scale.y = 0.10;
    //         marker.scale.z = 0.10;

    //         // Set the color -- be sure to set alpha to something non-zero!
    //         marker.color.r = 0.12f;
    //         marker.color.g = 0.56f;
    //         marker.color.b = 1.0f;
    //         marker.color.a = 1.0;
            
    //         marker.lifetime = ros::Duration();

    //         // Publish the marker
    //         while (marker_pub.getNumSubscribers() < 1 && img_pub.getNumSubscribers() < 1)
    //         {
    //             if (!ros::ok())
    //             {
    //             return 0;
    //             }
    //             ROS_WARN_ONCE("Please create a subscriber to the marker/image");
    //             sleep(1);
    //         }

    //         img_pub.publish(msg);
    //         marker_pub.publish(marker);
    //         ros::spinOnce();

    //         //r.sleep();
    //     }
    // //}
    return 0;
    
}

vector<string> read_filenames(string dir_ent){
    vector<string> file_names;
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir(dir_ent.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            file_names.push_back(dir_ent + string(ent->d_name));
        }
        closedir (dir);
    }else{
        // If the directory could not be opened
        std::cout << "Directory could not be opened" <<endl;
    }
    // Sorting the vector of strings so it is alphabetically ordered
    sort(file_names.begin(), file_names.end());
    file_names.erase(file_names.begin(), file_names.begin()+2);

    return file_names;
}

Mat readCameraMat(std::string dir_calibrarionFile){
    Mat cameraMat;
    cv::FileStorage opencv_file(dir_calibrarionFile, cv::FileStorage::READ);
    if (opencv_file.isOpened()){
        opencv_file["cameraMatrix"] >> cameraMat;
        opencv_file.release();
    }
    return cameraMat;
}

vector<DMatch> getGoodMatches(vector< vector< DMatch> > matches){
    vector<DMatch> goodMatches;
    // Use Nearest-Neighbor Ratio to determine "good" matches
    for (std::vector<std::vector<cv::DMatch> >::const_iterator it = matches.begin(); it != matches.end(); ++it) {
        if (it->size() > 1 && (*it)[0].distance / (*it)[1].distance < NEAREST_NEIGHBOR_RATIO) {
            DMatch m = (*it)[0];
            goodMatches.push_back(m);       // save good matches here                           
        }
    }
    return goodMatches;
}

array<vector<KeyPoint>,2> getGoodKeypoints(vector<DMatch> goodMatches, array< vector< KeyPoint>, 2 > keypoints){
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

vector<DMatch> gridFiltering(vector<DMatch> goodMatches, vector<KeyPoint> keypoints){
    // Dimension of patch of the grid
    int stepx = DIMENSION_WIDTH / DIMENSION_PATCH; 
    int stepy = DIMENSION_HEIGHT / DIMENSION_PATCH;
    int best_distance = 100;

    vector<DMatch> grid_matches;
    DMatch best_match;
    for(int i=0; i<DIMENSION_PATCH; i++){
        for(int j=0; j<DIMENSION_PATCH; j++){
            best_distance = 100;
            for (auto m: goodMatches) {
                //-- Get the keypoints from the good matches
                if(keypoints[m.queryIdx].pt.x >= stepx*i && keypoints[m.queryIdx].pt.x < stepx*(i+1) &&
                keypoints[m.queryIdx].pt.y >= stepy*j && keypoints[m.queryIdx].pt.y < stepy*(j+1)){
                    if(m.distance < best_distance){
                        best_distance = m.distance;
                        best_match = m;
                    }
                    goodMatches.erase(goodMatches.begin() + m.queryIdx);  
                }
            }
            if(best_distance != 100)
                grid_matches.push_back(best_match);
        }
    }
    return grid_matches;
}

Mat getProjectionMat(Mat cameraMat, Mat rotationMat, Mat translationMat){
    // ProjectionMat = cameraMat * [Rotation | translation]
    Mat projectionMat;
    hconcat(rotationMat, translationMat, projectionMat);
    projectionMat = cameraMat * projectionMat;

    return projectionMat;
}

// Print the stats of a feature detector and descriptor
void showFeatureStats(std::string detectorName, int nfeatures[2], int nmatches){
    std::cout << "***************************************" << endl;
    std::cout << "Stats of feature detector: " << detectorName << std::endl;
    std::cout << "Number of features found in current image: " << nfeatures[0] << std::endl;
    std::cout << "Number of features found in next image: " << nfeatures[1] << std::endl;
    std::cout << "Number matches: " << nmatches << std::endl;
}

// Print the features tracked in an image
void showMovementFeatures(Mat imgShow, vector<Point2f>& points1, vector<Point2f>& points2){
    for(int i=0; i<points1.size(); i++){
        Point2f a = points1.at(i);
        Point2f b = points2.at(i);
        cv::circle(imgShow, a, 3, Scalar(0,255,0), -1, 8 , 0);
        cv::line(imgShow , a, b, (0,0,255),1);
    }
}

// TODO
// Print the number of inliers and outliers found by findEssentialMat()
void printLiers(Mat mask){
    int outliers = 0, inliers =0;
    // for(int i=0; i<mask.size(); i++){
    //     if(mask.at(i) == 0){
    //         outliers += 1;
    //     }else{
    //         inliers += 1;
    //     }
    // }
    std::cout << "Intliers: "<< inliers << endl;
    std::cout << "Outliers: "<< outliers << endl;
    std::cout << "Total: "<< inliers + outliers << endl;
}
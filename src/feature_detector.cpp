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

    // Creating indexes for current and next image
    int currImg = 0, nextImg = 1;
    GpuMat img[image_names.size()];

    // Obtaining camera matrix from calibration XML file
    // (Default: /include/calibration.xml. Must be called cameraMatrix within the file
    Mat cameraMatrix = readCameraMatrix(dir_calibration);
    if(cameraMatrix.empty()){
        std::cout << "Error reading calibration xml file." << endl;
        return -1;
    }

    // Read current and next image
    Mat currImgRes, nextImgRes;
    currImgRes = imread(image_names[currImg]);
    nextImgRes = imread(image_names[nextImg]);

    // Checking for errors
    if (!currImgRes.data){
        std::cout << "Error reading the image: " << currImg + 1 << std::endl; return -1;
    }
    if (!nextImgRes.data){
        std::cout << "Error reading the image: " << nextImg + 1 << std::endl; return -1;
    }

    // Resizing the images to 640 x 480
    resize(currImgRes, currImgRes, Size(DIMENSION_WIDTH, DIMENSION_HEIGHT), 0 ,0, CV_INTER_LINEAR);
    resize(nextImgRes, nextImgRes, Size(DIMENSION_WIDTH , DIMENSION_HEIGHT), 0 ,0, CV_INTER_LINEAR);

    // Converting the images to GRAY
    cvtColor(currImgRes,currImgRes,COLOR_BGR2GRAY);
    cvtColor(nextImgRes,nextImgRes,COLOR_BGR2GRAY);

    // Uploading the images to GpuMat
    img[currImg].upload(currImgRes);
    img[nextImg].upload(nextImgRes);

    // Feature detection section
    GpuMat keypointsGPU[2];
    GpuMat descriptorsGPU[2];
    GpuMat matchesGPU;
    vector< vector< DMatch> > matches;
    vector<KeyPoint> keypoints[2];
    vector<float> descriptors[2];
    int nfeatures[2], nmatches;

    // SURF as feature detector (Default detector)
    if(detector == 0 || detector == -1){
        detectorName = "SURF Detector";

        Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher();
        cv::cuda::SURF_CUDA surf;
        Ptr<cv::cuda::Feature2DAsync> detector;
        // Detecting keypoints and computing descriptors
        surf(img[currImg], GpuMat(), keypointsGPU[0], descriptorsGPU[0]);
        surf(img[nextImg], GpuMat(), keypointsGPU[1], descriptorsGPU[1]);
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
        orb->detectAndCompute(img[currImg], noArray(), keypoints[0], descriptorsGPU[0]);
        orb->detectAndCompute(img[nextImg], noArray(), keypoints[1], descriptorsGPU[1]);
        // Matching descriptors
        matcher->knnMatch(descriptorsGPU[0], descriptorsGPU[1], matches, 2);
    }
    
    // Obtain good matches (delete outliers)
    vector<DMatch> goodMatches;
    goodMatches = getGoodMatches(matches);

    // Apply grid filtering of matches found (for homogenous sparse of features)
    goodMatches = gridFiltering(goodMatches, keypoints[0]);

    // Show results of feature detection (optional)
    if(feature_stats){
        Mat img_matches;
        drawMatches(Mat(img[currImg]), keypoints[0], Mat(img[nextImg]), keypoints[1], 
                        goodMatches, img_matches,Scalar::all(-1), Scalar::all(-1), 
                        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        nfeatures[0] = keypoints[0].size();
        nfeatures[1] = keypoints[1].size();
        nmatches = goodMatches.size();
        showFeatureStats(detectorName,nfeatures,nmatches);
        imshow(detectorName, img_matches);
        waitKey(0);     
    }

    // Pose estimation section


    // Resize the images
    Mat img_1, img_2;

    char dataSetPath[300];
    char filename1[300];
    char filename2[300];
    int calling;


    // Change the file path according to where your dataset is saved before running
    // Get the file names of the first two images
    std::sprintf(filename1, "/home/fabio/Documents/datasets/kitti/odometry/00/image_2/%06d.png", 60);
    std::sprintf(filename2, "/home/fabio/Documents/datasets/kitti/odometry/00/image_2/%06d.png", 61);
    
    // Read the first two images from the dataset
    // cv::Mat img_1_color = cv::imread("/home/fabio/Documents/datasets/crazyhorse/P1000968.JPG");
    // cv::Mat img_2_color = cv::imread("/home/fabio/Documents/datasets/crazyhorse/P1000971.JPG");
    cv::Mat img_1_color = cv::imread(filename1);
    cv::Mat img_2_color = cv::imread(filename2);

    // Check for errors
    if (!img_1_color.data || !img_2_color.data){
        std::cout << "(!) Error reading images " << std::endl; return -1;
    }

    // Resize the images
    // resize(img_1_color, img_1_color, cv::Size(1240,376), 0, 0, cv::INTER_CUBIC);
    // resize(img_2_color, img_2_color, cv::Size(1240,376), 0, 0, cv::INTER_CUBIC);

    // Convert the two images to grayscale
    cv::cvtColor(img_1_color, img_1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_2_color, img_2, cv::COLOR_BGR2GRAY);

    // Start clock variable for time measurement
    // int start_s = clock();  

    // Initialize stats of feature detector
    struct Stats stats = _initStats("SHI_T");

    // Feature detection and tracking
    vector<Point2f> points1, points2;
    vector<uchar> status;
    //getCornerEdges(img_1, img_2, points1, points2);
    featureDetectionTracking(img_1,img_2, points1, points2, status, "FAST");
    // Stop clock variable
    // int stop_s = clock();
    // stats.exec_time = calculateTime(start_s, stop_s);

    // Obtain Essential Matrix with the Five-Point Algorithm (David Nister, 2004)
    Mat EssentialMat, mask;
    double focal = 7.188560000000e+02; 
    Point2d pp = Point2d(6.071928000000e+02, 1.852157000000e+02);
    Mat K = (Mat_<double>(3,3) << focal, 0, 6.071928000000e+02, 0, focal,  1.852157000000e+02, 0, 0, 1);

    EssentialMat = findEssentialMat(points1, points2, focal, pp, cv::RANSAC, 0.999, 3.0, mask);
    //std::cout << EssentialMat << endl;
    // TODO
    // Show number of inliers/outliers found by findEssentialMat()
    // printLiers(mask);

    // Obtain Rotation matrix and translation vector
    Mat R, t;
    int inliers2;
    inliers2 = recoverPose(EssentialMat, points1, points2, R, t, focal, pp, mask);
    // std::cout << "ROTATION MATRIX" << endl;
    // std::cout << R << endl;
    // std::cout << "TRANSLATION VECTOR" << endl;
    // std::cout << t << endl;
    // std::cout << "Inliers: " << inliers2 << endl;

    // Obtain projection matrices
    Mat I3 = Mat::eye(3, 3, CV_64F);
    Mat v0 = Mat(3,1,CV_64F, double(0));
    Mat P1, P2;

    hconcat(I3, v0, P1);
    hconcat(R, t, P2);

    P1 = K * P1;
    P2 = K * P2;
    // P1 = K * [I3 | 0]
    // P2 = K2 * [R | t]

    // Compute depth of 3D points using triangulation
    Mat mapPoints;
    triangulatePoints(P1, P2, points1, points2, mapPoints);

    // Show features detected/tracked and stats information
    Mat imgShow = img_1_color.clone();
    //showFeatures(imgShow, points1, points2);
    //imshow("Output",imgShow);
    /*
    ros::init(argc, argv, "uw_slam");
    ros::NodeHandle n;
    ros::Rate r(1);
    ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 100);

    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher img_pub = it.advertise("camera/image",1);
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", imgShow).toImageMsg();

    // Set our initial shape type to be a cube
    uint32_t shape = visualization_msgs::Marker::CUBE;

    visualization_msgs::Marker marker;

    // Set the frame ID and timestamp.  See the TF tutorials for information on these.
    marker.header.frame_id = "/main_uw";
    marker.header.stamp = ros::Time::now();

    // Set the namespace and id for this marker.  This serves to create a unique ID
    // Any marker sent with the same namespace and id will overwrite the old one
    marker.ns = "uw_slam";

    // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    marker.type = shape;

    // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    marker.action = visualization_msgs::Marker::ADD;

    // Set the scale of the marker -- 1x1x1 here means 1m on a side
    marker.scale.x = 0.10;
    marker.scale.y = 0.10;
    marker.scale.z = 0.10;

    // Set the color -- be sure to set alpha to something non-zero!
    marker.color.r = 0.12f;
    marker.color.g = 0.56f;
    marker.color.b = 1.0f;
    marker.color.a = 1.0;

    marker.lifetime = ros::Duration();
    //while (ros::ok()){
        for(int i=0; i < mapPoints.cols; i++){
                        
            // Set the frame ID and timestamp.  See the TF tutorials for information on these.
            marker.header.frame_id = "/main_uw";
            marker.header.stamp = ros::Time::now();

            // Set the namespace and id for this marker.  This serves to create a unique ID
            // Any marker sent with the same namespace and id will overwrite the old one
            marker.ns = "uw_slam";
            marker.id = i;

            // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
            marker.type = shape;

            // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
            marker.action = visualization_msgs::Marker::ADD;

            // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
            marker.pose.position.x = -10 + (mapPoints.at<float>(2,i)/mapPoints.at<float>(3,i));
            marker.pose.position.y = 0 + (mapPoints.at<float>(0,i)/mapPoints.at<float>(3,i));
            marker.pose.position.z = 2 -(mapPoints.at<float>(1,i)/mapPoints.at<float>(3,i));
            marker.pose.position.x /= 2;
            marker.pose.position.y /= 2;
            marker.pose.position.z /= 2;

            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;

            // Set the scale of the marker -- 1x1x1 here means 1m on a side
            marker.scale.x = 0.10;
            marker.scale.y = 0.10;
            marker.scale.z = 0.10;

            // Set the color -- be sure to set alpha to something non-zero!
            marker.color.r = 0.12f;
            marker.color.g = 0.56f;
            marker.color.b = 1.0f;
            marker.color.a = 1.0;
            
            marker.lifetime = ros::Duration();

            // Publish the marker
            while (marker_pub.getNumSubscribers() < 1 && img_pub.getNumSubscribers() < 1)
            {
                if (!ros::ok())
                {
                return 0;
                }
                ROS_WARN_ONCE("Please create a subscriber to the marker/image");
                sleep(1);
            }

            img_pub.publish(msg);
            marker_pub.publish(marker);
            ros::spinOnce();

            //r.sleep();
        }
    //}*/
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

Mat readCameraMatrix(std::string dir_calibrarionFile){
    Mat cameraMatrix;
    cv::FileStorage opencv_file(dir_calibrarionFile, cv::FileStorage::READ);
    if (opencv_file.isOpened()){
        opencv_file["cameraMatrix"] >> cameraMatrix;
        opencv_file.release();
    }
    return cameraMatrix;
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

// Print the stats of a feature detector and descriptor
void showFeatureStats(std::string detectorName, int nfeatures[2], int nmatches){
    std::cout << "***************************************" << endl;
    std::cout << "Stats of feature detector: " << detectorName << std::endl;
    std::cout << "Number of features found in current image: " << nfeatures[0] << std::endl;
    std::cout << "Number of features found in next image: " << nfeatures[1] << std::endl;
    std::cout << "Number matches: " << nmatches << std::endl;
}

int getDistance(Point2f a, Point2f b){
    return sqrt(pow(b.y-a.y,2)+pow(b.x-a.x,2));
}

// Function to get Corners and Edges homogeneously in the images
void getCornerEdges(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2){

    // Patameters for Shi-Tomasi algorithm
    double qualityLevel = 0.98;
    double minDistance = 30;
    int maxCorners = 1;
    int blockSize = 20;
    bool useHarrisDetector = false;
    double k = 0.04;
    
    // Patch Size of each cell of the image
    int pS = 32;
    for(int x = 0; x < img_1.size().width - pS; x = x + pS){
        for(int y = 0; y < img_1.size().height- pS; y = y + pS){
            vector<Point2f> points_aux;
            Rect ROI(x,y,pS,pS);
            Mat img_aux = img_1(ROI);
            // imshow("HEY", img_aux);
            // waitKey();
            goodFeaturesToTrack(img_aux, points_aux, maxCorners, qualityLevel,  minDistance, Mat(), blockSize, useHarrisDetector, k);

            if(points_aux.size()==0){
                
            }else{
                points_aux[0].x += x;
                points_aux[0].y += y; 
                points1.push_back(points_aux[0]);
            }
        }
    }
    // Feature Tracking
    vector<float> err;
    vector<uchar> status;
    Size winSize = Size(21,21);
    TermCriteria termcrit = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    // Deleting points that the tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    int dThreshold = 60; 
    for(int i=0; i<status.size(); i++){
        Point2f pt1 = points1.at(i - indexCorrection);
        Point2f pt2 = points2.at(i - indexCorrection);
        int d = getDistance(pt1,pt2);
        if((status.at(i) == 0)||(pt2.x<0)||(pt2.y<0)||(d > dThreshold)){
            status.at(i) = 0;
            points1.erase (points1.begin() + (i - indexCorrection));
            points2.erase (points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}

// Function to detect and track features on two images (FAST | ORB | SURF | AKAZE) 
void featureDetectionTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status, std::string descriptor){
    if(descriptor=="FAST"){
        vector<KeyPoint> keypoints_1;
        int fast_threshold = 20;
        bool nonmaxSuppression = true;

        // Feature Detection
        FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
        KeyPoint::convert(keypoints_1, points1, vector<int>());

        // Saving the number of features detected
        //stats->n_features = points1.size();

        // Feature Tracking
        vector<float> err;
        Size winSize = Size(21,21);
        TermCriteria termcrit = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
        calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

        // Deleting points that the tracking failed or those who have gone outside the frame
        int indexCorrection = 0;
        for(int i=0; i<status.size(); i++){
            Point2f pt = points2.at(i - indexCorrection);
            if((status.at(i) == 0)||(pt.x<0)||(pt.y<0)){
                if((pt.x<0)||(pt.y<0)){
                    status.at(i) = 0;
                }
                points1.erase (points1.begin() + (i - indexCorrection));
                points2.erase (points2.begin() + (i - indexCorrection));
                indexCorrection++;
            }
        }
        // Saving the number of features tracked succesfully
        //stats->ok_features = points1.size();
    }

    if(descriptor=="ORB"){
        Mat descriptors_1, descriptors_2;
        vector<KeyPoint> keypoints_1, keypoints_2;
        vector<DMatch> matches, ok_matches;
        
        Ptr<FeatureDetector> orb_detector = cv::ORB::create(4000);
        Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
        
        // Feature Detection
        orb_detector->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);
        orb_detector->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);
        
        // Feature Tracking
        matcher->match (descriptors_1, descriptors_2, matches);

        // Deleting points that are too seperated
        double min_dist = 1000, max_dist = 0;
        for(int i=0; i<descriptors_1.rows; i++){
            double dist = matches[i].distance;
            if(dist < min_dist) min_dist = dist;
            if(dist > max_dist) max_dist = dist;
        }
        for(int i=0; i < descriptors_1.rows; i++){
            if(matches[i].distance <= max(2*min_dist, 30.0)){
                ok_matches.push_back(matches[i]);
            }
        }

        // Obtaining the coordinates of the ok_points
        int key1_index;
        int key2_index;
        for(int i=0; i<ok_matches.size(); i++){
            key1_index = ok_matches[i].queryIdx;
            key2_index = ok_matches[i].trainIdx;
            points1.push_back(keypoints_1[key1_index].pt);
            points2.push_back(keypoints_2[key2_index].pt);
        }

        // Uncomment to show an alternative image of the ok_points found between the two frames
        // Mat img_match, img_okmatch;
        // drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
        // drawMatches(img_1, keypoints_1, img_2, keypoints_2, ok_matches, img_okmatch);  
        // imshow("Matches", img_match);
        // imshow("OK Matches", img_okmatch);
        // waitKey(0);
        
        // Saving the number of features and ok_features detected
        //stats->n_features = matches.size();
        //stats->ok_features = ok_matches.size();
    }

    if(descriptor == "SURF"){
        int minHessian = 400;
        Mat descriptors_1, descriptors_2;
        vector<KeyPoint> keypoints_1, keypoints_2;
        vector<DMatch> matches, ok_matches;
        
        Ptr<SURF> surf_detector = SURF::create();
        surf_detector->setHessianThreshold(minHessian);

        Ptr<SurfDescriptorExtractor> surf_descriptor = SURF::create();
        FlannBasedMatcher matcher;

        // Feature Detection
        surf_detector->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);
        surf_detector->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);
        
        // Feature Tracking
        matcher.match (descriptors_1, descriptors_2, matches);

        // Deleting points that are too seperated
        double min_dist = 100, max_dist = 0;
        for(int i=0; i<descriptors_1.rows; i++){
            double dist = matches[i].distance;
            if(dist < min_dist) min_dist = dist;
            if(dist > max_dist) max_dist = dist;
        }
        for(int i=0; i < descriptors_1.rows; i++){
            if(matches[i].distance <= max(2*min_dist, 0.08)){
                ok_matches.push_back(matches[i]);
            }
        }

        // Obtaining the coordinates of the ok_points
        int key1_index;
        int key2_index;
        for(int i=0; i<ok_matches.size(); i++){
            key1_index = ok_matches[i].queryIdx;
            key2_index = ok_matches[i].trainIdx;
            points1.push_back(keypoints_1[key1_index].pt);
            points2.push_back(keypoints_2[key2_index].pt);
        }
        
        // Saving the number of features and ok_features detected
        //stats->n_features = matches.size();
        //stats->ok_features = ok_matches.size();
    }

    if(descriptor == "AKAZE"){
        Mat descriptors_1, descriptors_2;
        vector<KeyPoint> matched1, matched2, keypoints_1, keypoints_2, inliers1, inliers2;
        vector< vector<DMatch> > matches;
        vector<DMatch> ok_matches;
        
        Ptr<AKAZE> akaze = AKAZE::create();
        BFMatcher matcher(NORM_HAMMING);
        
        // Feature Detection
        akaze->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);
        akaze->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);

        // Feature Tracking
        matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);

        // Deleting points that are too seperated
        const float inlier_threshold = 120.0f;
        const float nn_match_ratio = 0.8f;
        for(size_t i = 0; i < matches.size(); i++) {
            DMatch first = matches[i][0];
            float dist1 = matches[i][0].distance;
            float dist2 = matches[i][1].distance;

            if(dist1 < nn_match_ratio * dist2) {
                matched1.push_back(keypoints_1[first.queryIdx]);
                matched2.push_back(keypoints_2[first.trainIdx]);
            }
        }
        for(unsigned i = 0; i < matched1.size(); i++) {
            Mat col = Mat::ones(3, 1, CV_64F);
            col.at<double>(0) = matched1[i].pt.x;
            col.at<double>(1) = matched1[i].pt.y;

            col /= col.at<double>(2);
            double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
                                pow(col.at<double>(1) - matched2[i].pt.y, 2));
            if(dist < inlier_threshold) {
                int new_i = static_cast<int>(inliers1.size());
                inliers1.push_back(matched1[i]);
                inliers2.push_back(matched2[i]);
                ok_matches.push_back(DMatch(new_i, new_i, 0));
            }
        }   

        // Obtaining the coordinates of the ok_points
        int key1_index;
        int key2_index;
        for(int i=0; i<ok_matches.size(); i++){
            key1_index = ok_matches[i].queryIdx;
            key2_index = ok_matches[i].trainIdx;
            points1.push_back(inliers1[key1_index].pt);
            points2.push_back(inliers2[key2_index].pt);
        }

        // Uncomment to show an alternative image of the ok_points found between the two frames
        // Mat img_match, img_okmatch;
        // drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
        // drawMatches(img_1, keypoints_1, img_2, keypoints_2, ok_matches, img_okmatch);  
        // imshow("Matches", img_match);
        // imshow("OK Matches", img_okmatch);
        // waitKey(0);
        
        // Saving the number of features and ok_features detected
        //stats->n_features = matches.size();
        //stats->ok_features = ok_matches.size();
    }
}

// Print the features tracked in an image
void showFeatures(Mat imgShow, vector<Point2f>& points1, vector<Point2f>& points2){
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


// Calculates the time in (ms) using the output of two clock() variables
int calculateTime(int start, int stop){
    return (stop - start)/double(CLOCKS_PER_SEC)*1000;
}





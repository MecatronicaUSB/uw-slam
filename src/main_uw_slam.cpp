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

#include "../include/Options.h"
#include "../include/System.h"

// C++ namespaces
using namespace uw;
using namespace cv;
using namespace std;
using namespace cv::cuda;

string imagesPath;
std::string calibrationPath;
cuda::DeviceInfo deviceInfo;

void showSettings(){
    cout << "CUDA enabled devices detected: " << deviceInfo.name() << endl;
    cout << "Directory of images: " << imagesPath << endl;
    cout << "Directory of calibration xml file: " << calibrationPath << "\n"<< endl;
}

int main ( int argc, char *argv[] ){

    cout << "===================================================" << endl;
    int nCuda = cuda::getCudaEnabledDeviceCount();
    if (nCuda > 0){
        cuda::setDevice(0);
    }
    else {
        cout << "No CUDA device detected" << endl;
        cout << "Exiting..." << endl;
        return -1;
    }

    // Parse section
    try{
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help){
        cout << parser;
        return 0;
    }
    catch (args::ParseError e){
        cerr << e.what() << endl;
        cerr << parser;
        return 1;
    }
    catch (args::ValidationError e){
        cerr << e.what() << endl;
        cerr << parser;
        return 1;
    }
    if (!dir_dataset) {
        cout<< "Introduce path of images as argument" << endl;
        cerr << "Use -h, --help command to see usage" << endl;
        return 1;
    }
    else {
        imagesPath = args::get(dir_dataset);
    }
    if(parse_calibration){
        calibrationPath = args::get(parse_calibration);
    }else{
        calibrationPath = "./sample/calibration.xml";
    }

    showSettings();

    // Create new System
    System* uwSystem = new System();
    // Add list of images names (with path)
    uwSystem->addListImages(imagesPath);
    // Calibrates system with certain Camera Model (currently only FOV) 
    uwSystem->Calibration(calibrationPath);
    
    cout << uwSystem->cameraModel->hOrg << endl;

    return 0;
}
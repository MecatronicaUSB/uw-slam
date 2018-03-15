# Monocular Underwater SLAM

Implementation of Monocular Simultaneous Localization and Mapping (SLAM) for underwater vehicles. Using OpenCV 3.2, CUDA 8.0 and ROS Kinect.

UW-SLAM is a free and open hardware licensed under the [GPL-3.0 License](https://en.wikipedia.org/wiki/GNU_General_Public_License).

## Table of Contents
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Directory Layout](#directory-layout)
- [License](#license)

## Requirements

- [OpenCV 3.2](http://opencv.org) and extra modules.
- [ROS Kinetic](http://wiki.ros.org/kinetic/Installation).
- [CUDA 8.0](https://developer.nvidia.com/cuda-toolkit-archive).
- [Eigen 3](http://eigen.tuxfamily.org/index.php?title=Main_Page).
- [Ceres solver](http://ceres-solver.org/).

## Getting Started

From a fresh Ubuntu 16.04 LTS, install the following dependencies:

#### OpenCV 3.2 with CUDA 8.0

Refer to [OpenCV and CUDA 8.0 installation instructions](https://github.com/MecatronicaUSB/uwimageproc/blob/master/INSTALL.md).

#### ROS Kinetic

Refer to [ROS Kinetic installation instructions](http://wiki.ros.org/kinetic/Installation).

#### Eigen 3

```bash
sudo apt-get install libsuitesparse-dev libeigen3-dev libboost-all-dev
```

#### Ceres solver

Refer to [Ceres solver installation](http://ceres-solver.org/installation.html#linux).

## Building UW-SLAM

Clone UW-SLAM repository in the `/src` folder of your catkin workspace:

```bash
cd <catkin_ws_directory>/src
git clone https://github.com/MecatronicaUSB/uw-slam.git
cd ..
catkin_make
```

## Usage

### General datasets

Run UW-SLAM on a dataset of images with known calibration parameters and dimentions of images. 

Modify the `calibration.xml` file in `/calibration` folder to specify the instrinsic parameters of the camera of the dataset to use. 
```bash
    -d <directory of images files>                  
    -c <directory of calibration.xml file>          (<uw-slam directory>/calibration/calibration.xml)
    -s <number of starting frame>                   (Default: 0)
```

Modify the `uw_slam.launch` file in `/launch` folder to specify the directory of files (Refer to `/calibration/calibrationTUM.xml` for proper configuration of the .xml file).
```bash
    <!-- Images dimensions (Input) -->
    <in_width  type_id="integer"> W </in_width>       (Input dimentions of images)
    <in_height type_id="integer"> H </in_height>      (Replace W: width, H: height)

    <!-- Images dimensions (Output) -->
    <out_width  type_id="integer"> W </out_width>     (Output desired dimentions of images)
    <out_height type_id="integer"> H </out_height>    (Replace W: width, H: height)

    <!-- Calibration Values of Dataset -->
    <calibration_values type_id="opencv-matrix">      (Intrinsic camera parameters)
    <rows>1</rows>                                    (Replace fx fy cx cy)
    <cols>4</cols>
    <dt>f</dt>
    <data>
        fx  fy  cx  cy </data></calibration_values> 

    <!-- Distortion coefficients -->
    <rectification type_id="opencv-matrix">          (Distortion parameters, optional)
    <rows>1</rows>                                   (Replace k1 k2 k3 k4)
    <cols>4</cols>                                   (If not: 0  0  0  1)
    <dt>f</dt>  
    <data>
        k1  k2  k3  k4 </data></rectification>
```

Run UW-SLAM for general datasets:
```bash
roslaunch uw_slam uw_slam.launch
```
### EUROC and TUM datasets

Currently, UW-SLAM supports ground-truth visualization along with UW-SLAM results for [TUM](https://vision.in.tum.de/data/datasets/mono-dataset?redirect=1) and [EUROC MAV](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) datasets for testing. For these datasets, a corresponding `calbration.xml` file is already created in the `/calibration` folder.

#### EUROC

For EUROC datasets, modify the args of the `uw_slamEUROC.launch` file in `/launch` folder to specify the directory of the files.
```bash
    -d <directory of images files>                  (<EUROC directory>/mav0/cam0/data/)
    -c <directory of calibrationEUROC.xml file>     (<uw-slam directory>/calibration/calibrationEUROC.xml)
    -s <number of starting frame>                   (Default: 0)
    --EUROC <directory of ground-truth poses file>  (<EUROC directory>/mav0/vic0/data.csv)
```
Run UW-SLAM for EUROC datasets:
```bash
roslaunch uw_slam uw_slamEUROC.launch
```
#### TUM

For TUM datasets, modify the args of the `uw_slamTUM.launch` file in `/launch` folder to specify the directory of the files.
```bash
    -d <directory of images files>                (<TUM directory>/rgb/)
    -c <directory of calibrationEUROC.xml file>   (<uw-slam directory>/calibration/calibrationTUM.xml)
    -s <number of starting frame>                 (Default: 0)
    --TUM <directory of ground-truth poses file>  (<TUM directory>/groundtruth.txt)
```
Run UW-SLAM for TUM datasets:
```bash
roslaunch uw_slam uw_slamTUM.launch
```
## Software Details

- Implementation done in C++ (CUDA optimization in progress).
- Using Rviz as visualization tool.

## Directory Layout

#### /src

Core .cpp files of UW-SLAM.

#### /include

Libraries .h files of UW-SLAM.
Argument parser library args.hxx ([Taywee/args](https://github.com/Taywee/args)).

#### /launch

Launch files of UW-SLAM for easy ROS Kinetic execution.

#### /thirdparty

[Sophus](https://github.com/strasdat/Sophus) library for Lie-Algebra space. 

#### /calibration

Calibration files. Included calibration for EUROC and TUM datasets.

## License

Copyright (c) 2017 Fabio Morales (<fabmoraleshidalgo@gmail.com>).

Release under the [GPL-3.0 License](LICENSE). 


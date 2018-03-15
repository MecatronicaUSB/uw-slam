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
- [CUDA 8.0](https://developer.nvidia.com/cuda-toolkit-archive).
- [Eigen 3]().
- [ROS Kinetic](http://wiki.ros.org/kinetic/Installation).

## Getting Started

Install the required dependencies:

### OpenCV 3.2 with CUDA 8.0

Please refer to [Installation OpenCV and CUDA](https://github.com/MecatronicaUSB/uwimageproc/blob/master/INSTALL.md).

### Eigen 3

```bash
sudo apt-get install libsuitesparse-dev libeigen3-dev libboost-all-dev
```

### Building UW-SLAM

Clone UW-SLAM repository in the /src folder of your catkin workspace:

```bash
cd <catkin_ws_directory>/src
git clone https://github.com/MecatronicaUSB/uw-slam.git
cd ..
catkin_make
```

## Usage

### Usage with general datasets

Run UW-SLAM on a dataset of images with known calibration parameters and dimentions of images. 

Modify the `calibration.xml` file in the `/sample` folder to specify the instrinsic parameters of the camera of the dataset to use.
Modify the `uw_slam.launch` file in the `/launch` folder to specify the directory of files. 


### Usage with EUROC and TUM datasets

Currently, UW-SLAM supports ground-truth visualization along with UW-SLAM results for [TUM](https://vision.in.tum.de/data/datasets/mono-dataset?redirect=1) and [EUROC MAV](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) datasets for testing. For these datasets, a corresponding `calbration.xml` file is already created in the `/sample` folder.

For EUROC datasets, modify the args of the `uw_slamEUROC.launch` file in `/launch` folder to specify the directory of the files.
```bash
    -d <directory of images files>
    -c <directory of calibrationEUROC.xml file>     (for EUROC, <uw-slam directory>/sample/calibrationEUROC.xml)
    -s <number of starting frame>                   (Default: 0)
    --EUROC <directory of ground-truth poses file>  (for EUROC, directory of data.csv)
```

For TUM datasets, modify the args of the `uw_slamTUM.launch` file in `/launch` folder to specify the directory of the files.
```bash
    -d <directory of images files>
    -c <directory of calibrationEUROC.xml file>   (for TUM, <uw-slam directory>/sample/calibrationTUM.xml)
    -s <number of starting frame>                 (Default: 0)
    --TUM <directory of ground-truth poses file>  (for TUM, directory of groundtruth.txt)
```

## Software Details

- Implementation done in C++ (CUDA optimization in progress).
- Using Rviz as visualization tool.

## Directory Layout

#### /src

Core files of uw-slam.

#### /include

Libraries files of uw-slam.


## License

Copyright (c) 2017 Fabio Morales (<fabmoraleshidalgo@gmail.com>).

Release under the [GPL-3.0 License](LICENSE). 


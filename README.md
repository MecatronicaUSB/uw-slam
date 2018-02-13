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
- [ROS Kinetic](http://wiki.ros.org/kinetic/Installation).

## Getting Started

This repository provides all you need to simulate and execute UW-SLAM.
```bash
cd <catkin_ws_directory>
git clone https://github.com/MecatronicaUSB/uw-slam.git
catkin_make
rosrun uw_slam main_uw_slam -d <images directory> -c <calibration file directory> 
```

## Software Details

- Implementation done in C++ using GPU optimization.
- Using Rviz as visualization tool.

## Directory Layout

#### /src

Core files of uw-slam.

#### /include

Libraries files of uw-slam.


## License

Copyright (c) 2017 Fabio Morales (<fabmoraleshidalgo@gmail.com>).

Release under the [GPL-3.0 License](LICENSE). 


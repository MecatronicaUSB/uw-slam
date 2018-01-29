# Monocular Underwater SLAM

Implementation of Monocular Simultaneous Localization and Mapping (SLAM) for underwater vehicles. Using OpenCV 3.2 and ROS Kinect.

UW-SLAM is a free and open hardware licensed under the [GPL-3.0 License](https://en.wikipedia.org/wiki/GNU_General_Public_License)

## Table of Contents
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Directory Layout](#directory-layout)
- [License](#license)

## Requirements

- *OpenCV 3.0+*
- *OpenCV extra modules 3.0+*
- [*ROS Kinetic*](http://wiki.ros.org/kinetic/Installation)

## Getting Started

This repository provides all you need to simulate and execute UW-SLAM.
```bash
cd <catkin_ws_directory>
git clone https://github.com/MecatronicaUSB/uw-slam.git
catkin_make
rosrun uw-slam uw-slam
```

## Software Details

- Implementation done in C++.
- Using Rviz as visualization tool.

## Directory Layout

#### README.md

This file.

#### main

Main script for simulation .

## License

Copyright (c) 2017 Fabio Morales (<fabmoraleshidalgo@gmail.com>).

Release under the [GPL-3.0 License](LICENSE).


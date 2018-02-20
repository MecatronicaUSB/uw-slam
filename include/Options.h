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
#include "Eigen/Core"
#include "sophus/sim3.hpp"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

namespace uw{

typedef Sophus::SE3d SE3;
typedef Sophus::Sim3d Sim3;
typedef Sophus::SO3d SO3;

typedef Eigen::Matrix<double,3,3> Mat33;
typedef Eigen::Matrix<double,4,4> Mat44;

// Global constants
extern const int PYRAMID_LEVELS;
extern int BLOCK_SIZE;
extern double GRADIENT_THRESHOLD;

}
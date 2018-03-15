cmake_minimum_required(VERSION 2.8.3)
project(uw_slam)

## Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)

# Required libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  visualization_msgs
  image_transport
  cv_bridge
  message_generation
)
find_package(OpenCV REQUIRED
NO_MODULE
PATHS /usr/local
NO_DEFAULT_PATH)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

set(CMAKE_BUILD_TYPE Debug)

if(CUDA_FOUND)
    INCLUDE(FindCUDA)
  # If the package has been found, several variables will
  # be set, you can find the full list with descriptions
  # in the CUDAConfig.cmake file.
  # Print some message showing some of them
  message(STATUS "CUDA library status:")
  message(STATUS "    toolkit root dir: ${CUDA_TOOLKIT_ROOT_DIR}")
  message(STATUS "    version: ${CUDA_VERSION}")
  message(STATUS "    libraries: ${CUDA_LIBRARIES}")
  message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
endif(CUDA_FOUND)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  visualization_msgs
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES uw_slam
   CATKIN_DEPENDS roscpp visualization_msgs message_runtime
#  DEPENDS system_lib
)

# Source files
set(uw_slam_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/main_uw_slam.cpp
    ${PROJECT_SOURCE_DIR}/src/System.cpp
    ${PROJECT_SOURCE_DIR}/src/Tracker.cpp
    ${PROJECT_SOURCE_DIR}/src/CameraModel.cpp
    ${PROJECT_SOURCE_DIR}/src/Options.cpp 
    ${PROJECT_SOURCE_DIR}/src/Visualizer.cpp 
    ${PROJECT_SOURCE_DIR}/src/LeastSquares.cpp 
    
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/thirdparty
)

add_library(uw ${uw_slam_SOURCE_FILES})

add_executable(main_uw_slam ${PROJECT_SOURCE_DIR}/src/main_uw_slam.cpp)

## IMPORTANT - TO WORK WITH CUDA AND ROS (CUDA INSTALLED IN ANOTHER OPENCV)
string(REPLACE "/opt/ros/kinetic/lib/libopencv_core3.so.3.3;" "" catkin_LIBRARIES "${catkin_LIBRARIES}")

target_link_libraries(main_uw_slam uw ${OpenCV_LIBS} ${CERES_LIBRARIES} ${catkin_LIBRARIES} ${CUDA_LIBRARIES})

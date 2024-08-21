cmake_minimum_required(VERSION 3.0.2)
project(moveint_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  rospy
  tf2
  tf2_msgs
  tf2_ros
  nav_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES moveint_ros
#  CATKIN_DEPENDS image_transport rospy sensor_msgs std_msgs tf2 tf2_msgs tf2_ros visualization_msgs
#  DEPENDS system_lib
)


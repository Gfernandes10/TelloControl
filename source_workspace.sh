#!/bin/bash
# Source ROS2 Foxy and the workspace, set Gazebo model path, and source Gazebo environment

source /opt/ros/foxy/setup.bash
source install/setup.bash
export GAZEBO_MODEL_PATH=${PWD}/install/tello_gazebo/share/tello_gazebo/models
source /usr/share/gazebo/setup.sh

echo "Environment ready! You can now run Gazebo and ROS2 launch files."

#!/bin/bash
# Script to setup and build the Tello Gazebo ROS2 workspace using vcs and colcon
set -e

# Source ROS2 Foxy
source /opt/ros/foxy/setup.bash

# Go to workspace root
cd "$(dirname "$0")/.."

# Import all repositories listed in repos.yaml
mkdir -p src
vcs import src < 01_Builds/repos_gazebo.yaml

# Instala dependências do sistema dos pacotes
rosdep install --from-paths src --ignore-src -r -y

# Build workspace
colcon build --symlink-install --cmake-clean-cache

source install/setup.bash

# Configurations
export GAZEBO_MODEL_PATH=${PWD}/install/tello_gazebo/share/tello_gazebo/models
source /usr/share/gazebo/setup.sh
if ! grep -q 'export MY_WORKSPACE_NAME="TelloControl"' ~/.bashrc; then
  echo 'export MY_WORKSPACE_NAME="TelloControl"' >> ~/.bashrc
fi

echo "Build complete! Source the workspace with: source install/setup.bash"

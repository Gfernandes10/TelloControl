#!/bin/bash

# Script to check and install ROS2, configure environment if ROS1 is installed,
# and install all dependencies required for package development

set -e

# Function to display warning
function warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# Check if ROS2 is installed
if command -v ros2 &> /dev/null; then
    echo "ROS2 is already installed."
else
    # Check if ROS1 is installed
    if command -v roscore &> /dev/null; then
        warning "ROS1 detected! The environment will be adjusted to prioritize ROS2."
        # Try to comment out any ROS1 sourcing in .bashrc
        if grep -q "source /opt/ros/.*setup.bash" ~/.bashrc; then
            sed -i '/source \/opt\/ros\//s/^/#/' ~/.bashrc
            echo "ROS1 sourcing lines commented out in ~/.bashrc."
        fi
    fi

    # Install ROS2 (Foxy for Ubuntu 20.04)
    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository universe
    sudo apt update && sudo apt install -y curl gnupg lsb-release
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt update
    sudo apt install -y ros-foxy-desktop

    # Add ROS2 sourcing to .bashrc
    if ! grep -q "source /opt/ros/foxy/setup.bash" ~/.bashrc; then
        echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
        echo "ROS2 sourcing added to ~/.bashrc."
    fi
fi

# Install general development dependencies
# Add any additional dependencies your packages require below
sudo apt update
sudo apt install -y \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-argcomplete \
    build-essential \
    git \
    cmake \
    python3-vcstool


# Initialize rosdep if not already initialized
if [ ! -f "/etc/ros/rosdep/sources.list.d/20-default.list" ]; then
    sudo rosdep init
fi
rosdep update

# Placeholder: install any other dependencies required by your packages below
# Example: sudo apt install -y <your-dependency>
# Example: pip3 install <your-python-package>
sudo apt install -y gazebo11 libgazebo11 libgazebo11-dev ros-foxy-gazebo-ros-pkgs
sudo apt install -y libasio-dev
sudo apt install -y ros-foxy-cv-bridge ros-foxy-camera-calibration-parsers 
sudo apt install -y libignition-rendering3
pip3 install transformations

# End of script

echo "ROS2 and all general development dependencies are installed! Add any extra dependencies to this script as needed. Open a new terminal to use ROS2."

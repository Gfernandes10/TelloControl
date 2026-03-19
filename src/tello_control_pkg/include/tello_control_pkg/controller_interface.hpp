#pragma once

#include <string>

#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "tello_control_pkg/msg/control_reference.hpp"

namespace tello_control_pkg
{

class ControllerInterface
{
public:
  virtual ~ControllerInterface() = default;

  virtual std::string name() const = 0;

  virtual void reset() {}

  virtual geometry_msgs::msg::Twist compute_command(
    const nav_msgs::msg::Odometry & odom_msg,
    const tello_control_pkg::msg::ControlReference & reference_msg) = 0;
};

}  // namespace tello_control_pkg

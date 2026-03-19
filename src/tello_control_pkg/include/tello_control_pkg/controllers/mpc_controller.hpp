#pragma once

#include "tello_control_pkg/controller_interface.hpp"

namespace tello_control_pkg
{

class MPCController : public ControllerInterface
{
public:
  std::string name() const override;

  geometry_msgs::msg::Twist compute_command(
    const nav_msgs::msg::Odometry & odom_msg,
    const tello_control_pkg::msg::ControlReference & reference_msg) override;
};

}  // namespace tello_control_pkg

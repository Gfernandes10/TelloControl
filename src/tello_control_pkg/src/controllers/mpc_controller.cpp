#include "tello_control_pkg/controllers/mpc_controller.hpp"

namespace tello_control_pkg
{

std::string MPCController::name() const
{
  return "MPC";
}

geometry_msgs::msg::Twist MPCController::compute_command(
  const nav_msgs::msg::Odometry & /*odom_msg*/,
  const tello_control_pkg::msg::ControlReference & /*reference_msg*/)
{
  // TODO(gabriel): implement MPC optimization/control law.
  return geometry_msgs::msg::Twist();
}

}  // namespace tello_control_pkg

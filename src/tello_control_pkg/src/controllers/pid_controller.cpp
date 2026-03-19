#include "tello_control_pkg/controllers/pid_controller.hpp"

#include <algorithm>
#include <cmath>

namespace tello_control_pkg
{

PIDController::PIDController(const PIDConfig & config)
: config_(config)
{
}

std::string PIDController::name() const
{
  return "PID";
}

void PIDController::reset()
{
  int_ex_ = 0.0;
  int_ey_ = 0.0;
  int_ez_ = 0.0;
  int_eyaw_ = 0.0;
  has_last_compute_time_ = false;
}

void PIDController::set_config(const PIDConfig & config)
{
  config_ = config;
}

PIDConfig PIDController::get_config() const
{
  return config_;
}

double PIDController::normalize_angle(double angle_rad)
{
  while (angle_rad > M_PI) {
    angle_rad -= 2.0 * M_PI;
  }
  while (angle_rad < -M_PI) {
    angle_rad += 2.0 * M_PI;
  }
  return angle_rad;
}

double PIDController::clamp(double value, double min_value, double max_value)
{
  return std::max(min_value, std::min(value, max_value));
}

geometry_msgs::msg::Twist PIDController::compute_command(
  const nav_msgs::msg::Odometry & odom_msg,
  const tello_control_pkg::msg::ControlReference & reference_msg)
{
  auto now = std::chrono::steady_clock::now();
  double dt = 0.02;
  if (has_last_compute_time_) {
    dt = std::chrono::duration<double>(now - last_compute_time_).count();
    dt = clamp(dt, 0.001, 0.2);
  } else {
    has_last_compute_time_ = true;
  }
  last_compute_time_ = now;

  const double x = odom_msg.pose.pose.position.x;
  const double y = odom_msg.pose.pose.position.y;
  const double z = odom_msg.pose.pose.position.z;

  const double vx = odom_msg.twist.twist.linear.x;
  const double vy = odom_msg.twist.twist.linear.y;
  const double vz = odom_msg.twist.twist.linear.z;
  const double wyaw = odom_msg.twist.twist.angular.z;

  const auto & q = odom_msg.pose.pose.orientation;
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  const double yaw = std::atan2(siny_cosp, cosy_cosp);

  const double ex = reference_msg.x - x;
  const double ey = reference_msg.y - y;
  const double ez = reference_msg.z - z;
  const double eyaw = normalize_angle(reference_msg.yaw - yaw);

  int_ex_ += ex * dt;
  int_ey_ += ey * dt;
  int_ez_ += ez * dt;
  int_eyaw_ += eyaw * dt;

  int_ex_ = clamp(int_ex_, -config_.int_xy_limit, config_.int_xy_limit);
  int_ey_ = clamp(int_ey_, -config_.int_xy_limit, config_.int_xy_limit);
  int_ez_ = clamp(int_ez_, -config_.int_z_limit, config_.int_z_limit);
  int_eyaw_ = clamp(int_eyaw_, -config_.int_yaw_limit, config_.int_yaw_limit);

  geometry_msgs::msg::Twist cmd;
  cmd.linear.x = config_.kp_xy * ex + config_.ki_xy * int_ex_ - config_.kd_xy * vx;
  cmd.linear.y = config_.kp_xy * ey + config_.ki_xy * int_ey_ - config_.kd_xy * vy;
  cmd.linear.z = config_.kp_z * ez + config_.ki_z * int_ez_ - config_.kd_z * vz;
  cmd.angular.z = config_.kp_yaw * eyaw + config_.ki_yaw * int_eyaw_ - config_.kd_yaw * wyaw;

  cmd.linear.x = clamp(cmd.linear.x, -config_.cmd_xy_limit, config_.cmd_xy_limit);
  cmd.linear.y = clamp(cmd.linear.y, -config_.cmd_xy_limit, config_.cmd_xy_limit);
  cmd.linear.z = clamp(cmd.linear.z, -config_.cmd_z_limit, config_.cmd_z_limit);
  cmd.angular.z = clamp(cmd.angular.z, -config_.cmd_yaw_limit, config_.cmd_yaw_limit);

  return cmd;
}

}  // namespace tello_control_pkg

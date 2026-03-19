#pragma once

#include <chrono>

#include "tello_control_pkg/controller_interface.hpp"

namespace tello_control_pkg
{

struct PIDConfig
{
  double kp_xy{0.9};
  double ki_xy{0.03};
  double kd_xy{0.20};

  double kp_z{1.1};
  double ki_z{0.05};
  double kd_z{0.25};

  double kp_yaw{1.0};
  double ki_yaw{0.02};
  double kd_yaw{0.10};

  double int_xy_limit{0.8};
  double int_z_limit{0.6};
  double int_yaw_limit{0.6};

  double cmd_xy_limit{0.35};
  double cmd_z_limit{0.35};
  double cmd_yaw_limit{0.8};
};

class PIDController : public ControllerInterface
{
public:
  explicit PIDController(const PIDConfig & config = PIDConfig{});

  std::string name() const override;
  void reset() override;
  void set_config(const PIDConfig & config);
  PIDConfig get_config() const;

  geometry_msgs::msg::Twist compute_command(
    const nav_msgs::msg::Odometry & odom_msg,
    const tello_control_pkg::msg::ControlReference & reference_msg) override;

private:
  static double normalize_angle(double angle_rad);
  static double clamp(double value, double min_value, double max_value);

  PIDConfig config_{};

  // Integrator states
  double int_ex_{0.0};
  double int_ey_{0.0};
  double int_ez_{0.0};
  double int_eyaw_{0.0};

  std::chrono::steady_clock::time_point last_compute_time_{};
  bool has_last_compute_time_{false};
};

}  // namespace tello_control_pkg

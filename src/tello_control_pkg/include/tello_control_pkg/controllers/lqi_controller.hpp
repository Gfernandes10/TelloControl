#pragma once

#include <array>
#include <chrono>
#include <string>
#include <vector>

#include "tello_control_pkg/controller_interface.hpp"

namespace tello_control_pkg
{

struct LQIConfig
{
  // "manual" or "synthesize"
  std::string mode{"synthesize"};

  // Manual gain matrix (row-major 4x12).
  std::vector<double> k{
    0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.35, 0.0, 0.0, 0.0, 0.5
  };

  // Nominal parameters for the plant model:
  // q_ddot = Lambda(psi) * q_dot + Gamma(psi) * v_cmd
  // with:
  //   Lambda(psi) using gamma2,gamma4,gamma6,gamma8
  //   Gamma(psi) using gamma1,gamma3,gamma5,gamma7
  double gamma1{3.75};
  double gamma2{-1.10};
  double gamma3{3.75};
  double gamma4{-1.10};
  double gamma5{2.68};
  double gamma6{-0.75};
  double gamma7{1.42};
  double gamma8{-2.06};

  // Discretization step used in synthesis mode.
  double design_dt{0.02};

  // Diagonal weights used in synthesis mode.
  std::vector<double> q_diag{
    6.0, 1.0, 6.0, 1.0, 8.0, 1.5, 4.0, 0.8, 12.0, 12.0, 16.0, 10.0};
  std::vector<double> r_diag{0.8, 0.8, 1.0, 0.6};

  // Integrator and command limits.
  double int_xy_limit{0.25};
  double int_z_limit{0.25};
  double int_yaw_limit{0.10};

  double cmd_xy_limit{0.25};
  double cmd_z_limit{0.30};
  double cmd_yaw_limit{0.35};

  // Feedforward trim in mapped command space (added before command clamp).
  double trim_x{0.0};
  double trim_y{0.0};
  double trim_z{0.0};
  double trim_yaw{0.0};

  // If true, freezes Z integrator update when Z command is saturated and
  // the current Z error would push command further into saturation.
  bool xy_conditional_anti_windup{false};
  bool z_conditional_anti_windup{true};
  bool yaw_conditional_anti_windup{true};

  // Runtime law:
  // u = control_sign * K * x_aug
  // x_aug = [ex, evx, ey, evy, ez, evz, eyaw, evyaw, iex, iey, iez, ieyaw]
  // where e = state - reference.
  double control_sign{-1.0};
};

struct LQIDebugSnapshot
{
  bool valid{false};
  double dt{0.0};

  // Measured state
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double yaw{0.0};
  double vx{0.0};
  double vy{0.0};
  double vz{0.0};
  double wyaw{0.0};

  // Reference actually used inside controller
  double x_ref{0.0};
  double y_ref{0.0};
  double z_ref{0.0};
  double yaw_ref{0.0};

  // Error state
  double ex{0.0};
  double evx{0.0};
  double ey{0.0};
  double evy{0.0};
  double ez{0.0};
  double evz{0.0};
  double eyaw{0.0};
  double evyaw{0.0};

  // Integrator values before and after clamping
  double int_ex_raw{0.0};
  double int_ey_raw{0.0};
  double int_ez_raw{0.0};
  double int_eyaw_raw{0.0};
  double int_ex{0.0};
  double int_ey{0.0};
  double int_ez{0.0};
  double int_eyaw{0.0};

  // Internal vectors
  std::array<double, 12> x_aug{};
  std::array<double, 4> u_virtual{};
  std::array<double, 4> u_p{};
  std::array<double, 4> u_d{};
  std::array<double, 4> u_i{};
  std::array<double, 4> u_trim{};
  std::array<double, 4> v_raw{};
  std::array<double, 4> v_p{};
  std::array<double, 4> v_d{};
  std::array<double, 4> v_i{};
  std::array<double, 4> v_trim{};
  std::array<double, 4> v_total_unclamped{};
  std::array<double, 4> cmd_clamped{};

  // Saturation flags
  bool sat_int_ex{false};
  bool sat_int_ey{false};
  bool sat_int_ez{false};
  bool sat_int_eyaw{false};
  bool sat_cmd_x{false};
  bool sat_cmd_y{false};
  bool sat_cmd_z{false};
  bool sat_cmd_yaw{false};
  bool aw_x_integrator_blocked{false};
  bool aw_y_integrator_blocked{false};
  bool aw_z_integrator_blocked{false};
  bool aw_yaw_integrator_blocked{false};
};

class LQIController : public ControllerInterface
{
public:
  explicit LQIController(const LQIConfig & config = LQIConfig{});

  std::string name() const override;
  void reset() override;
  void set_config(const LQIConfig & config);
  LQIConfig get_config() const;
  LQIDebugSnapshot get_last_debug_snapshot() const;

  geometry_msgs::msg::Twist compute_command(
    const nav_msgs::msg::Odometry & odom_msg,
    const tello_control_pkg::msg::ControlReference & reference_msg) override;

private:
  static double normalize_angle(double angle_rad);
  static double clamp(double value, double min_value, double max_value);

  std::array<double, 4> build_virtual_control(const std::array<double, 12> & x_aug) const;
  std::array<double, 4> map_virtual_to_command(
    const std::array<double, 4> & u_virtual,
    double yaw_rad) const;

  LQIConfig config_{};
  double int_ex_{0.0};
  double int_ey_{0.0};
  double int_ez_{0.0};
  double int_eyaw_{0.0};
  std::chrono::steady_clock::time_point last_compute_time_{};
  bool has_last_compute_time_{false};
  LQIDebugSnapshot last_debug_snapshot_{};
};

}  // namespace tello_control_pkg

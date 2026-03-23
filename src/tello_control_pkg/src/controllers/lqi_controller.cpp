#include "tello_control_pkg/controllers/lqi_controller.hpp"

#include <algorithm>
#include <cmath>

namespace tello_control_pkg
{

LQIController::LQIController(const LQIConfig & config)
: config_(config)
{
}

std::string LQIController::name() const
{
  return "LQI";
}

void LQIController::reset()
{
  int_ex_ = 0.0;
  int_ey_ = 0.0;
  int_ez_ = 0.0;
  int_eyaw_ = 0.0;
  has_last_compute_time_ = false;
}

void LQIController::reset_xy_integrators()
{
  int_ex_ = 0.0;
  int_ey_ = 0.0;
}

void LQIController::reset_yaw_integrator()
{
  int_eyaw_ = 0.0;
}

void LQIController::set_config(const LQIConfig & config)
{
  config_ = config;
}

LQIConfig LQIController::get_config() const
{
  return config_;
}

LQIDebugSnapshot LQIController::get_last_debug_snapshot() const
{
  return last_debug_snapshot_;
}

double LQIController::normalize_angle(double angle_rad)
{
  while (angle_rad > M_PI) {
    angle_rad -= 2.0 * M_PI;
  }
  while (angle_rad < -M_PI) {
    angle_rad += 2.0 * M_PI;
  }
  return angle_rad;
}

double LQIController::clamp(double value, double min_value, double max_value)
{
  return std::max(min_value, std::min(value, max_value));
}

std::array<double, 4> LQIController::build_virtual_control(const std::array<double, 12> & x_aug) const
{
  std::array<double, 4> u_virtual{0.0, 0.0, 0.0, 0.0};
  if (config_.k.size() != 48U) {
    return u_virtual;
  }

  for (std::size_t row = 0; row < 4; ++row) {
    double acc = 0.0;
    for (std::size_t col = 0; col < 12; ++col) {
      acc += config_.k[row * 12 + col] * x_aug[col];
    }
    u_virtual[row] = config_.control_sign * acc;
  }
  return u_virtual;
}

std::array<double, 4> LQIController::map_virtual_to_command(
  const std::array<double, 4> & u_virtual,
  double yaw_rad) const
{
  std::array<double, 4> v_cmd{0.0, 0.0, 0.0, 0.0};
  constexpr double kEps = 1e-9;

  if (
    std::abs(config_.gamma1) < kEps || std::abs(config_.gamma3) < kEps ||
    std::abs(config_.gamma5) < kEps || std::abs(config_.gamma7) < kEps)
  {
    return v_cmd;
  }

  const double c = std::cos(yaw_rad);
  const double s = std::sin(yaw_rad);

  // v = Gamma(psi)^-1 * u, for Gamma defined in the model.
  v_cmd[0] = (c * u_virtual[0] + s * u_virtual[1]) / config_.gamma1;
  v_cmd[1] = (-s * u_virtual[0] + c * u_virtual[1]) / config_.gamma3;
  v_cmd[2] = u_virtual[2] / config_.gamma5;
  v_cmd[3] = u_virtual[3] / config_.gamma7;
  return v_cmd;
}

geometry_msgs::msg::Twist LQIController::compute_command(
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

  // LQI state uses e = state - reference, matching the synthesis scripts.
  const double ex = x - reference_msg.x;
  const double ey = y - reference_msg.y;
  const double ez = z - reference_msg.z;
  const double eyaw = normalize_angle(yaw - reference_msg.yaw);

  const double evx = vx;
  const double evy = vy;
  const double evz = vz;
  const double evyaw = wyaw;

  const double int_ex_prev = int_ex_;
  const double int_ey_prev = int_ey_;
  const double int_ez_prev = int_ez_;
  const double int_eyaw_prev = int_eyaw_;
  const double int_ex_raw = int_ex_ + ex * dt;
  const double int_ey_raw = int_ey_ + ey * dt;
  const double int_ez_raw = int_ez_ + ez * dt;
  const double int_eyaw_raw = int_eyaw_ + eyaw * dt;
  int_ex_ = clamp(int_ex_raw, -config_.int_xy_limit, config_.int_xy_limit);
  int_ey_ = clamp(int_ey_raw, -config_.int_xy_limit, config_.int_xy_limit);
  int_ez_ = clamp(int_ez_raw, -config_.int_z_limit, config_.int_z_limit);
  int_eyaw_ = clamp(int_eyaw_raw, -config_.int_yaw_limit, config_.int_yaw_limit);

  std::array<double, 12> x_aug{
    ex, evx, ey, evy, ez, evz, eyaw, evyaw, int_ex_, int_ey_, int_ez_, int_eyaw_};
  std::array<double, 4> u_p{0.0, 0.0, 0.0, 0.0};
  std::array<double, 4> u_d{0.0, 0.0, 0.0, 0.0};
  std::array<double, 4> u_i{0.0, 0.0, 0.0, 0.0};
  std::array<double, 4> u_virtual{0.0, 0.0, 0.0, 0.0};
  std::array<double, 4> v_p{0.0, 0.0, 0.0, 0.0};
  std::array<double, 4> v_d{0.0, 0.0, 0.0, 0.0};
  std::array<double, 4> v_i{0.0, 0.0, 0.0, 0.0};
  std::array<double, 4> v_raw{0.0, 0.0, 0.0, 0.0};
  std::array<double, 4> v_trim{config_.trim_x, config_.trim_y, config_.trim_z, config_.trim_yaw};
  std::array<double, 4> v_total_unclamped{0.0, 0.0, 0.0, 0.0};
  geometry_msgs::msg::Twist cmd;

  auto recompute_with_current_integrators = [&]() {
      x_aug = {ex, evx, ey, evy, ez, evz, eyaw, evyaw, int_ex_, int_ey_, int_ez_, int_eyaw_};
      u_p = {0.0, 0.0, 0.0, 0.0};
      u_d = {0.0, 0.0, 0.0, 0.0};
      u_i = {0.0, 0.0, 0.0, 0.0};
      u_virtual = {0.0, 0.0, 0.0, 0.0};

      if (config_.k.size() == 48U) {
        for (std::size_t row = 0; row < 4; ++row) {
          const std::size_t base = row * 12U;
          const double up = (
            config_.k[base + 0U] * x_aug[0U] + config_.k[base + 2U] * x_aug[2U] +
            config_.k[base + 4U] * x_aug[4U] + config_.k[base + 6U] * x_aug[6U]);
          const double ud = (
            config_.k[base + 1U] * x_aug[1U] + config_.k[base + 3U] * x_aug[3U] +
            config_.k[base + 5U] * x_aug[5U] + config_.k[base + 7U] * x_aug[7U]);
          const double ui = (
            config_.k[base + 8U] * x_aug[8U] + config_.k[base + 9U] * x_aug[9U] +
            config_.k[base + 10U] * x_aug[10U] + config_.k[base + 11U] * x_aug[11U]);
          u_p[row] = config_.control_sign * up;
          u_d[row] = config_.control_sign * ud;
          u_i[row] = config_.control_sign * ui;
          u_virtual[row] = u_p[row] + u_d[row] + u_i[row];
        }
      }

      v_p = map_virtual_to_command(u_p, yaw);
      v_d = map_virtual_to_command(u_d, yaw);
      v_i = map_virtual_to_command(u_i, yaw);
      v_raw = map_virtual_to_command(u_virtual, yaw);
      for (std::size_t i = 0; i < 4; ++i) {
        v_total_unclamped[i] = v_raw[i] + v_trim[i];
      }

      cmd.linear.x = clamp(v_total_unclamped[0], -config_.cmd_xy_limit, config_.cmd_xy_limit);
      cmd.linear.y = clamp(v_total_unclamped[1], -config_.cmd_xy_limit, config_.cmd_xy_limit);
      cmd.linear.z = clamp(v_total_unclamped[2], -config_.cmd_z_limit, config_.cmd_z_limit);
      cmd.angular.z = clamp(v_total_unclamped[3], -config_.cmd_yaw_limit, config_.cmd_yaw_limit);
    };

  recompute_with_current_integrators();

  auto pushes_further_into_saturation = [](double cmd_unclamped, double limit, double error) {
      constexpr double kSatEps = 1e-9;
      const bool sat_high = (cmd_unclamped - limit) > kSatEps;
      const bool sat_low = (-limit - cmd_unclamped) > kSatEps;
      const bool pushes_further_high = sat_high && (error < 0.0);
      const bool pushes_further_low = sat_low && (error > 0.0);
      return pushes_further_high || pushes_further_low;
    };

  bool aw_x_integrator_blocked = false;
  bool aw_y_integrator_blocked = false;
  bool aw_z_integrator_blocked = false;
  bool aw_yaw_integrator_blocked = false;
  bool needs_recompute = false;

  if (config_.xy_conditional_anti_windup) {
    if (pushes_further_into_saturation(v_total_unclamped[0], config_.cmd_xy_limit, ex)) {
      int_ex_ = int_ex_prev;
      aw_x_integrator_blocked = true;
      needs_recompute = true;
    }
    if (pushes_further_into_saturation(v_total_unclamped[1], config_.cmd_xy_limit, ey)) {
      int_ey_ = int_ey_prev;
      aw_y_integrator_blocked = true;
      needs_recompute = true;
    }
  }
  if (config_.z_conditional_anti_windup &&
      pushes_further_into_saturation(v_total_unclamped[2], config_.cmd_z_limit, ez))
  {
    int_ez_ = int_ez_prev;
    aw_z_integrator_blocked = true;
    needs_recompute = true;
  }
  if (config_.yaw_conditional_anti_windup &&
      pushes_further_into_saturation(v_total_unclamped[3], config_.cmd_yaw_limit, eyaw))
  {
    int_eyaw_ = int_eyaw_prev;
    aw_yaw_integrator_blocked = true;
    needs_recompute = true;
  }

  if (needs_recompute) {
    recompute_with_current_integrators();
  }

  constexpr double kEps = 1e-9;
  const std::array<double, 4> u_trim{0.0, 0.0, 0.0, 0.0};
  last_debug_snapshot_.valid = true;
  last_debug_snapshot_.dt = dt;
  last_debug_snapshot_.x = x;
  last_debug_snapshot_.y = y;
  last_debug_snapshot_.z = z;
  last_debug_snapshot_.yaw = yaw;
  last_debug_snapshot_.vx = vx;
  last_debug_snapshot_.vy = vy;
  last_debug_snapshot_.vz = vz;
  last_debug_snapshot_.wyaw = wyaw;
  last_debug_snapshot_.x_ref = reference_msg.x;
  last_debug_snapshot_.y_ref = reference_msg.y;
  last_debug_snapshot_.z_ref = reference_msg.z;
  last_debug_snapshot_.yaw_ref = reference_msg.yaw;
  last_debug_snapshot_.ex = ex;
  last_debug_snapshot_.evx = evx;
  last_debug_snapshot_.ey = ey;
  last_debug_snapshot_.evy = evy;
  last_debug_snapshot_.ez = ez;
  last_debug_snapshot_.evz = evz;
  last_debug_snapshot_.eyaw = eyaw;
  last_debug_snapshot_.evyaw = evyaw;
  last_debug_snapshot_.int_ex_raw = int_ex_raw;
  last_debug_snapshot_.int_ey_raw = int_ey_raw;
  last_debug_snapshot_.int_ez_raw = int_ez_raw;
  last_debug_snapshot_.int_eyaw_raw = int_eyaw_raw;
  last_debug_snapshot_.int_ex = int_ex_;
  last_debug_snapshot_.int_ey = int_ey_;
  last_debug_snapshot_.int_ez = int_ez_;
  last_debug_snapshot_.int_eyaw = int_eyaw_;
  last_debug_snapshot_.x_aug = x_aug;
  last_debug_snapshot_.u_p = u_p;
  last_debug_snapshot_.u_d = u_d;
  last_debug_snapshot_.u_i = u_i;
  last_debug_snapshot_.u_trim = u_trim;
  last_debug_snapshot_.u_virtual = u_virtual;
  last_debug_snapshot_.v_p = v_p;
  last_debug_snapshot_.v_d = v_d;
  last_debug_snapshot_.v_i = v_i;
  last_debug_snapshot_.v_trim = v_trim;
  last_debug_snapshot_.v_raw = v_raw;
  last_debug_snapshot_.v_total_unclamped = v_total_unclamped;
  last_debug_snapshot_.cmd_clamped = {
    cmd.linear.x,
    cmd.linear.y,
    cmd.linear.z,
    cmd.angular.z
  };
  last_debug_snapshot_.sat_int_ex = std::abs(int_ex_raw - int_ex_) > kEps;
  last_debug_snapshot_.sat_int_ey = std::abs(int_ey_raw - int_ey_) > kEps;
  last_debug_snapshot_.sat_int_ez = std::abs(clamp(int_ez_raw, -config_.int_z_limit, config_.int_z_limit) - int_ez_raw) > kEps;
  last_debug_snapshot_.sat_int_eyaw = std::abs(int_eyaw_raw - int_eyaw_) > kEps;
  last_debug_snapshot_.sat_cmd_x = std::abs(v_total_unclamped[0] - cmd.linear.x) > kEps;
  last_debug_snapshot_.sat_cmd_y = std::abs(v_total_unclamped[1] - cmd.linear.y) > kEps;
  last_debug_snapshot_.sat_cmd_z = std::abs(v_total_unclamped[2] - cmd.linear.z) > kEps;
  last_debug_snapshot_.sat_cmd_yaw = std::abs(v_total_unclamped[3] - cmd.angular.z) > kEps;
  last_debug_snapshot_.aw_x_integrator_blocked = aw_x_integrator_blocked;
  last_debug_snapshot_.aw_y_integrator_blocked = aw_y_integrator_blocked;
  last_debug_snapshot_.aw_z_integrator_blocked = aw_z_integrator_blocked;
  last_debug_snapshot_.aw_yaw_integrator_blocked = aw_yaw_integrator_blocked;
  return cmd;
}

}  // namespace tello_control_pkg

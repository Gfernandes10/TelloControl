#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include "tello_control_pkg/controller_interface.hpp"
#include "tello_control_pkg/controllers/lqi_controller.hpp"
#include "tello_control_pkg/controllers/mpc_controller.hpp"
#include "tello_control_pkg/controllers/pid_controller.hpp"
#include "tello_control_pkg/msg/control_reference.hpp"
#include "tello_control_pkg/msg/control_status.hpp"

namespace tello_control_pkg
{

class TelloControlNode : public rclcpp::Node
{
public:
  TelloControlNode()
  : Node("tello_control_node")
  {
    this->declare_parameter<std::string>("controller_type", "pid");
    this->declare_parameter<std::string>("odom_topic", "/tello/filtered_pose");
    this->declare_parameter<std::string>("command_topic", "/tello/cmd_vel");
    this->declare_parameter<std::string>("reference_topic", "/tello_control/reference");
    this->declare_parameter<bool>("reference_yaw_in_degrees", true);
    this->declare_parameter<double>("yaw_command_sign", 1.0);
    this->declare_parameter<bool>("transform_world_xy_to_body", true);
    this->declare_parameter<double>("xy_command_sign_x", 1.0);
    this->declare_parameter<double>("xy_command_sign_y", 1.0);
    this->declare_parameter<bool>("xy_command_swap", false);
    this->declare_parameter<bool>("enable_control_xy", true);
    this->declare_parameter<bool>("enable_control_z", true);
    this->declare_parameter<bool>("enable_control_yaw", true);
    this->declare_parameter<std::string>("control_enable_service_name", "/tello_control/set_enabled");
    this->declare_parameter<std::string>("control_status_topic", "/tello_control/status");
    this->declare_parameter<bool>("start_enabled", false);
    this->declare_parameter<double>("odom_watchdog_timeout_sec", 0.3);
    this->declare_parameter<double>("odom_watchdog_check_period_sec", 0.05);
    this->declare_parameter<bool>("lqi.debug_csv_enabled", true);
    declare_pid_parameters();
    declare_lqi_parameters();

    this->get_parameter("controller_type", controller_type_);
    this->get_parameter("odom_topic", odom_topic_);
    this->get_parameter("command_topic", command_topic_);
    this->get_parameter("reference_topic", reference_topic_);
    this->get_parameter("reference_yaw_in_degrees", reference_yaw_in_degrees_);
    this->get_parameter("yaw_command_sign", yaw_command_sign_);
    this->get_parameter("transform_world_xy_to_body", transform_world_xy_to_body_);
    this->get_parameter("xy_command_sign_x", xy_command_sign_x_);
    this->get_parameter("xy_command_sign_y", xy_command_sign_y_);
    this->get_parameter("xy_command_swap", xy_command_swap_);
    this->get_parameter("enable_control_xy", enable_control_xy_);
    this->get_parameter("enable_control_z", enable_control_z_);
    this->get_parameter("enable_control_yaw", enable_control_yaw_);
    this->get_parameter("control_enable_service_name", control_enable_service_name_);
    this->get_parameter("control_status_topic", control_status_topic_);
    this->get_parameter("start_enabled", enabled_);
    this->get_parameter("odom_watchdog_timeout_sec", odom_watchdog_timeout_sec_);
    this->get_parameter("odom_watchdog_check_period_sec", odom_watchdog_check_period_sec_);
    this->get_parameter("lqi.debug_csv_enabled", lqi_debug_csv_enabled_);
    load_pid_parameters();
    load_lqi_parameters();
    normalize_lqi_mode(lqi_config_.mode);
    last_odom_rx_time_ = this->now();

    if (!is_valid_axis_command_sign(xy_command_sign_x_)) {
      RCLCPP_FATAL(get_logger(), "Invalid xy_command_sign_x=%.6f (must be finite and non-zero)", xy_command_sign_x_);
      throw std::runtime_error("invalid xy_command_sign_x");
    }
    if (!is_valid_axis_command_sign(xy_command_sign_y_)) {
      RCLCPP_FATAL(get_logger(), "Invalid xy_command_sign_y=%.6f (must be finite and non-zero)", xy_command_sign_y_);
      throw std::runtime_error("invalid xy_command_sign_y");
    }

    normalize_controller_type(controller_type_);
    controller_ = make_controller(controller_type_);
    if (!controller_) {
      RCLCPP_FATAL(
        get_logger(),
        "Invalid controller_type '%s'. Valid options are: pid, lqi, mpc.",
        controller_type_.c_str());
      throw std::runtime_error("invalid controller_type");
    }

    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(command_topic_, 10);
    control_status_pub_ = this->create_publisher<tello_control_pkg::msg::ControlStatus>(
      control_status_topic_,
      rclcpp::QoS(1).reliable().transient_local());
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 10, std::bind(&TelloControlNode::odom_callback, this, std::placeholders::_1));
    reference_sub_ = this->create_subscription<tello_control_pkg::msg::ControlReference>(
      reference_topic_, 10, std::bind(&TelloControlNode::reference_callback, this, std::placeholders::_1));
    control_enable_srv_ = this->create_service<std_srvs::srv::SetBool>(
      control_enable_service_name_,
      std::bind(
        &TelloControlNode::handle_set_enabled, this, std::placeholders::_1, std::placeholders::_2));
    odom_watchdog_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(odom_watchdog_check_period_sec_),
      std::bind(&TelloControlNode::check_odom_watchdog, this));

    parameter_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&TelloControlNode::on_parameters_set, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "tello_control_node started. controller_type=%s, enabled=%s, odom_topic=%s, command_topic=%s, "
      "reference_yaw_unit=%s, yaw_command_sign=%.3f, transform_world_xy_to_body=%s, "
      "xy_command_sign_x=%.3f, xy_command_sign_y=%.3f, xy_command_swap=%s, odom_watchdog_timeout_sec=%.3f",
      controller_->name().c_str(), enabled_ ? "true" : "false", odom_topic_.c_str(), command_topic_.c_str(),
      reference_yaw_in_degrees_ ? "deg" : "rad",
      yaw_command_sign_,
      transform_world_xy_to_body_ ? "true" : "false",
      xy_command_sign_x_,
      xy_command_sign_y_,
      xy_command_swap_ ? "true" : "false",
      odom_watchdog_timeout_sec_);
    RCLCPP_INFO(
      get_logger(),
      "Axis enable flags: XY=%s Z=%s YAW=%s",
      enable_control_xy_ ? "true" : "false",
      enable_control_z_ ? "true" : "false",
      enable_control_yaw_ ? "true" : "false");
    publish_control_status("startup");
    maybe_log_lqi_params_event("startup", lqi_config_);
  }

private:
  static void normalize_controller_type(std::string & value)
  {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
  }

  static void normalize_lqi_mode(std::string & value)
  {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
  }

  std::unique_ptr<ControllerInterface> make_controller(const std::string & controller_type) const
  {
    if (controller_type == "pid") {
      return std::make_unique<PIDController>(pid_config_);
    }
    if (controller_type == "lqi") {
      return std::make_unique<LQIController>(lqi_config_);
    }
    if (controller_type == "mpc") {
      return std::make_unique<MPCController>();
    }
    return nullptr;
  }

  rcl_interfaces::msg::SetParametersResult on_parameters_set(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "success";

    std::string next_controller_type = controller_type_;
    PIDConfig next_pid_config = pid_config_;
    LQIConfig next_lqi_config = lqi_config_;
    bool controller_type_changed = false;
    bool pid_config_changed = false;
    bool lqi_config_changed = false;
    bool reference_yaw_unit_changed = false;
    bool yaw_command_sign_changed = false;
    bool transform_world_xy_to_body_changed = false;
    bool xy_command_sign_x_changed = false;
    bool xy_command_sign_y_changed = false;
    bool xy_command_swap_changed = false;
    bool axis_enable_xy_changed = false;
    bool axis_enable_z_changed = false;
    bool axis_enable_yaw_changed = false;
    bool lqi_debug_csv_enabled_changed = false;
    bool next_reference_yaw_in_degrees = reference_yaw_in_degrees_;
    double next_yaw_command_sign = yaw_command_sign_;
    bool next_transform_world_xy_to_body = transform_world_xy_to_body_;
    double next_xy_command_sign_x = xy_command_sign_x_;
    double next_xy_command_sign_y = xy_command_sign_y_;
    bool next_xy_command_swap = xy_command_swap_;
    bool next_enable_control_xy = enable_control_xy_;
    bool next_enable_control_z = enable_control_z_;
    bool next_enable_control_yaw = enable_control_yaw_;
    bool next_lqi_debug_csv_enabled = lqi_debug_csv_enabled_;

    for (const auto & param : parameters) {
      if (param.get_name() == "controller_type") {
        std::string candidate = param.as_string();
        normalize_controller_type(candidate);
        if (!make_controller(candidate)) {
          result.successful = false;
          result.reason = "controller_type must be one of: pid, lqi, mpc";
          return result;
        }
        next_controller_type = candidate;
        controller_type_changed = true;
        continue;
      }
      if (param.get_name() == "reference_yaw_in_degrees") {
        next_reference_yaw_in_degrees = param.as_bool();
        reference_yaw_unit_changed = true;
        continue;
      }
      if (param.get_name() == "yaw_command_sign") {
        next_yaw_command_sign = param.as_double();
        yaw_command_sign_changed = true;
        continue;
      }
      if (param.get_name() == "transform_world_xy_to_body") {
        next_transform_world_xy_to_body = param.as_bool();
        transform_world_xy_to_body_changed = true;
        continue;
      }
      if (param.get_name() == "xy_command_sign_x") {
        next_xy_command_sign_x = param.as_double();
        xy_command_sign_x_changed = true;
        continue;
      }
      if (param.get_name() == "xy_command_sign_y") {
        next_xy_command_sign_y = param.as_double();
        xy_command_sign_y_changed = true;
        continue;
      }
      if (param.get_name() == "xy_command_swap") {
        next_xy_command_swap = param.as_bool();
        xy_command_swap_changed = true;
        continue;
      }
      if (param.get_name() == "enable_control_xy") {
        next_enable_control_xy = param.as_bool();
        axis_enable_xy_changed = true;
        continue;
      }
      if (param.get_name() == "enable_control_z") {
        next_enable_control_z = param.as_bool();
        axis_enable_z_changed = true;
        continue;
      }
      if (param.get_name() == "enable_control_yaw") {
        next_enable_control_yaw = param.as_bool();
        axis_enable_yaw_changed = true;
        continue;
      }
      if (param.get_name() == "lqi.debug_csv_enabled") {
        next_lqi_debug_csv_enabled = param.as_bool();
        lqi_debug_csv_enabled_changed = true;
        continue;
      }
      if (param.get_name() == "pid.kp_xy") { next_pid_config.kp_xy = param.as_double(); pid_config_changed = true; continue; }
      if (param.get_name() == "pid.ki_xy") { next_pid_config.ki_xy = param.as_double(); pid_config_changed = true; continue; }
      if (param.get_name() == "pid.kd_xy") { next_pid_config.kd_xy = param.as_double(); pid_config_changed = true; continue; }
      if (param.get_name() == "pid.kp_z") { next_pid_config.kp_z = param.as_double(); pid_config_changed = true; continue; }
      if (param.get_name() == "pid.ki_z") { next_pid_config.ki_z = param.as_double(); pid_config_changed = true; continue; }
      if (param.get_name() == "pid.kd_z") { next_pid_config.kd_z = param.as_double(); pid_config_changed = true; continue; }
      if (param.get_name() == "pid.kp_yaw") { next_pid_config.kp_yaw = param.as_double(); pid_config_changed = true; continue; }
      if (param.get_name() == "pid.ki_yaw") { next_pid_config.ki_yaw = param.as_double(); pid_config_changed = true; continue; }
      if (param.get_name() == "pid.kd_yaw") { next_pid_config.kd_yaw = param.as_double(); pid_config_changed = true; continue; }
      if (param.get_name() == "pid.int_xy_limit") { next_pid_config.int_xy_limit = param.as_double(); pid_config_changed = true; continue; }
      if (param.get_name() == "pid.int_z_limit") { next_pid_config.int_z_limit = param.as_double(); pid_config_changed = true; continue; }
      if (param.get_name() == "pid.int_yaw_limit") { next_pid_config.int_yaw_limit = param.as_double(); pid_config_changed = true; continue; }
      if (param.get_name() == "pid.cmd_xy_limit") { next_pid_config.cmd_xy_limit = param.as_double(); pid_config_changed = true; continue; }
      if (param.get_name() == "pid.cmd_z_limit") { next_pid_config.cmd_z_limit = param.as_double(); pid_config_changed = true; continue; }
      if (param.get_name() == "pid.cmd_yaw_limit") { next_pid_config.cmd_yaw_limit = param.as_double(); pid_config_changed = true; continue; }
      if (param.get_name() == "lqi.mode") {
        next_lqi_config.mode = param.as_string();
        normalize_lqi_mode(next_lqi_config.mode);
        lqi_config_changed = true;
        continue;
      }
      if (param.get_name() == "lqi.k") { next_lqi_config.k = param.as_double_array(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.gamma1") { next_lqi_config.gamma1 = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.gamma2") { next_lqi_config.gamma2 = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.gamma3") { next_lqi_config.gamma3 = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.gamma4") { next_lqi_config.gamma4 = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.gamma5") { next_lqi_config.gamma5 = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.gamma6") { next_lqi_config.gamma6 = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.gamma7") { next_lqi_config.gamma7 = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.gamma8") { next_lqi_config.gamma8 = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.design_dt") { next_lqi_config.design_dt = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.q_diag") { next_lqi_config.q_diag = param.as_double_array(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.r_diag") { next_lqi_config.r_diag = param.as_double_array(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.int_xy_limit") { next_lqi_config.int_xy_limit = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.int_z_limit") { next_lqi_config.int_z_limit = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.int_yaw_limit") { next_lqi_config.int_yaw_limit = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.cmd_xy_limit") { next_lqi_config.cmd_xy_limit = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.cmd_z_limit") { next_lqi_config.cmd_z_limit = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.cmd_yaw_limit") { next_lqi_config.cmd_yaw_limit = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.trim_x") { next_lqi_config.trim_x = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.trim_y") { next_lqi_config.trim_y = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.trim_z") { next_lqi_config.trim_z = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.trim_yaw") { next_lqi_config.trim_yaw = param.as_double(); lqi_config_changed = true; continue; }
      if (param.get_name() == "lqi.xy_conditional_anti_windup") {
        next_lqi_config.xy_conditional_anti_windup = param.as_bool();
        lqi_config_changed = true;
        continue;
      }
      if (param.get_name() == "lqi.z_conditional_anti_windup") {
        next_lqi_config.z_conditional_anti_windup = param.as_bool();
        lqi_config_changed = true;
        continue;
      }
      if (param.get_name() == "lqi.yaw_conditional_anti_windup") {
        next_lqi_config.yaw_conditional_anti_windup = param.as_bool();
        lqi_config_changed = true;
        continue;
      }
      if (param.get_name() == "lqi.control_sign") { next_lqi_config.control_sign = param.as_double(); lqi_config_changed = true; continue; }
    }

    if (!is_valid_pid_config(next_pid_config)) {
      result.successful = false;
      result.reason = "Invalid PID configuration: limits must be > 0 and gains must be >= 0";
      return result;
    }
    if (!is_valid_lqi_config(next_lqi_config)) {
      result.successful = false;
      result.reason = "Invalid LQI configuration";
      return result;
    }

    if (
      !controller_type_changed && !pid_config_changed && !lqi_config_changed &&
      !reference_yaw_unit_changed && !yaw_command_sign_changed &&
      !transform_world_xy_to_body_changed && !xy_command_sign_x_changed &&
      !xy_command_sign_y_changed && !xy_command_swap_changed &&
      !axis_enable_xy_changed && !axis_enable_z_changed && !axis_enable_yaw_changed &&
      !lqi_debug_csv_enabled_changed)
    {
      return result;
    }

    if (yaw_command_sign_changed && !is_valid_yaw_command_sign(next_yaw_command_sign)) {
      result.successful = false;
      result.reason = "yaw_command_sign must be finite and non-zero";
      return result;
    }
    if (xy_command_sign_x_changed && !is_valid_axis_command_sign(next_xy_command_sign_x)) {
      result.successful = false;
      result.reason = "xy_command_sign_x must be finite and non-zero";
      return result;
    }
    if (xy_command_sign_y_changed && !is_valid_axis_command_sign(next_xy_command_sign_y)) {
      result.successful = false;
      result.reason = "xy_command_sign_y must be finite and non-zero";
      return result;
    }

    {
      std::lock_guard<std::mutex> lock(controller_mutex_);
      const bool xy_enable_rising_edge =
        axis_enable_xy_changed && !enable_control_xy_ && next_enable_control_xy;
      const bool yaw_enable_rising_edge =
        axis_enable_yaw_changed && !enable_control_yaw_ && next_enable_control_yaw;

      if (controller_type_changed) {
        controller_type_ = next_controller_type;
        controller_ = make_controller(controller_type_);
        if (!controller_) {
          result.successful = false;
          result.reason = "Failed to instantiate requested controller";
          return result;
        }
      }

      if (pid_config_changed) {
        pid_config_ = next_pid_config;
        if (controller_type_ == "pid") {
          auto * pid_controller = dynamic_cast<PIDController *>(controller_.get());
          if (pid_controller) {
            pid_controller->set_config(pid_config_);
          }
        }
      }
      if (lqi_config_changed) {
        lqi_config_ = next_lqi_config;
        if (controller_type_ == "lqi") {
          auto * lqi_controller = dynamic_cast<LQIController *>(controller_.get());
          if (lqi_controller) {
            lqi_controller->set_config(lqi_config_);
          }
        }
      }
      if (reference_yaw_unit_changed) {
        reference_yaw_in_degrees_ = next_reference_yaw_in_degrees;
      }
      if (yaw_command_sign_changed) {
        yaw_command_sign_ = next_yaw_command_sign;
      }
      if (transform_world_xy_to_body_changed) {
        transform_world_xy_to_body_ = next_transform_world_xy_to_body;
      }
      if (xy_command_sign_x_changed) {
        xy_command_sign_x_ = next_xy_command_sign_x;
      }
      if (xy_command_sign_y_changed) {
        xy_command_sign_y_ = next_xy_command_sign_y;
      }
      if (xy_command_swap_changed) {
        xy_command_swap_ = next_xy_command_swap;
      }
      if (controller_type_ == "lqi") {
        auto * lqi_controller = dynamic_cast<LQIController *>(controller_.get());
        if (lqi_controller) {
          if (xy_enable_rising_edge) {
            lqi_controller->reset_xy_integrators();
          }
          if (yaw_enable_rising_edge) {
            lqi_controller->reset_yaw_integrator();
          }
        }
      }
      if (axis_enable_xy_changed) {
        enable_control_xy_ = next_enable_control_xy;
      }
      if (axis_enable_z_changed) {
        enable_control_z_ = next_enable_control_z;
      }
      if (axis_enable_yaw_changed) {
        enable_control_yaw_ = next_enable_control_yaw;
      }
      if (lqi_debug_csv_enabled_changed) {
        lqi_debug_csv_enabled_ = next_lqi_debug_csv_enabled;
      }
    }

    if (controller_type_changed) {
      RCLCPP_INFO(get_logger(), "Controller switched to: %s", controller_type_.c_str());
    }
    if (pid_config_changed) {
      RCLCPP_INFO(get_logger(), "PID gains/limits updated via parameters.");
    }
    if (lqi_config_changed) {
      RCLCPP_INFO(get_logger(), "LQI gains/model/limits updated via parameters.");
    }
    if (reference_yaw_unit_changed) {
      RCLCPP_INFO(
        get_logger(),
        "Reference yaw unit updated: %s",
        reference_yaw_in_degrees_ ? "degrees" : "radians");
    }
    if (yaw_command_sign_changed) {
      RCLCPP_INFO(get_logger(), "Yaw command sign updated: %.3f", yaw_command_sign_);
    }
    if (transform_world_xy_to_body_changed) {
      RCLCPP_INFO(
        get_logger(),
        "Transform world XY to body frame updated: %s",
        transform_world_xy_to_body_ ? "true" : "false");
    }
    if (xy_command_sign_x_changed) {
      RCLCPP_INFO(get_logger(), "XY command sign X updated: %.3f", xy_command_sign_x_);
    }
    if (xy_command_sign_y_changed) {
      RCLCPP_INFO(get_logger(), "XY command sign Y updated: %.3f", xy_command_sign_y_);
    }
    if (xy_command_swap_changed) {
      RCLCPP_INFO(
        get_logger(),
        "XY command swap updated: %s",
        xy_command_swap_ ? "true" : "false");
    }
    if (axis_enable_xy_changed) {
      RCLCPP_INFO(
        get_logger(),
        "Axis control enable updated: XY=%s",
        enable_control_xy_ ? "true" : "false");
      if (enable_control_xy_ && controller_type_ == "lqi") {
        RCLCPP_INFO(get_logger(), "LQI XY integrators reset on XY enable.");
      }
    }
    if (axis_enable_z_changed) {
      RCLCPP_INFO(
        get_logger(),
        "Axis control enable updated: Z=%s",
        enable_control_z_ ? "true" : "false");
    }
    if (axis_enable_yaw_changed) {
      RCLCPP_INFO(
        get_logger(),
        "Axis control enable updated: YAW=%s",
        enable_control_yaw_ ? "true" : "false");
      if (enable_control_yaw_ && controller_type_ == "lqi") {
        RCLCPP_INFO(get_logger(), "LQI yaw integrator reset on yaw enable.");
      }
    }
    if (lqi_debug_csv_enabled_changed) {
      RCLCPP_INFO(
        get_logger(),
        "LQI debug CSV logging updated: %s",
        lqi_debug_csv_enabled_ ? "true" : "false");
      if (!lqi_debug_csv_enabled_) {
        std::lock_guard<std::mutex> lock(lqi_debug_csv_mutex_);
        if (lqi_debug_csv_file_.is_open()) {
          lqi_debug_csv_file_.flush();
          lqi_debug_csv_file_.close();
        }
        if (lqi_params_events_csv_file_.is_open()) {
          lqi_params_events_csv_file_.flush();
          lqi_params_events_csv_file_.close();
        }
      }
    }
    if (
      lqi_config_changed || reference_yaw_unit_changed || yaw_command_sign_changed ||
      transform_world_xy_to_body_changed || xy_command_sign_x_changed ||
      xy_command_sign_y_changed || xy_command_swap_changed ||
      axis_enable_xy_changed || axis_enable_z_changed || axis_enable_yaw_changed)
    {
      maybe_log_lqi_params_event("param_update", lqi_config_);
    }

    return result;
  }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr odom_msg)
  {
    geometry_msgs::msg::Twist cmd;
    LQIDebugSnapshot lqi_debug_snapshot;
    bool has_lqi_debug_snapshot = false;
    {
      std::lock_guard<std::mutex> lock(controller_mutex_);
      last_odom_rx_time_ = this->now();
      odom_watchdog_tripped_ = false;
      last_odom_msg_ = *odom_msg;
      has_last_odom_ = true;
      if (!enabled_) {
        return;
      }
      if (!has_received_reference_) {
        set_reference_from_odom_locked(*odom_msg);
        RCLCPP_INFO_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "No external reference received yet. Holding current pose as reference.");
      }
      cmd = controller_->compute_command(*odom_msg, reference_msg_);

      const auto & q = odom_msg->pose.pose.orientation;
      const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
      const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
      const double yaw = std::atan2(siny_cosp, cosy_cosp);

      double cmd_x = cmd.linear.x;
      double cmd_y = cmd.linear.y;
      if (transform_world_xy_to_body_ && controller_type_ != "lqi") {
        const double cos_yaw = std::cos(yaw);
        const double sin_yaw = std::sin(yaw);
        const double cmd_body_x = cos_yaw * cmd_x + sin_yaw * cmd_y;
        const double cmd_body_y = -sin_yaw * cmd_x + cos_yaw * cmd_y;
        cmd_x = cmd_body_x;
        cmd_y = cmd_body_y;
      }
      if (xy_command_swap_) {
        std::swap(cmd_x, cmd_y);
      }
      cmd.linear.x = cmd_x * xy_command_sign_x_;
      cmd.linear.y = cmd_y * xy_command_sign_y_;
      cmd.angular.z *= yaw_command_sign_;
      if (!enable_control_xy_) {
        cmd.linear.x = 0.0;
        cmd.linear.y = 0.0;
      }
      if (!enable_control_z_) {
        cmd.linear.z = 0.0;
      }
      if (!enable_control_yaw_) {
        cmd.angular.z = 0.0;
      }

      if (controller_type_ == "lqi") {
        auto * lqi_controller = dynamic_cast<LQIController *>(controller_.get());
        if (lqi_controller) {
          lqi_debug_snapshot = lqi_controller->get_last_debug_snapshot();
          has_lqi_debug_snapshot = lqi_debug_snapshot.valid;
        }
      }
    }
    cmd_vel_pub_->publish(cmd);
    if (has_lqi_debug_snapshot) {
      maybe_log_lqi_debug_snapshot(lqi_debug_snapshot, cmd);
    }
  }

  void reference_callback(const tello_control_pkg::msg::ControlReference::SharedPtr msg)
  {
    constexpr double kDegToRad = 0.017453292519943295;
    std::lock_guard<std::mutex> lock(controller_mutex_);
    reference_msg_ = *msg;
    if (reference_yaw_in_degrees_) {
      reference_msg_.yaw *= kDegToRad;
    }
    has_received_reference_ = true;
  }

  void handle_set_enabled(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
    std::shared_ptr<std_srvs::srv::SetBool::Response> res)
  {
    set_controller_enabled(req->data, "service");
    res->success = true;
    res->message = req->data ? "controller enabled" : "controller disabled and reset";
  }

  void set_controller_enabled(bool enabled, const char * source)
  {
    bool changed_state = false;
    bool should_publish_zero = false;
    {
      std::lock_guard<std::mutex> lock(controller_mutex_);
      changed_state = (enabled_ != enabled);
      enabled_ = enabled;
      if (enabled_) {
        odom_watchdog_tripped_ = false;
        if (!has_received_reference_ && has_last_odom_) {
          set_reference_from_odom_locked(last_odom_msg_);
          RCLCPP_INFO(
            get_logger(),
            "Controller enabled without external reference. Using current odom pose as hold reference.");
        }
      }
      if (!enabled && controller_) {
        controller_->reset();
        should_publish_zero = true;
      }
    }
    if (changed_state) {
      RCLCPP_INFO(
        get_logger(),
        "Controller %s by %s.",
        enabled ? "enabled" : "disabled",
        source);
    }
    if (should_publish_zero) {
      cmd_vel_pub_->publish(geometry_msgs::msg::Twist());
    }
    publish_control_status(source);
  }

  void check_odom_watchdog()
  {
    bool should_trip = false;
    double elapsed_sec = 0.0;
    {
      std::lock_guard<std::mutex> lock(controller_mutex_);
      if (!enabled_ || odom_watchdog_tripped_) {
        return;
      }
      elapsed_sec = (this->now() - last_odom_rx_time_).seconds();
      if (elapsed_sec > odom_watchdog_timeout_sec_) {
        odom_watchdog_tripped_ = true;
        should_trip = true;
      }
    }

    if (should_trip) {
      RCLCPP_ERROR(
        get_logger(),
        "Odometry watchdog timeout (%.3fs > %.3fs). Disabling controller and sending zero command.",
        elapsed_sec,
        odom_watchdog_timeout_sec_);
      set_controller_enabled(false, "odom_watchdog");
    }
  }

  void publish_control_status(const std::string & reason)
  {
    tello_control_pkg::msg::ControlStatus status_msg;
    {
      std::lock_guard<std::mutex> lock(controller_mutex_);
      status_msg.enabled = enabled_;
    }
    status_msg.reason = reason;
    control_status_pub_->publish(status_msg);
  }

  bool ensure_lqi_debug_csv_open_locked()
  {
    if (lqi_debug_csv_file_.is_open()) {
      return true;
    }

    const char * home_dir = std::getenv("HOME");
    if (home_dir == nullptr) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Cannot open LQI debug CSV: HOME environment variable is not set.");
      return false;
    }

    const char * workspace_name = std::getenv("MY_WORKSPACE_NAME");
    std::string workspace_folder = "TelloControl";
    if (workspace_name != nullptr && std::string(workspace_name).size() > 0U) {
      workspace_folder = workspace_name;
    }

    std::time_t now = std::time(nullptr);
    std::tm now_tm{};
    std::tm * now_tm_ptr = std::localtime(&now);
    if (now_tm_ptr == nullptr) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Cannot open LQI debug CSV: failed to get local time.");
      return false;
    }
    now_tm = *now_tm_ptr;

    std::ostringstream timestamp;
    timestamp << std::put_time(&now_tm, "%Y%m%d_%H%M");
    const std::filesystem::path dir_path =
      std::filesystem::path(home_dir) / workspace_folder / "csvLogs" / timestamp.str() / "tello_control_pkg";

    std::error_code ec;
    std::filesystem::create_directories(dir_path, ec);
    if (ec) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Cannot create LQI debug CSV directory '%s': %s",
        dir_path.string().c_str(),
        ec.message().c_str());
      return false;
    }

    lqi_debug_csv_path_ = (dir_path / "lqi_debug.csv").string();
    lqi_debug_csv_file_.open(lqi_debug_csv_path_, std::ios::out | std::ios::trunc);
    if (!lqi_debug_csv_file_.is_open()) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Cannot open LQI debug CSV file '%s'",
        lqi_debug_csv_path_.c_str());
      return false;
    }

    lqi_debug_csv_file_
      << ",timestamp"
      << ",topic"
      << ",controller_type"
      << ",dt"
      << ",x,dx,y,dy,z,dz,yaw,wyaw"
      << ",x_ref,y_ref,z_ref,yaw_ref"
      << ",ex,evx,ey,evy,ez,evz,eyaw,evyaw"
      << ",int_ex_raw,int_ey_raw,int_ez_raw,int_eyaw_raw"
      << ",int_ex,int_ey,int_ez,int_eyaw"
      << ",x_aug_0,x_aug_1,x_aug_2,x_aug_3,x_aug_4,x_aug_5,x_aug_6,x_aug_7,x_aug_8,x_aug_9,x_aug_10,x_aug_11"
      << ",u_virtual_x,u_virtual_y,u_virtual_z,u_virtual_yaw"
      << ",u_p_x,u_p_y,u_p_z,u_p_yaw"
      << ",u_d_x,u_d_y,u_d_z,u_d_yaw"
      << ",u_i_x,u_i_y,u_i_z,u_i_yaw"
      << ",u_trim_x,u_trim_y,u_trim_z,u_trim_yaw"
      << ",v_raw_x,v_raw_y,v_raw_z,v_raw_yaw"
      << ",v_p_x,v_p_y,v_p_z,v_p_yaw"
      << ",v_d_x,v_d_y,v_d_z,v_d_yaw"
      << ",v_i_x,v_i_y,v_i_z,v_i_yaw"
      << ",v_trim_x,v_trim_y,v_trim_z,v_trim_yaw"
      << ",v_total_unclamped_x,v_total_unclamped_y,v_total_unclamped_z,v_total_unclamped_yaw"
      << ",cmd_controller_x,cmd_controller_y,cmd_controller_z,cmd_controller_yaw"
      << ",cmd_published_x,cmd_published_y,cmd_published_z,cmd_published_yaw"
      << ",sat_int_ex,sat_int_ey,sat_int_ez,sat_int_eyaw"
      << ",sat_cmd_x,sat_cmd_y,sat_cmd_z,sat_cmd_yaw"
      << ",aw_x_integrator_blocked,aw_y_integrator_blocked,aw_z_integrator_blocked,aw_yaw_integrator_blocked"
      << ",enable_control_xy,enable_control_z,enable_control_yaw"
      << ",reference_yaw_in_degrees,yaw_command_sign,transform_world_xy_to_body,xy_command_sign_x,xy_command_sign_y,xy_command_swap"
      << ",lqi_cmd_xy_limit,lqi_cmd_z_limit,lqi_cmd_yaw_limit"
      << ",lqi_int_xy_limit,lqi_int_z_limit,lqi_int_yaw_limit"
      << ",lqi_trim_x,lqi_trim_y,lqi_trim_z,lqi_trim_yaw"
      << ",lqi_xy_conditional_anti_windup,lqi_z_conditional_anti_windup,lqi_yaw_conditional_anti_windup"
      << ",lqi_control_sign"
      << ",lqi_gamma1,lqi_gamma2,lqi_gamma3,lqi_gamma4,lqi_gamma5,lqi_gamma6,lqi_gamma7,lqi_gamma8"
      << "\n";
    lqi_debug_csv_file_.flush();

    RCLCPP_INFO(get_logger(), "LQI debug CSV logging enabled: %s", lqi_debug_csv_path_.c_str());
    return true;
  }

  bool ensure_lqi_params_events_csv_open_locked()
  {
    if (lqi_params_events_csv_file_.is_open()) {
      return true;
    }

    const char * home_dir = std::getenv("HOME");
    if (home_dir == nullptr) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Cannot open LQI params-events CSV: HOME environment variable is not set.");
      return false;
    }

    const char * workspace_name = std::getenv("MY_WORKSPACE_NAME");
    std::string workspace_folder = "TelloControl";
    if (workspace_name != nullptr && std::string(workspace_name).size() > 0U) {
      workspace_folder = workspace_name;
    }

    std::time_t now = std::time(nullptr);
    std::tm now_tm{};
    std::tm * now_tm_ptr = std::localtime(&now);
    if (now_tm_ptr == nullptr) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Cannot open LQI params-events CSV: failed to get local time.");
      return false;
    }
    now_tm = *now_tm_ptr;

    std::ostringstream timestamp;
    timestamp << std::put_time(&now_tm, "%Y%m%d_%H%M");
    const std::filesystem::path dir_path =
      std::filesystem::path(home_dir) / workspace_folder / "csvLogs" / timestamp.str() / "tello_control_pkg";

    std::error_code ec;
    std::filesystem::create_directories(dir_path, ec);
    if (ec) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Cannot create LQI params-events CSV directory '%s': %s",
        dir_path.string().c_str(),
        ec.message().c_str());
      return false;
    }

    lqi_params_events_csv_path_ = (dir_path / "lqi_params_events.csv").string();
    lqi_params_events_csv_file_.open(lqi_params_events_csv_path_, std::ios::out | std::ios::trunc);
    if (!lqi_params_events_csv_file_.is_open()) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Cannot open LQI params-events CSV file '%s'",
        lqi_params_events_csv_path_.c_str());
      return false;
    }

    lqi_params_events_csv_file_
      << ",timestamp"
      << ",event"
      << ",controller_type"
      << ",mode"
      << ",design_dt"
      << ",control_sign"
      << ",reference_yaw_in_degrees,yaw_command_sign,transform_world_xy_to_body,xy_command_sign_x,xy_command_sign_y,xy_command_swap"
      << ",enable_control_xy,enable_control_z,enable_control_yaw"
      << ",gamma1,gamma2,gamma3,gamma4,gamma5,gamma6,gamma7,gamma8"
      << ",int_xy_limit,int_z_limit,int_yaw_limit"
      << ",cmd_xy_limit,cmd_z_limit,cmd_yaw_limit"
      << ",trim_x,trim_y,trim_z,trim_yaw"
      << ",xy_conditional_anti_windup,z_conditional_anti_windup,yaw_conditional_anti_windup"
      << ",q_diag_0,q_diag_1,q_diag_2,q_diag_3,q_diag_4,q_diag_5,q_diag_6,q_diag_7,q_diag_8,q_diag_9,q_diag_10,q_diag_11"
      << ",r_diag_0,r_diag_1,r_diag_2,r_diag_3"
      << ",k_0,k_1,k_2,k_3,k_4,k_5,k_6,k_7,k_8,k_9,k_10,k_11"
      << ",k_12,k_13,k_14,k_15,k_16,k_17,k_18,k_19,k_20,k_21,k_22,k_23"
      << ",k_24,k_25,k_26,k_27,k_28,k_29,k_30,k_31,k_32,k_33,k_34,k_35"
      << ",k_36,k_37,k_38,k_39,k_40,k_41,k_42,k_43,k_44,k_45,k_46,k_47"
      << "\n";
    lqi_params_events_csv_file_.flush();

    RCLCPP_INFO(get_logger(), "LQI params-events CSV logging enabled: %s", lqi_params_events_csv_path_.c_str());
    return true;
  }

  void maybe_log_lqi_params_event(const std::string & event_name, const LQIConfig & cfg)
  {
    if (!lqi_debug_csv_enabled_) {
      return;
    }

    std::lock_guard<std::mutex> lock(lqi_debug_csv_mutex_);
    if (!ensure_lqi_params_events_csv_open_locked()) {
      return;
    }

    lqi_params_events_csv_file_ << ","
                                << std::fixed << std::setprecision(6) << this->now().seconds()
                                << "," << event_name
                                << "," << controller_type_
                                << "," << cfg.mode
                                << "," << cfg.design_dt
                                << "," << cfg.control_sign
                                << "," << (reference_yaw_in_degrees_ ? 1 : 0)
                                << "," << yaw_command_sign_
                                << "," << (transform_world_xy_to_body_ ? 1 : 0)
                                << "," << xy_command_sign_x_
                                << "," << xy_command_sign_y_
                                << "," << (xy_command_swap_ ? 1 : 0)
                                << "," << (enable_control_xy_ ? 1 : 0)
                                << "," << (enable_control_z_ ? 1 : 0)
                                << "," << (enable_control_yaw_ ? 1 : 0)
                                << "," << cfg.gamma1
                                << "," << cfg.gamma2
                                << "," << cfg.gamma3
                                << "," << cfg.gamma4
                                << "," << cfg.gamma5
                                << "," << cfg.gamma6
                                << "," << cfg.gamma7
                                << "," << cfg.gamma8
                                << "," << cfg.int_xy_limit
                                << "," << cfg.int_z_limit
                                << "," << cfg.int_yaw_limit
                                << "," << cfg.cmd_xy_limit
                                << "," << cfg.cmd_z_limit
                                << "," << cfg.cmd_yaw_limit
                                << "," << cfg.trim_x
                                << "," << cfg.trim_y
                                << "," << cfg.trim_z
                                << "," << cfg.trim_yaw
                                << "," << (cfg.xy_conditional_anti_windup ? 1 : 0)
                                << "," << (cfg.z_conditional_anti_windup ? 1 : 0)
                                << "," << (cfg.yaw_conditional_anti_windup ? 1 : 0);

    for (std::size_t i = 0; i < 12; ++i) {
      lqi_params_events_csv_file_ << "," << (i < cfg.q_diag.size() ? cfg.q_diag[i] : 0.0);
    }
    for (std::size_t i = 0; i < 4; ++i) {
      lqi_params_events_csv_file_ << "," << (i < cfg.r_diag.size() ? cfg.r_diag[i] : 0.0);
    }
    for (std::size_t i = 0; i < 48; ++i) {
      lqi_params_events_csv_file_ << "," << (i < cfg.k.size() ? cfg.k[i] : 0.0);
    }

    lqi_params_events_csv_file_ << "\n";
    lqi_params_events_csv_file_.flush();
  }

  void maybe_log_lqi_debug_snapshot(
    const LQIDebugSnapshot & snapshot,
    const geometry_msgs::msg::Twist & cmd_published)
  {
    if (!lqi_debug_csv_enabled_) {
      return;
    }
    if (!snapshot.valid) {
      return;
    }

    std::lock_guard<std::mutex> lock(lqi_debug_csv_mutex_);
    if (!ensure_lqi_debug_csv_open_locked()) {
      return;
    }

    lqi_debug_csv_file_ << ","
                        << std::fixed << std::setprecision(6) << this->now().seconds()
                        << "," << command_topic_
                        << "," << controller_type_
                        << "," << snapshot.dt
                        << "," << snapshot.x
                        << "," << snapshot.vx
                        << "," << snapshot.y
                        << "," << snapshot.vy
                        << "," << snapshot.z
                        << "," << snapshot.vz
                        << "," << snapshot.yaw
                        << "," << snapshot.wyaw
                        << "," << snapshot.x_ref
                        << "," << snapshot.y_ref
                        << "," << snapshot.z_ref
                        << "," << snapshot.yaw_ref
                        << "," << snapshot.ex
                        << "," << snapshot.evx
                        << "," << snapshot.ey
                        << "," << snapshot.evy
                        << "," << snapshot.ez
                        << "," << snapshot.evz
                        << "," << snapshot.eyaw
                        << "," << snapshot.evyaw
                        << "," << snapshot.int_ex_raw
                        << "," << snapshot.int_ey_raw
                        << "," << snapshot.int_ez_raw
                        << "," << snapshot.int_eyaw_raw
                        << "," << snapshot.int_ex
                        << "," << snapshot.int_ey
                        << "," << snapshot.int_ez
                        << "," << snapshot.int_eyaw;

    for (double value : snapshot.x_aug) {
      lqi_debug_csv_file_ << "," << value;
    }
    for (double value : snapshot.u_virtual) {
      lqi_debug_csv_file_ << "," << value;
    }
    for (double value : snapshot.u_p) {
      lqi_debug_csv_file_ << "," << value;
    }
    for (double value : snapshot.u_d) {
      lqi_debug_csv_file_ << "," << value;
    }
    for (double value : snapshot.u_i) {
      lqi_debug_csv_file_ << "," << value;
    }
    for (double value : snapshot.u_trim) {
      lqi_debug_csv_file_ << "," << value;
    }
    for (double value : snapshot.v_raw) {
      lqi_debug_csv_file_ << "," << value;
    }
    for (double value : snapshot.v_p) {
      lqi_debug_csv_file_ << "," << value;
    }
    for (double value : snapshot.v_d) {
      lqi_debug_csv_file_ << "," << value;
    }
    for (double value : snapshot.v_i) {
      lqi_debug_csv_file_ << "," << value;
    }
    for (double value : snapshot.v_trim) {
      lqi_debug_csv_file_ << "," << value;
    }
    for (double value : snapshot.v_total_unclamped) {
      lqi_debug_csv_file_ << "," << value;
    }
    for (double value : snapshot.cmd_clamped) {
      lqi_debug_csv_file_ << "," << value;
    }

    lqi_debug_csv_file_ << "," << cmd_published.linear.x
                        << "," << cmd_published.linear.y
                        << "," << cmd_published.linear.z
                        << "," << cmd_published.angular.z
                        << "," << (snapshot.sat_int_ex ? 1 : 0)
                        << "," << (snapshot.sat_int_ey ? 1 : 0)
                        << "," << (snapshot.sat_int_ez ? 1 : 0)
                        << "," << (snapshot.sat_int_eyaw ? 1 : 0)
                        << "," << (snapshot.sat_cmd_x ? 1 : 0)
                        << "," << (snapshot.sat_cmd_y ? 1 : 0)
                        << "," << (snapshot.sat_cmd_z ? 1 : 0)
                        << "," << (snapshot.sat_cmd_yaw ? 1 : 0)
                        << "," << (snapshot.aw_x_integrator_blocked ? 1 : 0)
                        << "," << (snapshot.aw_y_integrator_blocked ? 1 : 0)
                        << "," << (snapshot.aw_z_integrator_blocked ? 1 : 0)
                        << "," << (snapshot.aw_yaw_integrator_blocked ? 1 : 0)
                        << "," << (enable_control_xy_ ? 1 : 0)
                        << "," << (enable_control_z_ ? 1 : 0)
                        << "," << (enable_control_yaw_ ? 1 : 0)
                        << "," << (reference_yaw_in_degrees_ ? 1 : 0)
                        << "," << yaw_command_sign_
                        << "," << (transform_world_xy_to_body_ ? 1 : 0)
                        << "," << xy_command_sign_x_
                        << "," << xy_command_sign_y_
                        << "," << (xy_command_swap_ ? 1 : 0)
                        << "," << lqi_config_.cmd_xy_limit
                        << "," << lqi_config_.cmd_z_limit
                        << "," << lqi_config_.cmd_yaw_limit
                        << "," << lqi_config_.int_xy_limit
                        << "," << lqi_config_.int_z_limit
                        << "," << lqi_config_.int_yaw_limit
                        << "," << lqi_config_.trim_x
                        << "," << lqi_config_.trim_y
                        << "," << lqi_config_.trim_z
                        << "," << lqi_config_.trim_yaw
                        << "," << (lqi_config_.xy_conditional_anti_windup ? 1 : 0)
                        << "," << (lqi_config_.z_conditional_anti_windup ? 1 : 0)
                        << "," << (lqi_config_.yaw_conditional_anti_windup ? 1 : 0)
                        << "," << lqi_config_.control_sign
                        << "," << lqi_config_.gamma1
                        << "," << lqi_config_.gamma2
                        << "," << lqi_config_.gamma3
                        << "," << lqi_config_.gamma4
                        << "," << lqi_config_.gamma5
                        << "," << lqi_config_.gamma6
                        << "," << lqi_config_.gamma7
                        << "," << lqi_config_.gamma8
                        << "\n";
    lqi_debug_csv_file_.flush();
  }

  void declare_pid_parameters()
  {
    this->declare_parameter<double>("pid.kp_xy", pid_config_.kp_xy);
    this->declare_parameter<double>("pid.ki_xy", pid_config_.ki_xy);
    this->declare_parameter<double>("pid.kd_xy", pid_config_.kd_xy);
    this->declare_parameter<double>("pid.kp_z", pid_config_.kp_z);
    this->declare_parameter<double>("pid.ki_z", pid_config_.ki_z);
    this->declare_parameter<double>("pid.kd_z", pid_config_.kd_z);
    this->declare_parameter<double>("pid.kp_yaw", pid_config_.kp_yaw);
    this->declare_parameter<double>("pid.ki_yaw", pid_config_.ki_yaw);
    this->declare_parameter<double>("pid.kd_yaw", pid_config_.kd_yaw);
    this->declare_parameter<double>("pid.int_xy_limit", pid_config_.int_xy_limit);
    this->declare_parameter<double>("pid.int_z_limit", pid_config_.int_z_limit);
    this->declare_parameter<double>("pid.int_yaw_limit", pid_config_.int_yaw_limit);
    this->declare_parameter<double>("pid.cmd_xy_limit", pid_config_.cmd_xy_limit);
    this->declare_parameter<double>("pid.cmd_z_limit", pid_config_.cmd_z_limit);
    this->declare_parameter<double>("pid.cmd_yaw_limit", pid_config_.cmd_yaw_limit);
  }

  void declare_lqi_parameters()
  {
    this->declare_parameter<std::string>("lqi.mode", lqi_config_.mode);
    this->declare_parameter<std::vector<double>>("lqi.k", lqi_config_.k);
    this->declare_parameter<double>("lqi.gamma1", lqi_config_.gamma1);
    this->declare_parameter<double>("lqi.gamma2", lqi_config_.gamma2);
    this->declare_parameter<double>("lqi.gamma3", lqi_config_.gamma3);
    this->declare_parameter<double>("lqi.gamma4", lqi_config_.gamma4);
    this->declare_parameter<double>("lqi.gamma5", lqi_config_.gamma5);
    this->declare_parameter<double>("lqi.gamma6", lqi_config_.gamma6);
    this->declare_parameter<double>("lqi.gamma7", lqi_config_.gamma7);
    this->declare_parameter<double>("lqi.gamma8", lqi_config_.gamma8);
    this->declare_parameter<double>("lqi.design_dt", lqi_config_.design_dt);
    this->declare_parameter<std::vector<double>>("lqi.q_diag", lqi_config_.q_diag);
    this->declare_parameter<std::vector<double>>("lqi.r_diag", lqi_config_.r_diag);
    this->declare_parameter<double>("lqi.int_xy_limit", lqi_config_.int_xy_limit);
    this->declare_parameter<double>("lqi.int_z_limit", lqi_config_.int_z_limit);
    this->declare_parameter<double>("lqi.int_yaw_limit", lqi_config_.int_yaw_limit);
    this->declare_parameter<double>("lqi.cmd_xy_limit", lqi_config_.cmd_xy_limit);
    this->declare_parameter<double>("lqi.cmd_z_limit", lqi_config_.cmd_z_limit);
    this->declare_parameter<double>("lqi.cmd_yaw_limit", lqi_config_.cmd_yaw_limit);
    this->declare_parameter<double>("lqi.trim_x", lqi_config_.trim_x);
    this->declare_parameter<double>("lqi.trim_y", lqi_config_.trim_y);
    this->declare_parameter<double>("lqi.trim_z", lqi_config_.trim_z);
    this->declare_parameter<double>("lqi.trim_yaw", lqi_config_.trim_yaw);
    this->declare_parameter<bool>(
      "lqi.xy_conditional_anti_windup",
      lqi_config_.xy_conditional_anti_windup);
    this->declare_parameter<bool>(
      "lqi.z_conditional_anti_windup",
      lqi_config_.z_conditional_anti_windup);
    this->declare_parameter<bool>(
      "lqi.yaw_conditional_anti_windup",
      lqi_config_.yaw_conditional_anti_windup);
    this->declare_parameter<double>("lqi.control_sign", lqi_config_.control_sign);
  }

  void load_pid_parameters()
  {
    this->get_parameter("pid.kp_xy", pid_config_.kp_xy);
    this->get_parameter("pid.ki_xy", pid_config_.ki_xy);
    this->get_parameter("pid.kd_xy", pid_config_.kd_xy);
    this->get_parameter("pid.kp_z", pid_config_.kp_z);
    this->get_parameter("pid.ki_z", pid_config_.ki_z);
    this->get_parameter("pid.kd_z", pid_config_.kd_z);
    this->get_parameter("pid.kp_yaw", pid_config_.kp_yaw);
    this->get_parameter("pid.ki_yaw", pid_config_.ki_yaw);
    this->get_parameter("pid.kd_yaw", pid_config_.kd_yaw);
    this->get_parameter("pid.int_xy_limit", pid_config_.int_xy_limit);
    this->get_parameter("pid.int_z_limit", pid_config_.int_z_limit);
    this->get_parameter("pid.int_yaw_limit", pid_config_.int_yaw_limit);
    this->get_parameter("pid.cmd_xy_limit", pid_config_.cmd_xy_limit);
    this->get_parameter("pid.cmd_z_limit", pid_config_.cmd_z_limit);
    this->get_parameter("pid.cmd_yaw_limit", pid_config_.cmd_yaw_limit);
  }

  void load_lqi_parameters()
  {
    this->get_parameter("lqi.mode", lqi_config_.mode);
    this->get_parameter("lqi.k", lqi_config_.k);
    this->get_parameter("lqi.gamma1", lqi_config_.gamma1);
    this->get_parameter("lqi.gamma2", lqi_config_.gamma2);
    this->get_parameter("lqi.gamma3", lqi_config_.gamma3);
    this->get_parameter("lqi.gamma4", lqi_config_.gamma4);
    this->get_parameter("lqi.gamma5", lqi_config_.gamma5);
    this->get_parameter("lqi.gamma6", lqi_config_.gamma6);
    this->get_parameter("lqi.gamma7", lqi_config_.gamma7);
    this->get_parameter("lqi.gamma8", lqi_config_.gamma8);
    this->get_parameter("lqi.design_dt", lqi_config_.design_dt);
    this->get_parameter("lqi.q_diag", lqi_config_.q_diag);
    this->get_parameter("lqi.r_diag", lqi_config_.r_diag);
    this->get_parameter("lqi.int_xy_limit", lqi_config_.int_xy_limit);
    this->get_parameter("lqi.int_z_limit", lqi_config_.int_z_limit);
    this->get_parameter("lqi.int_yaw_limit", lqi_config_.int_yaw_limit);
    this->get_parameter("lqi.cmd_xy_limit", lqi_config_.cmd_xy_limit);
    this->get_parameter("lqi.cmd_z_limit", lqi_config_.cmd_z_limit);
    this->get_parameter("lqi.cmd_yaw_limit", lqi_config_.cmd_yaw_limit);
    this->get_parameter("lqi.trim_x", lqi_config_.trim_x);
    this->get_parameter("lqi.trim_y", lqi_config_.trim_y);
    this->get_parameter("lqi.trim_z", lqi_config_.trim_z);
    this->get_parameter("lqi.trim_yaw", lqi_config_.trim_yaw);
    this->get_parameter(
      "lqi.xy_conditional_anti_windup",
      lqi_config_.xy_conditional_anti_windup);
    this->get_parameter(
      "lqi.z_conditional_anti_windup",
      lqi_config_.z_conditional_anti_windup);
    this->get_parameter(
      "lqi.yaw_conditional_anti_windup",
      lqi_config_.yaw_conditional_anti_windup);
    this->get_parameter("lqi.control_sign", lqi_config_.control_sign);
  }

  static bool is_valid_pid_config(const PIDConfig & cfg)
  {
    const bool gains_ok =
      cfg.kp_xy >= 0.0 && cfg.ki_xy >= 0.0 && cfg.kd_xy >= 0.0 &&
      cfg.kp_z >= 0.0 && cfg.ki_z >= 0.0 && cfg.kd_z >= 0.0 &&
      cfg.kp_yaw >= 0.0 && cfg.ki_yaw >= 0.0 && cfg.kd_yaw >= 0.0;
    const bool limits_ok =
      cfg.int_xy_limit > 0.0 && cfg.int_z_limit > 0.0 && cfg.int_yaw_limit > 0.0 &&
      cfg.cmd_xy_limit > 0.0 && cfg.cmd_z_limit > 0.0 && cfg.cmd_yaw_limit > 0.0;
    return gains_ok && limits_ok;
  }

  static bool is_valid_lqi_config(const LQIConfig & cfg)
  {
    const bool mode_ok = (cfg.mode == "manual" || cfg.mode == "synthesize");
    if (!mode_ok) {
      return false;
    }

    const auto all_finite = [](const std::vector<double> & values) {
      return std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); });
    };

    if (
      !std::isfinite(cfg.gamma1) || !std::isfinite(cfg.gamma2) ||
      !std::isfinite(cfg.gamma3) || !std::isfinite(cfg.gamma4) ||
      !std::isfinite(cfg.gamma5) || !std::isfinite(cfg.gamma6) ||
      !std::isfinite(cfg.gamma7) || !std::isfinite(cfg.gamma8))
    {
      return false;
    }
    if (cfg.design_dt <= 0.0 || !std::isfinite(cfg.design_dt)) {
      return false;
    }
    if (cfg.q_diag.size() != 12 || !all_finite(cfg.q_diag)) {
      return false;
    }
    if (std::any_of(cfg.q_diag.begin(), cfg.q_diag.end(), [](double q) { return q < 0.0; })) {
      return false;
    }
    if (cfg.r_diag.size() != 4 || !all_finite(cfg.r_diag)) {
      return false;
    }
    if (std::any_of(cfg.r_diag.begin(), cfg.r_diag.end(), [](double r) { return r <= 0.0; })) {
      return false;
    }
    const bool limits_ok =
      std::isfinite(cfg.int_xy_limit) && cfg.int_xy_limit > 0.0 &&
      std::isfinite(cfg.int_z_limit) && cfg.int_z_limit > 0.0 &&
      std::isfinite(cfg.int_yaw_limit) && cfg.int_yaw_limit > 0.0 &&
      std::isfinite(cfg.cmd_xy_limit) && cfg.cmd_xy_limit > 0.0 &&
      std::isfinite(cfg.cmd_z_limit) && cfg.cmd_z_limit > 0.0 &&
      std::isfinite(cfg.cmd_yaw_limit) && cfg.cmd_yaw_limit > 0.0 &&
      std::isfinite(cfg.trim_x) &&
      std::isfinite(cfg.trim_y) &&
      std::isfinite(cfg.trim_z) &&
      std::isfinite(cfg.trim_yaw) &&
      std::isfinite(cfg.control_sign) && std::abs(cfg.control_sign) > 1e-9;
    if (!limits_ok) {
      return false;
    }

    if (cfg.mode == "manual") {
      if (cfg.k.size() != 48 || !all_finite(cfg.k)) {
        return false;
      }
    }
    return true;
  }

  static bool is_valid_yaw_command_sign(double value)
  {
    return std::isfinite(value) && std::abs(value) > 1e-9;
  }

  static bool is_valid_axis_command_sign(double value)
  {
    return std::isfinite(value) && std::abs(value) > 1e-9;
  }

  void set_reference_from_odom_locked(const nav_msgs::msg::Odometry & odom_msg)
  {
    const auto & pose = odom_msg.pose.pose;
    reference_msg_.x = pose.position.x;
    reference_msg_.y = pose.position.y;
    reference_msg_.z = pose.position.z;

    const auto & q = pose.orientation;
    const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    reference_msg_.yaw = std::atan2(siny_cosp, cosy_cosp);
  }

  std::string controller_type_;
  std::string odom_topic_;
  std::string command_topic_;
  std::string reference_topic_;
  bool reference_yaw_in_degrees_{true};
  double yaw_command_sign_{1.0};
  bool transform_world_xy_to_body_{true};
  double xy_command_sign_x_{1.0};
  double xy_command_sign_y_{1.0};
  bool xy_command_swap_{false};
  bool enable_control_xy_{true};
  bool enable_control_z_{true};
  bool enable_control_yaw_{true};
  std::string control_enable_service_name_;
  std::string control_status_topic_;
  bool enabled_{true};
  double odom_watchdog_timeout_sec_{0.3};
  double odom_watchdog_check_period_sec_{0.05};
  bool lqi_debug_csv_enabled_{true};
  bool odom_watchdog_tripped_{false};
  PIDConfig pid_config_{};
  LQIConfig lqi_config_{};
  std::mutex controller_mutex_;
  tello_control_pkg::msg::ControlReference reference_msg_;
  nav_msgs::msg::Odometry last_odom_msg_;
  bool has_last_odom_{false};
  bool has_received_reference_{false};
  rclcpp::Time last_odom_rx_time_;

  std::unique_ptr<ControllerInterface> controller_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<tello_control_pkg::msg::ControlStatus>::SharedPtr control_status_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<tello_control_pkg::msg::ControlReference>::SharedPtr reference_sub_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr control_enable_srv_;
  rclcpp::TimerBase::SharedPtr odom_watchdog_timer_;
  OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;
  std::mutex lqi_debug_csv_mutex_;
  std::ofstream lqi_debug_csv_file_;
  std::string lqi_debug_csv_path_;
  std::ofstream lqi_params_events_csv_file_;
  std::string lqi_params_events_csv_path_;
};

}  // namespace tello_control_pkg

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<tello_control_pkg::TelloControlNode>());
  rclcpp::shutdown();
  return 0;
}

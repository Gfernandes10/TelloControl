from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os


def generate_launch_description():
    controller_type = LaunchConfiguration('controller_type')
    odom_topic = LaunchConfiguration('odom_topic')
    command_topic = LaunchConfiguration('command_topic')
    reference_topic = LaunchConfiguration('reference_topic')
    reference_yaw_in_degrees = LaunchConfiguration('reference_yaw_in_degrees')
    yaw_command_sign = LaunchConfiguration('yaw_command_sign')
    transform_world_xy_to_body = LaunchConfiguration('transform_world_xy_to_body')
    xy_command_sign_x = LaunchConfiguration('xy_command_sign_x')
    xy_command_sign_y = LaunchConfiguration('xy_command_sign_y')
    xy_command_swap = LaunchConfiguration('xy_command_swap')
    enable_control_xy = LaunchConfiguration('enable_control_xy')
    enable_control_z = LaunchConfiguration('enable_control_z')
    enable_control_yaw = LaunchConfiguration('enable_control_yaw')
    control_enable_service_name = LaunchConfiguration('control_enable_service_name')
    control_status_topic = LaunchConfiguration('control_status_topic')
    start_enabled = LaunchConfiguration('start_enabled')
    pid_params_file = LaunchConfiguration('pid_params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'controller_type',
            default_value='pid',
            description='Controller type: pid, lqi, or mpc',
        ),
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/tello/filtered_pose',
            description='Odometry topic for control feedback',
        ),
        DeclareLaunchArgument(
            'command_topic',
            default_value='/tello/cmd_vel',
            description='Command topic published as geometry_msgs/Twist',
        ),
        DeclareLaunchArgument(
            'reference_topic',
            default_value='/tello_control/reference',
            description='Topic (tello_control_pkg/msg/ControlReference) used as controller reference',
        ),
        DeclareLaunchArgument(
            'reference_yaw_in_degrees',
            default_value='true',
            description='Interpret ControlReference.yaw as degrees (true) or radians (false).',
        ),
        DeclareLaunchArgument(
            'yaw_command_sign',
            default_value='1.0',
            description='Global sign applied to cmd.angular.z from any active controller (+1 or -1).',
        ),
        DeclareLaunchArgument(
            'transform_world_xy_to_body',
            default_value='true',
            description='Rotate controller XY command from world frame to body frame using odometry yaw.',
        ),
        DeclareLaunchArgument(
            'xy_command_sign_x',
            default_value='1.0',
            description='Global sign applied to cmd.linear.x after world->body transform (+1 or -1).',
        ),
        DeclareLaunchArgument(
            'xy_command_sign_y',
            default_value='1.0',
            description='Global sign applied to cmd.linear.y after world->body transform (+1 or -1).',
        ),
        DeclareLaunchArgument(
            'xy_command_swap',
            default_value='false',
            description='Swap cmd.linear.x and cmd.linear.y after world->body transform.',
        ),
        DeclareLaunchArgument(
            'enable_control_xy',
            default_value='true',
            description='Enable control on XY axes (true/false).',
        ),
        DeclareLaunchArgument(
            'enable_control_z',
            default_value='true',
            description='Enable control on Z axis (true/false).',
        ),
        DeclareLaunchArgument(
            'enable_control_yaw',
            default_value='false',
            description='Enable control on YAW axis (true/false).',
        ),
        DeclareLaunchArgument(
            'control_enable_service_name',
            default_value='/tello_control/set_enabled',
            description='Service (std_srvs/SetBool) to enable or disable the controller',
        ),
        DeclareLaunchArgument(
            'control_status_topic',
            default_value='/tello_control/status',
            description='Topic (tello_control_pkg/msg/ControlStatus) with controller state',
        ),
        DeclareLaunchArgument(
            'start_enabled',
            default_value='false',
            description='Start controller enabled (true/false)',
        ),
        DeclareLaunchArgument(
            'pid_params_file',
            default_value=os.path.join(
                get_package_share_directory('tello_control_pkg'),
                'config',
                'pid_indoor_safe.yaml',
            ),
            description='YAML file with controller parameters (PID or LQI).',
        ),
        Node(
            package='tello_control_pkg',
            executable='tello_control_node',
            output='screen',
            parameters=[
                pid_params_file,
                {
                    'controller_type': controller_type,
                    'odom_topic': odom_topic,
                    'command_topic': command_topic,
                    'reference_topic': reference_topic,
                    'reference_yaw_in_degrees': reference_yaw_in_degrees,
                    'yaw_command_sign': yaw_command_sign,
                    'transform_world_xy_to_body': transform_world_xy_to_body,
                    'xy_command_sign_x': xy_command_sign_x,
                    'xy_command_sign_y': xy_command_sign_y,
                    'xy_command_swap': xy_command_swap,
                    'enable_control_xy': enable_control_xy,
                    'enable_control_z': enable_control_z,
                    'enable_control_yaw': enable_control_yaw,
                    'control_enable_service_name': control_enable_service_name,
                    'control_status_topic': control_status_topic,
                    'start_enabled': start_enabled,
                }
            ],
        ),
    ])

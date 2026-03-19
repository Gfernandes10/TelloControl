cmake_minimum_required(VERSION 3.5)
project(tello_control_pkg)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(Eigen3 REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ControlReference.msg"
  "msg/ControlStatus.msg"
)

add_executable(tello_control_node
  src/tello_control_node.cpp
  src/controllers/pid_controller.cpp
  src/controllers/lqi_controller.cpp
  src/controllers/mpc_controller.cpp
)

target_include_directories(tello_control_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(tello_control_node
  rclcpp
  rcl_interfaces
  geometry_msgs
  nav_msgs
  std_srvs
)
target_link_libraries(tello_control_node Eigen3::Eigen)
add_dependencies(tello_control_node ${PROJECT_NAME}__rosidl_typesupport_cpp)
rosidl_target_interfaces(tello_control_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

install(TARGETS tello_control_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)

ament_package()

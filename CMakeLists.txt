cmake_minimum_required(VERSION 3.8)
project(stereo_point)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(message_filters REQUIRED)
find_package(image_geometry REQUIRED)

# 创建可执行文件
add_executable(stereo_to_pointcloud src/stereo_to_pointcloud.cpp)

# 链接所需的库
ament_target_dependencies(stereo_to_pointcloud
  rclcpp
  sensor_msgs
  image_transport
  cv_bridge
  pcl_conversions
  pcl_ros
  message_filters
  image_geometry
)

# 安装可执行文件
install(TARGETS stereo_to_pointcloud
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

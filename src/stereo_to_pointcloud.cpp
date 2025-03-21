#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <image_geometry/stereo_camera_model.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

class RealsensePointCloud : public rclcpp::Node
{
public:
  RealsensePointCloud() : Node("realsense_pointcloud")
  {
    // 声明参数
    declare_parameter("queue_size", 10);
    declare_parameter("max_depth", 10.0); // 最大深度值（单位：米）

    // 初始化订阅者
    left_image_sub_.subscribe(this, "/camera/camera/infra1/image_rect_raw");
    right_image_sub_.subscribe(this, "/camera/camera/infra2/image_rect_raw");
    left_info_sub_.subscribe(this, "/camera/camera/infra1/camera_info");
    right_info_sub_.subscribe(this, "/camera/camera/infra2/camera_info");

    // 配置同步策略
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(get_parameter("queue_size").as_int()),
        left_image_sub_, right_image_sub_,
        left_info_sub_, right_info_sub_);
    sync_->registerCallback(
        std::bind(&RealsensePointCloud::imageCallback, this,
                  std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4));

    // 点云发布者
    pointcloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("pointcloud", 10);

    RCLCPP_INFO(get_logger(), "Realsense PointCloud node initialized.");
  }

private:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image,
      sensor_msgs::msg::CameraInfo, sensor_msgs::msg::CameraInfo>;

  void imageCallback(
      const sensor_msgs::msg::Image::ConstSharedPtr& left_image,
      const sensor_msgs::msg::Image::ConstSharedPtr& right_image,
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr& left_info,
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr& right_info)
  {
    try {
      // 转换图像为OpenCV格式
      cv::Mat left = cv_bridge::toCvShare(left_image, "mono8")->image;
      cv::Mat right = cv_bridge::toCvShare(right_image, "mono8")->image;

      // 初始化立体相机模型
      image_geometry::StereoCameraModel stereo_model;
      stereo_model.fromCameraInfo(*left_info, *right_info); // 使用 fromCameraInfo 初始化

      // 计算视差图
      cv::Mat disparity;
      cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(128, 21);
      stereo->compute(left, right, disparity);

      // 转换为浮点型视差
      disparity.convertTo(disparity, CV_32F, 1.0 / 16.0);

      // 生成3D点云
      cv::Mat_<cv::Vec3f> points3D;
      cv::reprojectImageTo3D(disparity, points3D, stereo_model.reprojectionMatrix(), true); // 使用 reprojectionMatrix

      // 转换为PCL点云
      pcl::PointCloud<pcl::PointXYZ> cloud;
      cloud.width = points3D.cols;
      cloud.height = points3D.rows;
      cloud.is_dense = false;
      cloud.resize(cloud.width * cloud.height);

      const float max_depth = get_parameter("max_depth").as_double();

      #pragma omp parallel for
      for (int v = 0; v < points3D.rows; ++v) {
        for (int u = 0; u < points3D.cols; ++u) {
          const cv::Vec3f& point = points3D(v, u);
          if (point[2] > max_depth || point[2] <= 0) continue;
          cloud.at(u, v) = pcl::PointXYZ(point[0], point[1], point[2]);
        }
      }

      // 发布点云
      sensor_msgs::msg::PointCloud2 msg;
      pcl::toROSMsg(cloud, msg);
      msg.header = left_image->header;
      pointcloud_pub_->publish(msg);
    }
    catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "CV Bridge error: %s", e.what());
    }
    catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Processing error: %s", e.what());
    }
  }

  // 成员变量
  message_filters::Subscriber<sensor_msgs::msg::Image> left_image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> right_image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> left_info_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> right_info_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RealsensePointCloud>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
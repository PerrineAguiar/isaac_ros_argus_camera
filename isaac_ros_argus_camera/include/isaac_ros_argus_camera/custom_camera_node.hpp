#ifndef ISAAC_ROS_ARGUS_CAMERA__CUSTOM_CAMERA_NODE_HPP_
#define ISAAC_ROS_ARGUS_CAMERA__CUSTOM_CAMERA_NODE_HPP_

#include <string>

#include "isaac_ros_argus_camera/argus_camera_node.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace argus
{

class CustomCameraNode : public ArgusCameraNode
{
public:
  explicit CustomCameraNode(
    const rclcpp::NodeOptions & options);

  ~CustomCameraNode() override;

private:
  std::string optical_frame_name_;

  std::string camera_info_url_;

  sensor_msgs::msg::CameraInfo::SharedPtr camera_info_;
};

}  // namespace argus
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_ARGUS_CAMERA__CUSTOM_CAMERA_NODE_HPP_

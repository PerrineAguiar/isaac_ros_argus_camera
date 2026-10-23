// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_argus_camera/argus_camera_custom_node.hpp"
#include <cuda_runtime.h>
#include "gxf/multimedia/video.hpp"
#include <opencv2/opencv.hpp>


namespace nvidia
{
namespace isaac_ros
{
namespace argus
{
namespace
{

constexpr char OUTPUT_COMPONENT_KEY_LEFT_IMAGE[] = "sink_left_image/sink";
constexpr char OUTPUT_TOPIC_NAME_LEFT_IMAGE[] = "left/image_raw";

constexpr char OUTPUT_COMPONENT_KEY_LEFT_CAMERAINFO[] = "sink_left_camera_info/sink";
constexpr char OUTPUT_TOPIC_NAME_LEFT_CAMERAINFO[] = "left/camera_info";

constexpr char OUTPUT_DEFAULT_IMAGE_FORMAT[] = "nitros_image_rgb8";
constexpr char OUTPUT_DEFAULT_CAMERAINFO_FORMAT[] = "nitros_camera_info";

constexpr char APP_YAML_FILENAME[] = "config/argus_camera_mono_node.yaml";

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {OUTPUT_COMPONENT_KEY_LEFT_IMAGE,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = OUTPUT_DEFAULT_IMAGE_FORMAT,
      .topic_name = OUTPUT_TOPIC_NAME_LEFT_IMAGE
    }
  },
  {OUTPUT_COMPONENT_KEY_LEFT_CAMERAINFO,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = OUTPUT_DEFAULT_CAMERAINFO_FORMAT,
      .topic_name = OUTPUT_TOPIC_NAME_LEFT_CAMERAINFO
    }
  }
};
#pragma GCC diagnostic pop

constexpr char PACKAGE_NAME[] = "isaac_ros_argus_camera";
const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {
  "config/isaac_ros_argus_camera_mono_spec.yaml",
};
const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"isaac_ros_gxf", "gxf/lib/serialization/libgxf_serialization.so"},
  {"gxf_isaac_gxf_helpers", "gxf/lib/libgxf_isaac_gxf_helpers.so"},
  {"gxf_isaac_sight", "gxf/lib/libgxf_isaac_sight.so"},
  {"gxf_isaac_atlas", "gxf/lib/libgxf_isaac_atlas.so"},
  {"gxf_isaac_messages", "gxf/lib/libgxf_isaac_messages.so"},
  {"gxf_isaac_tensorops", "gxf/lib/libgxf_isaac_tensorops.so"},
  {"gxf_isaac_rectify", "gxf/lib/libgxf_isaac_rectify.so"},
  {"gxf_isaac_timestamp_correlator", "gxf/lib/libgxf_isaac_timestamp_correlator.so"},
  {"gxf_isaac_argus", "gxf/lib/libgxf_isaac_argus.so"},
  {"gxf_isaac_message_compositor", "gxf/lib/libgxf_isaac_message_compositor.so"}
};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {
  "config/namespace_injector_rule.yaml"
};
}  // namespace

// void ArgusCustomNode::CustomImageCallback(
//   const gxf_context_t context,
//   nitros::NitrosTypeBase & msg,
//   const std::string frame_name)
// {
//   auto msg_entity =
//     nvidia::gxf::Entity::Shared(context, msg.handle);

//   if (!msg_entity) {
//     RCLCPP_ERROR(
//       get_logger(),
//       "Failed to create GXF entity");
//     return;
//   }

//   RCLCPP_INFO(
//     get_logger(),
//     "Custom callback received image");
// }

void ArgusCustomNode::CustomImageCallback(
  const gxf_context_t context,
  nitros::NitrosTypeBase & msg,
  const std::string frame_name)
{
  (void)frame_name;

  auto msg_entity =
    nvidia::gxf::Entity::Shared(context, msg.handle);

  if (!msg_entity) {
    RCLCPP_ERROR(
      get_logger(),
      "Failed to create GXF entity");

    return;
  }

  // Get the VideoBuffer containing the RGB_U8 image.
  auto video_buffer =
    msg_entity->get<nvidia::gxf::VideoBuffer>();

  if (!video_buffer) {
    RCLCPP_ERROR(
      get_logger(),
      "Could not find VideoBuffer in image message");

    return;
  }

  const auto & video =
    video_buffer.value();
  const auto info = video->video_frame_info();

  const uint32_t width = info.width;
  const uint32_t height = info.height;

  // RCLCPP_INFO(
  //   get_logger(),
  //   "Custom callback: width=%u height=%u",
  //   width,
  //   height);


  // RCLCPP_INFO(
  //   get_logger(),
  //   "Custom callback received image8 -CHECK AND WORK HERE");

  // auto * gpu_ptr = video->pointer();

  // RCLCPP_INFO(
  //   get_logger(),
  //   "VideoBuffer GPU pointer: %p",
  //   static_cast<void *>(gpu_ptr));

  // const auto storage_type =
  //   video->storage_type();

  // RCLCPP_INFO(
  //   get_logger(),
  //   "VideoBuffer storage type: %d",
  //    static_cast<int>(storage_type));
  auto * gpu_ptr = video->pointer();

  const size_t buffer_size =
    video->size();

  std::vector<uint8_t> cpu_buffer(buffer_size);

  cudaError_t result =
    cudaMemcpy(
      cpu_buffer.data(),
      gpu_ptr,
      buffer_size,
      cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    RCLCPP_ERROR(
      get_logger(),
      "cudaMemcpyDeviceToHost failed: %s",
      cudaGetErrorString(result));

    return;
  }

  RCLCPP_INFO(
    get_logger(),
    "GPU -> CPU copy successful: %zu bytes",
    buffer_size);

  const size_t expected_size =
    static_cast<size_t>(width) *
    static_cast<size_t>(height) *
    3;

  RCLCPP_INFO(
    get_logger(),
    "CPU buffer size: %zu, expected RGB size: %zu",
    buffer_size,
    expected_size);

  // --------------------------------------------------------------------------
  // Get the actual GXF/Argus acquisition timestamp.
  // --------------------------------------------------------------------------

  uint64_t timestamp_ns = 0;

  auto gxf_timestamp =
    msg_entity->get<nvidia::gxf::Timestamp>();

  if (!gxf_timestamp) {
    // Fallback to a component explicitly named "timestamp".
    gxf_timestamp =
      msg_entity->get<nvidia::gxf::Timestamp>("timestamp");
  }

  if (gxf_timestamp) {
    timestamp_ns =
      gxf_timestamp.value()->acqtime;
  } else {
    RCLCPP_WARN(
      get_logger(),
      "[ArgusCameraNode] Failed to get capture timestamp");
  }



  const std::string filename =
    "/tmp/argus_frame_" +
    std::to_string(camera_id_) +
    std::to_string(timestamp_ns) +
    ".png";

  // std::ofstream file(
  //   filename,
  //   std::ios::binary);

  // if (!file) {
  //   RCLCPP_ERROR(
  //     get_logger(),
  //     "Failed to open file: %s",
  //     filename.c_str());

  //   return;
  // }

  // file.write(
  //   reinterpret_cast<const char *>(cpu_buffer.data()),
  //   static_cast<std::streamsize>(cpu_buffer.size()));

  // file.close();

  cv::Mat rgb_image(
    height,
    width,
    CV_8UC3,
    cpu_buffer.data());

  cv::Mat bgr_image;
  cv::cvtColor(rgb_image, bgr_image, cv::COLOR_RGB2BGR);

  cv::imwrite(filename, bgr_image);


  RCLCPP_INFO(
    get_logger(),
    "Saved RGB frame to %s",
    filename.c_str());



  // RCLCPP_INFO(
  //   get_logger(),
  //   "Received image: %ux%u",
  //   width,
  //   height);
}


ArgusCustomNode::ArgusCustomNode(const rclcpp::NodeOptions & options)
: ArgusCameraNode(
    options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME)
{
  camera_id_ = declare_parameter<int>("camera_id", 0);
  module_id_ = declare_parameter<int>("module_id", 0);
  mode_ = declare_parameter<int>("mode", 0);
  framerate_ = declare_parameter<int>("framerate", 30);
  fsync_type_ = declare_parameter<int>("fsync_type", 1);
  use_hw_timestamp_ = declare_parameter<bool>("use_hw_timestamp", false);
  camera_link_frame_name_ = declare_parameter<std::string>("camera_link_frame_name", "camera");
  optical_frame_name_ = declare_parameter<std::string>("optical_frame_name", "camera_optical");
  camera_info_url_ = declare_parameter<std::string>("camera_info_url", "");
  publish_tf_ = declare_parameter<bool>("publish_tf", true);

  // Load camera info from a file if provided
  if (!camera_info_url_.empty()) {
    camera_info_ = loadCameraInfoFromFile(camera_info_url_);
    RCLCPP_INFO(
      get_logger(), "[ArgusCustomNode] Loaded camera info from \"%s\"", camera_info_url_.c_str());
  }

  // Adding callback for left image
  config_map_[OUTPUT_COMPONENT_KEY_LEFT_IMAGE].callback =
    std::bind(
    &ArgusCustomNode::CustomImageCallback, this,
    std::placeholders::_1, std::placeholders::_2, optical_frame_name_);

  // Adding callback for left camera_info
  config_map_[OUTPUT_COMPONENT_KEY_LEFT_CAMERAINFO].callback =
    std::bind(
    &ArgusCameraNode::ArgusCameraInfoCallback, this,
    std::placeholders::_1, std::placeholders::_2, camera_link_frame_name_, optical_frame_name_,
    camera_info_);

  startNitrosNode();
}

ArgusCustomNode::~ArgusCustomNode() = default;

}  // namespace argus
}  // namespace isaac_ros
}  // namespace nvidia

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::argus::ArgusCustomNode)

// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_ARGUS_CAMERA__ARGUS_CAMERA_THREAD_NODE_HPP_
#define ISAAC_ROS_ARGUS_CAMERA__ARGUS_CAMERA_THREAD_NODE_HPP_

#include <cuda_runtime.h>
#include <gxf/core/gxf.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "isaac_ros_argus_camera/argus_camera_node.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace argus
{

// Frame metadata for disk queue
struct PendingDiskFrame
{
  std::string filename;
  std::vector<uint8_t> data;
};

// Reusable GPU frame buffer
struct FrameBuffer
{
  uint8_t * gpu_data{nullptr};
  size_t size{0};
  uint32_t width{0};
  uint32_t height{0};
  size_t stride{0};
  uint64_t timestamp_ns{0};
  std::string frame_name;
  bool in_use{false};
};

class ArgusThreadNode : public ArgusCameraNode
{
public:
  explicit ArgusThreadNode(const rclcpp::NodeOptions & options);
  virtual ~ArgusThreadNode();

private:
  static constexpr size_t NUM_BUFFERS = 30;
  static constexpr size_t NUM_WRITERS = 4;
  static constexpr size_t MAX_WRITE_QUEUE_SIZE = 15;

  void CustomImageCallback(
    const gxf_context_t context,
    nitros::NitrosTypeBase & msg,
    const std::string frame_name);

  bool initializeBuffers(uint32_t width, uint32_t height, size_t stride);
  void destroyBuffers();

  bool writeJpegFile(
    const std::string & filename,
    const unsigned char * jpeg_data,
    size_t jpeg_size);

  void writerLoop(uint32_t writer_id);
  void diskWriterLoop();

  // ROS parameters
  int camera_id_{0};
  int module_id_{0};
  int mode_{0};
  int framerate_{30};
  int fsync_type_{1};
  bool use_hw_timestamp_{false};
  std::string camera_link_frame_name_{"camera"};
  std::string optical_frame_name_{"camera_optical"};
  std::string camera_info_url_{""};
  bool publish_tf_{true};
  std::string save_path_{"./output_frames/"};

  sensor_msgs::msg::CameraInfo camera_info_;

  // Buffer management
  std::mutex initialization_mutex_;
  bool buffers_initialized_{false};
  uint32_t buffer_width_{0};
  uint32_t buffer_height_{0};
  size_t buffer_stride_{0};
  size_t buffer_size_{0};

  FrameBuffer buffers_[NUM_BUFFERS];

  std::queue<size_t> free_queue_;
  std::mutex free_mutex_;

  std::queue<size_t> write_queue_;
  std::mutex write_mutex_;
  std::condition_variable write_cv_;

  // Asynchronous Disk Storage Queue
  std::queue<PendingDiskFrame> disk_queue_;
  std::mutex disk_mutex_;
  std::condition_variable disk_cv_;

  // Thread controls
  std::atomic<bool> running_{false};
  std::vector<std::thread> writer_threads_;
  std::thread disk_writer_thread_;

  cudaStream_t copy_stream_{nullptr};
};

}  // namespace argus
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_ARGUS_CAMERA__ARGUS_CAMERA_THREAD_NODE_HPP_
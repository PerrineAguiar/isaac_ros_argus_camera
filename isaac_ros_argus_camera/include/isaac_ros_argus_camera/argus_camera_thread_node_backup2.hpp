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

#ifndef ISAAC_ROS_ARGUS_CAMERA__ARGUS_CAMERA_THREAD_NODE_HPP_
#define ISAAC_ROS_ARGUS_CAMERA__ARGUS_CAMERA_THREAD_NODE_HPP_

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <nvjpeg.h>

#include "gxf/core/entity.hpp"

#include "isaac_ros_argus_camera/argus_camera_node.hpp"
#include "isaac_ros_nitros/nitros_node.hpp"

#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace argus
{

class ArgusThreadNode : public ArgusCameraNode
{
public:
  explicit ArgusThreadNode(
    const rclcpp::NodeOptions & options);

  ~ArgusThreadNode() override;

private:

  // ==========================================================================
  // Configuration
  // ==========================================================================

  static constexpr size_t NUM_BUFFERS = 6;
  static constexpr size_t NUM_WRITERS = 1;
  static constexpr size_t MAX_WRITE_QUEUE_SIZE = NUM_BUFFERS;

  // ==========================================================================
  // Application-owned GPU buffer
  //
  // IMPORTANT:
  //
  // This memory belongs to ArgusThreadNode, NOT Argus.
  // The Argus entity is released after the callback copies the image here.
  // ==========================================================================

  struct FrameBuffer
  {
    uint8_t * gpu_data = nullptr;

    size_t size = 0;

    uint32_t width = 0;
    uint32_t height = 0;

    size_t stride = 0;

    uint64_t timestamp_ns = 0;

    std::string frame_name;

    bool in_use = false;
  };

  std::array<FrameBuffer, NUM_BUFFERS> buffers_;

  // ==========================================================================
  // Queues
  // ==========================================================================

  // Buffers available for a new camera frame.
  std::queue<size_t> free_queue_;

  // Buffers waiting for JPEG workers.
  std::queue<size_t> write_queue_;

  // ==========================================================================
  // Synchronization
  // ==========================================================================

  std::mutex buffer_mutex_;

  std::mutex write_mutex_;

  std::condition_variable write_cv_;

  // ==========================================================================
  // Worker threads
  // ==========================================================================

  std::vector<std::thread> writer_threads_;

  std::atomic<bool> running_{true};

  // ==========================================================================
  // CUDA
  // ==========================================================================

  cudaStream_t copy_stream_ = nullptr;

  // ==========================================================================
  // Initialization
  // ==========================================================================

  std::mutex initialization_mutex_;

  bool buffers_initialized_ = false;

  uint32_t buffer_width_ = 0;
  uint32_t buffer_height_ = 0;
  size_t buffer_stride_ = 0;
  size_t buffer_size_ = 0;

  bool initializeBuffers(
    uint32_t width,
    uint32_t height,
    size_t stride);

  void destroyBuffers();

  // ==========================================================================
  // Writer
  // ==========================================================================

  void writerLoop(
    uint32_t writer_id);

  bool writeJpegFile(
    const std::string & filename,
    const unsigned char * jpeg_data,
    size_t jpeg_size);

  // ==========================================================================
  // Callback
  // ==========================================================================

  void CustomImageCallback(
    const gxf_context_t context,
    nitros::NitrosTypeBase & msg,
    const std::string frame_name);

  // ==========================================================================
  // Existing members
  // ==========================================================================

  std::string optical_frame_name_;

  std::string camera_info_url_;

  sensor_msgs::msg::CameraInfo::SharedPtr camera_info_;
};

}  // namespace argus
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_ARGUS_CAMERA__ARGUS_CAMERA_THREAD_NODE_HPP_

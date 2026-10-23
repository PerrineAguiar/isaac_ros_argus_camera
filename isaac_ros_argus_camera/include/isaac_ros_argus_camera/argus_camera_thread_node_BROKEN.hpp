// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef ISAAC_ROS_ARGUS_CAMERA__ARGUS_CAMERA_THREAD_NODE_HPP_
#define ISAAC_ROS_ARGUS_CAMERA__ARGUS_CAMERA_THREAD_NODE_HPP_

#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "isaac_ros_argus_camera/argus_camera_node.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace argus
{

class ArgusThreadNode : public ArgusCameraNode
{
public:
  explicit ArgusThreadNode(const rclcpp::NodeOptions & options);
  ~ArgusThreadNode() override;

private:
  static constexpr size_t NUM_BUFFERS = 32;
  static constexpr size_t NUM_WRITERS = 2;
  static constexpr size_t MAX_WRITE_QUEUE_SIZE = 64;
  static constexpr int JPEG_QUALITY = 90;

  struct FrameBuffer
  {
    unsigned char * gpu_data{nullptr};
    size_t size{0};
    uint32_t width{0};
    uint32_t height{0};
    size_t pitch{0};
    uint64_t timestamp_ns{0};
    std::string frame_name;
    bool in_use{false};
  };

  struct NvJpegJob
  {
    size_t buffer_index{0};
    uint64_t timestamp_ns{0};
    std::string frame_name;
    std::string output_filepath;
  };

  // Directories & Output
  std::string output_dir_{"./output_frames"};
  std::string csv_file_path_{"./output_frames/saved_frames.csv"};

  // CUDA & JPEG Resources
  cudaStream_t copy_stream_{nullptr};

  // Buffer Pool
  std::array<FrameBuffer, NUM_BUFFERS> buffers_{};
  std::queue<size_t> free_queue_;
  std::mutex buffer_mutex_;
  std::mutex initialization_mutex_;
  bool buffers_initialized_{false};

  uint32_t buffer_width_{0};
  uint32_t buffer_height_{0};
  size_t buffer_size_{0};
  size_t buffer_pitch_{0};

  // Queue & Worker Threads
  std::queue<NvJpegJob> write_queue_;
  std::mutex write_mutex_;
  std::condition_variable write_cv_;
  std::vector<std::thread> writer_threads_;
  std::atomic<bool> running_{false};

  // Thread-Safe CSV Logging
  std::mutex csv_mutex_;
  std::ofstream csv_file_;

  // Statistics
  std::atomic<uint64_t> acquired_buffers_{0};
  std::atomic<uint64_t> queued_buffers_{0};
  std::atomic<uint64_t> copied_buffers_{0};
  std::atomic<uint64_t> encoded_buffers_{0};
  std::atomic<uint64_t> released_buffers_{0};
  std::atomic<uint64_t> dropped_buffers_{0};

  // Buffer management
  bool initializeBuffers(uint32_t width, uint32_t height, size_t pitch, size_t buffer_size);
  void destroyBuffers();
  bool acquireFreeBuffer(size_t & buffer_index);
  void releaseBuffer(size_t buffer_index);

  // Argus image callback
  void CustomImageCallback(
    gxf_context_t context,
    nitros::NitrosTypeBase & msg,
    std::string frame_name);

  // JPEG worker
  void writerLoop(uint32_t writer_id);

  bool encodeJpeg(
    uint32_t writer_id,
    const NvJpegJob & job,
    cudaStream_t encode_stream,
    nvjpegHandle_t nvjpeg_handle,
    nvjpegEncoderState_t encoder_state,
    nvjpegEncoderParams_t encoder_params,
    std::vector<unsigned char> & jpeg_bitstream);

  bool writeJpegFile(
    const std::string & filename,
    const std::vector<unsigned char> & jpeg_data);

  void logToCsv(uint64_t timestamp_ns, const std::string & frame_name, const std::string & filepath);
};

}  // namespace argus
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_ARGUS_CAMERA__ARGUS_CAMERA_THREAD_NODE_HPP_
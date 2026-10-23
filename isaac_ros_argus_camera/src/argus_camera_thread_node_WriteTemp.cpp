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

#include "isaac_ros_argus_camera/argus_camera_thread_node.hpp"

#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "gxf/multimedia/video.hpp"

#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace argus
{
namespace
{

constexpr char OUTPUT_COMPONENT_KEY_LEFT_IMAGE[] =
  "sink_left_image/sink";

constexpr char OUTPUT_COMPONENT_KEY_LEFT_CAMERAINFO[] =
  "sink_left_camera_info/sink";

constexpr char OUTPUT_TOPIC_NAME_LEFT_CAMERAINFO[] =
  "left/camera_info";

constexpr char OUTPUT_DEFAULT_CAMERAINFO_FORMAT[] =
  "nitros_camera_info";

constexpr char APP_YAML_FILENAME[] =
  "config/argus_camera_mono_node.yaml";

constexpr char PACKAGE_NAME[] =
  "isaac_ros_argus_camera";

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {
    OUTPUT_COMPONENT_KEY_LEFT_CAMERAINFO,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = OUTPUT_DEFAULT_CAMERAINFO_FORMAT,
      .topic_name = OUTPUT_TOPIC_NAME_LEFT_CAMERAINFO
    }
  }
};

#pragma GCC diagnostic pop

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
  {"gxf_isaac_timestamp_correlator",
    "gxf/lib/libgxf_isaac_timestamp_correlator.so"},
  {"gxf_isaac_argus", "gxf/lib/libgxf_isaac_argus.so"},
  {"gxf_isaac_message_compositor",
    "gxf/lib/libgxf_isaac_message_compositor.so"}
};

const std::vector<std::string> GENERATOR_RULE_FILENAMES = {
  "config/namespace_injector_rule.yaml"
};

}  // namespace


// ============================================================================
// Constructor
// ============================================================================

ArgusThreadNode::ArgusThreadNode(
  const rclcpp::NodeOptions & options)
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
  camera_id_ =
    declare_parameter<int>(
      "camera_id",
      0);

  module_id_ =
    declare_parameter<int>(
      "module_id",
      0);

  mode_ =
    declare_parameter<int>(
      "mode",
      0);

  framerate_ =
    declare_parameter<int>(
      "framerate",
      30);

  fsync_type_ =
    declare_parameter<int>(
      "fsync_type",
      1);

  use_hw_timestamp_ =
    declare_parameter<bool>(
      "use_hw_timestamp",
      false);

  camera_link_frame_name_ =
    declare_parameter<std::string>(
      "camera_link_frame_name",
      "camera");

  optical_frame_name_ =
    declare_parameter<std::string>(
      "optical_frame_name",
      "camera_optical");

  camera_info_url_ =
    declare_parameter<std::string>(
      "camera_info_url",
      "");

  publish_tf_ =
    declare_parameter<bool>(
      "publish_tf",
      true);

  if (has_parameter("save_path")) {
    get_parameter("save_path", save_path_);
  } else {
    save_path_ = declare_parameter<std::string>("save_path", "./output_frames/");
  }

  if (!save_path_.empty() && save_path_.back() != '/') {
    save_path_ += "/";
  }

  std::filesystem::create_directories(save_path_);

  // --------------------------------------------------------------------------
  // Camera info
  // --------------------------------------------------------------------------

  if (!camera_info_url_.empty()) {

    auto loaded_info = loadCameraInfoFromFile(camera_info_url_);
    if (loaded_info) {
      camera_info_ = *loaded_info;
    }

    RCLCPP_INFO(
      get_logger(),
      "[ArgusThreadNode] Loaded camera info from \"%s\"",
      camera_info_url_.c_str());
  }

  // --------------------------------------------------------------------------
  // CUDA copy stream
  // --------------------------------------------------------------------------

  cudaError_t cuda_result =
    cudaStreamCreateWithFlags(
      &copy_stream_,
      cudaStreamNonBlocking);

  if (cuda_result != cudaSuccess) {

    throw std::runtime_error(
      std::string("cudaStreamCreate failed: ") +
      cudaGetErrorString(cuda_result));
  }

  // --------------------------------------------------------------------------
  // Start JPEG workers & Async Disk Writer Thread
  // --------------------------------------------------------------------------

  running_.store(true);

  disk_writer_thread_ =
    std::thread(&ArgusThreadNode::diskWriterLoop, this);

  for (size_t i = 0; i < NUM_WRITERS; ++i) {

    writer_threads_.emplace_back(
      &ArgusThreadNode::writerLoop,
      this,
      static_cast<uint32_t>(i));
  }

  RCLCPP_INFO(
    get_logger(),
    "Started %zu nvJPEG worker threads & async disk writer for camera_id=%d",
    NUM_WRITERS,
    camera_id_);

  // --------------------------------------------------------------------------
  // Image callback
  // --------------------------------------------------------------------------

  std::bind(&ArgusCameraNode::ArgusCameraInfoCallback, this,
            std::placeholders::_1, std::placeholders::_2,
            camera_link_frame_name_, optical_frame_name_,
            std::make_shared<sensor_msgs::msg::CameraInfo>(camera_info_));

  // --------------------------------------------------------------------------
  // Camera info callback
  // --------------------------------------------------------------------------

  config_map_["camera_info"].callback = std::bind(
    &ArgusCameraNode::ArgusCameraInfoCallback,
    this,
    std::placeholders::_1,
    std::placeholders::_2,
    camera_link_frame_name_,
    optical_frame_name_,
    std::make_shared<sensor_msgs::msg::CameraInfo>(camera_info_));

  // --------------------------------------------------------------------------
  // Start Nitros node
  // --------------------------------------------------------------------------

  startNitrosNode();
}

ArgusThreadNode::~ArgusThreadNode()
{
  running_.store(false);

  write_cv_.notify_all();
  disk_cv_.notify_all();

  for (auto & thread : writer_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  if (disk_writer_thread_.joinable()) {
    disk_writer_thread_.join();
  }

  destroyBuffers();

  if (copy_stream_ != nullptr) {
    cudaStreamDestroy(copy_stream_);
  }
}


// ============================================================================
// Initialize application-owned GPU buffers
// ============================================================================

bool ArgusThreadNode::initializeBuffers(
  uint32_t width,
  uint32_t height,
  size_t stride)
{
  std::lock_guard<std::mutex> lock(
    initialization_mutex_);

  if (buffers_initialized_) {

    if (width != buffer_width_ ||
        height != buffer_height_ ||
        stride != buffer_stride_)
    {
      RCLCPP_ERROR(
        get_logger(),
        "Image dimensions changed after initialization: "
        "old=%ux%u stride=%zu new=%ux%u stride=%zu",
        buffer_width_,
        buffer_height_,
        buffer_stride_,
        width,
        height,
        stride);

      return false;
    }

    return true;
  }

  const size_t allocation_size =
    stride * static_cast<size_t>(height);

  if (allocation_size == 0) {

    RCLCPP_ERROR(
      get_logger(),
      "Invalid GPU buffer size");

    return false;
  }

  RCLCPP_INFO(
    get_logger(),
    "Initializing GPU buffer pool: camera_id=%d buffers=%zu "
    "width=%u height=%u stride=%zu size=%zu bytes",
    camera_id_,
    NUM_BUFFERS,
    width,
    height,
    stride,
    allocation_size);

  std::lock_guard<std::mutex> free_lock(free_mutex_);

  for (size_t i = 0; i < NUM_BUFFERS; ++i) {

    cudaError_t cuda_result =
      cudaMalloc(
        reinterpret_cast<void **>(&buffers_[i].gpu_data),
        allocation_size);

    if (cuda_result != cudaSuccess) {

      RCLCPP_ERROR(
        get_logger(),
        "cudaMalloc failed for buffer %zu: %s",
        i,
        cudaGetErrorString(cuda_result));

      for (size_t j = 0; j < i; ++j) {
        if (buffers_[j].gpu_data != nullptr) {
          cudaFree(buffers_[j].gpu_data);
          buffers_[j].gpu_data = nullptr;
        }
      }

      return false;
    }

    buffers_[i].size = allocation_size;
    buffers_[i].width = width;
    buffers_[i].height = height;
    buffers_[i].stride = stride;
    buffers_[i].timestamp_ns = 0;
    buffers_[i].frame_name.clear();
    buffers_[i].in_use = false;

    free_queue_.push(i);
  }

  buffer_width_ = width;
  buffer_height_ = height;
  buffer_stride_ = stride;
  buffer_size_ = allocation_size;
  buffers_initialized_ = true;

  RCLCPP_INFO(
    get_logger(),
    "GPU buffer pool initialized: camera_id=%d %zu buffers",
    camera_id_,
    NUM_BUFFERS);

  return true;
}


// ============================================================================
// Destroy application-owned GPU buffers
// ============================================================================

void ArgusThreadNode::destroyBuffers()
{
  std::lock_guard<std::mutex> lock(
    initialization_mutex_);

  if (!buffers_initialized_) {
    return;
  }

  for (auto & buffer : buffers_) {
    if (buffer.gpu_data != nullptr) {
      cudaError_t cuda_result = cudaFree(buffer.gpu_data);
      if (cuda_result != cudaSuccess) {
        RCLCPP_ERROR(
          get_logger(),
          "cudaFree failed: %s",
          cudaGetErrorString(cuda_result));
      }
      buffer.gpu_data = nullptr;
    }
    buffer.size = 0;
    buffer.in_use = false;
  }

  {
    std::lock_guard<std::mutex> free_lock(free_mutex_);
    while (!free_queue_.empty()) {
      free_queue_.pop();
    }
  }

  {
    std::lock_guard<std::mutex> write_lock(write_mutex_);
    while (!write_queue_.empty()) {
      write_queue_.pop();
    }
  }

  buffers_initialized_ = false;
  buffer_width_ = 0;
  buffer_height_ = 0;
  buffer_stride_ = 0;
  buffer_size_ = 0;
}


// ============================================================================
// Image callback
// ============================================================================

void ArgusThreadNode::CustomImageCallback(
  const gxf_context_t context,
  nitros::NitrosTypeBase & msg,
  const std::string frame_name)
{
  if (!running_.load()) {
    return;
  }

  const char camera_id = frame_name[frame_name.find("argus_mono_") + 11];

  const std::filesystem::path trigger_file =
      std::string("/tmp/CAM_RECORD_FLAG_") + camera_id;

  if (!std::filesystem::exists(trigger_file)) {
      RCLCPP_WARN_THROTTLE(
          get_logger(),
          *get_clock(),
          10000,
          "%s NOT EXISTING - SKIP SAVING",
          trigger_file.c_str());

      return;
  }

  auto msg_entity =
    nvidia::gxf::Entity::Shared(
      context,
      msg.handle);

  if (!msg_entity) {
    RCLCPP_ERROR(get_logger(), "Failed to create GXF entity");
    return;
  }

  auto video_buffer =
    msg_entity.value().get<nvidia::gxf::VideoBuffer>();

  if (!video_buffer) {
    RCLCPP_ERROR(get_logger(), "Could not find VideoBuffer in image message");
    return;
  }

  const auto & video = video_buffer.value();
  const auto info = video->video_frame_info();

  const uint32_t width = info.width;
  const uint32_t height = info.height;

  if (info.color_planes.empty()) {
    RCLCPP_ERROR(get_logger(), "No color planes in image");
    return;
  }

  const auto & plane = info.color_planes[0];
  const uint32_t source_stride = plane.stride;
  const uint32_t bytes_per_pixel = plane.bytes_per_pixel;

  if (bytes_per_pixel != 3) {
    RCLCPP_ERROR(get_logger(), "Expected RGB bpp=3, got %u", bytes_per_pixel);
    return;
  }

  if (source_stride < width * 3U) {
    RCLCPP_ERROR(
      get_logger(),
      "Invalid source stride=%u width*3=%u",
      source_stride,
      width * 3U);
    return;
  }

  auto * source_gpu_ptr = video->pointer();
  if (source_gpu_ptr == nullptr) {
    RCLCPP_ERROR(get_logger(), "Source GPU pointer is null");
    return;
  }

  if (!initializeBuffers(width, height, source_stride)) {
    return;
  }

  size_t buffer_index = 0;

  {
    std::lock_guard<std::mutex> lock(free_mutex_);

    if (free_queue_.empty()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        1000,
        "GPU buffer pool exhausted for camera_id=%d; dropping frame",
        camera_id_);
      return;
    }

    buffer_index = free_queue_.front();
    free_queue_.pop();
    buffers_[buffer_index].in_use = true;
  }

  FrameBuffer & buffer = buffers_[buffer_index];

  uint64_t timestamp_ns = 0;
  auto gxf_timestamp = msg_entity.value().get<nvidia::gxf::Timestamp>();

  if (!gxf_timestamp) {
    gxf_timestamp = msg_entity.value().get<nvidia::gxf::Timestamp>("timestamp");
  }

  if (gxf_timestamp) {
    timestamp_ns = gxf_timestamp.value()->acqtime;
  } else {
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      5000,
      "Failed to get capture timestamp");
  }

  const auto copy_start = std::chrono::steady_clock::now();

  cudaError_t cuda_result =
    cudaMemcpy2D(
      buffer.gpu_data,
      buffer.stride,
      source_gpu_ptr,
      source_stride,
      static_cast<size_t>(width) * 3U,
      height,
      cudaMemcpyDeviceToDevice);

  if (cuda_result != cudaSuccess) {
    RCLCPP_ERROR(
      get_logger(),
      "GPU copy failed for buffer %zu: %s",
      buffer_index,
      cudaGetErrorString(cuda_result));

    {
      std::lock_guard<std::mutex> lock(free_mutex_);
      buffer.in_use = false;
      free_queue_.push(buffer_index);
    }
    return;
  }

  const auto copy_done = std::chrono::steady_clock::now();

  buffer.timestamp_ns = timestamp_ns;
  buffer.frame_name = frame_name;

  {
    std::lock_guard<std::mutex> lock(write_mutex_);

    if (write_queue_.size() >= MAX_WRITE_QUEUE_SIZE) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        1000,
        "JPEG queue full for camera_id=%d (%zu), dropping copied frame",
        camera_id_,
        write_queue_.size());

      std::lock_guard<std::mutex> free_lock(free_mutex_);
      buffer.in_use = false;
      free_queue_.push(buffer_index);
      return;
    }

    write_queue_.push(buffer_index);
  }

  write_cv_.notify_one();

  const auto copy_ms =
    std::chrono::duration_cast<std::chrono::microseconds>(
      copy_done - copy_start).count();

  RCLCPP_DEBUG(
    get_logger(),
    "camera_id=%d copied frame into buffer=%zu in %ld us",
    camera_id_,
    buffer_index,
    copy_ms);
}


// ============================================================================
// Write JPEG file (Non-blocking queue push)
// ============================================================================

bool ArgusThreadNode::writeJpegFile(
  const std::string & filename,
  const unsigned char * jpeg_data,
  size_t jpeg_size)
{
  {
    std::lock_guard<std::mutex> lock(disk_mutex_);
    disk_queue_.push(PendingDiskFrame{
      filename,
      std::vector<uint8_t>(jpeg_data, jpeg_data + jpeg_size)
    });
  }

  disk_cv_.notify_one();
  return true;
}


// ============================================================================
// Dedicated Async Disk Writer Loop (RAM-buffering burst flusher)
// ============================================================================

void ArgusThreadNode::diskWriterLoop()
{
  constexpr size_t BATCH_BURST_COUNT = 20;

  while (running_.load()) {
    std::vector<PendingDiskFrame> batch;

    {
      std::unique_lock<std::mutex> lock(disk_mutex_);

      // Wait until 20 frames build up OR 1000ms elapses
      disk_cv_.wait_for(lock, std::chrono::milliseconds(1000), [this]() {
        return disk_queue_.size() >= BATCH_BURST_COUNT || !running_.load();
      });

      while (!disk_queue_.empty()) {
        batch.push_back(std::move(disk_queue_.front()));
        disk_queue_.pop();
      }
    }

    // Perform bulk disk writes while unlocked
    if (!batch.empty()) {
      for (const auto & frame : batch) {
        std::filesystem::path file_path(frame.filename);
        std::error_code ec;
        std::filesystem::create_directories(file_path.parent_path(), ec);

        FILE * outfile = fopen(frame.filename.c_str(), "wb");
        if (outfile != nullptr) {
          fwrite(frame.data.data(), 1, frame.data.size(), outfile);
          fclose(outfile);
        } else {
          RCLCPP_ERROR(
            get_logger(),
            "Failed to open JPEG file: %s",
            frame.filename.c_str());
        }
      }
    }
  }
}


// ============================================================================
// JPEG worker
// ============================================================================

void ArgusThreadNode::writerLoop(uint32_t writer_id)
{
  RCLCPP_INFO(
    get_logger(),
    "nvJPEG worker %u starting for camera_id=%d",
    writer_id,
    camera_id_);

  cudaStream_t cuda_stream = nullptr;
  cudaError_t cuda_result =
    cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);

  if (cuda_result != cudaSuccess) {
    RCLCPP_ERROR(
      get_logger(),
      "Worker %u: cudaStreamCreate failed: %s",
      writer_id,
      cudaGetErrorString(cuda_result));
    return;
  }

  nvjpegHandle_t nvjpeg_handle = nullptr;
  nvjpegStatus_t nv_result = nvjpegCreateSimple(&nvjpeg_handle);

  if (nv_result != NVJPEG_STATUS_SUCCESS) {
    RCLCPP_ERROR(
      get_logger(),
      "Worker %u: nvjpegCreateSimple failed: %d",
      writer_id,
      static_cast<int>(nv_result));
    cudaStreamDestroy(cuda_stream);
    return;
  }

  nvjpegEncoderState_t encoder_state = nullptr;
  nv_result = nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, cuda_stream);

  if (nv_result != NVJPEG_STATUS_SUCCESS) {
    RCLCPP_ERROR(
      get_logger(),
      "Worker %u: nvjpegEncoderStateCreate failed: %d",
      writer_id,
      static_cast<int>(nv_result));
    nvjpegDestroy(nvjpeg_handle);
    cudaStreamDestroy(cuda_stream);
    return;
  }

  nvjpegEncoderParams_t encoder_params = nullptr;
  nv_result = nvjpegEncoderParamsCreate(nvjpeg_handle, &encoder_params, cuda_stream);

  if (nv_result != NVJPEG_STATUS_SUCCESS) {
    RCLCPP_ERROR(
      get_logger(),
      "Worker %u: nvjpegEncoderParamsCreate failed: %d",
      writer_id,
      static_cast<int>(nv_result));
    nvjpegEncoderStateDestroy(encoder_state);
    nvjpegDestroy(nvjpeg_handle);
    cudaStreamDestroy(cuda_stream);
    return;
  }

  nv_result = nvjpegEncoderParamsSetQuality(encoder_params, 99, cuda_stream);
  if (nv_result != NVJPEG_STATUS_SUCCESS) {
    RCLCPP_ERROR(
      get_logger(),
      "Worker %u: nvjpegEncoderParamsSetQuality failed: %d",
      writer_id,
      static_cast<int>(nv_result));
    nvjpegEncoderParamsDestroy(encoder_params);
    nvjpegEncoderStateDestroy(encoder_state);
    nvjpegDestroy(nvjpeg_handle);
    cudaStreamDestroy(cuda_stream);
    return;
  }

  nv_result = nvjpegEncoderParamsSetSamplingFactors(
    encoder_params, NVJPEG_CSS_420, cuda_stream);

  if (nv_result != NVJPEG_STATUS_SUCCESS) {
    RCLCPP_ERROR(
      get_logger(),
      "Worker %u: nvjpegEncoderParamsSetSamplingFactors failed: %d",
      writer_id,
      static_cast<int>(nv_result));
    nvjpegEncoderParamsDestroy(encoder_params);
    nvjpegEncoderStateDestroy(encoder_state);
    nvjpegDestroy(nvjpeg_handle);
    cudaStreamDestroy(cuda_stream);
    return;
  }

  RCLCPP_INFO(
    get_logger(),
    "nvJPEG worker %u initialized for camera_id=%d",
    writer_id,
    camera_id_);

  // ==========================================================================
  // Worker loop
  // ==========================================================================

  while (true) {
    size_t buffer_index = 0;

    {
      std::unique_lock<std::mutex> lock(write_mutex_);

      write_cv_.wait(lock, [this]() {
        return !write_queue_.empty() || !running_.load();
      });

      if (!running_.load() && write_queue_.empty()) {
        break;
      }

      buffer_index = write_queue_.front();
      write_queue_.pop();
    }

    FrameBuffer & buffer = buffers_[buffer_index];

    if (buffer.gpu_data == nullptr) {
      RCLCPP_ERROR(
        get_logger(),
        "Worker %u: buffer %zu GPU memory is null",
        writer_id,
        buffer_index);

      std::lock_guard<std::mutex> lock(free_mutex_);
      buffer.in_use = false;
      free_queue_.push(buffer_index);
      continue;
    }

    const uint32_t width = buffer.width;
    const uint32_t height = buffer.height;
    const size_t stride = buffer.stride;
    const uint64_t timestamp_ns = buffer.timestamp_ns;

    nvjpegImage_t image{};
    image.channel[0] = reinterpret_cast<unsigned char *>(buffer.gpu_data);
    image.pitch[0] = stride;

    nv_result = nvjpegEncodeImage(
      nvjpeg_handle,
      encoder_state,
      encoder_params,
      &image,
      NVJPEG_INPUT_RGBI,
      static_cast<int>(width),
      static_cast<int>(height),
      cuda_stream);

    if (nv_result != NVJPEG_STATUS_SUCCESS) {
      RCLCPP_ERROR(
        get_logger(),
        "Worker %u: nvjpegEncodeImage failed: %d",
        writer_id,
        static_cast<int>(nv_result));

      std::lock_guard<std::mutex> lock(free_mutex_);
      buffer.in_use = false;
      free_queue_.push(buffer_index);
      continue;
    }

    size_t jpeg_size = 0;
    nv_result = nvjpegEncodeRetrieveBitstream(
      nvjpeg_handle, encoder_state, nullptr, &jpeg_size, cuda_stream);

    if (nv_result != NVJPEG_STATUS_SUCCESS || jpeg_size == 0) {
      RCLCPP_ERROR(
        get_logger(),
        "Worker %u: failed to get JPEG size: status=%d size=%zu",
        writer_id,
        static_cast<int>(nv_result),
        jpeg_size);

      std::lock_guard<std::mutex> lock(free_mutex_);
      buffer.in_use = false;
      free_queue_.push(buffer_index);
      continue;
    }

    std::vector<unsigned char> jpeg_data(jpeg_size);
    size_t actual_jpeg_size = jpeg_size;

    nv_result = nvjpegEncodeRetrieveBitstream(
      nvjpeg_handle,
      encoder_state,
      jpeg_data.data(),
      &actual_jpeg_size,
      cuda_stream);

    if (nv_result != NVJPEG_STATUS_SUCCESS) {
      RCLCPP_ERROR(
        get_logger(),
        "Worker %u: nvjpegEncodeRetrieveBitstream failed: %d",
        writer_id,
        static_cast<int>(nv_result));

      std::lock_guard<std::mutex> lock(free_mutex_);
      buffer.in_use = false;
      free_queue_.push(buffer_index);
      continue;
    }

    cuda_result = cudaStreamSynchronize(cuda_stream);

    if (cuda_result != cudaSuccess) {
      RCLCPP_ERROR(
        get_logger(),
        "Worker %u: cudaStreamSynchronize failed: %s",
        writer_id,
        cudaGetErrorString(cuda_result));

      std::lock_guard<std::mutex> lock(free_mutex_);
      buffer.in_use = false;
      free_queue_.push(buffer_index);
      continue;
    }

    // ------------------------------------------------------------------------
    // Construct 100-second bin path & filename
    // ------------------------------------------------------------------------
    constexpr uint64_t NS_PER_100_SEC = 100000000000ULL;
    uint64_t folder_bin = timestamp_ns / NS_PER_100_SEC;
    std::filesystem::path folder_path =
      std::filesystem::path(save_path_) / std::to_string(folder_bin);

    const std::string filename =
      (folder_path / ("argus_frame_" +
                      std::to_string(camera_id_) +
                      "_" +
                      std::to_string(timestamp_ns) +
                      ".jpg")).string();

    // Fast non-blocking push to disk queue
    writeJpegFile(filename, jpeg_data.data(), actual_jpeg_size);

    // Immediately return buffer to pool
    {
      std::lock_guard<std::mutex> lock(free_mutex_);
      buffer.in_use = false;
      free_queue_.push(buffer_index);
    }
  }

  // Cleanup worker handles
  nvjpegEncoderParamsDestroy(encoder_params);
  nvjpegEncoderStateDestroy(encoder_state);
  nvjpegDestroy(nvjpeg_handle);
  cudaStreamDestroy(cuda_stream);
}

}  // namespace argus
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::argus::ArgusThreadNode)
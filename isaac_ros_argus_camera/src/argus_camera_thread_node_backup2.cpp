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

  // --------------------------------------------------------------------------
  // Camera info
  // --------------------------------------------------------------------------

  if (!camera_info_url_.empty()) {

    camera_info_ =
      loadCameraInfoFromFile(
        camera_info_url_);

    RCLCPP_INFO(
      get_logger(),
      "[ArgusThreadNode] Loaded camera info from \"%s\"",
      camera_info_url_.c_str());
  }

  // --------------------------------------------------------------------------
  // CUDA copy stream
  //
  // The actual copy is synchronized in the callback so that the Argus source
  // buffer can safely be released before the callback returns.
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
  // Start JPEG workers.
  // --------------------------------------------------------------------------

  running_.store(true);

  for (size_t i = 0; i < NUM_WRITERS; ++i) {

    writer_threads_.emplace_back(
      &ArgusThreadNode::writerLoop,
      this,
      static_cast<uint32_t>(i));
  }

  RCLCPP_INFO(
    get_logger(),
    "Started %zu nvJPEG writer threads for camera_id=%d",
    NUM_WRITERS,
    camera_id_);

  // --------------------------------------------------------------------------
  // Image callback
  // --------------------------------------------------------------------------

  config_map_[OUTPUT_COMPONENT_KEY_LEFT_IMAGE].callback =
    std::bind(
      &ArgusThreadNode::CustomImageCallback,
      this,
      std::placeholders::_1,
      std::placeholders::_2,
      optical_frame_name_);

  // --------------------------------------------------------------------------
  // Camera info callback
  // --------------------------------------------------------------------------

  config_map_[OUTPUT_COMPONENT_KEY_LEFT_CAMERAINFO].callback =
    std::bind(
      &ArgusCameraNode::ArgusCameraInfoCallback,
      this,
      std::placeholders::_1,
      std::placeholders::_2,
      camera_link_frame_name_,
      optical_frame_name_,
      camera_info_);

  // --------------------------------------------------------------------------
  // Start Nitros node
  // --------------------------------------------------------------------------

  startNitrosNode();
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

  // --------------------------------------------------------------------------
  // We allocate enough space for the complete source stride.
  // --------------------------------------------------------------------------

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

  // --------------------------------------------------------------------------
  // Allocate all buffers.
  // --------------------------------------------------------------------------

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

      // Clean up allocations already made.
      for (size_t j = 0; j < i; ++j) {

        if (buffers_[j].gpu_data != nullptr) {

          cudaFree(
            buffers_[j].gpu_data);

          buffers_[j].gpu_data =
            nullptr;
        }
      }

      return false;
    }

    buffers_[i].size =
      allocation_size;

    buffers_[i].width =
      width;

    buffers_[i].height =
      height;

    buffers_[i].stride =
      stride;

    buffers_[i].timestamp_ns =
      0;

    buffers_[i].frame_name.clear();

    buffers_[i].in_use =
      false;

    free_queue_.push(i);
  }

  buffer_width_ =
    width;

  buffer_height_ =
    height;

  buffer_stride_ =
    stride;

  buffer_size_ =
    allocation_size;

  buffers_initialized_ =
    true;

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

  // Workers must already be stopped before this is called.

  for (auto & buffer : buffers_) {

    if (buffer.gpu_data != nullptr) {

      cudaError_t cuda_result =
        cudaFree(
          buffer.gpu_data);

      if (cuda_result != cudaSuccess) {

        RCLCPP_ERROR(
          get_logger(),
          "cudaFree failed: %s",
          cudaGetErrorString(cuda_result));
      }

      buffer.gpu_data =
        nullptr;
    }

    buffer.size =
      0;

    buffer.in_use =
      false;
  }

  while (!free_queue_.empty()) {
    free_queue_.pop();
  }

  while (!write_queue_.empty()) {
    write_queue_.pop();
  }

  buffers_initialized_ =
    false;

  buffer_width_ =
    0;

  buffer_height_ =
    0;

  buffer_stride_ =
    0;

  buffer_size_ =
    0;
}


// ============================================================================
// Image callback
//
// IMPORTANT:
//
// We do NOT retain the Argus/GXF entity in the worker queue.
//
// The callback:
//
//   1. Gets a temporary GXF Entity.
//   2. Gets the Argus GPU pointer.
//   3. Gets an application-owned GPU buffer.
//   4. Copies the frame D2D.
//   5. Synchronizes the copy.
//   6. Releases the GXF entity.
//   7. Queues our own GPU buffer.
//
// This allows Argus to recycle its capture buffer immediately after the
// callback returns.
// ============================================================================

void ArgusThreadNode::CustomImageCallback(
  const gxf_context_t context,
  nitros::NitrosTypeBase & msg,
  const std::string frame_name)
{
  if (!running_.load()) {
    return;
  }

  // --------------------------------------------------------------------------
  // Create a temporary shared reference to the GXF entity.
  //
  // Shared() increments the entity reference count.
  // --------------------------------------------------------------------------

  auto msg_entity =
    nvidia::gxf::Entity::Shared(
      context,
      msg.handle);

  if (!msg_entity) {

    RCLCPP_ERROR(
      get_logger(),
      "Failed to create GXF entity");

    return;
  }

  // --------------------------------------------------------------------------
  // Get VideoBuffer.
  // --------------------------------------------------------------------------

  auto video_buffer =
    msg_entity.value().get<nvidia::gxf::VideoBuffer>();

  if (!video_buffer) {

    RCLCPP_ERROR(
      get_logger(),
      "Could not find VideoBuffer in image message");

    return;
  }

  const auto & video =
    video_buffer.value();

  const auto info =
    video->video_frame_info();

  const uint32_t width =
    info.width;

  const uint32_t height =
    info.height;

  // --------------------------------------------------------------------------
  // Validate plane.
  // --------------------------------------------------------------------------

  if (info.color_planes.empty()) {

    RCLCPP_ERROR(
      get_logger(),
      "No color planes in image");

    return;
  }

  const auto & plane =
    info.color_planes[0];

  const uint32_t source_stride =
    plane.stride;

  const uint32_t bytes_per_pixel =
    plane.bytes_per_pixel;

  if (bytes_per_pixel != 3) {

    RCLCPP_ERROR(
      get_logger(),
      "Expected RGB bpp=3, got %u",
      bytes_per_pixel);

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

  // --------------------------------------------------------------------------
  // GPU pointer.
  // --------------------------------------------------------------------------

  auto * source_gpu_ptr =
    video->pointer();

  if (source_gpu_ptr == nullptr) {

    RCLCPP_ERROR(
      get_logger(),
      "Source GPU pointer is null");

    return;
  }

  // --------------------------------------------------------------------------
  // Initialize our GPU pool on the first frame.
  // --------------------------------------------------------------------------

  if (!initializeBuffers(
      width,
      height,
      source_stride))
  {
    return;
  }

  // --------------------------------------------------------------------------
  // Get an application-owned buffer.
  //
  // NEVER block the Argus callback waiting for a buffer.
  // --------------------------------------------------------------------------

  size_t buffer_index = 0;

  {
    std::lock_guard<std::mutex> lock(
      buffer_mutex_);

    if (free_queue_.empty()) {

      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        1000,
        "GPU buffer pool exhausted for camera_id=%d; "
        "dropping frame",
        camera_id_);

      return;
    }

    buffer_index =
      free_queue_.front();

    free_queue_.pop();

    buffers_[buffer_index].in_use =
      true;
  }

  FrameBuffer & buffer =
    buffers_[buffer_index];

  // --------------------------------------------------------------------------
  // Timestamp.
  // --------------------------------------------------------------------------

  uint64_t timestamp_ns = 0;

  auto gxf_timestamp =
    msg_entity.value().get<nvidia::gxf::Timestamp>();

  if (!gxf_timestamp) {

    gxf_timestamp =
      msg_entity.value().get<nvidia::gxf::Timestamp>(
        "timestamp");
  }

  if (gxf_timestamp) {

    timestamp_ns =
      gxf_timestamp.value()->acqtime;

  } else {

    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      5000,
      "Failed to get capture timestamp");
  }

  // --------------------------------------------------------------------------
  // Copy Argus GPU memory into our GPU memory.
  //
  // This is deliberately synchronous with respect to the callback.
  //
  // The purpose is to guarantee that the Argus-owned source buffer is no
  // longer being accessed when the GXF entity is released.
  // --------------------------------------------------------------------------

  const auto copy_start =
    std::chrono::steady_clock::now();

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
      std::lock_guard<std::mutex> lock(
        buffer_mutex_);

      buffer.in_use =
        false;

      free_queue_.push(
        buffer_index);
    }

    return;
  }

  const auto copy_done =
    std::chrono::steady_clock::now();

  // --------------------------------------------------------------------------
  // The copy is complete.
  //
  // msg_entity goes out of scope at the end of this callback and Argus can
  // safely recycle its capture buffer.
  // --------------------------------------------------------------------------

  buffer.timestamp_ns =
    timestamp_ns;

  buffer.frame_name =
    frame_name;

  // --------------------------------------------------------------------------
  // Queue our application-owned buffer.
  // --------------------------------------------------------------------------

  {
    std::lock_guard<std::mutex> lock(
      write_mutex_);

    if (write_queue_.size() >= MAX_WRITE_QUEUE_SIZE) {

      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        1000,
        "JPEG queue full for camera_id=%d (%zu), "
        "dropping copied frame",
        camera_id_,
        write_queue_.size());

      // Return buffer to pool.
      std::lock_guard<std::mutex> buffer_lock(
        buffer_mutex_);

      buffer.in_use =
        false;

      free_queue_.push(
        buffer_index);

      return;
    }

    write_queue_.push(
      buffer_index);
  }

  write_cv_.notify_one();

  // --------------------------------------------------------------------------
  // Optional copy timing.
  // --------------------------------------------------------------------------

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
// Write JPEG file
// ============================================================================

bool ArgusThreadNode::writeJpegFile(
  const std::string & filename,
  const unsigned char * jpeg_data,
  size_t jpeg_size)
{
  FILE * outfile =
    fopen(
      filename.c_str(),
      "wb");

  if (outfile == nullptr) {

    RCLCPP_ERROR(
      get_logger(),
      "Failed to open JPEG file: %s",
      filename.c_str());

    return false;
  }

  const size_t written =
    fwrite(
      jpeg_data,
      1,
      jpeg_size,
      outfile);

  fclose(outfile);

  if (written != jpeg_size) {

    RCLCPP_ERROR(
      get_logger(),
      "JPEG write failed: file=%s expected=%zu written=%zu",
      filename.c_str(),
      jpeg_size,
      written);

    return false;
  }

  return true;
}


// ============================================================================
// JPEG worker
// ============================================================================

void ArgusThreadNode::writerLoop(
  uint32_t writer_id)
{
  RCLCPP_INFO(
    get_logger(),
    "nvJPEG worker %u starting for camera_id=%d",
    writer_id,
    camera_id_);

  // --------------------------------------------------------------------------
  // Each worker owns its own CUDA stream.
  // --------------------------------------------------------------------------

  cudaStream_t cuda_stream =
    nullptr;

  cudaError_t cuda_result =
    cudaStreamCreateWithFlags(
      &cuda_stream,
      cudaStreamNonBlocking);

  if (cuda_result != cudaSuccess) {

    RCLCPP_ERROR(
      get_logger(),
      "Worker %u: cudaStreamCreate failed: %s",
      writer_id,
      cudaGetErrorString(cuda_result));

    return;
  }

  // --------------------------------------------------------------------------
  // Each worker owns its own nvJPEG handle.
  // --------------------------------------------------------------------------

  nvjpegHandle_t nvjpeg_handle =
    nullptr;

  nvjpegStatus_t nv_result =
    nvjpegCreateSimple(
      &nvjpeg_handle);

  if (nv_result != NVJPEG_STATUS_SUCCESS) {

    RCLCPP_ERROR(
      get_logger(),
      "Worker %u: nvjpegCreateSimple failed: %d",
      writer_id,
      static_cast<int>(nv_result));

    cudaStreamDestroy(
      cuda_stream);

    return;
  }

  // --------------------------------------------------------------------------
  // Encoder state.
  // --------------------------------------------------------------------------

  nvjpegEncoderState_t encoder_state =
    nullptr;

  nv_result =
    nvjpegEncoderStateCreate(
      nvjpeg_handle,
      &encoder_state,
      cuda_stream);

  if (nv_result != NVJPEG_STATUS_SUCCESS) {

    RCLCPP_ERROR(
      get_logger(),
      "Worker %u: nvjpegEncoderStateCreate failed: %d",
      writer_id,
      static_cast<int>(nv_result));

    nvjpegDestroy(
      nvjpeg_handle);

    cudaStreamDestroy(
      cuda_stream);

    return;
  }

  // --------------------------------------------------------------------------
  // Encoder parameters.
  // --------------------------------------------------------------------------

  nvjpegEncoderParams_t encoder_params =
    nullptr;

  nv_result =
    nvjpegEncoderParamsCreate(
      nvjpeg_handle,
      &encoder_params,
      cuda_stream);

  if (nv_result != NVJPEG_STATUS_SUCCESS) {

    RCLCPP_ERROR(
      get_logger(),
      "Worker %u: nvjpegEncoderParamsCreate failed: %d",
      writer_id,
      static_cast<int>(nv_result));

    nvjpegEncoderStateDestroy(
      encoder_state);

    nvjpegDestroy(
      nvjpeg_handle);

    cudaStreamDestroy(
      cuda_stream);

    return;
  }

  // --------------------------------------------------------------------------
  // JPEG quality.
  // --------------------------------------------------------------------------

  nv_result =
    nvjpegEncoderParamsSetQuality(
      encoder_params,
      55,
      cuda_stream);

  if (nv_result != NVJPEG_STATUS_SUCCESS) {

    RCLCPP_ERROR(
      get_logger(),
      "Worker %u: nvjpegEncoderParamsSetQuality failed: %d",
      writer_id,
      static_cast<int>(nv_result));

    nvjpegEncoderParamsDestroy(
      encoder_params);

    nvjpegEncoderStateDestroy(
      encoder_state);

    nvjpegDestroy(
      nvjpeg_handle);

    cudaStreamDestroy(
      cuda_stream);

    return;
  }

  // --------------------------------------------------------------------------
  // JPEG 4:2:0.
  // --------------------------------------------------------------------------

  nv_result =
    nvjpegEncoderParamsSetSamplingFactors(
      encoder_params,
      NVJPEG_CSS_420,
      cuda_stream);

  if (nv_result != NVJPEG_STATUS_SUCCESS) {

    RCLCPP_ERROR(
      get_logger(),
      "Worker %u: nvjpegEncoderParamsSetSamplingFactors failed: %d",
      writer_id,
      static_cast<int>(nv_result));

    nvjpegEncoderParamsDestroy(
      encoder_params);

    nvjpegEncoderStateDestroy(
      encoder_state);

    nvjpegDestroy(
      nvjpeg_handle);

    cudaStreamDestroy(
      cuda_stream);

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

    // ------------------------------------------------------------------------
    // Get buffer index.
    // ------------------------------------------------------------------------

    {
      std::unique_lock<std::mutex> lock(
        write_mutex_);

      write_cv_.wait(
        lock,
        [this]() {
          return
            !write_queue_.empty() ||
            !running_.load();
        });

      if (!running_.load() &&
          write_queue_.empty())
      {
        break;
      }

      buffer_index =
        write_queue_.front();

      write_queue_.pop();
    }

    FrameBuffer & buffer =
      buffers_[buffer_index];

    // ------------------------------------------------------------------------
    // Validate buffer.
    // ------------------------------------------------------------------------

    if (buffer.gpu_data == nullptr) {

      RCLCPP_ERROR(
        get_logger(),
        "Worker %u: buffer %zu GPU memory is null",
        writer_id,
        buffer_index);

      std::lock_guard<std::mutex> lock(
        buffer_mutex_);

      buffer.in_use =
        false;

      free_queue_.push(
        buffer_index);

      continue;
    }

    const uint32_t width =
      buffer.width;

    const uint32_t height =
      buffer.height;

    const size_t stride =
      buffer.stride;

    const uint64_t timestamp_ns =
      buffer.timestamp_ns;

    // ------------------------------------------------------------------------
    // Build nvJPEG image.
    // ------------------------------------------------------------------------

    nvjpegImage_t image{};

    image.channel[0] =
      reinterpret_cast<unsigned char *>(
        buffer.gpu_data);

    image.channel[1] =
      nullptr;

    image.channel[2] =
      nullptr;

    image.channel[3] =
      nullptr;

    image.pitch[0] =
      stride;

    image.pitch[1] =
      0;

    image.pitch[2] =
      0;

    image.pitch[3] =
      0;

    // ------------------------------------------------------------------------
    // JPEG encode.
    // ------------------------------------------------------------------------

    const auto encode_start =
      std::chrono::steady_clock::now();

    nv_result =
      nvjpegEncodeImage(
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

      std::lock_guard<std::mutex> lock(
        buffer_mutex_);

      buffer.in_use =
        false;

      free_queue_.push(
        buffer_index);

      continue;
    }

    // ------------------------------------------------------------------------
    // Query JPEG size.
    // ------------------------------------------------------------------------

    size_t jpeg_size =
      0;

    nv_result =
      nvjpegEncodeRetrieveBitstream(
        nvjpeg_handle,
        encoder_state,
        nullptr,
        &jpeg_size,
        cuda_stream);

    if (nv_result != NVJPEG_STATUS_SUCCESS ||
        jpeg_size == 0)
    {
      RCLCPP_ERROR(
        get_logger(),
        "Worker %u: failed to get JPEG size: status=%d size=%zu",
        writer_id,
        static_cast<int>(nv_result),
        jpeg_size);

      std::lock_guard<std::mutex> lock(
        buffer_mutex_);

      buffer.in_use =
        false;

      free_queue_.push(
        buffer_index);

      continue;
    }

    // ------------------------------------------------------------------------
    // Allocate host JPEG buffer.
    // ------------------------------------------------------------------------

    std::vector<unsigned char> jpeg_data(
      jpeg_size);

    size_t actual_jpeg_size =
      jpeg_size;

    // ------------------------------------------------------------------------
    // Retrieve JPEG bitstream.
    // ------------------------------------------------------------------------

    nv_result =
      nvjpegEncodeRetrieveBitstream(
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

      std::lock_guard<std::mutex> lock(
        buffer_mutex_);

      buffer.in_use =
        false;

      free_queue_.push(
        buffer_index);

      continue;
    }

    // ------------------------------------------------------------------------
    // Wait for JPEG GPU work to finish before CPU accesses jpeg_data.
    // ------------------------------------------------------------------------

    cuda_result =
      cudaStreamSynchronize(
        cuda_stream);

    if (cuda_result != cudaSuccess) {

      RCLCPP_ERROR(
        get_logger(),
        "Worker %u: cudaStreamSynchronize failed: %s",
        writer_id,
        cudaGetErrorString(cuda_result));

      std::lock_guard<std::mutex> lock(
        buffer_mutex_);

      buffer.in_use =
        false;

      free_queue_.push(
        buffer_index);

      continue;
    }

    const auto encode_done =
      std::chrono::steady_clock::now();

    // ------------------------------------------------------------------------
    // Filename.
    // ------------------------------------------------------------------------

    const std::string filename =
      save_path_ +
      "argus_frame_" +
      std::to_string(camera_id_) +
      "_" +
      std::to_string(timestamp_ns) +
      ".jpg";

    // ------------------------------------------------------------------------
    // Write JPEG.
    // ------------------------------------------------------------------------

    const auto write_start =
      std::chrono::steady_clock::now();

    const bool write_success =
      writeJpegFile(
        filename,
        jpeg_data.data(),
        actual_jpeg_size);

    const auto write_done =
      std::chrono::steady_clock::now();

    // ------------------------------------------------------------------------
    // Return GPU buffer to pool.
    //
    // At this point nvJPEG has completely finished using it.
    // ------------------------------------------------------------------------

    {
      std::lock_guard<std::mutex> lock(
        buffer_mutex_);

      buffer.in_use =
        false;

      buffer.timestamp_ns =
        0;

      buffer.frame_name.clear();

      free_queue_.push(
        buffer_index);
    }

    if (!write_success) {
      continue;
    }

    // ------------------------------------------------------------------------
    // Timing.
    // ------------------------------------------------------------------------

    const auto encode_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
        encode_done - encode_start).count();

    const auto write_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
        write_done - write_start).count();

    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      1000,
      "JPEG camera=%d worker=%u %ux%u jpeg=%zu bytes "
      "encode=%ld ms write=%ld ms file=%s",
      camera_id_,
      writer_id,
      width,
      height,
      actual_jpeg_size,
      encode_ms,
      write_ms,
      filename.c_str());

    // ------------------------------------------------------------------------
    // Frame log.
    // ------------------------------------------------------------------------

    writeFrameLog(
      timestamp_ns,
      std::to_string(camera_id_));
  }

  // ==========================================================================
  // Worker shutdown
  // ==========================================================================

  RCLCPP_INFO(
    get_logger(),
    "Stopping nvJPEG worker %u camera_id=%d",
    writer_id,
    camera_id_);

  cuda_result =
    cudaStreamSynchronize(
      cuda_stream);

  if (cuda_result != cudaSuccess) {

    RCLCPP_ERROR(
      get_logger(),
      "Worker %u: final CUDA synchronize failed: %s",
      writer_id,
      cudaGetErrorString(cuda_result));
  }

  if (encoder_params != nullptr) {

    nvjpegEncoderParamsDestroy(
      encoder_params);

    encoder_params =
      nullptr;
  }

  if (encoder_state != nullptr) {

    nvjpegEncoderStateDestroy(
      encoder_state);

    encoder_state =
      nullptr;
  }

  if (nvjpeg_handle != nullptr) {

    nvjpegDestroy(
      nvjpeg_handle);

    nvjpeg_handle =
      nullptr;
  }

  if (cuda_stream != nullptr) {

    cudaStreamDestroy(
      cuda_stream);

    cuda_stream =
      nullptr;
  }

  RCLCPP_INFO(
    get_logger(),
    "nvJPEG worker %u stopped camera_id=%d",
    writer_id,
    camera_id_);
}


// ============================================================================
// Destructor
// ============================================================================

ArgusThreadNode::~ArgusThreadNode()
{
  RCLCPP_INFO(
    get_logger(),
    "Stopping ArgusThreadNode camera_id=%d",
    camera_id_);

  // --------------------------------------------------------------------------
  // Stop accepting new work.
  // --------------------------------------------------------------------------

  running_.store(false);

  write_cv_.notify_all();

  // --------------------------------------------------------------------------
  // Workers finish all queued JPEG jobs.
  // --------------------------------------------------------------------------

  for (auto & thread : writer_threads_) {

    if (thread.joinable()) {
      thread.join();
    }
  }

  writer_threads_.clear();

  // --------------------------------------------------------------------------
  // The workers are now completely finished with our GPU buffers.
  // --------------------------------------------------------------------------

  if (copy_stream_ != nullptr) {

    cudaError_t cuda_result =
      cudaStreamSynchronize(
        copy_stream_);

    if (cuda_result != cudaSuccess) {

      RCLCPP_ERROR(
        get_logger(),
        "Final copy stream synchronization failed: %s",
        cudaGetErrorString(cuda_result));
    }
  }

  destroyBuffers();

  // --------------------------------------------------------------------------
  // Destroy copy stream.
  // --------------------------------------------------------------------------

  if (copy_stream_ != nullptr) {

    cudaError_t cuda_result =
      cudaStreamDestroy(
        copy_stream_);

    if (cuda_result != cudaSuccess) {

      RCLCPP_ERROR(
        get_logger(),
        "cudaStreamDestroy failed: %s",
        cudaGetErrorString(cuda_result));
    }

    copy_stream_ =
      nullptr;
  }

  RCLCPP_INFO(
    get_logger(),
    "ArgusThreadNode stopped camera_id=%d",
    camera_id_);
}


}  // namespace argus
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::argus::ArgusThreadNode)

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

#pragma pack(push, 1)
struct FrameHeader {
  uint32_t magic = 0x4A504731; // "JPG1"
  uint32_t camera_id;
  uint32_t jpeg_size;
  uint64_t timestamp_ns;
};
#pragma pack(pop)

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
    save_path_ = declare_parameter<std::string>("save_path", "./default_thread_frames/");
  }

  if (!save_path_.empty() && save_path_.back() != '/') {
    save_path_ += "/";
  }

  std::filesystem::create_directories(save_path_);

  if (has_parameter("camera_name")) {
    get_parameter("camera_name", camera_name_);
  } else {
    camera_name_ = declare_parameter<std::string>("camera_name", "./output_frames/");
  }


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
    "Started %zu nvJPEG writer threads for camera_name=%s",
    NUM_WRITERS,
    camera_name_.c_str());

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
    "Initializing GPU buffer pool: buffers=%zu "
    "width=%u height=%u stride=%zu size=%zu bytes",
    NUM_BUFFERS,
    width,
    height,
    stride,
    allocation_size);

  // --------------------------------------------------------------------------
  // Allocate all buffers.
  // --------------------------------------------------------------------------

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
    "GPU buffer pool initialized: %zu buffers",
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
// ============================================================================

void ArgusThreadNode::CustomImageCallback(
  const gxf_context_t context,
  nitros::NitrosTypeBase & msg,
  const std::string frame_name)
{
  if (!running_.load()) {
    return;
  }

  const std::filesystem::path trigger_file =
      std::string("/tmp/CAM_RECORD_FLAG_") + camera_name_;

  if (camera_name_.empty()) {
      RCLCPP_WARN(this->get_logger(), "CALLBACK Parameter 'camera_name' is empty! Defaulting to 'unknown_camera'.");
      camera_name_ = "unknown_camera";
  }


  bool file_exists = std::filesystem::exists(trigger_file);

  if (!file_exists) {

      if (!was_missing_) {
          RCLCPP_WARN(
              get_logger(),
              "%s NOT EXISTING - SKIP SAVING",
              trigger_file.c_str());
          was_missing_ = true;
      }
      return;
  } 

  if (was_missing_) {
      RCLCPP_WARN(get_logger(), "%s SAVING.", trigger_file.c_str());
      was_missing_ = false;
  }

  // --------------------------------------------------------------------------
  // Create a temporary shared reference to the GXF entity.
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
  // Get an application-owned buffer (using free_mutex_ only)
  // --------------------------------------------------------------------------

  size_t buffer_index = 0;

  {
    std::lock_guard<std::mutex> lock(
      free_mutex_);

    if (free_queue_.empty()) {

      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        1000,
        "GPU buffer pool exhausted for camera_name=%s; "
        "dropping frame",
        camera_name_.c_str());

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
        free_mutex_);

      buffer.in_use =
        false;

      free_queue_.push(
        buffer_index);
    }

    return;
  }

  cudaDeviceSynchronize();

  const auto copy_done =
    std::chrono::steady_clock::now();

  buffer.timestamp_ns =
    timestamp_ns;

  buffer.frame_name =
    frame_name;

  // --------------------------------------------------------------------------
  // Queue our application-owned buffer (using write_mutex_ only)
  // --------------------------------------------------------------------------

  {
    std::lock_guard<std::mutex> lock(
      write_mutex_);

    if (write_queue_.size() >= MAX_WRITE_QUEUE_SIZE) {

      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        1000,
        "JPEG queue full for camera_name=%s (%zu), "
        "dropping copied frame",
        camera_name_.c_str(),
        write_queue_.size());

      // Return buffer to pool using free_mutex_
      std::lock_guard<std::mutex> free_lock(
        free_mutex_);

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
    "camera_name=%s copied frame into buffer=%zu in %ld us",
    camera_name_.c_str(),
    buffer_index,
    copy_ms);
}


// ============================================================================
// Write JPEG file
// ============================================================================

// bool ArgusThreadNode::writeJpegFile(
//   const std::string & filename,
//   const unsigned char * jpeg_data,
//   size_t jpeg_size)
// {
//   FILE * outfile =
//     fopen(
//       filename.c_str(),
//       "wb");

//   if (outfile == nullptr) {

//     RCLCPP_ERROR(
//       get_logger(),
//       "Failed to open JPEG file: %s",
//       filename.c_str());

//     return false;
//   }

//   const size_t written =
//     fwrite(
//       jpeg_data,
//       1,
//       jpeg_size,
//       outfile);

//   fclose(outfile);

//   if (written != jpeg_size) {

//     RCLCPP_ERROR(
//       get_logger(),
//       "JPEG write failed: file=%s expected=%zu written=%zu",
//       filename.c_str(),
//       jpeg_size,
//       written);

//     return false;
//   }

//   return true;
// }

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

// ============================================================================
// Continuous Stream JPEG Writer Loop
// ============================================================================
void ArgusThreadNode::writerLoop(uint32_t writer_id)
{
  RCLCPP_INFO(
    get_logger(),
    "nvJPEG stream worker %u starting for camera_name=%s",
    writer_id,
    camera_name_.c_str());

  // --------------------------------------------------------------------------
  // Initialize CUDA stream and nvJPEG objects per worker thread
  // --------------------------------------------------------------------------
  cudaStream_t cuda_stream = nullptr;
  cudaError_t cuda_result = cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);
  if (cuda_result != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "Worker %u: cudaStreamCreate failed: %s", writer_id, cudaGetErrorString(cuda_result));
    return;
  }

  nvjpegHandle_t nvjpeg_handle = nullptr;
  if (nvjpegCreateSimple(&nvjpeg_handle) != NVJPEG_STATUS_SUCCESS) {
    RCLCPP_ERROR(get_logger(), "Worker %u: nvjpegCreateSimple failed", writer_id);
    cudaStreamDestroy(cuda_stream);
    return;
  }

  nvjpegEncoderState_t encoder_state = nullptr;
  nvjpegEncoderParams_t encoder_params = nullptr;

  if (nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, cuda_stream) != NVJPEG_STATUS_SUCCESS ||
      nvjpegEncoderParamsCreate(nvjpeg_handle, &encoder_params, cuda_stream) != NVJPEG_STATUS_SUCCESS) {
    RCLCPP_ERROR(get_logger(), "Worker %u: nvjpeg state/params creation failed", writer_id);
    if (encoder_state) nvjpegEncoderStateDestroy(encoder_state);
    nvjpegDestroy(nvjpeg_handle);
    cudaStreamDestroy(cuda_stream);
    return;
  }

  nvjpegEncoderParamsSetQuality(encoder_params, 95, cuda_stream);
  nvjpegEncoderParamsSetSamplingFactors(encoder_params, NVJPEG_CSS_420, cuda_stream);

  // File handle for continuous binary streaming
  FILE * stream_file = nullptr;
  uint64_t current_chunk_bin = 0;

  RCLCPP_INFO(get_logger(), "nvJPEG worker %u fully initialized", writer_id);

  // ==========================================================================
  // Main Processing Loop
  // ==========================================================================
  while (running_.load()) {
    size_t buffer_index = 0;

    // ------------------------------------------------------------------------
    // 1. Pop next available buffer index from write_queue_
    // ------------------------------------------------------------------------
    {
      std::unique_lock<std::mutex> lock(write_mutex_);
      write_cv_.wait(
        lock,
        [this]() {
          return !write_queue_.empty() || !running_.load();
        });

      if (!running_.load() && write_queue_.empty()) {
        break;
      }

      buffer_index = write_queue_.front();
      write_queue_.pop();
    }

    FrameBuffer & buffer = buffers_[buffer_index];

    // Safety check against race conditions
    if (!buffer.in_use) {
      RCLCPP_ERROR(get_logger(), "Worker: Buffer %zu was not marked in_use! Skipping.", buffer_index);
      continue;
    }

    if (buffer.gpu_data == nullptr) {
      RCLCPP_ERROR(get_logger(), "Worker %u: buffer %zu GPU memory is null", writer_id, buffer_index);
      std::lock_guard<std::mutex> lock(free_mutex_);
      buffer.in_use = false;
      free_queue_.push(buffer_index);
      continue;
    }

    const uint32_t width = buffer.width;
    const uint32_t height = buffer.height;
    const size_t stride = buffer.stride;
    const uint64_t timestamp_ns = buffer.timestamp_ns;

    // ------------------------------------------------------------------------
    // 2. Perform GPU JPEG encoding
    // ------------------------------------------------------------------------
    nvjpegImage_t image{};
    image.channel[0] = reinterpret_cast<unsigned char *>(buffer.gpu_data);
    image.pitch[0] = stride;

    if (nvjpegEncodeImage(nvjpeg_handle, encoder_state, encoder_params, &image,
                          NVJPEG_INPUT_RGBI, static_cast<int>(width),
                          static_cast<int>(height), cuda_stream) != NVJPEG_STATUS_SUCCESS) {
      RCLCPP_ERROR(get_logger(), "Worker %u: nvjpegEncodeImage failed", writer_id);
      std::lock_guard<std::mutex> lock(free_mutex_);
      buffer.in_use = false;
      free_queue_.push(buffer_index);
      continue;
    }

    size_t jpeg_size = 0;
    if (nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, nullptr, &jpeg_size, cuda_stream) != NVJPEG_STATUS_SUCCESS || jpeg_size == 0) {
      std::lock_guard<std::mutex> lock(free_mutex_);
      buffer.in_use = false;
      free_queue_.push(buffer_index);
      continue;
    }

    std::vector<unsigned char> jpeg_data(jpeg_size);
    size_t actual_jpeg_size = jpeg_size;

    if (nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, jpeg_data.data(), &actual_jpeg_size, cuda_stream) != NVJPEG_STATUS_SUCCESS) {
      std::lock_guard<std::mutex> lock(free_mutex_);
      buffer.in_use = false;
      free_queue_.push(buffer_index);
      continue;
    }

    // Synchronize CUDA stream so CPU data vector is fully ready
    cudaStreamSynchronize(cuda_stream);

    // ------------------------------------------------------------------------
    // 3. IMMEDIATELY return buffer to free_queue_ before disk I/O
    //    (Prevents "GPU buffer pool exhausted" callback drops)
    // ------------------------------------------------------------------------

    // ------------------------------------------------------------------------
    // 4. File Rotation Logic
    // ------------------------------------------------------------------------
// ------------------------------------------------------------------------
    // 4. File Rotation Logic (Per-Camera Subfolders)
    // ------------------------------------------------------------------------
    constexpr uint64_t CHUNK_NS = 10000000000ULL; // 100 seconds
    uint64_t chunk_bin = timestamp_ns / CHUNK_NS;

    if (stream_file == nullptr || chunk_bin != current_chunk_bin) {
      if (stream_file != nullptr) {
        fclose(stream_file);
        stream_file = nullptr;
      }

      // 1. Build subfolder path for this camera: /tmp/camera_data/cam_1/
      std::filesystem::path camera_folder = 
        std::filesystem::path(save_path_) / ("cam_" + camera_name_);

      // 2. Ensure camera subfolder exists before opening file
      std::error_code ec;
      std::filesystem::create_directories(camera_folder, ec);

      // 3. Build sequence stamp & full file path
      uint64_t sequence_stamp = chunk_bin * CHUNK_NS;
      std::string stream_filename =
        (camera_folder / (std::to_string(sequence_stamp) + ".bin")).string();

      stream_file = fopen(stream_filename.c_str(), "ab");
      if (stream_file == nullptr) {
        RCLCPP_ERROR(get_logger(), "Worker %u: Failed to open stream file: %s", writer_id, stream_filename.c_str());
        continue;
      }

      current_chunk_bin = chunk_bin;
    }

    // ------------------------------------------------------------------------
    // 5. Append Frame Header & JPEG payload to continuous binary file
    // ------------------------------------------------------------------------
    FrameHeader header;
    header.camera_id = camera_id_;
    header.jpeg_size = static_cast<uint32_t>(actual_jpeg_size);
    header.timestamp_ns = timestamp_ns;

    if (fwrite(&header, sizeof(FrameHeader), 1, stream_file) == 1) {
      fwrite(jpeg_data.data(), 1, actual_jpeg_size, stream_file);
      writeFrameLog(timestamp_ns, camera_name_);
    } else {
      RCLCPP_ERROR(get_logger(), "Worker %u: Failed to write frame header to disk", writer_id);
    }

    {
      std::lock_guard<std::mutex> lock(free_mutex_);
      buffer.in_use = false;
      buffer.timestamp_ns = 0;
      buffer.frame_name.clear();
      free_queue_.push(buffer_index);
    }

  }

  // Cleanup file handle and CUDA/nvJPEG context on shutdown
  if (stream_file != nullptr) {
    fclose(stream_file);
  }

  nvjpegEncoderParamsDestroy(encoder_params);
  nvjpegEncoderStateDestroy(encoder_state);
  nvjpegDestroy(nvjpeg_handle);
  cudaStreamDestroy(cuda_stream);
}



// void ArgusThreadNode::writerLoop(
//   uint32_t writer_id)
// {
//   RCLCPP_INFO(
//     get_logger(),
//     "nvJPEG worker %u starting for camera_name=%s",
//     writer_id,
//     camera_name_.c_str());

//   // --------------------------------------------------------------------------
//   // Each worker owns its own CUDA stream.
//   // --------------------------------------------------------------------------

//   cudaStream_t cuda_stream =
//     nullptr;

//   cudaError_t cuda_result =
//     cudaStreamCreateWithFlags(
//       &cuda_stream,
//       cudaStreamNonBlocking);

//   if (cuda_result != cudaSuccess) {

//     RCLCPP_ERROR(
//       get_logger(),
//       "Worker %u: cudaStreamCreate failed: %s",
//       writer_id,
//       cudaGetErrorString(cuda_result));

//     return;
//   }

//   // --------------------------------------------------------------------------
//   // Each worker owns its own nvJPEG handle.
//   // --------------------------------------------------------------------------

//   nvjpegHandle_t nvjpeg_handle =
//     nullptr;

//   nvjpegStatus_t nv_result =
//     nvjpegCreateSimple(
//       &nvjpeg_handle);

//   if (nv_result != NVJPEG_STATUS_SUCCESS) {

//     RCLCPP_ERROR(
//       get_logger(),
//       "Worker %u: nvjpegCreateSimple failed: %d",
//       writer_id,
//       static_cast<int>(nv_result));

//     cudaStreamDestroy(
//       cuda_stream);

//     return;
//   }

//   // --------------------------------------------------------------------------
//   // Encoder state.
//   // --------------------------------------------------------------------------

//   nvjpegEncoderState_t encoder_state =
//     nullptr;

//   nv_result =
//     nvjpegEncoderStateCreate(
//       nvjpeg_handle,
//       &encoder_state,
//       cuda_stream);

//   if (nv_result != NVJPEG_STATUS_SUCCESS) {

//     RCLCPP_ERROR(
//       get_logger(),
//       "Worker %u: nvjpegEncoderStateCreate failed: %d",
//       writer_id,
//       static_cast<int>(nv_result));

//     nvjpegDestroy(
//       nvjpeg_handle);

//     cudaStreamDestroy(
//       cuda_stream);

//     return;
//   }

//   // --------------------------------------------------------------------------
//   // Encoder parameters.
//   // --------------------------------------------------------------------------

//   nvjpegEncoderParams_t encoder_params =
//     nullptr;

//   nv_result =
//     nvjpegEncoderParamsCreate(
//       nvjpeg_handle,
//       &encoder_params,
//       cuda_stream);

//   if (nv_result != NVJPEG_STATUS_SUCCESS) {

//     RCLCPP_ERROR(
//       get_logger(),
//       "Worker %u: nvjpegEncoderParamsCreate failed: %d",
//       writer_id,
//       static_cast<int>(nv_result));

//     nvjpegEncoderStateDestroy(
//       encoder_state);

//     nvjpegDestroy(
//       nvjpeg_handle);

//     cudaStreamDestroy(
//       cuda_stream);

//     return;
//   }

//   // --------------------------------------------------------------------------
//   // JPEG quality.
//   // --------------------------------------------------------------------------

//   nv_result =
//     nvjpegEncoderParamsSetQuality(
//       encoder_params,
//       99,
//       cuda_stream);

//   if (nv_result != NVJPEG_STATUS_SUCCESS) {

//     RCLCPP_ERROR(
//       get_logger(),
//       "Worker %u: nvjpegEncoderParamsSetQuality failed: %d",
//       writer_id,
//       static_cast<int>(nv_result));

//     nvjpegEncoderParamsDestroy(
//       encoder_params);

//     nvjpegEncoderStateDestroy(
//       encoder_state);

//     nvjpegDestroy(
//       nvjpeg_handle);

//     cudaStreamDestroy(
//       cuda_stream);

//     return;
//   }

//   // --------------------------------------------------------------------------
//   // JPEG 4:2:0.
//   // --------------------------------------------------------------------------

//   nv_result =
//     nvjpegEncoderParamsSetSamplingFactors(
//       encoder_params,
//       NVJPEG_CSS_420,
//       cuda_stream);

//   if (nv_result != NVJPEG_STATUS_SUCCESS) {

//     RCLCPP_ERROR(
//       get_logger(),
//       "Worker %u: nvjpegEncoderParamsSetSamplingFactors failed: %d",
//       writer_id,
//       static_cast<int>(nv_result));

//     nvjpegEncoderParamsDestroy(
//       encoder_params);

//     nvjpegEncoderStateDestroy(
//       encoder_state);

//     nvjpegDestroy(
//       nvjpeg_handle);

//     cudaStreamDestroy(
//       cuda_stream);

//     return;
//   }

//   RCLCPP_INFO(
//     get_logger(),
//     "nvJPEG worker %u initialized for camera_name=%s",
//     writer_id,
//     camera_name_.c_str());

//   // ==========================================================================
//   // Worker loop
//   // ==========================================================================

//   while (true) {

//     size_t buffer_index = 0;

//     // ------------------------------------------------------------------------
//     // Get buffer index using write_mutex_
//     // ------------------------------------------------------------------------

//     {
//       std::unique_lock<std::mutex> lock(
//         write_mutex_);

//       write_cv_.wait(
//         lock,
//         [this]() {
//           return
//             !write_queue_.empty() ||
//             !running_.load();
//         });

//       if (!running_.load() &&
//           write_queue_.empty())
//       {
//         break;
//       }

//       buffer_index =
//         write_queue_.front();

//       write_queue_.pop();
//     }

//     FrameBuffer & buffer =
//       buffers_[buffer_index];

//     // ------------------------------------------------------------------------
//     // Validate buffer.
//     // ------------------------------------------------------------------------

//     if (buffer.gpu_data == nullptr) {

//       RCLCPP_ERROR(
//         get_logger(),
//         "Worker %u: buffer %zu GPU memory is null",
//         writer_id,
//         buffer_index);

//       std::lock_guard<std::mutex> lock(
//         free_mutex_);

//       buffer.in_use =
//         false;

//       free_queue_.push(
//         buffer_index);

//       continue;
//     }

//     const uint32_t width =
//       buffer.width;

//     const uint32_t height =
//       buffer.height;

//     const size_t stride =
//       buffer.stride;

//     const uint64_t timestamp_ns =
//       buffer.timestamp_ns;

//     // ------------------------------------------------------------------------
//     // Build nvJPEG image.
//     // ------------------------------------------------------------------------

//     nvjpegImage_t image{};

//     image.channel[0] =
//       reinterpret_cast<unsigned char *>(
//         buffer.gpu_data);

//     image.channel[1] =
//       nullptr;

//     image.channel[2] =
//       nullptr;

//     image.channel[3] =
//       nullptr;

//     image.pitch[0] =
//       stride;

//     image.pitch[1] =
//       0;

//     image.pitch[2] =
//       0;

//     image.pitch[3] =
//       0;

//     // ------------------------------------------------------------------------
//     // JPEG encode.
//     // ------------------------------------------------------------------------

//     const auto encode_start =
//       std::chrono::steady_clock::now();

//     nv_result =
//       nvjpegEncodeImage(
//         nvjpeg_handle,
//         encoder_state,
//         encoder_params,
//         &image,
//         NVJPEG_INPUT_RGBI,
//         static_cast<int>(width),
//         static_cast<int>(height),
//         cuda_stream);

//     if (nv_result != NVJPEG_STATUS_SUCCESS) {

//       RCLCPP_ERROR(
//         get_logger(),
//         "Worker %u: nvjpegEncodeImage failed: %d",
//         writer_id,
//         static_cast<int>(nv_result));

//       std::lock_guard<std::mutex> lock(
//         free_mutex_);

//       buffer.in_use =
//         false;

//       free_queue_.push(
//         buffer_index);

//       continue;
//     }

//     // ------------------------------------------------------------------------
//     // Query JPEG size.
//     // ------------------------------------------------------------------------

//     size_t jpeg_size =
//       0;

//     nv_result =
//       nvjpegEncodeRetrieveBitstream(
//         nvjpeg_handle,
//         encoder_state,
//         nullptr,
//         &jpeg_size,
//         cuda_stream);

//     if (nv_result != NVJPEG_STATUS_SUCCESS ||
//         jpeg_size == 0)
//     {
//       RCLCPP_ERROR(
//         get_logger(),
//         "Worker %u: failed to get JPEG size: status=%d size=%zu",
//         writer_id,
//         static_cast<int>(nv_result),
//         jpeg_size);

//       std::lock_guard<std::mutex> lock(
//         free_mutex_);

//       buffer.in_use =
//         false;

//       free_queue_.push(
//         buffer_index);

//       continue;
//     }

//     // ------------------------------------------------------------------------
//     // Allocate host JPEG buffer.
//     // ------------------------------------------------------------------------

//     std::vector<unsigned char> jpeg_data(
//       jpeg_size);

//     size_t actual_jpeg_size =
//       jpeg_size;

//     // ------------------------------------------------------------------------
//     // Retrieve JPEG bitstream.
//     // ------------------------------------------------------------------------

//     nv_result =
//       nvjpegEncodeRetrieveBitstream(
//         nvjpeg_handle,
//         encoder_state,
//         jpeg_data.data(),
//         &actual_jpeg_size,
//         cuda_stream);

//     if (nv_result != NVJPEG_STATUS_SUCCESS) {

//       RCLCPP_ERROR(
//         get_logger(),
//         "Worker %u: nvjpegEncodeRetrieveBitstream failed: %d",
//         writer_id,
//         static_cast<int>(nv_result));

//       std::lock_guard<std::mutex> lock(
//         free_mutex_);

//       buffer.in_use =
//         false;

//       free_queue_.push(
//         buffer_index);

//       continue;
//     }

//     // ------------------------------------------------------------------------
//     // Wait for JPEG GPU work to finish before CPU accesses jpeg_data.
//     // ------------------------------------------------------------------------

//     cuda_result =
//       cudaStreamSynchronize(
//         cuda_stream);

//     if (cuda_result != cudaSuccess) {

//       RCLCPP_ERROR(
//         get_logger(),
//         "Worker %u: cudaStreamSynchronize failed: %s",
//         writer_id,
//         cudaGetErrorString(cuda_result));

//       std::lock_guard<std::mutex> lock(
//         free_mutex_);

//       buffer.in_use =
//         false;

//       free_queue_.push(
//         buffer_index);

//       continue;
//     }

//     const auto encode_done =
//       std::chrono::steady_clock::now();

//     // ------------------------------------------------------------------------
//     // Filename.
//     // ------------------------------------------------------------------------

//     // const std::string filename =
//     //   save_path_ +
//     //   "argus_frame_" +
//     //   camera_name_ +
//     //   "_" +
//     //   std::to_string(timestamp_ns) +
//     //   ".jpg";


//     // // 1. Get current local system time (minute resolution)
//     // auto now = std::chrono::system_clock::now();
//     // std::time_t time_t_now = std::chrono::system_clock::to_time_t(now);
//     // std::tm tm_now = *std::localtime(&time_t_now);

//     // std::ostringstream minute_folder_ss;
//     // minute_folder_ss << std::put_time(&tm_now, "%Y-%m-%d_%H-%M");

//     // // 2. Build full minute-wise folder path (e.g., "/tmp/camera_data/2026-10-07_16-55/")
//     // std::filesystem::path folder_path = std::filesystem::path(save_path_) / minute_folder_ss.str();

//     // // 3. Ensure directory exists before constructing filename
//     // std::error_code ec;
//     // std::filesystem::create_directories(folder_path, ec);

//     // // 4. Build final filename path
//     // const std::string filename =
//     //   (folder_path / ("argus_frame_" +
//     //                   camera_name_ +
//     //                   "_" +
//     //                   std::to_string(timestamp_ns) +
//     //                   ".jpg")).string();

//     // 100 seconds in nanoseconds (100 * 10^9)
//     constexpr uint64_t NS_PER_10_SEC = 10000000000ULL;

//     // Integer division isolates the 100-second bin (e.g., 4161... / 10^11 = 41)
//     uint64_t folder_bin = timestamp_ns / NS_PER_10_SEC;

//     std::string folder_name = std::to_string(folder_bin);

//     // 1. Build directory path: /tmp/camera_data/41/
//     std::filesystem::path folder_path = std::filesystem::path(save_path_) / folder_name;

//     // 2. Ensure directory exists
//     std::error_code ec;
//     std::filesystem::create_directories(folder_path, ec);

//     // 3. Build final file path: /tmp/camera_data/41/argus_frame_1_4161879387000.jpg
//     const std::string filename =
//       (folder_path / ("argus_frame_" +
//                       camera_name_ +
//                       "_" +
//                       std::to_string(timestamp_ns) +
//                       ".jpg")).string();

//     // ------------------------------------------------------------------------
//     // Write JPEG.
//     // ------------------------------------------------------------------------

//     const auto write_start =
//       std::chrono::steady_clock::now();

//     const bool write_success =
//       writeJpegFile(
//         filename,
//         jpeg_data.data(),
//         actual_jpeg_size);

//     const auto write_done =
//       std::chrono::steady_clock::now();

//     // ------------------------------------------------------------------------
//     // Return GPU buffer to pool using free_mutex_
//     // ------------------------------------------------------------------------

//     {
//       std::lock_guard<std::mutex> lock(
//         free_mutex_);

//       buffer.in_use =
//         false;

//       buffer.timestamp_ns =
//         0;

//       buffer.frame_name.clear();

//       free_queue_.push(
//         buffer_index);
//     }

//     if (!write_success) {
//       continue;
//     }

//     // ------------------------------------------------------------------------
//     // Timing.
//     // ------------------------------------------------------------------------

//     const auto encode_ms =
//       std::chrono::duration_cast<std::chrono::milliseconds>(
//         encode_done - encode_start).count();

//     const auto write_ms =
//       std::chrono::duration_cast<std::chrono::milliseconds>(
//         write_done - write_start).count();

//     RCLCPP_INFO_THROTTLE(
//       get_logger(),
//       *get_clock(),
//       10000,
//       "JPEG camera=%d worker=%u %ux%u jpeg=%zu bytes "
//       "encode=%ld ms write=%ld ms file=%s",
//       camera_name_.c_str(),
//       writer_id,
//       width,
//       height,
//       actual_jpeg_size,
//       encode_ms,
//       write_ms,
//       filename.c_str());

//     // ------------------------------------------------------------------------
//     // Frame log.
//     // ------------------------------------------------------------------------

//     writeFrameLog(
//       timestamp_ns,
//       camera_name_);
//   }

//   // --------------------------------------------------------------------------
//   // Cleanup worker resources upon loop exit
//   // --------------------------------------------------------------------------

//   nvjpegEncoderParamsDestroy(encoder_params);
//   nvjpegEncoderStateDestroy(encoder_state);
//   nvjpegDestroy(nvjpeg_handle);
//   cudaStreamDestroy(cuda_stream);
// }

// // ============================================================================
// // Single Batch Binary Container Header
// // ============================================================================
// struct BatchHeader {
//   uint32_t magic = 0x4A504742; // "JPGB"
//   uint32_t num_frames = 0;
//   uint32_t frame_sizes[4];
//   uint64_t timestamps[4];
// };

// // ============================================================================
// // JPEG worker (Processes batches of 4 frames into a single file)
// // ============================================================================
// void ArgusThreadNode::writerLoop(uint32_t writer_id)
// {
//   RCLCPP_INFO(get_logger(), "nvJPEG worker %u starting for camera_name=%s", writer_id, camera_name_.c_str());

//   constexpr size_t BATCH_SIZE = 4;

//   cudaStream_t cuda_stream = nullptr;
//   cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);

//   nvjpegHandle_t nvjpeg_handle = nullptr;
//   nvjpegCreateSimple(&nvjpeg_handle);

//   nvjpegEncoderState_t encoder_state = nullptr;
//   nvjpegEncoderParams_t encoder_params = nullptr;

//   nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, cuda_stream);
//   nvjpegEncoderParamsCreate(nvjpeg_handle, &encoder_params, cuda_stream);
//   nvjpegEncoderParamsSetQuality(encoder_params, 99, cuda_stream);
//   nvjpegEncoderParamsSetSamplingFactors(encoder_params, NVJPEG_CSS_420, cuda_stream);

//   while (true) {
//     std::vector<size_t> batch_indices;
//     batch_indices.reserve(BATCH_SIZE);

//     // 1. Wait for AT LEAST 4 frames (or flush remaining on shutdown)
//     {
//       std::unique_lock<std::mutex> lock(write_mutex_);

//       write_cv_.wait(
//         lock,
//         [this, BATCH_SIZE]() {
//           return write_queue_.size() >= BATCH_SIZE || !running_.load();
//         });

//       if (!running_.load() && write_queue_.empty()) {
//         break;
//       }

//       while (!write_queue_.empty() && batch_indices.size() < BATCH_SIZE) {
//         batch_indices.push_back(write_queue_.front());
//         write_queue_.pop();
//       }
//     }

//     if (batch_indices.empty()) {
//       continue;
//     }

//     const size_t current_batch_size = batch_indices.size();
//     BatchHeader header;
//     header.num_frames = static_cast<uint32_t>(current_batch_size);

//     std::vector<std::vector<unsigned char>> encoded_jpegs(current_batch_size);

//     // 2. Sequential CUDA-stream queued batch encoding
//     for (size_t i = 0; i < current_batch_size; ++i) {
//       size_t buffer_index = batch_indices[i];
//       FrameBuffer & buffer = buffers_[buffer_index];

//       header.timestamps[i] = buffer.timestamp_ns;

//       nvjpegImage_t image{};
//       image.channel[0] = reinterpret_cast<unsigned char *>(buffer.gpu_data);
//       image.pitch[0] = buffer.stride;

//       nvjpegStatus_t status = nvjpegEncodeImage(
//         nvjpeg_handle,
//         encoder_state,
//         encoder_params,
//         &image,
//         NVJPEG_INPUT_RGBI,
//         static_cast<int>(buffer.width),
//         static_cast<int>(buffer.height),
//         cuda_stream);

//       if (status == NVJPEG_STATUS_SUCCESS) {
//         size_t jpeg_size = 0;
//         nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, nullptr, &jpeg_size, cuda_stream);

//         encoded_jpegs[i].resize(jpeg_size);
//         size_t actual_size = jpeg_size;

//         nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, encoded_jpegs[i].data(), &actual_size, cuda_stream);
//         header.frame_sizes[i] = static_cast<uint32_t>(actual_size);
//       }
//     }

//     // Synchronize stream so CPU bitstream buffer data is ready
//     cudaStreamSynchronize(cuda_stream);

//     // 3. Return GPU buffers to free_queue IMMEDIATELY (Prevents callback pool exhaustion)
//     {
//       std::lock_guard<std::mutex> lock(free_mutex_);
//       for (size_t buffer_index : batch_indices) {
//         buffers_[buffer_index].in_use = false;
//         buffers_[buffer_index].timestamp_ns = 0;
//         buffers_[buffer_index].frame_name.clear();
//         free_queue_.push(buffer_index);
//       }
//     }

//     // 4. Single continuous write operation to NVMe (Fixes NVMe overheating)
//     constexpr uint64_t NS_PER_100_SEC = 100000000000ULL;
//     uint64_t folder_bin = header.timestamps[0] / NS_PER_100_SEC;
//     std::filesystem::path folder_path = std::filesystem::path(save_path_) / std::to_string(folder_bin);

//     std::error_code ec;
//     std::filesystem::create_directories(folder_path, ec);

//     std::string batch_filename = (folder_path / ("batch_cam_" + camera_name_ + "_" +
//                                   std::to_string(header.timestamps[0]) + ".bin")).string();

//     FILE * outfile = fopen(batch_filename.c_str(), "wb");
//     if (outfile != nullptr) {
//       fwrite(&header, sizeof(BatchHeader), 1, outfile);

//       for (size_t i = 0; i < header.num_frames; ++i) {
//         if (!encoded_jpegs[i].empty()) {
//           fwrite(encoded_jpegs[i].data(), 1, header.frame_sizes[i], outfile);
//         }
//       }
//       fclose(outfile);

//       for (size_t i = 0; i < header.num_frames; ++i) {
//         writeFrameLog(header.timestamps[i], camera_name_);
//       }
//     }
//   }

//   // Cleanup
//   nvjpegEncoderParamsDestroy(encoder_params);
//   nvjpegEncoderStateDestroy(encoder_state);
//   nvjpegDestroy(nvjpeg_handle);
//   cudaStreamDestroy(cuda_stream);
// }


// ============================================================================
// Destructor
// ============================================================================

ArgusThreadNode::~ArgusThreadNode()
{
  // 1. Signal workers to stop and unblock waiting threads
  running_.store(false);
  write_cv_.notify_all();

  // 2. Join all worker threads
  for (auto & thread : writer_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  // 3. Destroy CUDA stream
  if (copy_stream_ != nullptr) {
    cudaStreamDestroy(copy_stream_);
    copy_stream_ = nullptr;
  }

  // 4. Free GPU buffer pool allocations
  destroyBuffers();
}


}  // namespace argus
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::argus::ArgusThreadNode)
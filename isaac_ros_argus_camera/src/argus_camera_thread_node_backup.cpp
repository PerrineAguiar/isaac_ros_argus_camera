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

#include <fstream>
#include <string>
#include <vector>

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

constexpr char PACKAGE_NAME[] =
  "isaac_ros_argus_camera";

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
// Initialize pinned CPU buffers
// ============================================================================

bool ArgusThreadNode::initializeBuffers(
  uint32_t width,
  uint32_t height,
  size_t buffer_size)
{
  RCLCPP_INFO(
    get_logger(),
    "Initializing %zu pinned buffers",
    NUM_BUFFERS);

  RCLCPP_INFO(
    get_logger(),
    "Frame dimensions: %ux%u",
    width,
    height);

  RCLCPP_INFO(
    get_logger(),
    "Frame buffer size: %zu bytes",
    buffer_size);

  // --------------------------------------------------------------------------
  // Create CUDA stream.
  // --------------------------------------------------------------------------

  cudaError_t result =
    cudaStreamCreate(&copy_stream_);

  if (result != cudaSuccess) {
    RCLCPP_ERROR(
      get_logger(),
      "cudaStreamCreate failed: %s",
      cudaGetErrorString(result));

    return false;
  }

  // --------------------------------------------------------------------------
  // Allocate pinned host buffers.
  // --------------------------------------------------------------------------

  for (size_t i = 0; i < NUM_BUFFERS; ++i) {

    auto & buffer = buffers_[i];

    buffer.width = width;
    buffer.height = height;
    buffer.size = buffer_size;

    result =
      cudaHostAlloc(
        reinterpret_cast<void **>(&buffer.data),
        buffer_size,
        cudaHostAllocDefault);

    if (result != cudaSuccess) {

      RCLCPP_ERROR(
        get_logger(),
        "cudaHostAlloc failed for buffer %zu: %s",
        i,
        cudaGetErrorString(result));

      return false;
    }

    // ------------------------------------------------------------------------
    // Event used to determine when GPU -> CPU copy has completed.
    // ------------------------------------------------------------------------

    result =
      cudaEventCreateWithFlags(
        &buffer.copy_done,
        cudaEventDisableTiming);

    if (result != cudaSuccess) {

      RCLCPP_ERROR(
        get_logger(),
        "cudaEventCreate failed for buffer %zu: %s",
        i,
        cudaGetErrorString(result));

      return false;
    }

    // Buffer starts as free.
    free_queue_.push(i);
  }

  RCLCPP_INFO(
    get_logger(),
    "Pinned buffer initialization complete");

  return true;
}


// ============================================================================
// Image callback
// ============================================================================

void ArgusThreadNode::CustomImageCallback(
  const gxf_context_t context,
  nitros::NitrosTypeBase & msg,
  const std::string frame_name)
{

  if (!running_) {
    return;
  }

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
    msg_entity->get<nvidia::gxf::VideoBuffer>();

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

  const size_t buffer_size =
    video->size();


  // RCLCPP_INFO(
  //   get_logger(),
  //   "VideoBuffer: %ux%u format=%d storage_type=%d size=%lu pointer=%p",
  //   info.width,
  //   info.height,
  //   static_cast<int>(info.color_format),
  //   static_cast<int>(video->storage_type()),
  //   static_cast<unsigned long>(video->size()),
  //   static_cast<void *>(video->pointer()));

  // RCLCPP_INFO(
  //   get_logger(),
  //     "width=%u height=%u format=%d planes=%zu",
  //     info.width,
  //     info.height,
  //     static_cast<int>(info.color_format),
  //     info.color_planes.size());

  // for (size_t i = 0; i < info.color_planes.size(); ++i) {
  //   RCLCPP_INFO(
  //     get_logger(),
  //         "plane[%zu]: stride=%u width=%u height=%u bpp=%u offset=%u size=%lu",
  //         i,
  //         info.color_planes[i].stride,
  //         info.color_planes[i].width,
  //         info.color_planes[i].height,
  //         info.color_planes[i].bytes_per_pixel,
  //         info.color_planes[i].offset,
  //         info.color_planes[i].size);
  }



  // --------------------------------------------------------------------------
  // Get Argus/GXF acquisition timestamp.
  // --------------------------------------------------------------------------

  uint64_t timestamp_ns = 0;

  auto gxf_timestamp =
    msg_entity->get<nvidia::gxf::Timestamp>();

  if (!gxf_timestamp) {
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

  // --------------------------------------------------------------------------
  // Buffer
  // --------------------------------------------------------------------------

  // if (!buffers_initialized_) {

  //   if (!initializeBuffers(
  //       width,
  //       height,
  //       buffer_size))
  //   {
  //     RCLCPP_ERROR(
  //       get_logger(),
  //       "Failed to initialize pinned buffers");

  //     return;
  //   }

  //   buffers_initialized_ = true;

  //   // // Start writer thread after buffers have been initialized.
  //   // writer_thread_ =
  //   //   std::thread(
  //   //     &ArgusThreadNode::writerLoop,
  //   //     this);

  //   // RCLCPP_INFO(
  //   //   get_logger(),
  //   //   "Writer thread started");
  //   constexpr size_t NUM_WRITERS = 4;

  //   for (size_t i = 0; i < NUM_WRITERS; ++i) {
  //     writer_threads_.emplace_back(
  //       &ArgusThreadNode::writerLoop,
  //       this);
  //   }

  //   RCLCPP_INFO(
  //     get_logger(),
  //     "Started %zu writer threads",
  //     NUM_WRITERS);

  // }

  if (!buffers_initialized_) {
    std::lock_guard<std::mutex> lock(initialization_mutex_);

    if (!buffers_initialized_) {
      if (!initializeBuffers(
          width,
          height,
          buffer_size))
      {
        RCLCPP_ERROR(
          get_logger(),
          "Failed to initialize pinned buffers");

        return;
      }

      buffers_initialized_ = true;

      for (size_t i = 0; i < NUM_WRITERS; ++i) {
        writer_threads_.emplace_back(
          &ArgusThreadNode::writerLoop,
          this,
          i);
      }

      RCLCPP_INFO(
        get_logger(),
        "Started %zu writer threads",
        NUM_WRITERS);
    }
  }
   
  size_t buffer_index;

  // {
  //   std::unique_lock<std::mutex> lock(
  //     buffer_mutex_);

  //   if (free_queue_.empty()) {
  // RCLCPP_WARN(
  //   get_logger(),
  //   "BUFFER FULL: waiting for a free buffer...");
  //   }
      

  //   buffer_cv_.wait(
  //     lock,
  //     [this]() {
  //       return
  //         !free_queue_.empty() ||
  //         !running_;
  //     });

  //   if (!running_) {
  //     return;
  //   }

  //   buffer_index =
  //     free_queue_.front();

  //   free_queue_.pop();
  // }

  {
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    if (free_queue_.empty()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        1000,
        "Buffer pool full, dropping frame");

      return;
    }

    buffer_index = free_queue_.front();
    free_queue_.pop();
  }



  auto & buffer =
    buffers_[buffer_index];

  if (buffer_size > buffer.size) {

    RCLCPP_ERROR(
      get_logger(),
      "Incoming frame is larger than pinned buffer: "
      "%zu > %zu",
      buffer_size,
      buffer.size);

    {
      std::lock_guard<std::mutex> lock(
        buffer_mutex_);

      free_queue_.push(buffer_index);
    }

    buffer_cv_.notify_one();

    return;
  }

  // --------------------------------------------------------------------------
  // Store metadata.
  // --------------------------------------------------------------------------

  buffer.width =
    width;

  buffer.height =
    height;

  buffer.size =
    buffer_size;

  buffer.timestamp_ns =
    timestamp_ns;

  buffer.frame_name =
    frame_name;

  // --------------------------------------------------------------------------
  // Get GPU pointer.
  // --------------------------------------------------------------------------

  auto * gpu_ptr =
    video->pointer();

  if (gpu_ptr == nullptr) {

    RCLCPP_ERROR(
      get_logger(),
      "VideoBuffer GPU pointer is null");

    {
      std::lock_guard<std::mutex> lock(
        buffer_mutex_);

      free_queue_.push(buffer_index);
    }

    buffer_cv_.notify_one();

    return;
  }

  cudaError_t result =
    cudaMemcpyAsync(
      buffer.data,
      gpu_ptr,
      buffer_size,
      cudaMemcpyDeviceToHost,
      copy_stream_);

  if (result != cudaSuccess) {

    RCLCPP_ERROR(
      get_logger(),
      "cudaMemcpyAsync failed: %s",
      cudaGetErrorString(result));

    {
      std::lock_guard<std::mutex> lock(
        buffer_mutex_);

      free_queue_.push(buffer_index);
    }

    buffer_cv_.notify_one();

    return;
  }

  // --------------------------------------------------------------------------
  // Record event after the async copy.
  // --------------------------------------------------------------------------

  result =
    cudaEventRecord(
      buffer.copy_done,
      copy_stream_);

  if (result != cudaSuccess) {

    RCLCPP_ERROR(
      get_logger(),
      "cudaEventRecord failed: %s",
      cudaGetErrorString(result));

    {
      std::lock_guard<std::mutex> lock(
        buffer_mutex_);

      free_queue_.push(buffer_index);
    }

    buffer_cv_.notify_one();

    return;
  }

  // --------------------------------------------------------------------------
  // Put buffer into writer queue.
  // --------------------------------------------------------------------------

  {
    std::lock_guard<std::mutex> lock(
      write_mutex_);

    write_queue_.push(
      buffer_index);

    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      5000,
      "write_queue length: %zu",
      write_queue_.size());

  }

  write_cv_.notify_one();
  // RCLCPP_INFO(
  //   get_logger(),
  //   "Timestamp: %lu",
  //   static_cast<unsigned long>(timestamp_ns));

  
  // --------------------------------------------------------------------------
  // Callback returns immediately.
  // --------------------------------------------------------------------------
}


// ============================================================================
// Writer thread
// ============================================================================

void ArgusThreadNode::writerLoop(
  uint32_t writer_id)
{
  RCLCPP_INFO(
    get_logger(),
    "Frame writer thread running");

  while (true) {

    size_t buffer_index;

    // ------------------------------------------------------------------------
    // Get next buffer from write queue.
    // ------------------------------------------------------------------------

    {
      std::unique_lock<std::mutex> lock(
        write_mutex_);

      write_cv_.wait(
        lock,
        [this]() {
          return
            !write_queue_.empty() ||
            !running_;
        });

      if (!running_ &&
          write_queue_.empty())
      {
        break;
      }

      buffer_index =
        write_queue_.front();

      write_queue_.pop();
    }

    auto & buffer =
      buffers_[buffer_index];

    // ------------------------------------------------------------------------
    // Wait for GPU -> CPU copy.
    // ------------------------------------------------------------------------
    const auto start = std::chrono::steady_clock::now();
    cudaError_t result =
      cudaEventSynchronize(
        buffer.copy_done);
    const auto cuda_done = std::chrono::steady_clock::now();

    if (result != cudaSuccess) {

      RCLCPP_ERROR(
        get_logger(),
        "cudaEventSynchronize failed: %s",
        cudaGetErrorString(result));

    } else {

      // ----------------------------------------------------------------------
      // Write raw RGB image.
      // ----------------------------------------------------------------------

      // std::lock_guard<std::mutex> lock(frame_file_mutex_);
    
      // // .RGB SAVE
      // const std::string filename =
      //   save_path_ +
      //   "argus_frame_" +
      //   std::to_string(camera_id_) +
      //   "_" +
      //   std::to_string(buffer.timestamp_ns) +
      //   ".rgb";

    
      // std::ofstream file(
      //   filename,
      //   std::ios::binary);

      // if (!file) {

      //   RCLCPP_ERROR(
      //     get_logger(),
      //     "Failed to open file: %s",
      //     filename.c_str());

      // } else {

      //   // file.write(
      //   //   reinterpret_cast<const char *>(
      //   //     buffer.data),
      //   //   static_cast<std::streamsize>(
      //   //     buffer.size));
      //   file.write(
      //       reinterpret_cast<const char*>(buffer.data),
      //       static_cast<std::streamsize>(buffer.size / 4));


      //   if (!file.good()) {

      //     RCLCPP_ERROR(
      //       get_logger(),
      //       "Failed writing file: %s",
      //       filename.c_str());

      //   } 
      // }

      // ----------------------------------------------------------------------
      // .png SAVE
      // ----------------------------------------------------------------------

      // const std::string filename =
      //   save_path_ +
      //   "argus_frame_" +
      //   std::to_string(camera_id_) +
      //   "_" +
      //   std::to_string(buffer.timestamp_ns) +
      //   ".png";


      // cv::Mat rgb_image(
      //   2064,
      //   2472,
      //   CV_8UC3,
      //   buffer.data);

      // cv::Mat bgr_image;
      // cv::cvtColor(rgb_image, bgr_image, cv::COLOR_RGB2BGR);

      // cv::imwrite(filename, bgr_image);
      
      // // ----------------------------------------------------------------------
      // // .jpgs SAVE
      // // ----------------------------------------------------------------------
      // const std::string filename =
      //   save_path_ +
      //   "argus_frame_" +
      //   std::to_string(camera_id_) +
      //   "_" +
      //   std::to_string(buffer.timestamp_ns) +
      //   ".jpg";

      // cv::Mat rgb_image(
      //     2064,
      //     2472,
      //     CV_8UC3,
      //     buffer.data);
                
      // cv::Mat bgr_image;
      // cv::cvtColor(rgb_image, bgr_image, cv::COLOR_RGB2BGR);

      // const auto color_done = std::chrono::steady_clock::now();

      // std::vector<int> params = {
      //     cv::IMWRITE_JPEG_QUALITY, 25
      // };

      // // cv::imwrite(filename, bgr_image, params);

      // // writeFrameLog(
      // //   buffer.timestamp_ns,
      // //   std::to_string(writer_id));  

      // std::vector<uchar> encoded;

      // cv::imencode(
      //     ".jpg",
      //     bgr_image,
      //     encoded,
      //     params);

      // const auto encode_done = std::chrono::steady_clock::now();

      // std::ofstream file(
      //     filename,
      //     std::ios::binary);

      // file.write(
      //     reinterpret_cast<const char*>(encoded.data()),
      //     static_cast<std::streamsize>(encoded.size()));

      // file.close();

      // const auto write_done = std::chrono::steady_clock::now();

      // const auto cuda_ms =
      //   std::chrono::duration_cast<std::chrono::milliseconds>(
      //     cuda_done - start).count();

      // auto color_ms =
      //   std::chrono::duration_cast<std::chrono::milliseconds>(
      //     color_done - cuda_done).count();

      // auto encode_ms =
      //   std::chrono::duration_cast<std::chrono::milliseconds>(
      //     encode_done - color_done).count();

      // auto write_ms =
      //   std::chrono::duration_cast<std::chrono::milliseconds>(
      //     write_done - encode_done).count();

      // RCLCPP_INFO_THROTTLE(
      //   get_logger(),
      //   *get_clock(),
      //   1000,
      //   "writer thread=%u cuda=%ld ms color=%ld ms encode=%ld ms write=%ld ms",
      //   writer_id,
      //   cuda_ms,
      //   color_ms,
      //   encode_ms,
      //   write_ms);

      const std::string filename =
        save_path_ +
        "argus_frame_" +
        std::to_string(camera_id_) +
        "_" +
        std::to_string(buffer.timestamp_ns) +
        ".jpg";

      FILE * outfile = fopen(filename.c_str(), "wb");

      if (!outfile) {
        RCLCPP_ERROR(
          get_logger(),
          "Failed to open JPEG file: %s",
          filename.c_str());
      } else {

        jpeg_compress_struct cinfo{};
        jpeg_error_mgr jerr{};

        cinfo.err = jpeg_std_error(&jerr);

        jpeg_create_compress(&cinfo);

        jpeg_stdio_dest(&cinfo, outfile);

        cinfo.image_width =
          static_cast<JDIMENSION>(buffer.width);

        cinfo.image_height =
          static_cast<JDIMENSION>(buffer.height);

        // Your buffer is RGB, so tell libjpeg directly.
        cinfo.input_components = 3;
        cinfo.in_color_space = JCS_RGB;

        jpeg_set_defaults(&cinfo);

        // 25, 50, 85, 95 etc.
        jpeg_set_quality(&cinfo, 95, TRUE);

        jpeg_start_compress(&cinfo, TRUE);

        while (cinfo.next_scanline < cinfo.image_height) {

          JSAMPROW row_pointer =
            reinterpret_cast<JSAMPROW>(
              buffer.data +
              cinfo.next_scanline * buffer.width * 3);

          jpeg_write_scanlines(
            &cinfo,
            &row_pointer,
            1);
        }

        jpeg_finish_compress(&cinfo);
        jpeg_destroy_compress(&cinfo);

        fclose(outfile);
      }

      writeFrameLog(
        buffer.timestamp_ns,
        std::to_string(camera_id_));  

      const auto write_done = std::chrono::steady_clock::now();

      const auto cuda_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
          cuda_done - start).count();

      auto write_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
          write_done - cuda_done).count();

      RCLCPP_INFO_THROTTLE(
        get_logger(),
        *get_clock(),
        1000,
        "writer thread=%u cuda=%ld ms encode&write=%ld",
        writer_id,
        cuda_ms,
        write_ms);


    }
    

    // ------------------------------------------------------------------------
    // Buffer is now free again.
    // ------------------------------------------------------------------------

    {
      std::lock_guard<std::mutex> lock(
        buffer_mutex_);

      free_queue_.push(
        buffer_index);
    }

    buffer_cv_.notify_one();
  }

  RCLCPP_INFO(
    get_logger(),
    "Frame writer thread stopped (thread_id=%u)",
    writer_id);
}

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

  // const std::string filename =
  //   save_path_ +
  //   "camera_" +
  //   std::to_string(camera_id_) +
  //   ".rgb";

  // frame_file_.open(
  //   filename,
  //   std::ios::binary | std::ios::out | std::ios::trunc);

  // if (!frame_file_.is_open()) {
  //   RCLCPP_ERROR(
  //     get_logger(),
  //     "Failed to open frame file: %s",
  //     filename.c_str());
  // } else {
  //   RCLCPP_INFO(
  //     get_logger(),
  //     "Writing frames to: %s",
  //     filename.c_str());
  // }

  // --------------------------------------------------------------------------
  // Load camera info.
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
  // Image callback.
  // --------------------------------------------------------------------------

  config_map_[OUTPUT_COMPONENT_KEY_LEFT_IMAGE].callback =
    std::bind(
      &ArgusThreadNode::CustomImageCallback,
      this,
      std::placeholders::_1,
      std::placeholders::_2,
      optical_frame_name_);

  // --------------------------------------------------------------------------
  // Camera info callback.
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
  // Start Nitros node.
  // --------------------------------------------------------------------------

  startNitrosNode();
}


// ============================================================================
// Destructor
// ============================================================================

ArgusThreadNode::~ArgusThreadNode()
{
  RCLCPP_INFO(
    get_logger(),
    "Stopping ArgusThreadNode");

  // --------------------------------------------------------------------------
  // Tell writer thread to stop.
  // --------------------------------------------------------------------------

  running_.store(false);

  write_cv_.notify_all();
  buffer_cv_.notify_all();

  // --------------------------------------------------------------------------
  // Wait for writer thread.
  // --------------------------------------------------------------------------

  for (auto & thread : writer_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  writer_threads_.clear();

  // Now make absolutely sure the CUDA copy stream is finished.
  if (copy_stream_ != nullptr) {

    cudaError_t result =
      cudaStreamSynchronize(copy_stream_);

      if (result != cudaSuccess) {
      RCLCPP_ERROR(
        get_logger(),
        "cudaStreamSynchronize during shutdown failed: %s",
        cudaGetErrorString(result));
    }
  }

  // --------------------------------------------------------------------------
  // Destroy CUDA events and pinned memory.
  // --------------------------------------------------------------------------

  for (auto & buffer : buffers_) {
    std::lock_guard<std::mutex> lock(frame_file_mutex_);

    if (frame_file_.is_open()) {
      frame_file_.close();
    }
    if (buffer.copy_done != nullptr) {

      cudaEventDestroy(
        buffer.copy_done);

      buffer.copy_done =
        nullptr;
    }

    if (buffer.data != nullptr) {

      cudaFreeHost(
        buffer.data);

      buffer.data =
        nullptr;
    }
  }

  // --------------------------------------------------------------------------
  // Destroy CUDA stream.
  // --------------------------------------------------------------------------

  if (copy_stream_ != nullptr) {

    cudaStreamDestroy(
      copy_stream_);

    copy_stream_ =
      nullptr;
  }

  RCLCPP_INFO(
    get_logger(),
    "ArgusThreadNode stopped");
}

}  // namespace argus
}  // namespace isaac_ros
}  // namespace nvidia

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::argus::ArgusThreadNode)

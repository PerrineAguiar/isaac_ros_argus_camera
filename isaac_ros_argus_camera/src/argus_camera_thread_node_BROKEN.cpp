// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_argus_camera/argus_camera_thread_node.hpp"

#include <filesystem>
#include <iomanip>
#include <sstream>

#include "gxf/multimedia/video.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace argus
{

// Pass the required default Nitros configuration parameters down to ArgusCameraNode
ArgusThreadNode::ArgusThreadNode(const rclcpp::NodeOptions & options)
: ArgusCameraNode(
    options,
    "ArgusThreadNode",
    nitros::NitrosPublisherSubscriberConfigMap{},
    std::vector<std::string>{},
    std::vector<std::string>{},
    std::vector<std::string>{},
    std::vector<std::pair<std::string, std::string>>{},
    "isaac_ros_argus_camera")  // <--- Changed from "" to "isaac_ros_argus_camera"
{
  // Declare parameters
  output_dir_ = this->declare_parameter<std::string>("output_dir", "./output_frames");
  csv_file_path_ = this->declare_parameter<std::string>("csv_file_path", "./output_frames/saved_frames.csv");

  // Create target directory if needed
  std::filesystem::create_directories(output_dir_);

  // Initialize CSV File
  {
    std::lock_guard<std::mutex> lock(csv_mutex_);
    csv_file_.open(csv_file_path_, std::ios::out | std::ios::app);
    if (csv_file_.is_open() && std::filesystem::file_size(csv_file_path_) == 0) {
      csv_file_ << "timestamp_ns,frame_name,filepath\n";
      csv_file_.flush();
    }
  }

  // Create CUDA Stream
  cudaError_t cuda_err = cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking);
  if (cuda_err != cudaSuccess) {
    RCLCPP_ERROR(this->get_logger(), "Failed to create CUDA copy stream: %s", cudaGetErrorString(cuda_err));
  }

  // Start Worker Threads
  running_ = true;
  for (uint32_t i = 0; i < NUM_WRITERS; ++i) {
    writer_threads_.emplace_back(&ArgusThreadNode::writerLoop, this, i);
  }

  RCLCPP_INFO(this->get_logger(), "ArgusThreadNode initialized successfully.");
}

ArgusThreadNode::~ArgusThreadNode()
{
  // Signal workers to stop
  running_ = false;
  write_cv_.notify_all();

  // Join workers
  for (std::thread & thread : writer_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  // Clean up CUDA Stream & Buffers
  if (copy_stream_) {
    cudaStreamDestroy(copy_stream_);
    copy_stream_ = nullptr;
  }

  destroyBuffers();

  // Close CSV
  {
    std::lock_guard<std::mutex> lock(csv_mutex_);
    if (csv_file_.is_open()) {
      csv_file_.close();
    }
  }

  RCLCPP_INFO(this->get_logger(), "ArgusThreadNode shut down cleanly.");
}

bool ArgusThreadNode::initializeBuffers(uint32_t width, uint32_t height, size_t pitch, size_t buffer_size)
{
  std::lock_guard<std::mutex> lock(buffer_mutex_);

  buffer_width_ = width;
  buffer_height_ = height;
  buffer_pitch_ = pitch;
  buffer_size_ = buffer_size;

  while (!free_queue_.empty()) {
    free_queue_.pop();
  }

  for (size_t i = 0; i < NUM_BUFFERS; ++i) {
    cudaError_t err = cudaMalloc(&buffers_[i].gpu_data, buffer_size_);
    if (err != cudaSuccess) {
      RCLCPP_ERROR(this->get_logger(), "Failed to allocate GPU memory for buffer %zu: %s", i, cudaGetErrorString(err));
      return false;
    }

    buffers_[i].size = buffer_size_;
    buffers_[i].width = width;
    buffers_[i].height = height;
    buffers_[i].pitch = pitch;
    buffers_[i].in_use = false;

    free_queue_.push(i);
  }

  buffers_initialized_ = true;
  RCLCPP_INFO(this->get_logger(), "Allocated %zu GPU buffers (%u x %u).", NUM_BUFFERS, width, height);
  return true;
}

void ArgusThreadNode::destroyBuffers()
{
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  for (size_t i = 0; i < NUM_BUFFERS; ++i) {
    if (buffers_[i].gpu_data) {
      cudaFree(buffers_[i].gpu_data);
      buffers_[i].gpu_data = nullptr;
    }
    buffers_[i].in_use = false;
  }
  while (!free_queue_.empty()) {
    free_queue_.pop();
  }
  buffers_initialized_ = false;
}

bool ArgusThreadNode::acquireFreeBuffer(size_t & buffer_index)
{
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  if (free_queue_.empty()) {
    return false;
  }

  buffer_index = free_queue_.front();
  free_queue_.pop();
  buffers_[buffer_index].in_use = true;
  acquired_buffers_++;
  return true;
}

void ArgusThreadNode::releaseBuffer(size_t buffer_index)
{
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  buffers_[buffer_index].in_use = false;
  free_queue_.push(buffer_index);
  released_buffers_++;
}

void ArgusThreadNode::CustomImageCallback(
  gxf_context_t context,
  nitros::NitrosTypeBase & msg,
  std::string frame_name)
{
  if (!running_) return;

  // Use msg.handle to retrieve the GXF entity UID
  gxf_uid_t eid = msg.handle;
  
  auto video_buffer = nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>::Create(context, eid);
  if (!video_buffer) {
    RCLCPP_ERROR(this->get_logger(), "Failed to get VideoBuffer handle from GXF entity.");
    return;
  }

  auto video_header = video_buffer.value()->video_frame_info();
  uint32_t width = video_header.width;
  uint32_t height = video_header.height;
  size_t pitch = video_header.color_planes[0].stride;
  size_t buffer_size = video_buffer.value()->size();
  const unsigned char * src_gpu_ptr = video_buffer.value()->pointer();

  // Lazy Initialization of GPU Buffer Pool
  if (!buffers_initialized_) {
    std::lock_guard<std::mutex> init_lock(initialization_mutex_);
    if (!buffers_initialized_) {
      if (!initializeBuffers(width, height, pitch, buffer_size)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize buffer pool.");
        return;
      }
    }
  }

  // Acquire Buffer
  size_t buffer_index = 0;
  if (!acquireFreeBuffer(buffer_index)) {
    dropped_buffers_++;
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Drop frame: No free buffers available!");
    return;
  }

  // Asynchronous Copy (GPU -> Private GPU Buffer)
  FrameBuffer & buf = buffers_[buffer_index];
  buf.width = width;
  buf.height = height;
  buf.pitch = pitch;
  buf.timestamp_ns = this->now().nanoseconds();
  buf.frame_name = frame_name;

  cudaError_t err = cudaMemcpy2DAsync(
    buf.gpu_data, pitch,
    src_gpu_ptr, pitch,
    width * 3, // Stride based on image format
    height,
    cudaMemcpyDeviceToDevice,
    copy_stream_);

  if (err != cudaSuccess) {
    RCLCPP_ERROR(this->get_logger(), "cudaMemcpy2DAsync failed: %s", cudaGetErrorString(err));
    releaseBuffer(buffer_index);
    return;
  }

  // Synchronize stream so Argus can reclaim src_gpu_ptr immediately
  cudaStreamSynchronize(copy_stream_);
  copied_buffers_++;

  // Enqueue Job for Writer Workers
  {
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (write_queue_.size() >= MAX_WRITE_QUEUE_SIZE) {
      dropped_buffers_++;
      releaseBuffer(buffer_index);
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Drop frame: Write queue is full!");
      return;
    }

    std::ostringstream filename_ss;
    filename_ss << output_dir_ << "/frame_" << buf.timestamp_ns << ".jpg";

    NvJpegJob job;
    job.buffer_index = buffer_index;
    job.timestamp_ns = buf.timestamp_ns;
    job.frame_name = frame_name;
    job.output_filepath = filename_ss.str();

    write_queue_.push(job);
    queued_buffers_++;
  }

  write_cv_.notify_one();
}

void ArgusThreadNode::writerLoop(uint32_t writer_id)
{
  cudaStream_t encode_stream;
  cudaStreamCreateWithFlags(&encode_stream, cudaStreamNonBlocking);

  nvjpegHandle_t nvjpeg_handle;
  nvjpegEncoderState_t encoder_state;
  nvjpegEncoderParams_t encoder_params;

  nvjpegCreateSimple(&nvjpeg_handle);
  nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, encode_stream);
  nvjpegEncoderParamsCreate(nvjpeg_handle, &encoder_params, encode_stream);
  nvjpegEncoderParamsSetQuality(encoder_params, JPEG_QUALITY, encode_stream);

  std::vector<unsigned char> jpeg_bitstream;

  while (running_ || !write_queue_.empty()) {
    NvJpegJob job;
    {
      std::unique_lock<std::mutex> lock(write_mutex_);
      write_cv_.wait(lock, [this]() {
        return !running_ || !write_queue_.empty();
      });

      if (!running_ && write_queue_.empty()) {
        break;
      }

      job = write_queue_.front();
      write_queue_.pop();
    }

    if (encodeJpeg(writer_id, job, encode_stream, nvjpeg_handle, encoder_state, encoder_params, jpeg_bitstream)) {
      if (writeJpegFile(job.output_filepath, jpeg_bitstream)) {
        logToCsv(job.timestamp_ns, job.frame_name, job.output_filepath);
        encoded_buffers_++;
      }
    }

    releaseBuffer(job.buffer_index);
  }

  nvjpegEncoderParamsDestroy(encoder_params);
  nvjpegEncoderStateDestroy(encoder_state);
  nvjpegDestroy(nvjpeg_handle);
  cudaStreamDestroy(encode_stream);
}

bool ArgusThreadNode::encodeJpeg(
  uint32_t writer_id,
  const NvJpegJob & job,
  cudaStream_t encode_stream,
  nvjpegHandle_t nvjpeg_handle,
  nvjpegEncoderState_t encoder_state,
  nvjpegEncoderParams_t encoder_params,
  std::vector<unsigned char> & jpeg_bitstream)
{
  const FrameBuffer & buf = buffers_[job.buffer_index];

  nvjpegImage_t nvjpeg_image;
  nvjpeg_image.channel[0] = buf.gpu_data;
  nvjpeg_image.pitch[0] = buf.pitch;

  nvjpegStatus_t status = nvjpegEncodeImage(
    nvjpeg_handle,
    encoder_state,
    encoder_params,
    &nvjpeg_image,
    NVJPEG_INPUT_RGB,
    buf.width,
    buf.height,
    encode_stream);

  if (status != NVJPEG_STATUS_SUCCESS) {
    RCLCPP_ERROR(this->get_logger(), "[Writer %u] nvjpegEncodeImage failed: %d", writer_id, status);
    return false;
  }

  cudaStreamSynchronize(encode_stream);

  size_t length = 0;
  nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, NULL, &length, encode_stream);
  cudaStreamSynchronize(encode_stream);

  jpeg_bitstream.resize(length);
  nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, jpeg_bitstream.data(), &length, encode_stream);
  cudaStreamSynchronize(encode_stream);

  return true;
}

bool ArgusThreadNode::writeJpegFile(const std::string & filename, const std::vector<unsigned char> & jpeg_data)
{
  std::ofstream file(filename, std::ios::out | std::ios::binary);
  if (!file.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to open file for writing: %s", filename.c_str());
    return false;
  }

  file.write(reinterpret_cast<const char *>(jpeg_data.data()), jpeg_data.size());
  file.close();
  return true;
}

void ArgusThreadNode::logToCsv(uint64_t timestamp_ns, const std::string & frame_name, const std::string & filepath)
{
  std::lock_guard<std::mutex> lock(csv_mutex_);
  if (csv_file_.is_open()) {
    csv_file_ << timestamp_ns << "," << frame_name << "," << filepath << "\n";
    csv_file_.flush();
  }
}

}  // namespace argus
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::argus::ArgusThreadNode)
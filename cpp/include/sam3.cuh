#pragma once

#include "sam3.hpp"
#include <filesystem>
#include <fstream>
#include <memory>
#include "cuda_runtime.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "prepost.cuh"

struct PinnedMemoryDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }
};


#define MAX_DIMS 8

static void cuda_check(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::stringstream ss;
        ss << "Error in " << msg << ": " << cudaGetErrorString(err);
        throw std::runtime_error(ss.str());
    }
}

class TRTLogger : public nvinfer1::ILogger
{
    public:
        void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity <= nvinfer1::ILogger::Severity::kVERBOSE)
        {
            std::cout << "\033[1;35m [TRT] " << msg << "\033[0m\n";
        }
    }
};

inline TRTLogger trt_logger{};

class SAM3_PCS
{
public:
    SAM3_PCS(const std::string engine_path, const float vis_alpha, const float prob_threshold);
    ~SAM3_PCS();
    bool infer_on_image(const cv::Mat& input, cv::Mat& result, SAM3_VISUALIZATION vis_type);
    bool run_blind_inference();
    void pin_opencv_matrices(cv::Mat& input_mat, cv::Mat& result_mat);
    std::pair<cv::Mat, std::shared_ptr<void>> allocate_pinned_mat(int rows, int cols, int type);
    void setup_pinned_matrices(cv::Mat& input_mat, cv::Mat& result_mat);
    std::vector<void*> output_cpu;

private:
    bool is_zerocopy; //check if we can use zero-copy on this platform
    cudaStream_t sam3_stream;
    dim3 bsize;
    dim3 gsize;
    int in_width, in_height, opencv_inbytes;

    std::vector<void*> input_cpu;
    std::vector<void*> input_gpu;

    std::vector<void*> output_gpu;
    std::vector<size_t>output_sizes;

    void* opencv_input; // used only if dGPU
    uint8_t* gpu_result; // used for both
    uint8_t* zc_input; // used only if iGPU

    uint8_t* input_ptr; // placeholder for dGPU/iGPU ptr to pass into kernel
    float3* gpu_colpal;
    
    // Track registered OpenCV matrices for cleanup
    void* registered_input_data;
    void* registered_result_data;
    bool opencv_matrices_registered;
    void setup_color_palette();
    
    void check_zero_copy();
    void allocate_io_buffers();
    void load_engine();
    bool infer_on_dGPU(const cv::Mat& input, cv::Mat& result, SAM3_VISUALIZATION vis_type);
    bool infer_on_iGPU(const cv::Mat& input, cv::Mat& result, SAM3_VISUALIZATION vis_type);

    void visualize_on_dGPU(const cv::Mat& input, cv::Mat& result, SAM3_VISUALIZATION vis_type);
    const float _overlay_alpha, _probability_threshold;

    const std::string _engine_path;

    std::vector<std::string> _input_names;
    std::vector<std::string> _output_names;

    std::unique_ptr<nvinfer1::IRuntime> trt_runtime; // highest level
    std::unique_ptr<nvinfer1::ICudaEngine> trt_engine; // lower than runtime
    std::unique_ptr<nvinfer1::IExecutionContext> trt_ctx; // lower than engine
    
};
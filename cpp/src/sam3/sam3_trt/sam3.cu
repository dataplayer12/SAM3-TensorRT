#include "sam3.cuh"

SAM3_PCS::SAM3_PCS(const std::string engine_path, const float vis_alpha, const float prob_threshold)
    : _engine_path(engine_path)
    , _overlay_alpha(vis_alpha)
    , _probability_threshold(prob_threshold)
    , opencv_input(nullptr)
    , gpu_result(nullptr)
    , zc_input(nullptr)
    , gpu_colpal(nullptr)
    , registered_input_data(nullptr)
    , registered_result_data(nullptr)
    , opencv_matrices_registered(false)
    , opencv_inbytes(0)
{

    cuda_check(cudaStreamCreate(&sam3_stream), "creating CUDA stream for SAM3");
    
    trt_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(trt_logger));
    load_engine(); // after runtime is created, we can create the engine
    trt_ctx = std::unique_ptr<nvinfer1::IExecutionContext>(
        trt_engine->createExecutionContext());
    
    check_zero_copy(); // needed before allocating io buffers
    allocate_io_buffers();
    setup_color_palette();

    bsize.x=16;
    bsize.y=16;
}

std::pair<cv::Mat, std::shared_ptr<void>> SAM3_PCS::allocate_pinned_mat(int rows, int cols, int type)
{
    size_t bytes = rows * cols * CV_ELEM_SIZE(type);
    void* ptr = nullptr;
    
    cuda_check(cudaMallocHost(&ptr, bytes), " allocating pinned memory for Mat");
    
    auto deleter = [](void* p) { if (p) cudaFreeHost(p); };
    std::shared_ptr<void> mem_holder(ptr, deleter);
    
    cv::Mat mat(rows, cols, type, ptr);
    
    return std::make_pair(mat, mem_holder);
}

void SAM3_PCS::setup_pinned_matrices(cv::Mat& input_mat, cv::Mat& result_mat)
{
    opencv_inbytes = input_mat.total() * input_mat.elemSize();
    
    if (is_zerocopy)
    {
        cuda_check(cudaHostGetDevicePointer(&zc_input, input_mat.data, 0),
            " getting GPU pointer for pinned input Mat");
        
        cuda_check(cudaHostGetDevicePointer(&gpu_result, result_mat.data, 0),
            " getting GPU pointer for pinned result Mat");
    }
    else
    {
        if (opencv_input != nullptr)
        {
            cudaFree(opencv_input);
            cudaFree((void*)gpu_result);
        }
        cuda_check(cudaMalloc(&opencv_input, opencv_inbytes), " allocating opencv input memory on a dGPU system");
        cuda_check(cudaMalloc((void**)&gpu_result, opencv_inbytes), " allocating result memory on a dGPU system");
        cudaMemset(opencv_input, 0, opencv_inbytes);
        cudaMemset((void *)gpu_result, 0, opencv_inbytes);
    }
}

void SAM3_PCS::pin_opencv_matrices(cv::Mat& input_mat, cv::Mat& result_mat)
{
    opencv_inbytes = input_mat.total() * input_mat.elemSize();

    cuda_check(cudaHostRegister(
            input_mat.data,
            opencv_inbytes,
            cudaHostRegisterDefault),
            " pinning opencv input Mat on host"
        );
    
    // Track registered pointer for cleanup
    registered_input_data = input_mat.data;
    opencv_matrices_registered = true;
    
    // for most purposes the default flag is good enough, in my benchmarking
    // using others say readonly flag did not improve performance

    if (is_zerocopy)
    {
        cuda_check(cudaHostRegister(
            result_mat.data,
            opencv_inbytes,
            cudaHostRegisterDefault),
            " pinning opencv result Mat on host"
        );
        
        // Track registered pointer for cleanup
        registered_result_data = result_mat.data;

        cuda_check(cudaHostGetDevicePointer(
            &zc_input, input_mat.data, 0),
            " getting GPU pointer for input Mat");
        
        cuda_check(cudaHostGetDevicePointer(
            &gpu_result, result_mat.data, 0),
            " getting GPU pointer for result Mat");
    }
    else
    {
        // on dGPU allocate additional memory for input
        cuda_check(cudaMalloc(&opencv_input, opencv_inbytes), " allocating opencv input memory on a dGPU system");
        cuda_check(cudaMalloc((void**)&gpu_result, opencv_inbytes), " allocating result memory on a dGPU system");        
        cudaMemset(opencv_input, 0, opencv_inbytes);
        cudaMemset((void *)gpu_result, 0, opencv_inbytes);
    }
}

void SAM3_PCS::visualize_on_dGPU(const cv::Mat& input, cv::Mat& result, SAM3_VISUALIZATION vis_type)
{
    if (is_zerocopy)
    {
        input_ptr = zc_input;
    }
    else
    {
        input_ptr = static_cast<uint8_t*>(opencv_input);
    }

    if (vis_type == SAM3_VISUALIZATION::VIS_SEMANTIC_SEGMENTATION)
    {
        dim3 sbsize(16,16);
        dim3 sgsize;
        sgsize.x = (input.cols + THREAD_COARSENING_FACTOR*sbsize.x - 1) / (THREAD_COARSENING_FACTOR*sbsize.x);
        sgsize.y = (input.rows + THREAD_COARSENING_FACTOR*sbsize.y - 1) / (THREAD_COARSENING_FACTOR*sbsize.y);
        
        draw_semantic_seg_mask<<<sgsize, sbsize, 0, sam3_stream>>>(
            input_ptr,
            static_cast<float*>(output_gpu[1]),
            gpu_result,
            input.cols,
            input.rows,
            input.channels(),
            SAM3_OUTMASK_WIDTH,
            SAM3_OUTMASK_HEIGHT,
            _overlay_alpha,
            _probability_threshold,
            make_float3(0,185,118));
    }
    else if (vis_type == SAM3_VISUALIZATION::VIS_INSTANCE_SEGMENTATION)
    {
        dim3 ibsize(8,8,8); // 3D block
        dim3 igsize;

        igsize.x = (input.cols + THREAD_COARSENING_FACTOR*ibsize.x - 1) / (THREAD_COARSENING_FACTOR*ibsize.x);
        igsize.y = (input.rows + THREAD_COARSENING_FACTOR*ibsize.y - 1) / (THREAD_COARSENING_FACTOR*ibsize.y);
        // 2D grid

        size_t input_bytes = input.total() * input.elemSize();
        cuda_check(cudaMemcpyAsync((void *)gpu_result, 
            (void *)input_ptr, 
            input_bytes, 
            cudaMemcpyDeviceToDevice, 
            sam3_stream), " async memcpy for result during instance seg visualization");

        for (int _mask_channel_idx=0; _mask_channel_idx<200; _mask_channel_idx+=ibsize.z)
        {
            draw_instance_seg_mask<<<igsize, ibsize, 0, sam3_stream>>>(
                input_ptr,
                static_cast<float*>(output_gpu[0]),
                gpu_result,
                input.cols,
                input.rows,
                input.channels(),
                SAM3_OUTMASK_WIDTH,
                SAM3_OUTMASK_HEIGHT,
                _mask_channel_idx,
                _overlay_alpha,
                _probability_threshold,
                gpu_colpal);
        }
    }

    if (!is_zerocopy && vis_type == SAM3_VISUALIZATION::VIS_NONE)
    {
        cudaMemcpyAsync(output_cpu[0], output_gpu[0], output_sizes[0], cudaMemcpyDeviceToHost, sam3_stream);
        cudaMemcpyAsync(output_cpu[1], output_gpu[1], output_sizes[1], cudaMemcpyDeviceToHost, sam3_stream);
    }
    else if (!is_zerocopy)
    {
        size_t result_bytes = result.total() * result.elemSize();
        cudaMemcpyAsync((void*)result.data, (void*)gpu_result, result_bytes, cudaMemcpyDeviceToHost, sam3_stream);
    }
}

bool SAM3_PCS::infer_on_dGPU(const cv::Mat& input, cv::Mat& result, SAM3_VISUALIZATION vis_type)
{
    if (input.cols % 2 != 0 || input.rows % 2 != 0)
    {
        std::stringstream err;
        err << "Error: Input image dimensions must be even. Current size: " 
            << input.cols << "x" << input.rows 
            << ". Please resize the image to even dimensions before inference.";
        throw std::runtime_error(err.str());
    }
    
    size_t current_inbytes = input.total() * input.elemSize();
    
    if (current_inbytes > opencv_inbytes)
    {
        if (opencv_input != nullptr)
        {
            cudaFree(opencv_input);
            cudaFree((void*)gpu_result);
        }
        opencv_inbytes = current_inbytes;
        cuda_check(cudaMalloc(&opencv_input, opencv_inbytes), " reallocating opencv input memory");
        cuda_check(cudaMalloc((void**)&gpu_result, opencv_inbytes), " reallocating result memory");
    }
    
    gsize.x = (in_width + bsize.x - 1) / (THREAD_COARSENING_FACTOR*bsize.x);
    gsize.y = (in_height + bsize.y - 1) / (THREAD_COARSENING_FACTOR*bsize.y);

    cuda_check(
        cudaMemcpyAsync(
            opencv_input, input.data, current_inbytes, cudaMemcpyHostToDevice, sam3_stream)
        , " async memcpy of opencv image");

    pre_process_sam3<<<gsize, bsize, 0, sam3_stream>>>(
        static_cast<uint8_t*>(opencv_input),
        static_cast<float*>(input_gpu[0]),
        input.cols,
        input.rows,
        input.channels(),
        in_width,
        in_height);
    
    bool res = trt_ctx->enqueueV3(sam3_stream);

    visualize_on_dGPU(input, result, vis_type);
    cudaStreamSynchronize(sam3_stream);
    return res;
}

bool SAM3_PCS::infer_on_iGPU(const cv::Mat& input, cv::Mat& result, SAM3_VISUALIZATION vis_type)
{
    if (input.cols % 2 != 0 || input.rows % 2 != 0)
    {
        std::stringstream err;
        err << "Error: Input image dimensions must be even. Current size: " 
            << input.cols << "x" << input.rows 
            << ". Please resize the image to even dimensions before inference.";
        throw std::runtime_error(err.str());
    }
    
    gsize.x = (in_width + bsize.x - 1) / (THREAD_COARSENING_FACTOR*bsize.x);
    gsize.y = (in_height + bsize.y - 1) / (THREAD_COARSENING_FACTOR*bsize.y);

    pre_process_sam3<<<gsize, bsize, 0, sam3_stream>>>(
        zc_input,
        static_cast<float*>(input_gpu[0]),
        input.cols,
        input.rows,
        input.channels(),
        in_width,
        in_height);
    
    bool res = trt_ctx->enqueueV3(sam3_stream);
    visualize_on_dGPU(input, result, vis_type);
    cudaStreamSynchronize(sam3_stream);
    
    return res;
}

bool SAM3_PCS::infer_on_image(const cv::Mat& input, cv::Mat& result, SAM3_VISUALIZATION vis_type)
{
    if (is_zerocopy)
    {
        return infer_on_iGPU(input, result, vis_type);
    }

    return infer_on_dGPU(input, result, vis_type);
}

// only for engine benchmarking purposes
bool SAM3_PCS::run_blind_inference()
{
    bool res = trt_ctx->enqueueV3(sam3_stream);
    cudaStreamSynchronize(sam3_stream);
    return res;
}

void SAM3_PCS::load_engine()
{
    size_t file_size = std::filesystem::file_size(_engine_path);

    if (file_size==0)
    {
        std::stringstream err;
        err << "Engine file is empty";
        throw std::runtime_error(err.str());
    }

    char* engine_data = (char *)malloc(file_size);
    std::ifstream file(_engine_path, std::ios::binary);

    if (!file.is_open())
    {
        std::stringstream err;
        err << "File " << _engine_path << " could not be opened. Please check permissions\n";
        throw std::runtime_error(err.str());
    }

    file.read(engine_data, file_size);
    file.close();

    trt_engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        trt_runtime->deserializeCudaEngine(engine_data, file_size));
    
    free((void*)engine_data);
}

void SAM3_PCS::check_zero_copy()
{
    // ToDo: support multi GPU for x86
    int gpu, is_integrated;
    cuda_check(cudaGetDevice(&gpu), " getting GPU device");

    cuda_check(
        cudaDeviceGetAttribute(
        &is_integrated,
        cudaDevAttrIntegrated,
        gpu),
        " checking for zero-copy property");
    
    is_zerocopy = (is_integrated>0);

    if (is_zerocopy)
    {
        std::cout << "Running on a zero-copy platform. I/O binding buffers will be shared" << std::endl;
    }
    else
    {
        std::cout << "Running on dGPU. I/O binding buffers will be copied" << std::endl;
    }
}

void SAM3_PCS::allocate_io_buffers()
{
    int nb_io = trt_engine->getNbIOTensors();

    for (int io_idx = 0; io_idx <= nb_io - 1; io_idx++)
    {
        const char* name = trt_engine->getIOTensorName(io_idx);
        nvinfer1::TensorIOMode mode = trt_engine->getTensorIOMode(name);

        nvinfer1::Dims dims = trt_engine->getTensorShape(name);
        size_t nbytes = sizeof(trt_engine->getTensorDataType(name));
        
        for (int idx=0;idx < MAX_DIMS; idx++)
        {
            nbytes*=std::max(1, (int)dims.d[idx]);
        }

        void *cpu_buf, *gpu_buf;

        cuda_check(cudaHostAlloc(&cpu_buf, nbytes, cudaHostAllocMapped), 
            " allocating io CPU buffer");

        std::memset(cpu_buf, 0, nbytes);

        if (is_zerocopy)
        {
            cuda_check(
                cudaHostGetDevicePointer(&gpu_buf, cpu_buf, 0),
                " accessing shared CPU/GPU pointer on zero-copy platform");
        }
        else
        {
            // most likely x86
            cuda_check(cudaMalloc(&gpu_buf, nbytes), " allocating GPU memory on dGPU system");
            cudaMemset(gpu_buf, 0, nbytes);
        }

        trt_ctx->setTensorAddress(name, gpu_buf);

        if (mode == nvinfer1::TensorIOMode::kINPUT)
        {
            std::cout << "Found input tensor " << name << std::endl;
            _input_names.push_back(std::string(name));
            input_cpu.push_back(cpu_buf);
            input_gpu.push_back(gpu_buf);

            if (dims.d[1]==3) // typically 3 input channels for SAM like models
            {
                in_width = dims.d[3];
                in_height= dims.d[2];
                std::cout << "Input image has dimensions " 
                    << in_width
                    << " x "
                    << in_height
                    << std::endl;
            }
        }
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
        {
            std::cout << "Found output tensor " << name << std::endl;
            _output_names.push_back(std::string(name));
            output_cpu.push_back(cpu_buf);
            output_gpu.push_back(gpu_buf);
            output_sizes.push_back(nbytes);
        }
        else
        {
            std::cout << "I/O binding " 
                << name 
                << " has an unknown mode: " 
                << static_cast<int>(mode)
                << ", this is most likely a bug in the application"
                << std::endl;
        }

    }
}

void SAM3_PCS::setup_color_palette()
{
    cuda_check(cudaMalloc(&gpu_colpal, colpal.size()*sizeof(float3)), 
        " allocating color palette on GPU");
        
    cuda_check(cudaMemcpyAsync((void *)gpu_colpal, 
            (void *)colpal.data(), 
            colpal.size()*sizeof(float3), 
            cudaMemcpyHostToDevice, 
            sam3_stream), " async memcpy for color pallete");
    
    cudaStreamSynchronize(sam3_stream);
}

SAM3_PCS::~SAM3_PCS()
{
    // 1. Unregister pinned OpenCV matrices (if registered)
    if (opencv_matrices_registered)
    {
        if (registered_input_data != nullptr)
        {
            cudaHostUnregister(registered_input_data);
        }
        
        if (registered_result_data != nullptr)
        {
            cudaHostUnregister(registered_result_data);
        }
    }
    
    // 2. Free input_gpu buffers (allocated in allocate_io_buffers)
    for (auto& ptr : input_gpu)
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }

    // 3. Free output_gpu buffers (allocated in allocate_io_buffers)
    for (auto& ptr : output_gpu)
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }
    
    // 4. Free input_cpu buffers (allocated with cudaHostAlloc)
    for (auto& ptr : input_cpu)
    {
        if (ptr)
        {
            cudaFreeHost(ptr);
        }
    }
    
    // 5. Free output_cpu buffers (allocated with cudaHostAlloc)
    for (auto& ptr : output_cpu)
    {
        if (ptr)
        {
            cudaFreeHost(ptr);
        }
    }
    
    // 6. Free opencv_input (only allocated in dGPU mode)
    if (!is_zerocopy && opencv_input != nullptr)
    {
        cudaFree(opencv_input);
    }
    
    // 7. Free gpu_result (only if allocated separately in dGPU mode)
    // In zero-copy mode, gpu_result points to registered memory, not separately allocated
    if (!is_zerocopy && gpu_result != nullptr)
    {
        cudaFree(gpu_result);
    }
    
    // 8. Free color palette
    if (gpu_colpal != nullptr)
    {
        cudaFree(gpu_colpal);
    }
    
    // 9. Destroy CUDA stream
    if (sam3_stream)
    {
        cudaStreamDestroy(sam3_stream);
    }
}
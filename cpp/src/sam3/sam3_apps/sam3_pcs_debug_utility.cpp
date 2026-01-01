#include "sam3.hpp"
#include "sam3.cuh"
#include <chrono>
#include <thread>
#include <opencv2/imgproc.hpp>

void read_image_into_buffer(const std::string imgpath, char* raw_buffer, cv::Mat& buffer)
{
    size_t file_size = std::filesystem::file_size(imgpath);
    if (file_size==0)
    {
        std::stringstream err;
        err << "Image file is empty";
        throw std::runtime_error(err.str());
    }

    std::ifstream file(imgpath, std::ios::binary);

    if (!file.is_open())
    {
        std::stringstream err;
        err << "File " << imgpath << " could not be opened. Please check permissions\n";
        throw std::runtime_error(err.str());
    }

    file.read(raw_buffer, file_size);
    file.close();
    cv::imdecode(raw_buffer, cv::IMREAD_COLOR, &buffer);
}

void infer_one_image(SAM3_PCS& pcs, 
    const cv::Mat& img, 
    const cv::Mat& result, 
    const SAM3_VISUALIZATION vis,
    const std::string outfile)
{
    success = pcs.infer_on_image(img, result, visualize);

    if (vis == SAM3_VISUALIZATION::VIS_NONE)
    {
        cv::Mat seg = cv::Mat(SAM3_OUTMASK_WIDTH, SAM3_OUTMASK_HEIGHT, CV_32FC1, pcs.output_cpu[1]);
        // these are raw logits and should be passed through sigmoid before for any quantitative use
    }
    else
    {
        cv::imwrite(outfile, result);
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cout << "Usage: ./sam3_pcs_app image.jpg engine_path.engine" << std::endl;
        return 0;
    }

    const std::string infile = argv[1];
    std::string epath = argv[2];

    const float vis_alpha = 0.5;
    const float probability_threshold = 0.3;
    const SAM3_VISUALIZATION visualize = SAM3_VISUALIZATION::VIS_SEMANTIC_SEGMENTATION;

    SAM3_PCS pcs(epath, 0.5, 0.3);

    cv::Mat img = cv::imread(infile, 1); // input Mat
    cv::Mat result = cv::imread(infile, 1); // output Mat
    if (img.empty())
    {
        std::cout << "Image could not be read" << std::endl;
    }
    pcs.pin_opencv_input_mat(img);

    int MAX_ITER=10;
    auto start = std::chrono::system_clock::now();
    bool success=false;
    
    for (int idx=0; idx<MAX_ITER; idx++)
    {
        success = pcs.infer_on_image(img, result, visualize);

        if (visualize == SAM3_VISUALIZATION::VIS_NONE)
        {
            cv::Mat seg = cv::Mat(SAM3_OUTMASK_WIDTH, SAM3_OUTMASK_HEIGHT, CV_32FC1, pcs.output_cpu[1]);
            // these are raw logits and should be passed through sigmoid before for any quantitative use
        }
        else
        {
            cv::imwrite("sam3_result.jpg", result);
        }

    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff = end - start;
    int millis = diff.count() * 1000;
    float millis_per_iteration = millis/(MAX_ITER + 0.0);

    std::cout << "Engine inference latency = " << millis_per_iteration << " ms" << std::endl;
}
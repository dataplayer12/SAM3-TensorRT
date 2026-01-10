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
    
    cv::Mat raw_mat(1, static_cast<int>(file_size), CV_8UC1, raw_buffer); 
    // just a wrapper, minimal allocation

    cv::imdecode(raw_mat, cv::IMREAD_COLOR, &buffer);
}

void infer_one_image(SAM3_PCS& pcs, 
    const cv::Mat& img, 
    cv::Mat& result, 
    const SAM3_VISUALIZATION vis,
    const std::string outfile,
    bool benchmark_run)
{
    bool success = pcs.infer_on_image(img, result, vis);

    if (benchmark_run)
    {
        return;
    }

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
        std::cout << "Usage: ./sam3_pcs_app indir engine_path.engine <benchmark=false>" << std::endl;
        return 0;
    }

    const std::string in_dir = argv[1];
    std::string epath = argv[2];
    bool benchmark=false; // in benchmarking mode we dont save output images

    if (argc==4)
    {
        std::string b_arg = argv[3]; // should be 0 or 1
        try
        {
            benchmark = (b_arg == "1");
        }
        catch(const std::exception)
        {
            std::cout << "Unrecognized benchmark type " << argv[3] << std::endl;
        }
    }
    std::cout << "Benchmarking: " << benchmark << std::endl;

    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff;
    float millis_elapsed = 0.0; // int will overflow after ~650 hours

    const float vis_alpha = 0.3;
    const float probability_threshold = 0.5;
    const SAM3_VISUALIZATION visualize = SAM3_VISUALIZATION::VIS_SEMANTIC_SEGMENTATION;

    SAM3_PCS pcs(epath, vis_alpha, probability_threshold);

    cv::Mat img, result;
    char* raw_bytes;

    std::filesystem::create_directories("results");
    int num_images_read=0;

    for (const auto& fname : std::filesystem::directory_iterator(in_dir))
    {
        if (std::filesystem::is_regular_file(fname.path())) 
        {
            std::filesystem::path outfile = std::filesystem::path("results") / fname.path().filename();
            
            if (num_images_read==0)
            {
                cv::Mat tmp = cv::imread(fname.path(), cv::IMREAD_COLOR);
                raw_bytes = (char *)malloc(tmp.total()*tmp.elemSize());
                read_image_into_buffer(fname.path(), raw_bytes, img);
                result = cv::imread(fname.path(), cv::IMREAD_COLOR);
                pcs.pin_opencv_matrices(img, result);
            }
            else
            {
                read_image_into_buffer(fname.path(), raw_bytes, img);
            }
            start = std::chrono::system_clock::now();
            infer_one_image(pcs, img, result, visualize, outfile, benchmark);
            num_images_read++;
            end = std::chrono::system_clock::now();
            diff = end - start;
            millis_elapsed += (diff.count() * 1000);

            if (num_images_read>0 && num_images_read%10==0)
            {
                float msec_per_image = millis_elapsed/num_images_read;
                printf("Processed %d images at %f msec/image\n", num_images_read, msec_per_image);
            }
        }
    }
    
    // Clean up allocated memory
    if (raw_bytes != nullptr)
    {
        free(raw_bytes);
    }
    
    return 0;
}

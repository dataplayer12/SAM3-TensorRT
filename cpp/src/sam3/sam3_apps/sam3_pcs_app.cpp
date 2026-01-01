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
    const std::string outfile)
{
    bool success = pcs.infer_on_image(img, result, vis);

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
        std::cout << "Usage: ./sam3_pcs_app indir engine_path.engine" << std::endl;
        return 0;
    }

    const std::string in_dir = argv[1];
    std::string epath = argv[2];

    const float vis_alpha = 0.5;
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
                pcs.pin_opencv_input_mat(img);
                result = cv::imread(fname.path(), cv::IMREAD_COLOR);
            }
            else
            {
                read_image_into_buffer(fname.path(), raw_bytes, img);
            }
            infer_one_image(pcs, img, result, visualize, outfile);
            num_images_read++;
        }
    }
}
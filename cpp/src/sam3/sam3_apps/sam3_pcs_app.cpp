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
    int prev_width = 0, prev_height = 0;

    std::filesystem::create_directories("results");
    int num_images_read=0;

    // tokenized version of 'person'
    std::vector<int64_t> iid={49406,  2533, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407};
    
    std::vector<int64_t> iam={1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0};
        
    pcs.set_prompt(iid, iam);

    const int MAX_BENCHMARK_IMAGES = 100;

    for (const auto& fname : std::filesystem::directory_iterator(in_dir))
    {
        if (std::filesystem::is_regular_file(fname.path())) 
        {
            std::filesystem::path outfile = std::filesystem::path("results") / fname.path().filename();
            
            img = cv::imread(fname.path(), cv::IMREAD_COLOR);
            if (img.empty()) continue;

            result.create(img.rows, img.cols, img.type());

            if (img.cols != prev_width || img.rows != prev_height)
            {
                pcs.pin_opencv_matrices(img, result);
                prev_width = img.cols;
                prev_height = img.rows;
            }

            start = std::chrono::system_clock::now();
            infer_one_image(pcs, img, result, visualize, outfile, benchmark);
            num_images_read++;
            end = std::chrono::system_clock::now();
            diff = end - start;
            millis_elapsed += (diff.count() * 1000);

            if (num_images_read>0 && num_images_read%10==0)
            {
                printf("Processed %d images...\n", num_images_read);
            }

            if (num_images_read >= MAX_BENCHMARK_IMAGES) break;
        }
    }

    if (num_images_read > 0)
    {
        float msec_per_image = millis_elapsed/num_images_read;
        float est_1000_min = msec_per_image * 1000.0f / 1000.0f / 60.0f;
        printf("\n=== Benchmark Results ===\n");
        printf("Processed %d images in %.1f s\n", num_images_read, millis_elapsed/1000.0f);
        printf("Average: %.2f msec/image\n", msec_per_image);
        printf("Estimated time for 1000 images: %.1f min\n", est_1000_min);
    }
}

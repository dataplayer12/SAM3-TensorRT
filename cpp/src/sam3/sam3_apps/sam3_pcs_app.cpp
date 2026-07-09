#include "sam3.hpp"
#include "sam3.cuh"
#include <chrono>
#include <thread>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <array>
#include <memory>
#include <stdexcept>

// Helper to execute bash command and read stdout
std::string exec_python_tokenizer(const std::string& prompt) {
    // Assuming the app is run from /workspace/cpp/build or /workspace
    std::string cmd = "python3 /workspace/python/tokenize_prompt.py \"" + prompt + "\"";
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    // Remove trailing newlines
    result.erase(std::remove(result.begin(), result.end(), '\n'), result.end());
    return result;
}

void infer_one_image(SAM3_PCS& pcs, 
    const cv::Mat& img, 
    cv::Mat& result, 
    const SAM3_VISUALIZATION vis,
    const std::string outfile,
    const std::string prompt,
    bool benchmark_run)
{
    bool success = pcs.infer_on_image(img, result, vis);

    if (benchmark_run) return;

    if (vis == SAM3_VISUALIZATION::VIS_BBOX)
    {
        // CPU-side box coordinates and logits copied over by sam3.cu
        float* boxes = static_cast<float*>(pcs.output_cpu[2]);
        float* logits = static_cast<float*>(pcs.output_cpu[3]);
        int num_boxes = 200;
        
        for (int i=0; i < num_boxes; i++) {
            float logit = logits[i];
            float prob = 1.0f / (1.0f + std::exp(-logit));
            if (prob > 0.5f) { // threshold match
                float x1 = boxes[i * 4 + 0];
                float y1 = boxes[i * 4 + 1];
                int x_min = std::max(0, (int)(x1 * img.cols));
                int y_min = std::max(0, (int)(y1 * img.rows));
                
                // Draw text slightly above the bounding box
                // Reduced font scale from 0.9 to 0.5, thickness from 2 to 1
                cv::putText(result, prompt, cv::Point(x_min, std::max(15, y_min - 6)), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
        }
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
    if (argc < 4)
    {
        std::cout << "Usage: ./sam3_pcs_app indir engine_path.engine prompt <benchmark=0>" << std::endl;
        return 0;
    }

    const std::string in_dir = argv[1];
    std::string epath = argv[2];
    std::string prompt = argv[3];
    bool benchmark = false; // in benchmarking mode we dont save output images

    if (argc == 5)
    {
        benchmark = (std::string(argv[4]) == "1");
    }
    std::cout << "Target Prompt: " << prompt << std::endl;
    std::cout << "Benchmarking: " << benchmark << std::endl;

    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff;
    float millis_elapsed = 0.0;

    const float vis_alpha = 0.3;
    const float probability_threshold = 0.5;
    const SAM3_VISUALIZATION visualize = SAM3_VISUALIZATION::VIS_BBOX;

    SAM3_PCS pcs(epath, vis_alpha, probability_threshold);

    cv::Mat img, result;
    int prev_width = 0, prev_height = 0;

    std::filesystem::create_directories("results");
    int num_images_read=0;

    // Tokenize the prompt
    std::vector<int64_t> iid(32, 49407); // 49407 is usually the PAD/EOS token for SAM3 text encoder
    std::vector<int64_t> iam(32, 0);

    try {
        std::cout << "Calling Python to tokenize prompt: '" << prompt << "'..." << std::endl;
        std::string py_out = exec_python_tokenizer(prompt);
        
        if (py_out == "-1" || py_out.empty()) {
            throw std::runtime_error("Python tokenizer script failed.");
        }
        
        // Parse comma separated string
        std::vector<int32_t> ids;
        std::stringstream ss(py_out);
        std::string token;
        while (std::getline(ss, token, ',')) {
            ids.push_back(std::stoi(token));
        }
        
        for (size_t i = 0; i < ids.size() && i < 32; ++i) {
            iid[i] = ids[i];
            iam[i] = 1; // 1 for real tokens, 0 for pad
        }
        std::cout << "Successfully tokenized prompt into " << ids.size() << " tokens." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Tokenizer error: " << e.what() << std::endl;
        std::cout << "Falling back to 'person' tokens.\n";
        iid = {49406, 2533, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
               49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
               49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
               49407, 49407};
        iam = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0};
    }

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
            infer_one_image(pcs, img, result, visualize, outfile, prompt, benchmark);
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

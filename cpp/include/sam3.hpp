#pragma once

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <string>

typedef enum {
    TASK_PROMPTABLE_CONCEPT_SEGMENTATION,
    TASK_TRACKING
} SAM3_TASKS;

typedef enum {
    VIS_NONE,
    VIS_SEMANTIC_SEGMENTATION,
    VIS_INSTANCE_SEGMENTATION
} SAM3_VISUALIZATION;

typedef struct {
    float score;
    int box_x, box_y, box_w, box_h;
    std::vector<int> mask_x;
    std::vector<int> mask_y;
} SAM3_PCS_RESULT;


#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#define THREAD_COARSENING_FACTOR 2
// thread coarsening factor in x and y

#define SAM3_RESCALE_FACTOR 0.00392156862745098
#define SAM3_IMG_MEAN 0.5
#define SAM3_IMG_STD 0.5
#define SAM3_OUTMASK_WIDTH 288
#define SAM3_OUTMASK_HEIGHT 288
// taken from https://huggingface.co/facebook/sam3/blob/main/processor_config.json

__global__ void pre_process_sam3(
    uint8_t* src,
    float* dst,
    int src_width,
    int src_height,
    int src_channels,
    int dst_width,
    int dst_height);

__global__ void draw_semantic_seg_mask(
    uint8_t* src,
    float* mask,
    uint8_t* result,
    int src_width,
    int src_height,
    int src_channels,
    int mask_width,
    int mask_height,
    float mask_alpha,
    float prob_threshold,
    float3 color);

__global__ void draw_instance_seg_mask(
    uint8_t* src,
    float* mask,
    uint8_t* result,
    int src_width,
    int src_height,
    int src_channels,
    int mask_width,
    int mask_height,
    int mask_channel_idx,
    float mask_alpha,
    float prob_threshold,
    float3* color_palette
);

static std::vector<float3> colpal = {
    make_float3(  0, 185, 118), // teal (your original)
    make_float3(230, 159,   0), // orange
    make_float3( 86, 180, 233), // sky blue
    make_float3(240, 228,  66), // yellow
    make_float3(  0, 114, 178), // blue
    make_float3(213,  94,   0), // vermillion
    make_float3(204, 121, 167), // purple
    make_float3(  0, 158, 115), // green
    make_float3( 45,  76, 189), // royal blue
    make_float3(177,  89,  40), // brown
    make_float3(128,  62, 117), // deep purple
    make_float3( 52, 160, 164), // cyan-teal
    make_float3( 87,  87,  87), // neutral gray
    make_float3(233,  75,  60), // red
    make_float3( 60, 186,  84), // green
    make_float3(244, 194,  13), // gold
    make_float3( 72,  61, 139), // slate blue
    make_float3(255, 105, 180), // pink
    make_float3(  0, 206, 209), // turquoise
    make_float3(160,  82,  45), // sienna
};

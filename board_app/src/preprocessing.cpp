/*
 * Quick Draw Preprocessing — C++ implementation
 *
 * Pipeline matches the original Python preprocessing:
 * 1. Find ink bounding box (pixels < ink_threshold)
 * 2. Crop with margin
 * 3. Pad to square (centered)
 * 4. Invert (white strokes on black bg)
 * 5. Resize to model_size x model_size
 *    - Area-based (box filter) for downsampling — matches PIL/training
 *    - Bilinear for upsampling
 * 6. 3-channel RGB with sketch normalization: pixel/255 (0-1 scaling)
 */

#include "preprocessing.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <string>

/*
 * Area-based (box filter) resize for downsampling.
 * Averages all source pixels that map to each output pixel.
 * This matches PIL's LANCZOS/BILINEAR behavior for downscaling
 * and produces smoother, more accurate results than point/bilinear.
 */
static void area_resize(const uint8_t* src, int sw, int sh,
                         uint8_t* dst, int dw, int dh)
{
    float x_scale = static_cast<float>(sw) / dw;
    float y_scale = static_cast<float>(sh) / dh;

    for (int dy = 0; dy < dh; dy++) {
        float src_y0 = dy * y_scale;
        float src_y1 = (dy + 1) * y_scale;
        int iy0 = static_cast<int>(src_y0);
        int iy1 = std::min(static_cast<int>(std::ceil(src_y1)), sh);

        for (int dx = 0; dx < dw; dx++) {
            float src_x0 = dx * x_scale;
            float src_x1 = (dx + 1) * x_scale;
            int ix0 = static_cast<int>(src_x0);
            int ix1 = std::min(static_cast<int>(std::ceil(src_x1)), sw);

            float sum = 0.0f;
            float area = 0.0f;

            for (int iy = iy0; iy < iy1; iy++) {
                float wy = std::min(static_cast<float>(iy + 1), src_y1)
                         - std::max(static_cast<float>(iy), src_y0);
                for (int ix = ix0; ix < ix1; ix++) {
                    float wx = std::min(static_cast<float>(ix + 1), src_x1)
                             - std::max(static_cast<float>(ix), src_x0);
                    float weight = wx * wy;
                    sum += src[iy * sw + ix] * weight;
                    area += weight;
                }
            }

            dst[dy * dw + dx] = static_cast<uint8_t>(
                std::min(255.0f, std::max(0.0f, sum / area + 0.5f)));
        }
    }
}

/*
 * Bilinear resize for upsampling (when source < destination).
 */
static void bilinear_resize(const uint8_t* src, int sw, int sh,
                             uint8_t* dst, int dw, int dh)
{
    float x_ratio = static_cast<float>(sw) / dw;
    float y_ratio = static_cast<float>(sh) / dh;

    for (int dy = 0; dy < dh; dy++) {
        float gy = dy * y_ratio;
        int y0 = static_cast<int>(gy);
        int y1 = std::min(y0 + 1, sh - 1);
        float fy = gy - y0;

        for (int dx = 0; dx < dw; dx++) {
            float gx = dx * x_ratio;
            int x0 = static_cast<int>(gx);
            int x1 = std::min(x0 + 1, sw - 1);
            float fx = gx - x0;

            float p00 = src[y0 * sw + x0];
            float p10 = src[y0 * sw + x1];
            float p01 = src[y1 * sw + x0];
            float p11 = src[y1 * sw + x1];

            float val = p00 * (1 - fx) * (1 - fy)
                      + p10 * fx * (1 - fy)
                      + p01 * (1 - fx) * fy
                      + p11 * fx * fy;

            dst[dy * dw + dx] = static_cast<uint8_t>(
                std::min(255.0f, std::max(0.0f, val + 0.5f)));
        }
    }
}

bool preprocess_canvas(const uint8_t* gray, int width, int height,
                       const PreprocessConfig& cfg,
                       std::vector<float>& output)
{
    /* 1. Find ink bounding box */
    int x_min = width, x_max = -1;
    int y_min = height, y_max = -1;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (gray[y * width + x] < cfg.ink_threshold) {
                if (x < x_min) x_min = x;
                if (x > x_max) x_max = x;
                if (y < y_min) y_min = y;
                if (y > y_max) y_max = y;
            }
        }
    }

    /* No ink found */
    if (x_max < 0) return false;

    /* 2. Crop with margin */
    x_min = std::max(0, x_min - cfg.crop_margin);
    y_min = std::max(0, y_min - cfg.crop_margin);
    x_max = std::min(width - 1, x_max + cfg.crop_margin);
    y_max = std::min(height - 1, y_max + cfg.crop_margin);

    int cw = x_max - x_min + 1;
    int ch = y_max - y_min + 1;

    /* 3. Pad to square (center the drawing) */
    int side = std::max(cw, ch);
    int pad_left = (side - cw) / 2;
    int pad_top  = (side - ch) / 2;

    /* Reusable buffer — avoids heap allocation per request (single-threaded) */
    static std::vector<uint8_t> square;
    square.assign(side * side, 255); /* white fill */
    for (int y = 0; y < ch; y++) {
        std::memcpy(&square[(pad_top + y) * side + pad_left],
                    &gray[(y_min + y) * width + x_min],
                    cw);
    }

    /* 4. Invert: white strokes on black background */
    for (int i = 0; i < side * side; i++) {
        square[i] = 255 - square[i];
    }

    /* 5. Resize to model_size x model_size
     *    Use area-based for downsampling (matches PIL training),
     *    bilinear for upsampling (small drawings). */
    int ms = cfg.model_size;
    static std::vector<uint8_t> resized;
    resized.resize(ms * ms);

    if (side == ms) {
        std::memcpy(resized.data(), square.data(), ms * ms);
    } else if (side > ms) {
        /* Downsampling — area-based (box filter) */
        area_resize(square.data(), side, side, resized.data(), ms, ms);
    } else {
        /* Upsampling — bilinear */
        bilinear_resize(square.data(), side, side, resized.data(), ms, ms);
    }

    /* 6. Convert to 3-channel RGB (repeat grayscale) with sketch normalization.
     *    Layout: CHW [3, model_size, model_size]
     *    Formula: pixel/255 (simple 0-1 scaling, no mean/std adjustment)
     *    Matches training with --norm-mode sketch and DRP-AI compile mean=[0,0,0] std=[1,1,1]. */
    static const float mean[3] = {0.0f, 0.0f, 0.0f};
    static const float std_dev[3] = {1.0f, 1.0f, 1.0f};
    int pixels = ms * ms;
    output.resize(3 * pixels);
    for (int c = 0; c < 3; c++) {
        float inv_std = 1.0f / std_dev[c];
        for (int i = 0; i < pixels; i++) {
            output[c * pixels + i] = (resized[i] / 255.0f - mean[c]) * inv_std;
        }
    }

    /* Save debug image if requested (the resized grayscale BEFORE normalization) */
    g_last_debug_image = resized;

    return true;
}

/* Global: last preprocessed image for debug dump (set by preprocess_canvas) */
std::vector<uint8_t> g_last_debug_image;

bool save_debug_image(const uint8_t* data, int width, int height,
                      const std::string& path)
{
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) return false;

    /* PGM binary format (P5) — simple, no library needed */
    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    fwrite(data, 1, width * height, fp);
    fclose(fp);
    return true;
}

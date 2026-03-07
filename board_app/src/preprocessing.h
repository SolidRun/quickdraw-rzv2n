/*
 * Quick Draw Preprocessing — C++ implementation
 *
 * Receives raw grayscale canvas (uint8), produces normalized float32
 * tensor ready for DRP-AI inference.
 *
 * Pipeline: find ink → crop → pad to square → invert → resize → normalize
 */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <vector>
#include <cstdint>
#include <string>

/* Last preprocessed image (before normalization) — for debug dump */
extern std::vector<uint8_t> g_last_debug_image;

struct PreprocessConfig {
    int model_size = 128;          /* Target NxN */
    uint8_t ink_threshold = 245;   /* Pixels below this are "ink" */
    int crop_margin = 12;          /* Margin around ink bbox */
};

/*
 * Preprocess a raw grayscale canvas for Quick Draw classification.
 *
 * Input:  uint8 grayscale image (white bg=255, black ink=0), width x height
 * Output: float32 tensor [3 x model_size x model_size], CHW layout, sketch normalized (0-1)
 *
 * Returns true if ink was found and tensor was produced.
 * Returns false if canvas is blank (no ink).
 */
bool preprocess_canvas(const uint8_t* gray, int width, int height,
                       const PreprocessConfig& cfg,
                       std::vector<float>& output);

/*
 * Save a preprocessed grayscale image as PGM (Portable GrayMap) for debugging.
 * The image is the 128x128 result AFTER crop/pad/invert/resize but BEFORE normalization.
 *
 * PGM is a simple format readable by any image viewer — no external library needed.
 */
bool save_debug_image(const uint8_t* data, int width, int height,
                      const std::string& path);

#endif /* PREPROCESSING_H */

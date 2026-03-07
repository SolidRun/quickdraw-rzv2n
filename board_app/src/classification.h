#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

/*
 * Classification Post-Processing
 * Softmax + Top-K extraction for DRP-AI classification models
 */

#include <vector>

struct ClassResult {
    int class_id;
    float confidence;  /* probability after softmax */
};

/*
 * Apply softmax to raw logits, return probability vector.
 */
std::vector<float> softmax(const float* logits, int num_classes);

/*
 * Extract top-K results from a probability vector.
 * probs must already be normalized (e.g. from softmax).
 */
std::vector<ClassResult> top_k(const float* probs, int num_classes, int k);

/*
 * Apply softmax to logits and extract top-K results (convenience wrapper).
 */
std::vector<ClassResult> classify(const float* logits, int num_classes, int top_k);

#endif /* CLASSIFICATION_H */

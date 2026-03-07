/*
 * Classification Post-Processing Implementation
 * Softmax + Top-K extraction
 */

#include "classification.h"
#include <cmath>
#include <algorithm>

std::vector<float> softmax(const float* logits, int num_classes) {
    if (!logits || num_classes <= 0) return {};

    /* Find max for numerical stability */
    float max_val = logits[0];
    for (int i = 1; i < num_classes; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    /* Compute exp and sum */
    std::vector<float> probs(num_classes);
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }

    /* Normalize (guard against division by zero) */
    if (sum == 0.0f) {
        float uniform = 1.0f / num_classes;
        for (int i = 0; i < num_classes; i++) probs[i] = uniform;
        return probs;
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < num_classes; i++) {
        probs[i] *= inv_sum;
    }

    return probs;
}

std::vector<ClassResult> top_k(const float* probs, int num_classes, int k) {
    if (!probs || num_classes <= 0) return {};

    std::vector<ClassResult> results(num_classes);
    for (int i = 0; i < num_classes; i++) {
        results[i].class_id = i;
        results[i].confidence = probs[i];
    }

    k = std::min(k, num_classes);
    std::partial_sort(results.begin(), results.begin() + k, results.end(),
        [](const ClassResult& a, const ClassResult& b) {
            return a.confidence > b.confidence;
        });

    results.resize(k);
    return results;
}

std::vector<ClassResult> classify(const float* logits, int num_classes, int top_k_count) {
    auto probs = softmax(logits, num_classes);
    if (probs.empty()) return {};
    return top_k(probs.data(), num_classes, top_k_count);
}

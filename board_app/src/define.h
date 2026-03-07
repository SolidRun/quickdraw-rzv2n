#ifndef DEFINE_H
#define DEFINE_H

/*
 * Quick Draw DRP-AI Inference Server — Renesas RZ/V2N
 *
 * Unix socket server for classification inference.
 * Receives raw grayscale canvas, preprocesses in C++
 * (crop/pad/invert/resize/normalize), runs DRP-AI, returns JSON.
 *
 * Model-specific config (class names, input size) loaded at runtime
 * from labels.txt and CLI arguments — no rebuild needed to swap models.
 */

/* Defaults (overridable via CLI args) */
#define DEFAULT_MODEL_DIR   "qd_mobilenetv2"
#define DEFAULT_LABELS      "labels.txt"
#define DEFAULT_INPUT_W     128
#define DEFAULT_INPUT_H     128
#define DEFAULT_INPUT_C     3
#define DEFAULT_SOCKET      "/tmp/quickdraw.sock"
#define TOP_K               5
#define NUM_OUTPUTS         1

/* DRP-AI device */
#define DRPAI_DEV           "/dev/drpai0"

/*
 * DRP-AI frequency settings (RZ/V2N)
 *   DRP_MAX_FREQ: DRP core   — 2=420MHz, 3=315MHz, 4=252MHz, 5=210MHz
 *   DRPAI_FREQ:   AI-MAC     — 1=1GHz, 3=630MHz, 4=420MHz, 5=315MHz
 */
#define DRP_MAX_FREQ        2
#define DRPAI_FREQ          3

#endif /* DEFINE_H */

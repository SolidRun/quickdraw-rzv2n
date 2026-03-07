# Application

Single C++ binary (`app_quickdraw_gui`) that handles drawing input, preprocessing, DRP-AI inference, and display. No server process, no sockets, no Python.

---

## Architecture

```
Touch/mouse input
    |
    v
Cairo drawing surface (white bg, black ink)
    |
    v
Extract grayscale pixels from surface
    |
    v
C++ preprocessing:
  1. Find ink bounding box (pixels < 245)
  2. Crop with 12 px margin
  3. Pad to square (centered)
  4. Invert (white strokes on black bg)
  5. Resize to 128x128 (area-based downscale, bilinear upscale)
  6. Repeat to 3 channels (R=G=B)
  7. Normalize: pixel / 255.0 (0-1 scaling)
    |
    v
float32 [3, 128, 128] tensor (CHW)
    |
    v
DRP-AI3 hardware inference (~1 ms)
  Model: MobileNetV2, INT8, 345 classes
    |
    v
[1, 345] raw logits
    |
    v
Softmax -> temporal smoothing -> top-5
    |
    v
Display: prediction, confidence bars, AI commentary
```

### Preprocessing defaults

From `src/preprocessing.h`:

| Parameter | Value |
|-----------|-------|
| `model_size` | 128 |
| `ink_threshold` | 245 (pixels below this are considered ink) |
| `crop_margin` | 12 px |

### Classification

From `src/classification.cpp`:

- Softmax with overflow guard (subtracts max logit before exp)
- Top-K extraction (K=5, defined in `src/define.h` as `TOP_K`)

---

## Configuration

Two config files control the application. Both are editable without rebuilding.

### config.ini

Loaded by `main_gui.cpp` at startup. Controls DRP-AI hardware and model path.

```ini
[model]
model_dir = model
model_name = qd_mobilenetv2
labels = labels.txt
input_size = 128

[drpai]
# DRP core frequency index
#   2 = 420 MHz (fastest)
#   3 = 315 MHz
#   4 = 252 MHz
#   5 = 210 MHz
drp_max_freq = 2

# AI-MAC frequency index
#   1 = 1 GHz (fastest)
#   3 = 630 MHz
#   4 = 420 MHz
#   5 = 315 MHz
drpai_freq = 1

[ui]
config_json = config.json
```

### config.json

Loaded by `gui.cpp`. Controls UI layout, colors, and AI commentary.

#### UI settings

| Key | Value in config.json | Default in gui.h | Description |
|-----|---------------------|-------------------|-------------|
| `font_scale` | 1.2 | 1.2 | Text size multiplier |
| `brush_radius` | 8 | 8 | Drawing brush size (pixels) |
| `canvas_ratio` | 0.80 | 0.72 | Fraction of screen width for canvas |
| `canvas_padding` | 6 | 6 | Padding around canvas (pixels) |
| `auto_predict_delay_ms` | 500 | 500 | Delay after pen-up before auto-predict |
| `live_predict_interval_ms` | 500 | 0 | Predict interval while drawing (0 = disabled) |
| `fps` | 15 | 15 | GUI refresh rate |
| `max_history` | 8 | 8 | Number of history entries shown |
| `title` | "QUICK DRAW" | "QUICK DRAW" | Title bar text |
| `subtitle` | "DRP-AI3" | "DRP-AI3" | Title bar subtitle |
| `badge` | "RZ/V2N SR SOM" | "RZ/V2N SR SOM" | Title bar badge |

#### Model settings (in config.json)

| Key | Value | Description |
|-----|-------|-------------|
| `smooth_window` | 3 | Temporal smoothing: average last N predictions |
| `confidence_threshold` | 0.15 | Below this, display shows "Uncertain..." |

#### Colors

All colors are RGB arrays. Example: `"accent": [60, 200, 220]`.

| Key | RGB | Usage |
|-----|-----|-------|
| `background` | [46, 30, 30] | Main background |
| `title_bg` | [78, 45, 45] | Title bar |
| `text` | [230, 220, 220] | Primary text |
| `text_dim` | [160, 140, 140] | Secondary text |
| `accent` | [60, 200, 220] | Highlights, top prediction |
| `green` | [100, 220, 100] | Correct/YES |
| `red` | [230, 80, 80] | Wrong/NO |
| `yellow` | [240, 220, 60] | Warnings |
| `bar_bg` | [70, 50, 50] | Confidence bar background |
| `bar_fg` | [60, 140, 180] | Confidence bar fill |
| `panel_line` | [120, 100, 100] | Panel separator lines |
| `btn_clear` | [90, 60, 60] | CLEAR button |
| `btn_undo` | [90, 60, 60] | UNDO button |
| `btn_predict` | [40, 80, 120] | EXIT button |
| `btn_yes` | [60, 120, 60] | YES button |
| `btn_no` | [140, 60, 60] | NO button |
| `comment_bg` | [60, 45, 50] | AI comment background |
| `comment_text` | [200, 180, 220] | AI comment text |

#### AI Commentary

Enabled by default (`"enabled": true`).

| Setting | Value | Description |
|---------|-------|-------------|
| `confident_threshold` | 0.8 | Above this triggers "confident" comments |
| `uncertain_threshold` | 0.3 | Below this triggers "confused" comments |
| `no_repeat_buffer` | 8 | Prevent same comment within last 8 picks |
| `min_display_secs` | 0.5 | Minimum time before comment changes |

Comment categories and their triggers:

| Pool | Trigger |
|------|---------|
| `confident` | Top prediction above `confident_threshold` |
| `uncertain` | Top prediction between thresholds |
| `confused` | Top prediction below `uncertain_threshold` |
| `close_call` | Top two predictions within 10% of each other |
| `improving` | Confidence increasing while drawing |
| `changing` | Top prediction changes |
| `long_journey` | Many guesses before confident result |
| `first_stroke` | Very first stroke on empty canvas |
| `streak` | Ongoing correct streak |
| `wrong_feedback` | User presses NO |

Templates use variables: `{class}`, `{prob}`, `{prev}`, `{num_guesses}`, `{first_guess}`, `{streak}`.

---

## Controls

| Action | Touch | Keyboard |
|--------|-------|----------|
| Draw | Drag on canvas | Mouse drag |
| Clear canvas | CLEAR button | `c` |
| Undo last stroke | UNDO button | `z` |
| Mark correct | YES button | `y` |
| Mark wrong | NO button | `n` |
| Exit | EXIT button | `Esc` |
| Stop from terminal | -- | `Ctrl+C` |

---

## Constants

From `src/define.h`:

| Constant | Value |
|----------|-------|
| `DEFAULT_MODEL_DIR` | `"qd_mobilenetv2"` |
| `DEFAULT_LABELS` | `"labels.txt"` |
| `DEFAULT_INPUT_W` | 128 |
| `DEFAULT_INPUT_H` | 128 |
| `DEFAULT_INPUT_C` | 3 |
| `DEFAULT_SOCKET` | `"/tmp/quickdraw.sock"` |
| `TOP_K` | 5 |
| `NUM_OUTPUTS` | 1 |
| `DRPAI_DEV` | `"/dev/drpai0"` |
| `DRP_MAX_FREQ` | 2 (420 MHz) |
| `DRPAI_FREQ` | 3 (630 MHz) |

Note: `config.ini` overrides `DRPAI_FREQ` to 1 (1 GHz).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot open /dev/drpai0` | Run as root. `run.sh` auto-elevates with `su`. |
| `Failed to load model` | Verify `model/qd_mobilenetv2/` contains `deploy.so`, `deploy.json`, `deploy.params`. |
| No display | Check weston is running: `ls /run/user/*/wayland-*` |
| Touch not responding | Touch requires Wayland compositor. Verify weston. |
| Always shows "Uncertain..." | Lower `confidence_threshold` in `config.json`. |
| Wrong predictions | Normalization mismatch. Must be `[0,0,0]/[1,1,1]` in training, compilation, and app. |
| `libmera2_runtime.so not found` | `run.sh` installs on first run. Or: `cp lib/*.so /usr/lib64/ && ldconfig` |
| Docker container not running | `docker start drp-ai_tvm_v2n_container_ahmad` |
| Root-owned `build/` directory | `docker_build.sh` handles cleanup automatically. |

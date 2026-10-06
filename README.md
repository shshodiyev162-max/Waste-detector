# Waste Detector

Point a phone camera at a piece of waste and the detector names its material in real time: **biodegradable, cardboard, glass, metal, paper or plastic**.

A YOLO11n detector runs **entirely in the browser** through ONNX Runtime Web. There's no server and no uploads, and the camera feed never leaves the phone.

**▶ Live demo:** [ADD YOUR NETLIFY LINK HERE] (open it on a phone and allow camera access)

---

## Results

These are validation results for the current model (YOLO11n, 640px, trained on the merged, leak-free dataset):

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **All** | **0.906** | **0.863** | **0.899** | **0.670** |
| Biodegradable | 0.775 | 0.561 | 0.656 | 0.365 |
| Cardboard | 0.897 | 0.914 | 0.924 | 0.725 |
| Glass | 0.950 | 0.950 | 0.964 | 0.754 |
| Metal | 0.954 | 0.937 | 0.972 | 0.757 |
| Paper | 0.965 | 0.949 | 0.968 | 0.809 |
| Plastic | 0.896 | 0.870 | 0.912 | 0.608 |

These are validation-set numbers, drawn from the same distribution as the training data. This project already learned once that good validation metrics don't guarantee real-world performance (see below). Testing at a realistic distance through the deployed app is still in progress.

## How it works

```
Phone camera (getUserMedia, rear camera)
   → letterbox resize to 640×640 (same as training)
   → YOLO11n in ONNX Runtime Web (WASM, on-device)
   → decode raw output [1, 10, 8400] → confidence threshold → Non-Max Suppression
   → boxes + labels drawn over the live video
```

The whole app is a single static `index.html` hosted on Netlify. HTTPS is what makes phone camera access work.

**Why YOLO11n and the browser:** the nano model has 2.6M parameters and is built for edge inference, so it can run on a phone CPU. A larger model like YOLO11x would be far too slow there. Running in the browser avoids the server memory crashes of an earlier Gradio/Hugging Face deployment, and avoids building a native app.

## What went wrong, and what it taught me

The full story, with every number, is in **[Waste-detector-journal.md](Waste-detector-journal.md)**. The short version:

1. **TACO dataset → wrong framing.** About 45% of objects were smaller than 32×32 px in wide street scenes, while my use case is one object held up to a camera. Merging classes changed the labels but not the photos. I switched datasets.
2. **"Paper doesn't work" wasn't a data shortage.** Paper had 1,238 training images, but only about 1.2% of them had landed in validation. A broken split, not missing data. I re-split, rarest class first.
3. **Data leakage.** After the re-split, the model hit 0.64 mAP50 after **one epoch**, which isn't plausible. Roboflow's augmented copies of the same photo were landing in both train and valid. I re-split by source-photo group.
4. **A stale cache faked a retrain.** A retrain on filtered labels produced metrics identical to three decimal places, which is statistically impossible. Ultralytics had reused an old `.cache` file, so the retrain never actually happened.
5. **The real bug was distance.** On a live phone, the model kept calling things paper or cardboard. Measuring box sizes showed paper and cardboard boxes averaged 29–35% of the frame (versus 4–15% for the other classes), and the dataset was shot close-up while my camera was about 50 cm away. It was a **scale/domain gap**.
6. **The fix that worked:** switching to a medium-range dataset (Recyclable Waste Detection) and merging in biodegradable from the original dataset, with a class-ID remap and a leak-safe split. **mAP50 went from 0.670 to 0.899.**

**Known limitation:** biodegradable still lags (recall 0.56). Its images are dense scenes with about 20 objects per image, versus about 1 for the other classes, and I haven't found a better source for it yet.

## Datasets

- **Recyclable Waste Detection** (Roboflow Universe, `recycle/recyclable-waste-detection`): cardboard, glass, metal, paper, plastic
- **Garbage Classification 3** (Roboflow Universe, `material-identification/garbage-classification-3`): biodegradable class only
- Merged dataset: 10,650 images, split 80/20 per class with 0 source-photo overlap between train and valid

## Run it yourself

```bash
# any static server works; the camera needs localhost or HTTPS
python -m http.server 8000
# open http://localhost:8000 and allow camera access
```

`test.py` runs the exported `best.onnx` directly in Python, bypassing the browser. That's how I checked the model separately from the web pipeline.

## Tech

Ultralytics YOLO11n · PyTorch (CUDA, RTX 5060 Laptop) · ONNX / ONNX Runtime Web · vanilla JS + HTML · Netlify

**Earlier version:** [trash-sorting-v2](https://github.com/shshodiyev162-max/trash-sorting-v2), a ResNet50 image classifier (fastai + Gradio). This repo is the detector that replaced it.

---
Built by Shohjahon Shodiyev, 2026 · MIT License

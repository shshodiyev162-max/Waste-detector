# Waste Detector

A real-time waste-material detector that runs **entirely in the browser**. Point a phone camera at an item and it draws a box and labels the material: **biodegradable, cardboard, glass, metal, paper, or plastic**.

Built solo as an applied machine-learning project. The repo documents the whole path, including the dead ends, because most of the learning came from diagnosing why things failed.

> **Status: in development.** The model scores well on its validation set (mAP50 **0.899**), but live testing on a phone is still **inconsistent**: paper is often labelled cardboard, and cardboard sometimes fires on the whole frame. See [Limitations](#limitations). I'm not claiming a finished, field-reliable system.

**Live demo:** [waste-detector.netlify.app](https://waste-detector.netlify.app) (open on a phone and allow camera access)

**Earlier version:** [trash-sorting-v2](https://github.com/shshodiyev162-max/trash-sorting-v2), a ResNet50 image classifier (fastai + Gradio). This repo is the detector that replaced it.

---

## Contents

1. [What it does](#what-it-does)
2. [Repository layout](#repository-layout)
3. [Results at a glance](#results-at-a-glance)
4. [How the project evolved](#how-the-project-evolved)
5. [Final dataset](#final-dataset)
6. [Final model and training](#final-model-and-training)
7. [Final results](#final-results)
8. [The browser app](#the-browser-app)
9. [Reproducing this](#reproducing-this)
10. [Limitations](#limitations)
11. [Bugs found and fixed](#bugs-found-and-fixed)
12. [Lessons learned](#lessons-learned)
13. [Next steps](#next-steps)
14. [Datasets, credits and licences](#datasets-credits-and-licences)

---

## What it does

- Opens the rear camera (`getUserMedia`, `facingMode: "environment"`).
- Runs a **YOLO11n** detector exported to **ONNX** using `onnxruntime-web` (WASM). Nothing is uploaded; inference happens on the device.
- Letterboxes each frame to 640×640, decodes the raw YOLO output, applies a confidence threshold and Non-Max Suppression, and draws boxes over the live video.
- Deploys as a single static page on Netlify. There is no server.

## Repository layout

```
Waste-detector/
├── README.md                    # this file
├── Waste-detector-journal.md    # detailed phase-by-phase lab notebook (Phases 1–16)
├── index.html                   # self-contained camera + ONNX Runtime Web app
├── best.onnx                    # final exported detector (YOLO11n, 6 classes)
├── test.py                      # runs best.onnx in Python, bypassing the browser
├── images.jpg, images (1).jpg   # sample images used by test.py
└── LICENSE                      # MIT
```

Datasets, `runs/`, and training caches are not in the repo.

## Results at a glance

| Stage | Data | Model | Result |
|---|---|---|---|
| Classification baseline | TrashNet, 5,054 imgs, 6 cls | ResNet34 | ~89% accuracy |
| Classification v2 | 7,431 imgs, 8 cls | ResNet50 (fastai) | ~94% accuracy |
| Detection v1 | TACO, 1,453 imgs, 60 cls | YOLO11n @640 | mAP50 ≈ 0.28–0.30 |
| Detection v2 | TACO filtered, 38 cls | YOLO11n @1280 | mAP50 ≈ 0.21 |
| Detection v3 | Garbage Classification 3 (6 cls), leak-safe split | YOLO11n @640 | mAP50 **0.670** |
| **Detection v4 (final)** | **Merged dataset (6 cls), leak-safe split** | **YOLO11n @640** | **mAP50 0.899** |

The detection numbers are not on a shared benchmark: each stage used a different dataset and a different validation set, so they show the *direction* of progress, not a controlled comparison.

---

## How the project evolved

The short version: **classification worked, detection on the wrong data didn't, and the project became a series of diagnoses about data rather than models.** Full detail is in [`waste-detector-journal.md`](waste-detector-journal.md).

### Stage 0 — Classification experiments (before the detector)

| Experiment | Dataset | Model | Outcome |
|---|---|---|---|
| Material classifier | TrashNet: 5,054 images, 6 classes (metal 820, glass 1,002, plastic 964, paper 1,188, trash 274, cardboard 806) | ResNet34 | ~89% accuracy |
| Classifier v2 | 7,431 images, 8 classes | ResNet50 (fastai), saved as `waste_classifier_v2.pkl` | ~94% accuracy |

Classifiers need the object already cropped and centred, so they can't find waste in a live camera frame. That motivated moving to detection. An earlier Gradio / Hugging Face Spaces deployment of the classifier also ran out of memory, which pushed the later design toward client-side inference.

### Stage 1 — TACO detection (several attempts, none worked)

TACO is the standard open litter dataset (1,453 images in my copy, 60 categories, split 1,162 / 145 / 146).

| Attempt | Setup | Result |
|---|---|---|
| YOLO11n on native categories | 60 classes, 640 px, batch 16 (actual 8), patience 10, seed 42 | P 0.504 / R 0.308 / mAP50 0.296 / mAP50-95 0.233; a later evaluation gave P 0.337 / R 0.339 / mAP50 0.275 / mAP50-95 0.207 (stopped near epoch 52) |
| Merge into material buckets | plastic / glass / paper / cardboard | no meaningful improvement |
| Filter tiny boxes and rare classes | drop boxes < 16 px and classes with < 10 annotations: 60 → 52 → 38 classes; 1,419 images, 3,136 annotations, split 1,135 / 142 / 142 | — |
| Higher resolution | 1280 px, batch 2, 120 epochs, patience 20, AMP | P 0.585 / R 0.185 / mAP50 0.211 / mAP50-95 0.180 |

Per-class AP50 on the 1280 run showed the long tail clearly: clear plastic bottle 0.472, drink can 0.308, plastic film 0.150, glass bottle 0.025, broken glass 0.004.

**Diagnosis:** a *training/deployment distribution mismatch*, not a tuning problem. TACO photos are wide litter scenes (streets, beaches, forests) shot from a few metres; about 45% of annotated objects are smaller than 32×32 px. My use case is a close-up object held to a phone. Merging classes changes label names, not the photos, so no amount of class merging, resolution, or filtering could close that gap.

> Caveat: the 640 px and 1280 px runs are **not** a controlled comparison (different class sets, batch sizes, and filtering). Some checkpoint mappings and the 1,453 → 1,419 image change were not recorded precisely.

### Stage 2 — Dataset pivot

Switched to **Garbage Classification 3** (Roboflow Universe): 10,464 images, bounding-box annotated, 6 classes. Also considered and rejected:

- **Two-stage pipeline** (class-agnostic detector + ResNet50 classifier): simpler to ship one end-to-end detector.
- **YOLO11x**: too heavy for an 8 GB laptop GPU, and far too slow for in-browser WASM inference on a phone. YOLO11n (≈2.6 M parameters) was chosen deliberately.

### Stage 3 — Paper "doesn't work" (it was the split, not the data)

First run: mAP50 0.537, but **paper = 0.077**. The obvious guess was "not enough paper data". Counting labels directly showed 1,238 paper training images, more than several classes, but only **15 validation images** (33 instances). Paper's validation share was ~1.2% and plastic's ~8%, versus 23–30% for the other classes. The dataset was split badly, not short on data. Fix: pool everything and re-split, handling the rarest class first.

### Stage 4 — Data leakage

After the re-split, a clean from-scratch run hit mAP50 0.643 after **one epoch**, which is implausible. Roboflow exports several augmented copies of each source photo (shared filename prefix before `.rf.<hash>`). My re-split shuffled at file level, so near-duplicates landed on both sides of the split. Fix: group by source photo, then split by group.

**Clean result on this dataset: mAP50 0.670** (mAP50-95 0.472).

| Class | mAP50 | mAP50-95 | Recall |
|---|---|---|---|
| Biodegradable | 0.615 | 0.335 | 0.500 |
| Cardboard | 0.576 | 0.423 | 0.499 |
| Glass | 0.772 | 0.595 | 0.693 |
| Metal | 0.769 | 0.540 | 0.691 |
| Paper | 0.664 | 0.526 | 0.595 |
| Plastic | 0.627 | 0.413 | 0.587 |

### Stage 5 — Browser deployment

Exported to ONNX (`imgsz=640, simplify=True, opset=12`), verified the tensor shapes (input `images`, output `output0` = `[1, 10, 8400]`: 4 box values + 6 class scores × 8400 candidates), and built the web app. Two bugs here are described under [Bugs found and fixed](#bugs-found-and-fixed). Hosted on Netlify (HTTPS is required for phone camera access).

### Stage 6 — "It thinks everything is paper"

Live, the model called a face, a wall and a room "paper". I first ruled out the deployment by running the ONNX file directly in Python (a plastic bottle came back PLASTIC at 0.62). Many PAPER images in the dataset are magazine pages with faces, and a closed-set detector with no background class cannot say "none of these". A stratified check on 48 validation images (8 per class) scored **43/48 (~90%)**, matching training metrics, which proved the model was fine *on its own distribution*.

### Stage 7 — Real phone testing exposes a deeper problem

Real-world testing still failed. Measuring box sizes per class showed why:

| Class | Avg box area (% of frame) | Boxes > 50% of frame | Boxes > 80% of frame |
|---|---|---|---|
| Biodegradable | 4.2 | 1.2% | 0.4% |
| Cardboard | 28.9 | 23.3% | 12.6% |
| Glass | 15.4 | 6.0% | 1.2% |
| Metal | 11.8 | 5.9% | 1.7% |
| Paper | 34.6 | 30.3% | 18.3% |
| Plastic | 12.7 | 7.5% | 1.9% |

Paper and cardboard labels often cover most of the frame, so the model learned "big textured region = paper/cardboard". I filtered boxes over 80% for those two classes (904 label lines removed; 753 images left with no labels and kept as negative examples), then retrained.

**Result: metrics identical to the previous run to the third decimal.** That is statistically impossible for different data. Cause: Ultralytics' `.cache` file is keyed by path, and I had replaced labels in place at the same path, so the **stale cache was reused and the "retrain" trained on the old labels**. Fix: delete `*.cache` before retraining whenever label content changes.

### Stage 8 — Domain gap, again, in reverse

Independently of the cache bug, the main insight was that the original dataset's photos are mostly *close-up, filling the frame*, but a person holds an item at roughly arm's length (~50 cm). TACO's objects were too small/far; this dataset's were too large/close. Same class of problem, opposite direction. The strong 43/48 test could not reveal it because it drew from the training distribution.

Two fixes were weighed: (1) collect my own phone photos and fine-tune, which was infeasible without enough physical items; (2) switch to a dataset with more representative framing, which I did.

### Stage 9 — New dataset and merge

Chose **Recyclable Waste Detection** (Roboflow Universe, `recycle/recyclable-waste-detection`, v1): 8,373 images, 5 classes, medium-range single-object framing. Rejected `garbage_detection` (one class, pile-up scenes at varying distance) and `Waste Classification` (too sparse). It has no biodegradable class, so biodegradable images were extracted from the original dataset and merged in.

Class IDs differ between the datasets, and YOLO labels store IDs, not names, so merging without remapping would silently mislabel classes. Remap applied to the new dataset:

| Class | New-dataset ID | Unified ID |
|---|---|---|
| Biodegradable | — | 0 |
| Cardboard | 0 | 1 |
| Glass | 1 | 2 |
| Metal | 2 | 3 |
| Paper | 3 | 4 |
| Plastic | 4 | 5 |

The unified order matches the original project so the web app's `CLASS_NAMES` didn't change. Pooled counts reconciled exactly (8,371 + 2,279 = 10,650). The 8,371 vs the 8,373 images on the dataset page is a 2-file difference I did not trace.

### Stage 10 — Leak-safe resplit and final training

Grouped by source photo again: 10,650 files → **9,359 groups** (1,291 files were augmented duplicates). Split by group, rarest-class-first, ~20% validation per class. Verified directly: **0 overlapping groups** (7,490 train / 1,869 valid).

Epoch-1 mAP50 was already 0.569, which triggered the same suspicion as Stage 4, so the overlap check was run *before* trusting the result. It passed; the speed-up is consistent with better-matched data and a COCO-pretrained backbone. The run was accidentally interrupted by a `KeyboardInterrupt` at epoch 57; because Ultralytics saves `best.pt` each improving epoch, nothing was lost and `best.pt` was validated directly.

---

## Final dataset

Merged from two Roboflow Universe datasets (see [credits](#datasets-credits-and-licences)).

| | Files | Notes |
|---|---|---|
| Pooled | 10,650 | 9,359 source-photo groups |
| Train | 8,516 (8,510 used) | 6 files skipped as corrupt by Ultralytics |
| Valid | 2,134 (2,133 used) | 1 file skipped; 10,731 validation instances |

Ultralytics skipped those 7 files because of mixed segment/box label formats.

| Class | Images | Instances | Train imgs | Valid imgs | Valid % |
|---|---|---|---|---|---|
| Biodegradable | 2,279 | 45,358 | 1,822 | 457 | 20.1% |
| Cardboard | 1,584 | 1,785 | 1,271 | 313 | 19.8% |
| Glass | 1,583 | 1,600 | 1,267 | 316 | 20.0% |
| Metal | 1,725 | 1,767 | 1,380 | 345 | 20.0% |
| Paper | 1,728 | 1,745 | 1,380 | 348 | 20.1% |
| Plastic | 1,848 | 1,925 | 1,478 | 370 | 20.0% |

The five new-dataset classes have ~1 instance per image (single-object framing). Biodegradable keeps its original ~20 instances per image (dense scenes), a structural difference that matters in the results.

`data.yaml`:

```yaml
train: ../train/images
val: ../valid/images

nc: 6
names: ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
```

## Final model and training

- **Model:** Ultralytics YOLO11n, COCO-pretrained (`yolo11n.pt`), 6 classes.
- **Hardware:** RTX 5060 Laptop GPU (8 GB VRAM), 16 GB single-channel RAM, hence `cache=False`.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(
    data="GARBAGE-V2-RESPLIT/data.yaml",
    imgsz=640, batch=16, device=0, workers=4,
    patience=20, epochs=100, cache=False,
    project="runs/waste-detector", name="yolo11n_v2_newdata",
)

# Export for the browser
model.export(format="onnx", imgsz=640, simplify=True, opset=12)
```

Training was interrupted at epoch 57; `best.pt` was used.

## Final results

Validation of `best.pt` on the held-out, leak-free validation split (2,133 images, 10,731 instances):

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **All** | **0.906** | **0.863** | **0.899** | **0.670** |
| Biodegradable | 0.775 | 0.561 | 0.656 | 0.365 |
| Cardboard | 0.897 | 0.914 | 0.924 | 0.725 |
| Glass | 0.950 | 0.950 | 0.964 | 0.754 |
| Metal | 0.954 | 0.937 | 0.972 | 0.757 |
| Paper | 0.965 | 0.949 | 0.968 | 0.809 |
| Plastic | 0.896 | 0.870 | 0.912 | 0.608 |

**Progress on the same six classes** (mAP50):

| Class | Old dataset (Stage 4) | Merged dataset (final) |
|---|---|---|
| Biodegradable | 0.615 | 0.656 |
| Cardboard | 0.576 | 0.924 |
| Glass | 0.772 | 0.964 |
| Metal | 0.769 | 0.972 |
| Paper | 0.664 | 0.968 |
| Plastic | 0.627 | 0.912 |

Five of six classes gained 0.19–0.35 mAP50, which supports the domain-gap diagnosis: changing the training photos, not the model or hyperparameters, produced the improvement. Biodegradable barely moved because its images were carried over unchanged.

**Read these numbers carefully.** They are validation metrics from the *same distribution as training*. The original dataset also looked strong on validation (0.670 mAP50, 43/48 on a real-image test) and still failed on a phone. The phone is the real test, and it is not yet passing consistently.

## The browser app

`index.html` is one self-contained file:

1. `getUserMedia` opens the rear camera.
2. `onnxruntime-web` (WASM execution provider, loaded from a CDN) loads `best.onnx` (ONNX Runtime Web 1.19.0 from jsDelivr).
3. Each frame is **letterboxed** to 640×640 (aspect-preserving resize plus padding) to match training.
4. The `[1, 10, 8400]` output is decoded manually: for each of 8,400 candidates take the max class score, threshold at **0.4**, then run IoU-based NMS at **0.45**.
5. A `requestAnimationFrame` loop draws boxes and labels over the video.

Run locally (any static server works; the camera needs localhost or HTTPS):

```bash
python -m http.server 8000
# open http://localhost:8000 and allow camera access
```

`test.py` runs `best.onnx` directly in Python on a sample image and prints the top class ID and confidence. That is how I checked the model separately from the web pipeline.

Phones need HTTPS for the camera, so deploy to a static host (Netlify drag-and-drop works) rather than testing over a local IP.

## Reproducing this

```bash
# 1. Environment
pip install ultralytics roboflow onnx onnxruntime

# 2. Download the two Roboflow datasets (YOLO11 format; API key needed)
#    material-identification/garbage-classification-3  (v2)
#    recycle/recyclable-waste-detection                (v1)

# 3. Remap the new dataset's class IDs (0..4 -> 1..5), extract class-0
#    BIODEGRADABLE labels + images from the old set, and pool both.

# 4. Group files by source photo (filename prefix before ".rf.") and split
#    BY GROUP, rarest class first, ~20% validation, seed 42.
#    Verify the train/valid group intersection is empty.

# 5. Delete stale caches before every retrain
find . -name "*.cache" -delete

# 6. Train and export (see "Final model and training")
```

The full splitting and remapping code lives in my notebook; the logic is described in Stages 4, 9 and 10 above.

---

## Limitations

Stated plainly:

- **Real-device performance is inconsistent.** On a phone, paper is often called cardboard, and cardboard can still fire on the whole frame. Validation metrics did not translate cleanly.
- **Biodegradable is weak** (recall 0.561, mAP50-95 0.365). Its source data is dense multi-object scenes (~20 instances per image), unlike the rest. I never visually confirmed this explanation.
- **Box-area bias was never re-checked on the merged dataset.** The original set had large background-like paper/cardboard boxes; the new set was not audited for the same issue.
- **Closed-set detector with no background class.** Faces, walls and empty rooms get forced into the nearest label.
- **Same-distribution validation.** There is no independent test set of photos taken the way the app is actually used.
- **TACO-stage numbers are not fully traceable** (checkpoint mapping, class-mapping history, 1,453 → 1,419 images, inference speed not recorded).
- **Inference speed on phones is unmeasured.**
- **Training was stopped at epoch 57** by an accident, not by convergence criteria.
- **Material, not recyclability.** The classes are material guesses, not disposal advice.

## Bugs found and fixed

| Bug | Symptom | Cause | Fix |
|---|---|---|---|
| Bad validation split | Paper mAP50 0.077 | Paper had 15 val images (~1.2%) | Pool and re-split, rarest class first |
| Data leakage | mAP50 0.643 after 1 epoch | Augmented copies of one photo on both sides of the split | Split by source-photo group |
| Silent web page | Nothing happened, no errors | Hand-merge left `async function main()` unclosed, nesting later functions inside it | Inspect the full file; close the brace |
| Nothing rendered | Pipeline ran but no display | `.box` had width but no height; children were `position: absolute` | `aspect-ratio: 4 / 3` |
| Stale label cache | Retrain gave identical metrics | Ultralytics `.cache` keyed by path; labels changed in place | Delete `*.cache` before retrain |
| Class-ID mismatch (prevented) | Would silently mislabel classes | Different class order between datasets | Explicit remap table before merging |
| Mixed label formats | 7 files skipped | Segment and box labels mixed | Left as-is (negligible: 7 of 10,650) |
| Interrupted run | `KeyboardInterrupt` at epoch 57 | Accidental kernel interrupt | Validated `best.pt` from disk |

## Lessons learned

1. **Debug the data before the model.** Every major failure was in the data (split, leakage, framing), never in the architecture.
2. **Implausible results are signals, not wins.** Impossibly good (0.643 at epoch 1) and impossibly identical (same metrics to three decimals) results both meant a bug.
3. **Count it yourself.** The dataset page's summary hid the broken split; counting label files exposed it.
4. **Validation metrics measure consistency, not deployment.** A model can score ~90% on its own distribution and still fail on a phone.
5. **Match training data to deployment conditions** (distance, framing, object size).
6. **Silent failures are the worst ones.** Stale caches, ID mismatches and unclosed braces gave no error. Verify with hard checks (group overlap = 0, pooled counts sum exactly).
7. **Constrain the model to the target hardware early.** YOLO11n was chosen for in-browser inference on a phone.

## Next steps

- Collect a small **real-device test set** (photos taken the way the app is used) and report metrics on it separately from validation.
- Add a **confidence threshold and size guard** in the app to suppress whole-frame detections.
- Audit the merged dataset's **box-area distribution** per class.
- Find or build a better biodegradable source with single-object framing.
- Consider a **background/negative class** or negative images so non-waste isn't forced into a label.
- Measure on-device **latency** and consider quantisation.
- Consider the two-stage option (class-agnostic detector + classifier) if the single detector stays unreliable.

---

## Datasets, credits and licences

- **Garbage Classification 3**: Roboflow Universe, `material-identification/garbage-classification-3`, v2. Source of the biodegradable class and the first six-class experiments.
- **Recyclable Waste Detection**: Roboflow Universe, `recycle/recyclable-waste-detection`, v1. Source of cardboard, glass, metal, paper and plastic in the final model.
- **TACO** (Trash Annotations in Context): used for early experiments only. See the TACO project for its licence terms.
- **TrashNet**: used for the early classification baseline only. See its repository for licence terms.
- **YOLO11 / Ultralytics**: [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0 licence applies to the code and weights; check this before any commercial use).
- **ONNX Runtime Web**: Microsoft, loaded from a CDN in the browser.

Please check each dataset's own page for current licence and attribution requirements; the Roboflow datasets used here were published under CC BY 4.0 at the time I downloaded them.

MIT licence (see `LICENSE`). Built by Shohjahon Shodiyev, 2026.

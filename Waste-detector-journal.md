# Waste Detector — Project Journal

A record of the actual methodology behind this project: what was tried, what
failed, why, and how each failure was diagnosed. Kept deliberately honest
about dead ends — the debugging process here (particularly the data leakage
catch, the stale-cache dead end, and the domain-gap diagnosis) is as much a
demonstration of ML understanding as the final model is.

## Phase 1 — TACO dataset

**Goal:** detect and classify waste from a mobile phone camera.

Started with TACO (Trash Annotations in Context), the standard open litter
dataset: ~1,500 images, 60+ fine-grained litter categories, instance
segmentation masks.

Attempts made on TACO:
- Trained a ResNet50 classifier on TACO's annotated instances.
- Trained YOLO11n as a detector directly on TACO's 60+ native categories —
  poor performance, severe long-tail imbalance (a handful of categories
  carry most instances, many categories have only single digits to low
  dozens of examples).
- Combined the 60+ categories into coarser material buckets (plastic,
  glass, paper, cardboard) to reduce class count — performance did not
  meaningfully improve.
- Trained without combining classes, at 1280px resolution (rather than
  the usual 640px), specifically to help the model see TACO's very small
  object instances.
- Removed annotations below a small pixel-area threshold, on the theory
  that the tiniest/most ambiguous boxes were adding noise rather than
  useful signal.

None of these fixed the underlying problem.

### Diagnosis

TACO's images are wide real-world litter scenes — streets, beaches,
forests — shot from a few meters away, not close-up single-object photos.
Roughly 45% of TACO's annotated instances are smaller than 32x32 pixels.
Combining classes changes the *label count*, not the *underlying photos* —
the same small, cluttered, low-resolution crops remain, just grouped under
fewer names. This is a **training/deployment distribution mismatch**, not
a fixable data-processing problem: TACO was built for a different task
(spotting litter in a scene) than the one this project needs (identifying
a close-up object a phone camera is pointed directly at).

**Conclusion:** no amount of class-merging, resolution increase, or
annotation-filtering closes a gap that exists because the source photos
themselves don't resemble the deployment scenario. (This exact class of
problem — training distribution vs. deployment distribution — resurfaces
later, in Phase 13, in a subtler form that took much longer to diagnose.)

## Phase 2 — Dataset pivot

Rather than continuing to patch TACO, switched to a dataset whose photos
better match the deployment scenario: **Garbage Classification 3**
(Roboflow Universe, `material-identification/garbage-classification-3`,
v2) — 10,464 images, already bounding-box annotated, 6 material classes:
biodegradable, cardboard, glass, metal, paper, plastic.

Also considered and rejected:
- **A two-stage pipeline** (class-agnostic detector + the existing
  ResNet50 classifier) — would have sidestepped detector-side imbalance
  entirely, but a single end-to-end detector on a better-matched dataset
  was simpler to ship and deploy given the timeline.
- **YOLO11x** instead of nano — rejected on two grounds: training
  feasibility on an 8GB-VRAM laptop GPU, and inference speed once
  deployed to a browser running on a phone's CPU (WASM), where a
  57M-parameter model would very likely be too slow for a live-camera
  demo. YOLO11n (2.6M params) is specifically built for constrained/edge
  inference.

## Phase 3 — First training run on the new dataset

Config: `yolo11n.pt` pretrained start, `imgsz=640`, `batch=16`, `device=0`
(RTX 5060 Laptop, 8GB VRAM), `workers=4`, `patience=20`, `epochs=100`,
`cache=False` (default — 16GB system RAM is single-channel, so RAM
caching was avoided to reduce memory pressure).

Result: overall mAP50 **0.537**.

| Class | Instances (val) | mAP50 |
|---|---|---|
| Biodegradable | 13,637 | 0.612 |
| Cardboard | 1,292 | 0.619 |
| Glass | 2,380 | 0.819 |
| Metal | 1,360 | 0.692 |
| Paper | 33 | **0.077** |
| Plastic | 214 | 0.406 |

Paper was effectively non-functional. Initial hypothesis: paper simply
lacked enough training data.

### Diagnosis

Counted actual instances in the downloaded label files directly (not
Roboflow's dataset-page summary), across train + validation combined:

| Class | Train images | Train instances | Val images | Val instances |
|---|---|---|---|---|
| Biodegradable | 1,603 | 31,721 | 676 | 13,637 |
| Cardboard | 1,328 | 3,948 | 398 | 1,360 |
| Glass | 1,879 | 5,429 | 805 | 2,380 |
| Metal | 1,053 | 3,386 | 438 | 1,292 |
| Paper | 1,238 | 2,981 | 15 | 33 |
| Plastic | 1,019 | 4,146 | 91 | 214 |

**This overturned the "not enough data" hypothesis.** Paper had 1,238
training images — more than several other classes. The problem was the
*validation split*, not the *dataset*: paper's split ratio was ~1.2%
into validation, and plastic's was ~8%, while every other class sat in a
normal 23–30% band. This pattern (some classes' images clustered and
mis-split) is consistent with a dataset assembled from multiple
contributors, where a naive shuffle-based split can fail if a class's
images were uploaded in batches rather than uniformly.

**Fix:** pooled all images back together and re-split, processing the
rarest class first so its ~20% validation allocation couldn't be crowded
out by chance before it was assigned.

## Phase 4 — Data leakage discovery

After the first re-split, a follow-up fine-tuning run showed paper jump
from 0.077 to 0.661 mAP50 and plastic from 0.406 to 0.606 — but a second,
*clean* from-scratch run on the same re-split data hit 0.643 overall mAP50
after a single epoch. That result is implausible for genuine learning —
it should take dozens of epochs to reach numbers like that, not one.

### Diagnosis

Roboflow-exported datasets contain multiple augmented copies of each
original source photo (rotated, brightness-shifted, etc.), identifiable
by a shared filename prefix before `.rf.<hash>`. The first re-split
pooled and shuffled at the *individual file* level, with no awareness of
this grouping — meaning an augmented copy of a photo could land in
`train/` while a near-duplicate sibling of that same photo landed in
`valid/`. The model wasn't being validated on unseen data; it was
being tested on near-copies of what it had just trained on.

**Fix:** re-split again, this time grouping all augmented copies of a
source photo together before assigning train/valid, so no source photo's
siblings are split across both sides.

This "implausibly high result after very little training → suspect
leakage before celebrating" pattern recurred twice more later in the
project (Phase 16), and having internalized it here is what prevented it
from being missed the second time.

## Phase 5 — Final validated result (first dataset)

Same config as Phase 3, run clean from `yolo11n.pt` on the leak-safe
split, full run (no interruption, no fine-tuning restart).

Overall mAP50: **0.670** (mAP50-95: 0.472)

| Class | mAP50 | mAP50-95 | Recall |
|---|---|---|---|
| Biodegradable | 0.615 | 0.335 | 0.500 |
| Cardboard | 0.576 | 0.423 | 0.499 |
| Glass | 0.772 | 0.595 | 0.693 |
| Metal | 0.769 | 0.540 | 0.691 |
| Paper | 0.664 | 0.526 | 0.595 |
| Plastic | 0.627 | 0.413 | 0.587 |

Paper and plastic land almost exactly where the (leakage-adjacent,
fine-tuned) Phase 4 run put them — 0.664 vs. 0.661, 0.627 vs. 0.606 —
which cross-validates that the split fix, not the leakage or the
fine-tuning restart, was the real driver of the improvement.

### Open issue: biodegradable

Biodegradable has by far the most training data of any class (31,721
instances) yet still shows the lowest mAP50-95 (0.335) and a recall of
only 0.500 — the model misses about half of all biodegradable instances.
More data has not fixed this, which rules out volume as the explanation.

Working hypothesis: biodegradable images average roughly 20 labeled
instances per image, versus roughly 3 for every other class — suggesting
these are dense scenes with many small/overlapping objects, a
structurally harder detection problem than the largely single/few-object
photos in the other five classes. Never confirmed by direct visual
inspection of the training images; this pattern held all the way through
the dataset merge in Phase 15 as well (see below), and remains an
acknowledged, unresolved limitation rather than a solved problem.

## Environment

- RTX 5060 Laptop GPU, 8GB VRAM
- 16GB system RAM, single-channel
- `imgsz=640`, `batch=16`, `workers=4`, `cache=False`
- Ultralytics YOLO11n, PyTorch (CUDA build)
- Windows (native, moved off WSL for the webapp folder)

## Phase 6 — ONNX export and the browser detector

Exported the Phase 5 checkpoint: `model.export(format="onnx", imgsz=640,
simplify=True, opset=12)`. Verified input/output names (`images` /
`output0`) and output tensor shape (`[1, 10, 8400]` — 4 box coordinates +
6 class scores, 8400 candidate boxes at 640px) directly, rather than
assuming them.

Built a single self-contained HTML page: camera access via
`getUserMedia` (rear camera via `facingMode: "environment"`),
`onnxruntime-web` (WASM execution provider, loaded from CDN) for
client-side inference, manual letterboxing (aspect-ratio-preserving
resize + padding) to match the training-time preprocessing, manual
decoding of the raw YOLO output tensor (per-box max-class-score
extraction, confidence thresholding, Non-Max Suppression via IoU), and a
`requestAnimationFrame` loop drawing boxes over the live video.

Chosen over Gradio/Hugging Face Spaces (an earlier ResNet50 classifier
deployment on that stack hit an out-of-memory failure) and over a native
mobile app (new platform tooling, not worth the deadline risk). A
browser-side static deployment sidesteps server-side inference and its
memory-management failure modes entirely, and deploys as a single static
page to Netlify.

### Bug: silently broken page (missing brace)

After merging two versions of the script (a debug-logging version and a
full-pipeline version) by hand, the page loaded but did nothing — no
errors, no console output, not even a page-load `alert()` on line 1
firing. Diagnosed by requesting the complete current file content
directly rather than guessing from a description: the manual merge had
left `async function main()` unclosed, so every function defined after
it was nested inside `main()` instead of being a top-level declaration.
This is a class of bug that's invisible from the outside — the script
still *parses*, it just does something structurally different from what
it looks like — which is why pasting the full file, not a description of
it, was the only way to catch it.

### Bug: correct execution, nothing rendered

Once the brace bug was fixed, console logs confirmed every pipeline stage
running (camera stream acquired, model loaded, inference running,
"Running" status shown) — but the page displayed nothing. Root cause was
CSS, not JS: the `.box` container div had an explicit `width` but no
`height`, and its `video`/`canvas` children were `position: absolute`
(removed from normal document flow), so the parent collapsed to zero
height with nothing to force it open. Fixed by adding `aspect-ratio: 4 /
3` to `.box`, matching the camera's actual logged resolution (640x480).

## Phase 7 — The "it thinks everything is paper" investigation

Deployed and tested live: the model called a human face, a wall, and the
room behind the user "paper" with moderate-to-high confidence.

First hypothesis: the deployed pipeline was broken. Ruled this out by
testing the exported `.onnx` model directly in Python, bypassing the
browser entirely — a clean photo of a plastic bottle correctly predicted
PLASTIC at 0.62 confidence. The model and export were fundamentally
sound; the failure was specific to certain content.

Second hypothesis, then confirmed by inspecting the training images
directly: many PAPER-labeled photos in the source dataset are magazine
pages containing human faces, so the model had genuinely learned to
associate "a face-like region filling most of the frame" with the paper
class. Faces and empty rooms are also simply **out-of-distribution**
input for a closed-set detector with no "background/none of the above"
class — it has no way to say "I don't know," only to pick its
best-matching trained label, however poor that match is.

**Conclusion at the time:** not a bug to fix, since faces/rooms are
outside the actual use case (someone holding an item up to the camera).
This conclusion held up until Phase 10, where real phone testing on
non-face content showed the paper/cardboard problem was broader than
just magazine faces (see below).

### Validation via stratified test

To confirm the underlying model was sound rather than assume it, ran a
controlled test: 8 real validation images per class, single-class images
only (48 total), tallying true vs. predicted class through the isolated
Python ONNX pipeline. Result: **43/48 correct (~90%)**, consistent with
training-time mAP. This confirmed the model generalizes well *within
its training distribution* — the open question, not yet visible at this
point, was whether that distribution matched real deployment conditions.

## Phase 8 — Deployment

Hosted the webapp on Netlify (drag-and-drop static deploy). HTTPS is
required for camera access on phones, which Netlify provides by default,
solving the earlier same-WiFi/local-IP camera-permission issue
encountered during local testing.

## Phase 9 — Real phone testing surfaces a deeper issue

Live phone testing (not faces, not screens — attempted on real
surroundings) reproduced the "everything is paper/cardboard" behavior,
which ruled out the Phase 7 "it's just magazine faces" explanation as
incomplete. Direct re-inspection of the training dataset showed many
PAPER and CARDBOARD boxes covering large, background-like regions of
their frame rather than discrete objects — not limited to magazine-face
images.

### Quantifying the bias

Rather than accept the visual impression, wrote a script computing, per
class, the average bounding-box area as a percentage of frame and the
percentage of boxes exceeding 50% of frame:

| Class | Avg box area (% of frame) | % of boxes > 50% of frame |
|---|---|---|
| Biodegradable | 4.2 | 1.2% |
| Cardboard | 28.9 | 23.3% |
| Glass | 15.4 | 6.0% |
| Metal | 11.8 | 5.9% |
| Paper | 34.6 | 30.3% |
| Plastic | 12.7 | 7.5% |

This confirmed the hypothesis quantitatively: paper and cardboard have
roughly 2–8x larger average boxes than every other class, and 4–25x more
boxes exceeding half the frame. A genuine, provable dataset labeling
quality issue, not something inferred from a handful of anecdotes.

Re-ran at a stricter 80%-of-frame threshold to separate "a smaller subset
of extreme outliers" (mechanically filterable) from "most of the class
is like this" (not filterable, needs a different fix):

| Class | % of boxes > 80% of frame |
|---|---|
| Biodegradable | 0.4% |
| Cardboard | 12.6% |
| Glass | 1.2% |
| Metal | 1.7% |
| Paper | 18.3% |
| Plastic | 1.9% |

Five of six classes sit under 2% — background noise. Paper and cardboard
sit an order of magnitude higher, confirming this was a real, isolable
minority of the class, not a pervasive property of it — worth a targeted
mechanical fix rather than only a deployment-side threshold hack.

## Phase 10 — The filtering attempt and the stale-cache dead end

Filtered out label lines where a PAPER or CARDBOARD box exceeded 80% of
frame area (904 lines removed across the dataset; 753 images ended up
with zero remaining labels, kept as background/negative examples rather
than dropped, since a labeled-empty image teaches the model "there is
nothing to detect here" — directly useful against an over-triggering
model). Retrained from `yolo11n.pt` on the filtered labels.

**Result: identical metrics to the third decimal place across every
class**, matching the pre-filter Phase 5 run exactly. This is not
possible for two independently trained runs on genuinely different data
— the probability of an exact match by chance across six classes and
four metrics each is effectively zero.

### Diagnosis

Root cause: Ultralytics caches parsed label data into a `.cache` file
(e.g. `train.cache`) keyed largely by file path rather than by label
content. The filtering step replaced the contents of `train/labels`
in-place (same path, new content) via a folder rename, but a stale
`.cache` file from an earlier run at that same path was silently reused,
meaning the "retrained" model had, in fact, trained on the original
unfiltered labels the entire time — the retraining never actually
happened despite every surface signal (new run folder, no errors,
plausible-looking logs) suggesting it had.

This is a subtler failure mode than either of the two earlier bugs
(validation-split, data leakage): it produces *no error*, *no warning*,
and a completely normal-looking training log — the only tell was the
statistically impossible exact-match metrics, caught only because the
Phase 4 leakage bug had already established the habit of treating
suspiciously-good or suspiciously-identical results as a signal to
investigate rather than a result to report.

**Lesson, not yet re-applied at time of writing:** clear `.cache` files
before any retrain where label *content* changed but label *paths*
didn't.

## Phase 11 — Reconsidering the dataset itself

After the stale-cache confusion, and independent of it, direct
inspection of the dataset raised a further, distinct concern beyond the
paper/cardboard box-area bias already fixed: most of the dataset's
photos are shot close-up, filling most of the frame — a framing
convention that doesn't match how a person actually holds an object up
to a phone camera (roughly arm's length, ~50cm). This is a **scale/
distance domain gap**, structurally the same category of problem as
Phase 1's TACO mismatch, but inverted: TACO's objects were too small/far
for the deployment scenario, this dataset's objects were too large/close.

This diagnosis was reached independently, from real phone-testing
evidence (the same model that scored 43/48 in the isolated Python
stratified test performed poorly live on a phone) rather than from
metrics — a stratified test drawn from the same distribution as training
data can only ever validate consistency *within* that distribution, not
generalization to a different one. This is the correct and sharper
explanation for the real-world failures observed in Phase 9, superseding
the "just needs box-area filtering" framing as the complete picture.

Two remediation paths were weighed:
1. **Collect real phone/webcam photos and fine-tune** on them directly —
   the most surgical fix, but infeasible: no access to enough physical
   waste items to build even a small calibration set.
2. **Switch to a dataset whose photos are shot at a more representative
   distance/framing** — chosen instead.

## Phase 12 — Dataset research and selection

Searched Roboflow Universe for alternatives, explicitly screening
candidates for framing/distance rather than just class coverage or image
count, since that framing mismatch was now understood to be the actual
target to fix.

Rejected candidates:
- **garbage_detection** (single "garbage" class) — pileup/environmental
  scenes at varying distances, same wrong-framing problem as TACO.
- **Waste Classification** (2 visible classes) — too sparse for a 6-class
  project.

Selected: **Recyclable Waste Detection** (Roboflow Universe,
`recycle/recyclable-waste-detection`, v1) — 8,373 images, 5 classes
(Cardboard, Glass, Metal, Paper, Plastic), medium-range single-object
framing closer to the actual deployment scenario. Tradeoff: no
biodegradable class, so biodegradable had to be sourced separately from
the original dataset and merged in.

## Phase 13 — Merging datasets: class-ID remap

The new dataset's class order (`Cardboard=0, Glass=1, Metal=2, Paper=3,
Plastic=4`) does not match the original project's order
(`Biodegradable=0, Cardboard=1, Glass=2, Metal=3, Paper=4, Plastic=5`).
Since YOLO label files store class IDs, not names, merging label files
from two sources without remapping IDs first would silently mislabel
classes (e.g. a glass box copied in as class ID 1 would be read as
cardboard in the unified scheme) — a bug that would not throw an error
or look obviously wrong, only quietly corrupt training signal.

Built an explicit remap table before touching any files:

| Class | New dataset ID | Final unified ID |
|---|---|---|
| Biodegradable | (not present) | 0 |
| Cardboard | 0 | 1 |
| Glass | 1 | 2 |
| Metal | 2 | 3 |
| Paper | 3 | 4 |
| Plastic | 4 | 5 |

Kept the final order identical to the original project's order
specifically so the deployed webapp's `CLASS_NAMES` array needed no
changes.

Pooled all images from the new dataset (train+valid+test) into a staging
directory with labels rewritten through the remap table, then separately
extracted only the biodegradable-labeled lines and their images from the
original dataset (both its train and valid splits, since everything was
being resplit from scratch anyway) — filtering out any image with zero
biodegradable boxes, so only genuinely biodegradable-labeled content
carried over, with no cross-contamination from the other five classes
already covered by the new dataset. Verified no filename collisions
between the two sources by checking that pooled counts summed exactly
(8,371 + 2,279 = 10,650, confirmed).

Post-merge per-class counts:

| Class | Images | Instances |
|---|---|---|
| Biodegradable | 2,279 | 45,358 |
| Cardboard | 1,584 | 1,785 |
| Glass | 1,583 | 1,600 |
| Metal | 1,725 | 1,767 |
| Paper | 1,728 | 1,745 |
| Plastic | 1,848 | 1,925 |

The five new classes show a near-1:1 instance-to-image ratio, consistent
with the single-object framing the new dataset was chosen for.
Biodegradable retains its original ~20-instances-per-image density —
unchanged, since its source data didn't change, and its Phase 5 recall/
mAP weakness should be expected to persist for the same unresolved
reason.

## Phase 14 — Leak-safe resplit, again

Applied the same source-photo-grouping method from Phase 4 to the merged
pool, since both source datasets could independently contain Roboflow's
augmented-copy naming pattern. Of 10,650 total label files, only 9,359
distinct source-photo groups were found — 1,291 files were augmented
duplicates of a photo already present elsewhere in the pool, confirming
the same leak risk existed here and needed the same fix.

Split at the group level, rarest-class-first (processing order:
`GLASS, CARDBOARD, METAL, PAPER, PLASTIC, BIODEGRADABLE`), targeting 20%
validation per class:

| Class | Train images | Valid images | Valid % |
|---|---|---|---|
| Biodegradable | 1,822 | 457 | 20.1% |
| Cardboard | 1,271 | 313 | 19.8% |
| Glass | 1,267 | 316 | 20.0% |
| Metal | 1,380 | 345 | 20.0% |
| Paper | 1,380 | 348 | 20.1% |
| Plastic | 1,478 | 370 | 20.0% |

Every class landed within 0.3% of the 20% target — no repeat of the
Phase 3 validation-split bug.

## Phase 15 — Retraining on the merged dataset

Same config as every prior run: `yolo11n.pt`, `imgsz=640`, `batch=16`,
`device=0`, `workers=4`, `patience=20`, `epochs=100`, `cache=False`.

At epoch 1, overall mAP50 was already 0.569 (mAP50-95 0.357) — a result
high enough, this early, to immediately raise the same suspicion the
Phase 4 leakage bug had trained into the process. Rather than assume
either "this is fine" or "this is broken," ran the group-overlap check
directly against the actual split on disk:

```
Train groups: 7,490
Valid groups: 1,869
Overlapping source-photo groups: 0
```

Zero overlap — confirmed leak-free. By epoch 15, overall mAP50 reached
roughly **0.8**, clearly above the original dataset's fully-trained
ceiling of 0.670. Unlike the Phase 4 leak (which inflated the ceiling
itself while pretending to be genuine), this result is consistent with
what actually changed: a dataset with more consistent single-object
framing and a COCO-pretrained backbone that already knows how to
localize generic objects converges faster and higher without needing a
shortcut to do it.

### Interrupted run, recovered without loss

At epoch 57, an accidental kernel interrupt (`KeyboardInterrupt`) killed
the training process mid-run. No data was lost: Ultralytics writes
`last.pt` and `best.pt` to disk after every completed epoch, so the
checkpoint from the most recent improvement was already safely on disk.
Rather than resume the run, validated `best.pt` directly — the metrics
by epoch 15 had already suggested the model was near its ceiling, making
the remaining unrun epochs unlikely to matter much.

## Phase 16 — Final validated result (merged dataset)

Full validation of `best.pt` from the interrupted run:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **All (overall)** | 0.906 | 0.863 | 0.899 | 0.670 |
| Biodegradable | 0.775 | 0.561 | 0.656 | 0.365 |
| Cardboard | 0.897 | 0.914 | 0.924 | 0.725 |
| Glass | 0.950 | 0.950 | 0.964 | 0.754 |
| Metal | 0.954 | 0.937 | 0.972 | 0.757 |
| Paper | 0.965 | 0.949 | 0.968 | 0.809 |
| Plastic | 0.896 | 0.870 | 0.912 | 0.608 |

Five of six classes now sit at mAP50 0.91–0.97 with precision and recall
both in the 0.87–0.97 range — a substantial jump from the original
dataset's ceiling (Phase 5: overall mAP50 0.670) across every one of
those five classes. This is strong evidence the Phase 11 domain-gap
diagnosis (training-distribution framing/distance mismatch with the
deployment scenario) was the correct explanation for the original
dataset's real-world failures, not just a plausible story: fixing that
one property of the training data produced a large, consistent
improvement across every class the new dataset covers.

**Biodegradable remains the clear outlier** — mAP50 0.656, mAP50-95
0.365, recall 0.561. It improved modestly from Phase 5 (0.615/0.335/
0.500), but nowhere near what the other five classes gained, which is
consistent with the Phase 5 hypothesis: biodegradable's source data is
structurally different (dense scenes, ~20 instances/image vs. ~1 for
the other classes) and that property didn't change in the merge, since
biodegradable's images were carried over unmodified from the original
dataset. This is an honest, acknowledged limitation, not a solved
problem — the fix that worked for the other five classes (better
framing/distance match) was never applied to biodegradable's source
data, because no such alternative source was found or built.

One number worth being precise about in any write-up: these are
**validation-set metrics**, drawn from the same source distribution as
training. The entire reason this phase of the project happened is that
validation metrics from the *original* dataset (Phase 5: 0.670 mAP50,
and a 43/48 stratified real-image test) looked strong but did not
predict real-device performance. The same caveat applies here until
confirmed otherwise by testing on an actual physical object at a
realistic distance through the deployed webapp — a step intentionally
called out as still pending rather than folded into a claim of success,
in keeping with the lesson this exact project already had to learn once.

## Deployment

Browser-based static page (camera access + ONNX Runtime Web for
client-side inference), hosted on Netlify. Chosen over a native mobile
app and over Gradio/Hugging Face Spaces, as detailed in Phase 6.
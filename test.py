import onnxruntime as ort
import numpy as np
from PIL import Image

session = ort.InferenceSession("best.onnx")

# Pick a real image from your validation set that you know the true label for
img = Image.open("images (1).jpg").convert("RGB").resize((640, 640))
arr = np.array(img).astype(np.float32) / 255.0
arr = arr.transpose(2, 0, 1)[None, :, :, :]  # HWC -> CHW, add batch dim

outputs = session.run(None, {"images": arr})
output = outputs[0][0]  # shape (10, 8400)

# Find the single highest-confidence detection across all 8400 candidates
class_scores = output[4:, :]  # 6 classes x 8400
best_idx = np.unravel_index(np.argmax(class_scores), class_scores.shape)
class_id, box_idx = best_idx
print("Best class ID:", class_id, "confidence:", class_scores[class_id, box_idx])
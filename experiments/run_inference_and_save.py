# experiments/run_inference_and_save.py
# Run segmentation inference, save probability map and binary mask, list candidate components.
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import measure

from src.inference import seg_infer, extract_candidate_components, classify_and_prune

MODEL_PATH = "experiments/models/seg/seg_epoch10.pth"
IMAGE_PATH = "data/processed/case_000_img.nii.gz"
OUT_DIR = "experiments"

os.makedirs(OUT_DIR, exist_ok=True)

print("Running segmentation inference...")
prob = seg_infer(MODEL_PATH, IMAGE_PATH)  # returns numpy array (z,y,x)
print("Prob map shape:", prob.shape)

# Save probability map
prob_path = os.path.join(OUT_DIR, "pred_case_000_prob.nii.gz")
nib.save(nib.Nifti1Image(prob.astype("float32"), affine=np.eye(4)), prob_path)
print("Saved probability map to:", prob_path)

# Make a binary mask (threshold 0.5) and do small-object removal
mask = (prob >= 0.5).astype("uint8")
mask = ndimage.binary_opening(mask, structure=np.ones((3,3,3))).astype("uint8")
mask_path = os.path.join(OUT_DIR, "pred_case_000_mask.nii.gz")
nib.save(nib.Nifti1Image(mask.astype("uint8"), affine=np.eye(4)), mask_path)
print(f"Saved binary mask to: {mask_path} (voxels = {int(mask.sum())})")

# List connected components found
labels = measure.label(mask)
props = measure.regionprops(labels)
print("Connected components (candidates) found:", len(props))
for p in props:
    centroid = tuple(float(x) for x in p.centroid)
    print(f" - Label {p.label}: voxels={p.area}, centroid_vox={centroid}")

# If classifier exists, try pruning (optional)
cls_model_path = "experiments/models/cls/cls_final.pth"
if os.path.exists(cls_model_path):
    try:
        print("Classifier found; running classification + pruning...")
        img = nib.load(IMAGE_PATH).get_fdata().astype(np.float32)
        candidates = extract_candidate_components(prob, prob_thresh=0.3, min_voxels=10)
        kept = classify_and_prune(candidates, img, cls_model_path, cls_patch_size=32, cls_thresh=0.5)
        print("After classifier pruning, kept candidates:", len(kept))
        for k in kept:
            print(f"  centroid={k['centroid']}, score={k.get('score', None)}")
    except Exception as e:
        print("Classifier pruning failed:", e)
else:
    print("No classifier model found at", cls_model_path)

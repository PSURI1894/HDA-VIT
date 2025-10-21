# experiments/postprocess_filter.py
import os
import csv
import numpy as np
import nibabel as nib
from skimage import measure
from scipy import ndimage

PROB = "experiments/pred_case_000_prob.nii.gz"
IMG  = "data/processed/case_000_img.nii.gz"
OUT_DIR = "experiments/filtered"
os.makedirs(OUT_DIR, exist_ok=True)

# Parameters: tweak these
THRESHOLD = 0.30        # probability threshold to binarize
MIN_VOXELS = 20         # smallest component to keep
MAX_VOXELS = 50000      # largest component to keep (removes giant false positives)

print("Loading probability map:", PROB)
prob = nib.load(PROB).get_fdata()

# Optional: if you have a lung mask to restrict to lung region, set LUNG_MASK to path or None
LUNG_MASK = None  # e.g., "data/processed/case_000_mask.nii.gz"
if LUNG_MASK and os.path.exists(LUNG_MASK):
    lung = nib.load(LUNG_MASK).get_fdata().astype(bool)
    print("Applying lung mask to probability map.")
    prob = prob * lung

mask = (prob >= THRESHOLD).astype(np.uint8)
# optional light morphological cleaning (closing to fill holes)
mask = ndimage.binary_closing(mask, structure=np.ones((3,3,3))).astype(np.uint8)

labels = measure.label(mask)
props = measure.regionprops(labels)
print("Total components found (pre-filter):", len(props))

kept_mask = np.zeros_like(mask, dtype=np.uint8)
rows = []
for p in props:
    vox = p.area
    label = p.label
    centroid = tuple(float(x) for x in p.centroid)
    keep = (vox >= MIN_VOXELS) and (vox <= MAX_VOXELS)
    rows.append({
        "label": label,
        "voxels": int(vox),
        "centroid_z": centroid[0],
        "centroid_y": centroid[1],
        "centroid_x": centroid[2],
        "keep": keep
    })
    if keep:
        kept_mask[labels == label] = 1

# Save kept mask
out_mask = os.path.join(OUT_DIR, f"mask_th{int(THRESHOLD*100)}_min{MIN_VOXELS}_max{MAX_VOXELS}.nii.gz")
nib.save(nib.Nifti1Image(kept_mask.astype(np.uint8), affine=np.eye(4)), out_mask)
print("Saved filtered mask:", out_mask)

# Save CSV report
csv_path = os.path.join(OUT_DIR, "components_report.csv")
with open(csv_path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["label","voxels","centroid_z","centroid_y","centroid_x","keep"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
print("Saved components report:", csv_path)

# Summary
kept = sum(1 for r in rows if r["keep"])
print(f"Kept components: {kept} / {len(rows)}")

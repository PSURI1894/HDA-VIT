# experiments/threshold_analysis.py
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import measure
from src.quantify import compute_volume_mm3, equiv_sphere_diameter

PROB_PATH = "experiments/pred_case_000_prob.nii.gz"
IMG_PATH = "data/processed/case_000_img.nii.gz"
OUT_DIR = "experiments/thresholds"
spacing = (1.0, 1.0, 1.0)   # mm (adjust if your processed spacing differs)

os.makedirs(OUT_DIR, exist_ok=True)

prob = nib.load(PROB_PATH).get_fdata()
print("Loaded prob shape:", prob.shape)
img = nib.load(IMG_PATH).get_fdata()

thresholds = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1]
for th in thresholds:
    mask = (prob >= th).astype('uint8')
    # remove tiny islands (but keep small objects for now)
    mask_clean = ndimage.binary_opening(mask, structure=np.ones((3,3,3))).astype('uint8')
    mask_path = os.path.join(OUT_DIR, f"mask_th_{int(th*100)}.nii.gz")
    nib.save(nib.Nifti1Image(mask_clean, affine=np.eye(4)), mask_path)
    labels = measure.label(mask_clean)
    props = measure.regionprops(labels)
    print(f"TH={th:.2f}: voxels_total={(mask_clean.sum()):d}, components={len(props)}")
    # report top 10 components by voxel count
    comps = sorted(props, key=lambda p: p.area, reverse=True)
    for i,p in enumerate(comps[:10]):
        vox = p.area
        vol_mm3 = compute_volume_mm3((labels==p.label).astype(np.uint8), spacing)
        diam = equiv_sphere_diameter(vol_mm3)
        centroid = tuple(float(x) for x in p.centroid)
        print(f"  comp#{i+1}: vox={vox}, vol_mm3={vol_mm3:.1f}, equiv_diam_mm={diam:.1f}, centroid_vox={centroid}")
    print("-"*60)
print("Saved masks to", OUT_DIR)

# inspect_prob.py
import nibabel as nib
import numpy as np
p = nib.load("experiments/pred_case_000_prob.nii.gz").get_fdata()
print("prob shape:", p.shape)
print("min, max, mean:", float(p.min()), float(p.max()), float(p.mean()))
# percentiles
for q in [0,1,5,10,25,50,75,90,95,99,100]:
    print(f"p{q}: {np.percentile(p, q):.6f}")
# small histogram counts for bins
hist, bins = np.histogram(p.flatten(), bins=20, range=(0.0,1.0))
print("hist bins:", list(zip([round(b,2) for b in bins[:-1]], hist.tolist())))
# number of voxels above thresholds
for th in [0.9, 0.7, 0.5, 0.3, 0.1]:
    cnt = (p >= th).sum()
    print(f"voxels >= {th}: {int(cnt)}")

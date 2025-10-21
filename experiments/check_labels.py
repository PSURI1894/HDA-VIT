# experiments/check_labels.py
import os, glob
import nibabel as nib
import numpy as np

data_dir = "data/processed"
files = sorted(glob.glob(os.path.join(data_dir, "*_label*.nii*")) + glob.glob(os.path.join(data_dir, "*_mask*.nii*")))
if not files:
    print("No *_label or *_mask files found in", data_dir)
else:
    print("Found label/mask files:", len(files))
for p in files:
    img = nib.load(p).get_fdata()
    voxels = int((img != 0).sum())
    total = int(img.size)
    frac = voxels / total
    mins = float(img.min()); maxs = float(img.max())
    unique = np.unique(img) if img.size <= 1e6 else None
    print("----")
    print(p)
    print(f" shape={img.shape}, voxels_nonzero={voxels}, fraction={frac:.6f}, min={mins}, max={maxs}")
    if unique is not None:
        print(" unique values:", unique)
    # bounding box of positive voxels (if any)
    if voxels>0:
        coords = np.array(np.where(img!=0))
        zmin,zmax = coords[0].min(), coords[0].max()
        ymin,ymax = coords[1].min(), coords[1].max()
        xmin,xmax = coords[2].min(), coords[2].max()
        print(f" bbox z[{zmin},{zmax}] y[{ymin},{ymax}] x[{xmin},{xmax}]")

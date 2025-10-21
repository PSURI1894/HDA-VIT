# Synthetic demo generator to create small NIfTI volumes for quick testing
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import nibabel as nib

def _make_sphere(shape, center, radius, value=1):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = (zz-center[0])**2 + (yy-center[1])**2 + (xx-center[2])**2
    mask = dist <= radius**2
    arr = np.zeros(shape, dtype=np.float32)
    arr[mask] = value
    return arr, mask

def generate_sample_dataset(out_dir, n_cases=8, vol_size=128, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)
    for i in range(n_cases):
        vol = np.zeros((vol_size, vol_size, vol_size), dtype=np.float32)
        mask = np.zeros_like(vol, dtype=np.uint8)
        n_nod = np.random.randint(1,4)
        for n in range(n_nod):
            c = [np.random.randint(16, vol_size-16) for _ in range(3)]
            r = np.random.randint(4,12)
            sphere, sph_mask = _make_sphere(vol.shape, c, r, value=1.0)
            mask = (mask | sph_mask).astype(np.uint8)
            vol += sphere.astype(np.float32) * 0.9
        vol = gaussian_filter(vol, sigma=1.0)
        vol = vol + np.random.normal(0, 0.02, size=vol.shape)
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        img = nib.Nifti1Image(vol.astype('float32'), affine=np.eye(4))
        msk = nib.Nifti1Image(mask.astype('uint8'), affine=np.eye(4))
        nib.save(img, os.path.join(out_dir, f"case_{i:03d}_img.nii.gz"))
        nib.save(msk, os.path.join(out_dir, f"case_{i:03d}_label.nii.gz"))
    print(f"Saved {n_cases} synthetic cases to {out_dir}")

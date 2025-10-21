import numpy as np

def compute_volume_mm3(mask, spacing):
    voxel_vol = spacing[0]*spacing[1]*spacing[2]
    return mask.sum() * voxel_vol

def equiv_sphere_diameter(volume_mm3):
    return ((6*volume_mm3)/np.pi)**(1/3)

def mean_hu(mask, ct_arr):
    vals = ct_arr[mask>0]
    if vals.size == 0: return None
    return float(vals.mean())

def max_linear_dimension(mask, spacing):
    coords = np.array(np.where(mask))
    if coords.shape[1] < 2: return 0.0
    coords_mm = coords.T * np.array(spacing)
    d = np.linalg.norm(coords_mm[:,None,:] - coords_mm[None,:,:], axis=-1)
    return float(d.max())

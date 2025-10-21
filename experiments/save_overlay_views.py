# experiments/save_overlay_views.py
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

PROB_PATH = "experiments/pred_case_000_prob.nii.gz"
IMG_PATH = "data/processed/case_000_img.nii.gz"
OUT_DIR = "experiments/figs"
os.makedirs(OUT_DIR, exist_ok=True)

prob = nib.load(PROB_PATH).get_fdata()
img = nib.load(IMG_PATH).get_fdata()

# center slices
z,y,x = np.array(prob.shape)//2
slices = {
    f"axial_z{z}": (img[z,:,:], prob[z,:,:]),
    f"coronal_y{y}": (img[:,y,:], prob[:,y,:]),
    f"sagittal_x{x}": (img[:,:,x], prob[:,:,x])
}

for name, (im, p) in slices.items():
    plt.figure(figsize=(6,6))
    plt.imshow(im, cmap='gray')
    plt.imshow(p, cmap='Reds', alpha=0.45)
    plt.axis('off')
    outp = os.path.join(OUT_DIR, f"{name}.png")
    plt.savefig(outp, bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved", outp)

print("Check the images in", OUT_DIR)

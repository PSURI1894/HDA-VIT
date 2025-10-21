import os, torch, numpy as np, nibabel as nib
from monai.inferers import sliding_window_inference
from src.models.unet3d_monai import get_monai_unet
from src.models.resnet3d import ResNet3D
from skimage import measure
from scipy import ndimage

def seg_infer(model_path, image_nifti, device=None, roi_size=(96,96,96), sw_batch_size=2):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_monai_unet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    img = nib.load(image_nifti).get_fdata().astype(np.float32)
    input_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = sliding_window_inference(input_tensor, roi_size, sw_batch_size, model, device=device)
        prob = torch.sigmoid(pred).cpu().numpy()[0,0]
    return prob

def extract_candidate_components(prob_map, prob_thresh=0.3, min_voxels=10):
    mask = (prob_map >= prob_thresh).astype(np.uint8)
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)
    candidates = []
    for p in props:
        if p.area < min_voxels: continue
        mask_component = (labeled == p.label).astype(np.uint8)
        centroid = p.centroid
        bbox = p.bbox
        candidates.append({ "label":p.label, "area":p.area, "centroid":centroid, "bbox":bbox, "mask":mask_component })
    return candidates

def classify_and_prune(candidates, img_arr, cls_model_path, cls_patch_size=32, device=None, cls_thresh=0.5):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls = ResNet3D(num_classes=1, in_ch=1)
    cls.load_state_dict(torch.load(cls_model_path, map_location=device))
    cls.to(device).eval()
    kept = []
    for c in candidates:
        cz, cy, cx = [int(round(v)) for v in c['centroid']]
        half = cls_patch_size//2
        z0 = max(0, cz-half); z1 = z0+cls_patch_size
        y0 = max(0, cy-half); y1 = y0+cls_patch_size
        x0 = max(0, cx-half); x1 = x0+cls_patch_size
        sz = img_arr.shape
        if z1>sz[0] or y1>sz[1] or x1>sz[2]:
            z0 = max(0, min(z0, sz[0]-cls_patch_size)); y0 = max(0, min(y0, sz[1]-cls_patch_size)); x0 = max(0, min(x0, sz[2]-cls_patch_size))
            z1 = z0+cls_patch_size; y1=y0+cls_patch_size; x1=x0+cls_patch_size
        patch = img_arr[z0:z1, y0:y1, x0:x1]
        p = torch.from_numpy(patch[None,None,...]).float().to(device)
        with torch.no_grad():
            out = cls(p).cpu().numpy().squeeze()
            prob = 1.0 / (1.0 + np.exp(-out))
        if prob >= cls_thresh:
            c['score'] = float(prob)
            c['bbox_crop'] = (z0,z1,y0,y1,x0,x1)
            kept.append(c)
    kept = sorted(kept, key=lambda x: x['score'], reverse=True)
    return kept

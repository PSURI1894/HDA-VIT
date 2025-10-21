import os
import numpy as np
import SimpleITK as sitk
from src.io_utils import dicom_series_to_itk, resample_itk, sitk_to_numpy, numpy_to_sitk

def window_clip(arr_hu, min_hu=-1000, max_hu=400):
    arr = np.clip(arr_hu, min_hu, max_hu)
    arr = (arr - min_hu) / (max_hu - min_hu)
    return arr.astype(np.float32)

def simple_lung_mask(itk_img):
    arr = sitk.GetArrayFromImage(itk_img)
    mask = (arr < -320).astype(np.uint8)
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(itk_img)
    mask_img = sitk.BinaryMorphologicalClosing(mask_img, [5,5,3])
    mask_img = sitk.BinaryFillhole(mask_img)
    return sitk.Cast(mask_img, sitk.sitkUInt8)

def process_dicom_series(dicom_dir, out_image_path, out_mask_path, target_spacing=(1.0,1.0,1.0)):
    itk = dicom_series_to_itk(dicom_dir)
    resampled = resample_itk(itk, new_spacing=tuple(target_spacing), is_label=False)
    arr, spacing, origin = sitk_to_numpy(resampled)
    arr_windowed = window_clip(arr, -1000, 400)
    lung_mask_img = simple_lung_mask(resampled)
    lung_mask = sitk.GetArrayFromImage(lung_mask_img).astype(np.uint8)
    coords = np.where(lung_mask)
    if len(coords[0]) == 0:
        img = numpy_to_sitk(arr_windowed, spacing)
        sitk.WriteImage(img, out_image_path)
        mask_img = numpy_to_sitk(lung_mask, spacing)
        sitk.WriteImage(mask_img, out_mask_path)
        return out_image_path, out_mask_path
    zmin, zmax = coords[0].min(), coords[0].max()
    ymin, ymax = coords[1].min(), coords[1].max()
    xmin, xmax = coords[2].min(), coords[2].max()
    pad = 16
    z0 = max(0, zmin - pad); z1 = min(arr.shape[0]-1, zmax + pad)
    y0 = max(0, ymin - pad); y1 = min(arr.shape[1]-1, ymax + pad)
    x0 = max(0, xmin - pad); x1 = min(arr.shape[2]-1, xmax + pad)
    arr_crop = arr_windowed[z0:z1+1, y0:y1+1, x0:x1+1]
    mask_crop = lung_mask[z0:z1+1, y0:y1+1, x0:x1+1]
    sitk.WriteImage(numpy_to_sitk(arr_crop, spacing), out_image_path)
    sitk.WriteImage(numpy_to_sitk(mask_crop, spacing), out_mask_path)
    return out_image_path, out_mask_path

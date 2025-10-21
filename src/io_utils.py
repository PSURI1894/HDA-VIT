import SimpleITK as sitk
import numpy as np

def dicom_series_to_itk(dicom_dir):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {dicom_dir}")
    files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(files)
    img = reader.Execute()
    return img

def sitk_to_numpy(itk_image):
    arr = sitk.GetArrayFromImage(itk_image).astype(np.float32)
    spacing = itk_image.GetSpacing()[::-1]
    origin = itk_image.GetOrigin()[::-1]
    return arr, spacing, origin

def numpy_to_sitk(arr, spacing):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(spacing[::-1]))
    return img

def resample_itk(itk_img, new_spacing=(1.0,1.0,1.0), is_label=False):
    original_spacing = itk_img.GetSpacing()
    original_size = itk_img.GetSize()
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(new_spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(itk_img.GetDirection())
    resampler.SetOutputOrigin(itk_img.GetOrigin())
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
    resampled = resampler.Execute(itk_img)
    return resampled

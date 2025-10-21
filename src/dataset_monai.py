from monai.transforms import (
    LoadImaged, AddChanneld, Spacingd, Orientationd, ScaleIntensityRanged,
    CropForegroundd, RandCropByPosNegLabeld, ToTensord, RandFlipd, RandRotate90d,
    RandShiftIntensityd, RandAffined
)
from monai.data import CacheDataset, DataLoader
import multiprocessing

def get_seg_transforms(keys=("image","label"), target_spacing=(1.0,1.0,1.0), patch_size=(96,96,96)):
    train_transforms = [
        LoadImaged(keys=keys),
        AddChanneld(keys=keys),
        Spacingd(keys=keys, pixdim=target_spacing, mode=("bilinear","nearest")),
        Orientationd(keys=keys, axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=keys, source_key="image"),
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image","label"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandAffined(keys=["image","label"], prob=0.3, rotate_range=(0.15,0.15,0.15), scale_range=(0.1,0.1,0.1)),
        RandCropByPosNegLabeld(keys=keys, label_key="label", spatial_size=patch_size, pos=1, neg=1, num_samples=8, image_key="image", image_threshold=0),
        ToTensord(keys=keys)
    ]
    return train_transforms

def get_dataloader(data_list, transforms, batch_size=2, num_workers=None, cache_rate=0.2):
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count()//2)
    ds = CacheDataset(data=data_list, transform=transforms, cache_rate=cache_rate, num_workers=num_workers)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader

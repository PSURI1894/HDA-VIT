import os, yaml, json, torch, time
from torch.utils.data import DataLoader, Dataset
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from src.models.resnet3d import ResNet3D
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import nibabel as nib
from tqdm import tqdm

class PatchDataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels = labels
    def __len__(self):
        return len(self.patches)
    def __getitem__(self, idx):
        return torch.from_numpy(self.patches[idx]).float(), torch.tensor(self.labels[idx]).float()

def extract_positive_and_neg_patches(processed_dir, pos_radius=12, neg_samples_per_scan=50, patch_size=32):
    patches = []
    labels = []
    for img_path in sorted([p for p in os.listdir(processed_dir) if p.endswith("_img.nii.gz") or p.endswith("_img.nii")]):

        img = nib.load(os.path.join(processed_dir, img_path)).get_fdata().astype(np.float32)
        label = nib.load(os.path.join(processed_dir, img_path.replace("_img","_label"))).get_fdata().astype(np.uint8)
        from scipy import ndimage
        cc, n = ndimage.label(label)
        props = ndimage.find_objects(cc)
        for i in range(n):
            sl = props[i]
            cz = (sl[0].start + sl[0].stop)//2
            cy = (sl[1].start + sl[1].stop)//2
            cx = (sl[2].start + sl[2].stop)//2
            z0 = max(0, cz - patch_size//2); z1 = z0 + patch_size
            y0 = max(0, cy - patch_size//2); y1 = y0 + patch_size
            x0 = max(0, cx - patch_size//2); x1 = x0 + patch_size
            patch = img[z0:z1, y0:y1, x0:x1]
            if patch.shape == (patch_size,patch_size,patch_size):
                patches.append(np.expand_dims(patch,0))
                labels.append(1.0)
        tries = 0
        neg_count = 0
        while neg_count < neg_samples_per_scan and tries < neg_samples_per_scan*10:
            tries += 1
            cz = np.random.randint(patch_size//2, img.shape[0]-patch_size//2)
            cy = np.random.randint(patch_size//2, img.shape[1]-patch_size//2)
            cx = np.random.randint(patch_size//2, img.shape[2]-patch_size//2)
            slz = slice(cz-patch_size//2, cz+patch_size//2)
            sly = slice(cy-patch_size//2, cy+patch_size//2)
            slx = slice(cx-patch_size//2, cx+patch_size//2)
            if label[slz, sly, slx].sum() == 0:
                patch = img[slz, sly, slx]
                patches.append(np.expand_dims(patch,0))
                labels.append(0.0)
                neg_count += 1
    patches = np.stack(patches)
    labels = np.array(labels)
    return patches, labels

def train_cls(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    patches, labels = extract_positive_and_neg_patches(cfg['data']['processed_dir'], neg_samples_per_scan=cfg['training']['neg_per_scan'], patch_size=cfg['training']['patch_size'])
    idx = np.arange(len(labels)); np.random.shuffle(idx)
    patches = patches[idx]; labels = labels[idx]
    split = int(len(labels)*0.8)
    train_ds = PatchDataset(patches[:split], labels[:split])
    val_ds = PatchDataset(patches[split:], labels[split:])
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet3D(num_classes=1, in_ch=1).to(device)
    opt = Adam(model.parameters(), lr=cfg['training']['lr'])
    loss_fn = BCEWithLogitsLoss()
    scaler = GradScaler()
    for epoch in range(cfg['training']['max_epochs']):
        model.train()
        total_loss = 0.0
        for x,y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['max_epochs']}"):
            x = x.to(device); y = y.to(device)
            with autocast():
                out = model(x)
                out = out.squeeze(1)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            total_loss += float(loss)
        print(f"Epoch {epoch+1} train loss {total_loss/len(train_loader):.4f}")
    os.makedirs(cfg['training']['model_dir'], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg['training']['model_dir'], "cls_final.pth"))
    return os.path.join(cfg['training']['model_dir'], "cls_final.pth")

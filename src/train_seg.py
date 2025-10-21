import os, yaml, torch, time
from monai.losses import DiceLoss
from monai.utils import set_determinism
from src.dataset_monai import get_seg_transforms, get_dataloader
from src.models.unet3d_monai import get_monai_unet
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import json

def train_seg(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    set_determinism(seed=cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(cfg['data']['train_json']) as f:
        train_items = json.load(f)
    transforms = get_seg_transforms(keys=("image","label"), target_spacing=tuple(cfg['data']['spacing']), patch_size=tuple(cfg['training']['patch_size']))
    loader = get_dataloader(train_items, transforms, batch_size=cfg['training']['batch_size'], cache_rate=cfg['training'].get('cache_rate',0.2))
    model = get_monai_unet(in_channels=1, out_channels=1).to(device)
    loss_fn = DiceLoss(sigmoid=True)
    optimizer = Adam(model.parameters(), lr=cfg['training']['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['training']['max_epochs'])
    scaler = GradScaler()
    os.makedirs(cfg['training']['model_dir'], exist_ok=True)

    for epoch in range(cfg['training']['max_epochs']):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch in loader:
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            with autocast():
                outputs = model(imgs)
                loss = loss_fn(outputs, labels) + torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            epoch_loss += float(loss)
        scheduler.step()
        cp_path = os.path.join(cfg['training']['model_dir'], f"seg_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), cp_path)
        print(f"Epoch {epoch+1}/{cfg['training']['max_epochs']} loss:{epoch_loss/len(loader):.4f} time:{time.time()-t0:.1f}s")
    return os.path.join(cfg['training']['model_dir'], f"seg_epoch{cfg['training']['max_epochs']}.pth")

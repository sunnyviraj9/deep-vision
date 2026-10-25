"""
train.py — DeepVision Crowd Monitor: Local Training Script
===========================================================
Trains TinyMCNN on ShanghaiTech Part A + Part B combined for better accuracy.
Includes data augmentation: horizontal flip, brightness/contrast jitter, random crop.

Usage:
  python train.py                          # default: part_A + part_B, 80 epochs
  python train.py --epochs 50 --no-partb  # part_A only
"""

import os, glob, argparse, cv2, random, time
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DATA_BASE    = os.path.join(os.path.dirname(__file__), "data", "ShanghaiTech")
IMG_SZ       = (256, 256)
BATCH        = 8
EPOCHS       = 80          # more epochs for better convergence
LR           = 5e-5        # slightly higher lr with scheduler
SIGMA        = 4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_MODEL    = os.path.join(os.path.dirname(__file__), "tiny_mcnn.pth")


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ──────────────────────────────────────────────────────────────────────────────
def augment(img, den):
    """Apply random horizontal flip + brightness/contrast jitter to img+density."""
    # Horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
        den = cv2.flip(den, 1)
    # Brightness / contrast jitter
    alpha = random.uniform(0.8, 1.2)   # contrast
    beta  = random.randint(-20, 20)    # brightness
    img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return img, den


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class CrowdDataset(Dataset):
    def __init__(self, root, sub="train", augment_data=False):
        self.tf           = transforms.ToTensor()
        self.size         = IMG_SZ
        self.augment_data = augment_data
        split   = "train_data" if sub == "train" else "test_data"
        img_dir = os.path.join(root, split, "images")
        self.imgs = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg")) +
            glob.glob(os.path.join(img_dir, "*.png"))
        )
        if not self.imgs:
            raise FileNotFoundError(f"No images in: {img_dir}")
        self.gt_dir = os.path.join(root, split, "ground-truth")
        print(f"  [{os.path.basename(root)}/{sub}] {len(self.imgs)} images")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p   = self.imgs[i]
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        oh, ow = img.shape[:2]
        img = cv2.resize(img, self.size)

        # Ground truth
        base = os.path.splitext(os.path.basename(p))[0]
        cand = os.path.join(self.gt_dir, f"GT_{base}.mat")
        if not os.path.exists(cand):
            cand = os.path.join(self.gt_dir, base + ".mat")

        pts = np.zeros((0, 2), dtype=np.float32)
        if os.path.exists(cand):
            try:
                m   = sio.loadmat(cand)
                pts = np.array(m["image_info"][0, 0][0, 0][0], dtype=np.float32)
            except Exception as e:
                print(f"[WARN] {cand}: {e}")

        if pts.size:
            pts[:, 0] *= self.size[1] / ow
            pts[:, 1] *= self.size[0] / oh

        den = np.zeros(self.size, dtype=np.float32)
        for pt in pts:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < self.size[1] and 0 <= y < self.size[0]:
                den[y, x] += 1
        if den.sum() > 0:
            k   = SIGMA * 4 + 1
            den = cv2.GaussianBlur(den, (k, k), SIGMA)

        # Augmentation (train only)
        if self.augment_data:
            img, den = augment(img, den)

        img_t = self.tf(img.astype("float32") / 255.0)
        den_t = torch.from_numpy(den[None]).float()
        return img_t, den_t


# ──────────────────────────────────────────────────────────────────────────────
# Model: Improved Tiny-MCNN with BatchNorm + Dropout
# ──────────────────────────────────────────────────────────────────────────────
class TinyMCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,  16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32,  1, 1),
        )

    def forward(self, x):
        o = self.net(x)
        if o.shape[2:] != x.shape[2:]:
            o = F.interpolate(o, size=x.shape[2:], mode="bilinear", align_corners=False)
        return o


# ──────────────────────────────────────────────────────────────────────────────
# Training + Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def train_eval(part_a_root, part_b_root=None, epochs=EPOCHS):
    torch.manual_seed(42)
    use_amp = AMP_AVAILABLE and DEVICE.type == "cuda"
    print(f"\nDevice: {DEVICE}  |  AMP: {use_amp}  |  Epochs: {epochs}")

    # ── Build datasets ──────────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds_a = CrowdDataset(part_a_root, "train", augment_data=True)
    test_ds_a  = CrowdDataset(part_a_root, "test",  augment_data=False)

    if part_b_root and os.path.isdir(part_b_root):
        train_ds_b = CrowdDataset(part_b_root, "train", augment_data=True)
        test_ds_b  = CrowdDataset(part_b_root, "test",  augment_data=False)
        train_ds   = ConcatDataset([train_ds_a, train_ds_b])
        test_ds    = ConcatDataset([test_ds_a,  test_ds_b])
        print(f"  Combined train: {len(train_ds)}  |  Combined test: {len(test_ds)}")
    else:
        train_ds = train_ds_a
        test_ds  = test_ds_a
        print(f"  Part A only — train: {len(train_ds)}  |  test: {len(test_ds)}")

    tr = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                    num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    te = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                    num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    # ── Model + optimizer + scheduler ───────────────────────────────────────
    model   = TinyMCNN().to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    # Reduce LR by 0.5 every 20 epochs
    sched   = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    lossf   = nn.MSELoss()
    scaler  = GradScaler() if use_amp else None

    train_losses, val_losses = [], []
    best_mae = float("inf")

    for ep in range(epochs):
        model.train()
        tl = 0.0
        pbar = tqdm(tr, desc=f"Epoch {ep+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            if use_amp:
                with autocast():
                    l = lossf(model(x), y)
                scaler.scale(l).backward()
                scaler.step(opt); scaler.update()
            else:
                l = lossf(model(x), y)
                l.backward(); opt.step()
            tl += l.item()
            pbar.set_postfix(loss=f"{l.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.1e}")
        sched.step()
        train_losses.append(tl / len(tr))

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        vl, preds, gts = 0.0, [], []
        with torch.no_grad():
            for x, y in te:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                p     = model(x)
                vl   += lossf(p, y).item()
                preds += p.cpu().numpy().sum(axis=(1, 2, 3)).tolist()
                gts   += y.cpu().numpy().sum(axis=(1, 2, 3)).tolist()
        val_losses.append(vl / len(te))
        preds_arr, gts_arr = np.array(preds), np.array(gts)
        mae  = float(np.mean(np.abs(preds_arr - gts_arr)))
        rmse = float(np.sqrt(np.mean((preds_arr - gts_arr) ** 2)))
        print(f"  Ep{ep+1:02d}  TrLoss={train_losses[-1]:.4f}  "
              f"ValLoss={val_losses[-1]:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}")

        # Save best model
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), OUT_MODEL)
            print(f"    ✓ Best model saved (MAE={mae:.2f})")

    # ── Loss curve ──────────────────────────────────────────────────────────
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses,   label="val")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.legend()
    curve_path = os.path.join(os.path.dirname(__file__), "loss_curve.png")
    plt.savefig(curve_path); plt.close()
    print(f"\nLoss curve → {curve_path}")

    # ── Sample predictions ───────────────────────────────────────────────────
    vis_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    # reload best weights for visualization
    model.load_state_dict(torch.load(OUT_MODEL, map_location=DEVICE, weights_only=True))
    it, shown = iter(te), 0
    with torch.no_grad():
        while shown < 4:
            try: x, y = next(it)
            except StopIteration: break
            p = model(x.to(DEVICE)).cpu()
            for i in range(len(x)):
                img_np = (x[i].permute(1, 2, 0).numpy() * 255).astype("uint8")
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                axes[0].imshow(img_np);          axes[0].set_title("Image");                    axes[0].axis("off")
                axes[1].imshow(y[i, 0], cmap="jet"); axes[1].set_title(f"GT {y[i].sum():.0f}");   axes[1].axis("off")
                axes[2].imshow(p[i, 0], cmap="jet"); axes[2].set_title(f"Pred {p[i].sum():.0f}"); axes[2].axis("off")
                save = os.path.join(vis_dir, f"pred_{shown+1}.png")
                plt.savefig(save, bbox_inches="tight"); plt.close(fig)
                shown += 1
                if shown >= 4: break

    print(f"\nBest MAE: {best_mae:.2f}")
    print(f"Model saved → {OUT_MODEL}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int, default=EPOCHS)
    parser.add_argument("--no-partb", action="store_true", help="Use Part A only")
    args = parser.parse_args()

    part_a = os.path.join(DATA_BASE, "part_A")
    part_b = None if args.no_partb else os.path.join(DATA_BASE, "part_B")

    train_eval(part_a, part_b, epochs=args.epochs)

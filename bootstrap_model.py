"""
bootstrap_model.py
------------------
Trains TinyMCNN for a few epochs on synthetically generated crowd images
so the Streamlit dashboard gives sensible heat-map responses without
needing the real ShanghaiTech dataset.

Run once:  python bootstrap_model.py
Output:    tiny_mcnn.pth  (next to this script)
"""

import os, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SZ   = (256, 256)
N_SYNTH  = 400          # synthetic training samples
EPOCHS   = 15
BATCH    = 16
LR       = 1e-4
OUT      = os.path.join(os.path.dirname(__file__), "tiny_mcnn.pth")


# ── Synthetic dataset ─────────────────────────────────────────────────────────
class SyntheticCrowd(Dataset):
    """
    Each sample: a grey image with random bright blobs (simulated people) +
    a Gaussian-smoothed density map aligned to those blobs.
    """
    def __init__(self, n=N_SYNTH, size=IMG_SZ, seed=42):
        self.n, self.size = n, size
        rng = np.random.default_rng(seed)
        self.samples = []
        h, w = size
        for _ in range(n):
            # random number of people 5-120
            count = int(rng.integers(5, 121))
            img   = rng.integers(30, 80, (h, w, 3), dtype=np.uint8)
            den   = np.zeros((h, w), dtype=np.float32)
            for _ in range(count):
                cx = int(rng.integers(0, w))
                cy = int(rng.integers(0, h))
                r  = int(rng.integers(4, 12))
                # draw a soft blob on the image
                Y, X = np.ogrid[:h, :w]
                mask = (X - cx)**2 + (Y - cy)**2 <= r**2
                img[mask] = np.clip(img[mask].astype(int) + rng.integers(60, 130), 0, 255).astype(np.uint8)
                # place point on density map
                den[cy, cx] += 1.0
            # gaussian blur density
            import cv2
            den = cv2.GaussianBlur(den, (17, 17), 4) if den.sum() > 0 else den
            self.samples.append((img, den))

    def __len__(self):  return self.n

    def __getitem__(self, i):
        img, den = self.samples[i]
        tf  = transforms.ToTensor()
        return tf(img.astype("float32") / 255.0), torch.from_numpy(den[None]).float()


# ── Model ─────────────────────────────────────────────────────────────────────
class TinyMCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,  16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64,  1, 1),
        )

    def forward(self, x):
        o = self.net(x)
        if o.shape[2:] != x.shape[2:]:
            o = F.interpolate(o, size=x.shape[2:], mode="bilinear", align_corners=False)
        return o


# ── Train ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Generating {N_SYNTH} synthetic crowd images …")
    ds     = SyntheticCrowd(N_SYNTH)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

    model = TinyMCNN().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for ep in range(EPOCHS):
        total = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            l = loss_fn(model(x), y)
            l.backward(); opt.step()
            total += l.item()
        print(f"  Epoch {ep+1:02d}/{EPOCHS}  loss={total/len(loader):.5f}")

    torch.save(model.state_dict(), OUT)
    print(f"\nBootstrap model saved → {OUT}")
    print("The Streamlit app will now show density heat-maps.")
    print("\nTo get accurate counts, train on the real dataset:")
    print("  1. Download ShanghaiTech from https://www.kaggle.com/datasets/tthien/shanghaitech")
    print("  2. Extract into deep-vision/data/ShanghaiTech/")
    print("  3. Run:  python train.py")


if __name__ == "__main__":
    main()

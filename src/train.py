"""训练 RoadMark-binary 车道标线 U-Net。"""

import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from preprocess import CONFIG, ROOT, WEIGHTS


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBlock(3, 16)
        self.e2 = ConvBlock(16, 32)
        self.e3 = ConvBlock(32, 64)
        self.e4 = ConvBlock(64, 128)
        self.mid = ConvBlock(128, 256)
        self.d4 = ConvBlock(384, 128)
        self.d3 = ConvBlock(192, 64)
        self.d2 = ConvBlock(96, 32)
        self.d1 = ConvBlock(48, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        d = self.e4(self.pool(c))
        z = self.mid(self.pool(d))
        z = self.d4(
            torch.cat(
                [
                    F.interpolate(
                        z, d.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    d,
                ],
                1,
            )
        )
        z = self.d3(
            torch.cat(
                [
                    F.interpolate(
                        z, c.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    c,
                ],
                1,
            )
        )
        z = self.d2(
            torch.cat(
                [
                    F.interpolate(
                        z, b.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    b,
                ],
                1,
            )
        )
        z = self.d1(
            torch.cat(
                [
                    F.interpolate(
                        z, a.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    a,
                ],
                1,
            )
        )
        return self.head(z).squeeze(1)


class RoadMarkDataset(Dataset):
    def __init__(self, root):
        self.images = sorted((root / "train" / "ortho").glob("*.png"))
        self.masks = root / "train" / "mask"
        self.mean = np.array(CONFIG["training"]["image_mean"], np.float32)[
            None, None, :
        ]
        self.std = np.array(CONFIG["training"]["image_std"], np.float32)[None, None, :]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks / path.name), cv2.IMREAD_GRAYSCALE)
        if random.random() < 0.5:
            image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
        if random.random() < 0.5:
            image, mask = cv2.flip(image, 0), cv2.flip(mask, 0)
        turns = random.randrange(4)
        if turns:
            image = np.rot90(image, turns).copy()
            mask = np.rot90(mask, turns).copy()

        image = (image.astype(np.float32) / 255 - self.mean) / self.std
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        mask = torch.from_numpy((mask > 127).astype(np.float32))
        return image, mask


def train_lane(data_dir, epochs=None):
    device = torch.device(f"cuda:{CONFIG['device']}")
    dataset = RoadMarkDataset(ROOT / data_dir)
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["training"]["lane_batch"],
        shuffle=True,
        num_workers=CONFIG["training"]["lane_workers"],
        pin_memory=True,
        persistent_workers=True,
    )
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["training"]["lane_learning_rate"],
        weight_decay=CONFIG["training"]["lane_weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda")
    positive_weight = torch.tensor(
        CONFIG["training"]["lane_positive_weight"], device=device
    )

    count = epochs if epochs is not None else CONFIG["training"]["lane_epochs"]
    for epoch in range(count):
        model.train()
        losses = []
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images).float()
                bce = F.binary_cross_entropy_with_logits(
                    logits, masks, pos_weight=positive_weight
                )
                prob = logits.sigmoid()
                dice = (
                    1
                    - (
                        (2 * (prob * masks).sum((1, 2)) + 1)
                        / (prob.sum((1, 2)) + masks.sum((1, 2)) + 1)
                    ).mean()
                )
                loss = bce + dice
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

        print(f"第 {epoch + 1} 轮：loss {np.mean(losses):.4f}")

    destination = WEIGHTS["lane_training"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)
    return destination

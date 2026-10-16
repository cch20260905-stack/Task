# -*- coding: utf-8 -*-
"""ARIS 数据集读取：直接 mmap 读取 .npy，支持论文所述的随机裁剪 + 随机水平翻转。"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], np.float32)


class ARISDataset(Dataset):
    """x: (N,H,W,3) float64 0-1 伪彩图；y: (N,H,W) float64 高斯密度图（积分=鱼数）。

    train=True 时做随机裁剪 + 随机水平翻转（论文原话：only random cropping and
    random horizontal flipping）；val/test 用整图以保证评测尺寸与论文一致。
    """

    def __init__(self, root, split, crop=None, train=False, normalize=True, limit=0,
                 need_seg=False, aug=False):
        self.x = np.load(os.path.join(root, "Labelled_data", "x_%s.npy" % split),
                         mmap_mode="r")
        self.y = np.load(os.path.join(root, "Labelled_data", "y_%s.npy" % split),
                         mmap_mode="r")
        self.crop = crop
        self.train = train
        self.normalize = normalize
        self.limit = len(self.x) if limit <= 0 else min(limit, len(self.x))
        self.need_seg = need_seg
        self.aug = aug          # 额外数据增强(治 val/test 泛化 gap)：亮度/对比度抖动 + 随机擦除
        if need_seg:
            d = os.path.join(root, "seg_labels", split)
            if not os.path.isdir(d):
                raise FileNotFoundError("未找到分割标签目录 %s（先运行 ARIS/generate_seg_labels.py）" % d)
            self.seg_paths = [os.path.join(d, "%04d.npy" % i) for i in range(len(self.x))]

    def __len__(self):
        return self.limit

    def __getitem__(self, i):
        x = np.asarray(self.x[i], dtype=np.float32)          # H,W,3
        y = np.asarray(self.y[i], dtype=np.float32)          # H,W
        s = np.load(self.seg_paths[i]) if self.need_seg else None   # H,W uint8 0/1/2

        if self.train and self.crop:
            ch, cw = self.crop
            H, W = y.shape
            ch, cw = min(ch, H), min(cw, W)
            top = np.random.randint(0, H - ch + 1)
            left = np.random.randint(0, W - cw + 1)
            x = x[top:top + ch, left:left + cw]
            y = y[top:top + ch, left:left + cw]
            if s is not None:
                s = s[top:top + ch, left:left + cw]
            if np.random.rand() < 0.5:
                x = x[:, ::-1]
                y = y[:, ::-1]
                if s is not None:
                    s = s[:, ::-1]

        if self.train and self.aug:
            # 1) 亮度/对比度抖动：x 是 0-1 伪彩图，只动图像不动标签
            x = x * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.1, 0.1)
            m, M = x.mean(), x.mean() + x.std()
            x = (x - m) / max(M - m, 1e-6) * np.random.uniform(0.8, 1.2) + 0.5
            x = np.clip(x, 0, 1)
            # 2) 随机擦除：遮一小块图像(置 0)，密度/分割标签不变——
            #    被遮区域的鱼数贡献仍在 y 中，迫使模型依赖全局上下文而非局部死记
            if np.random.rand() < 0.5:
                H, W = y.shape
                eh, ew = np.random.randint(H // 8, H // 4 + 1), np.random.randint(W // 8, W // 4 + 1)
                t0, l0 = np.random.randint(0, H - eh + 1), np.random.randint(0, W - ew + 1)
                x[t0:t0 + eh, l0:l0 + ew] = 0.0

        x = np.ascontiguousarray(x.transpose(2, 0, 1))       # 3,H,W
        y = np.ascontiguousarray(y)
        if self.normalize:
            x = (x - IMG_MEAN[:, None, None]) / IMG_STD[:, None, None]

        if s is None:
            return torch.from_numpy(x), torch.from_numpy(y)[None]   # (3,H,W), (1,H,W)
        return (torch.from_numpy(x), torch.from_numpy(y)[None],
                torch.from_numpy(np.ascontiguousarray(s)).long())


def build_loaders(root, crop=(256, 256), batch=4, workers=0, limit=0, need_seg=False,
                  aug=False):
    from torch.utils.data import DataLoader
    tr = ARISDataset(root, "train", crop=crop, train=True, limit=limit,
                     need_seg=need_seg, aug=aug)
    va = ARISDataset(root, "val", crop=None, train=False, limit=limit, need_seg=need_seg)
    te = ARISDataset(root, "test", crop=None, train=False, limit=limit, need_seg=need_seg)
    return (
        DataLoader(tr, batch_size=batch, shuffle=True, num_workers=workers,
                   drop_last=True, pin_memory=True),
        DataLoader(va, batch_size=1, shuffle=False, num_workers=workers,
                   pin_memory=True),
        DataLoader(te, batch_size=1, shuffle=False, num_workers=workers,
                   pin_memory=True),
    )

# -*- coding: utf-8 -*-
"""通用工具：随机种子、日志、点图反推。"""
import os
import csv
import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def density_to_dots(y, thr=0.001, min_dist=2):
    """从高斯密度图反推点图 z（论文的 L_TV 需要 z，但数据集只给密度图）。

    做法：3x3 局部极大值 + 阈值 thr，并把峰值归一化到 rho=1/(2πσ²) 的强度，
    使 ∏z∏_1 尽量等于真实鱼数。密集重叠处会有漏检，属于已知近似。
    """
    import cv2
    y = np.asarray(y, dtype=np.float32)
    mx = cv2.dilate(y, np.ones((2 * min_dist + 1, 2 * min_dist + 1), np.uint8))
    peaks = (y >= mx) & (y > thr)
    z = np.zeros_like(y)
    z[peaks] = y[peaks]
    return z


class CSVLogger:
    def __init__(self, path, fields):
        self.path = path
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        if new:
            self.w.writeheader()

    def log(self, **kw):
        self.w.writerow(kw)
        self.f.flush()

    def close(self):
        self.f.close()

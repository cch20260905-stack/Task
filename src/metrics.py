# -*- coding: utf-8 -*-
"""评测指标：MAE / RMSE（计数口径，密度图积分 = 鱼数）。"""
import numpy as np
import torch


@torch.no_grad()
def count_from_density(d):
    return d.sum(dim=(1, 2, 3)).detach().cpu().numpy()


def mae_rmse(pred_counts, gt_counts):
    pred_counts = np.asarray(pred_counts, dtype=np.float64)
    gt_counts = np.asarray(gt_counts, dtype=np.float64)
    err = pred_counts - gt_counts
    return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean()))


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, v, k=1):
        self.sum += float(v) * k
        self.n += k

    @property
    def avg(self):
        return self.sum / self.n if self.n else 0.0

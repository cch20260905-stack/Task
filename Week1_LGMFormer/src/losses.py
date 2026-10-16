# -*- coding: utf-8 -*-
"""损失函数。论文式(12)(13)：L_count = L_C + 0.01·L_OT + 0.1·L_TV。

阶段说明：消融表第一行（Baseline）用 L2 密度回归损失即可；进入主方法阶段再切换到
DM-Count 改进版损失（论文 Table VIII 专门做了这组对比）。
"""
import torch
import torch.nn.functional as F


def mse_loss(pred, target, **kw):
    """L2 像素级密度图回归（消融表 baseline / Table VIII 的 L2 行）。"""
    return F.mse_loss(pred, target)


def count_loss(pred, target):
    """L_C：预测总数与真值总数的 L1 误差。"""
    return (pred.sum(dim=(1, 2, 3)) - target.sum(dim=(1, 2, 3))).abs().mean()


def tv_loss(pred, dots, sigma=4.0):
    """L_TV（论文式12）：用 σ=4 高斯平滑后的点图 G_σ(z) 与预测密度图比较。

    z 为未归一化点图（∏z∏_1 = 真实鱼数）。本项目中数据集只提供高斯密度图，
    点图由 utils.density_to_dots 从密度图反推，见该函数说明。
    """
    def blur(t):
        k = int(2 * round(3 * sigma) + 1)
        g = torch.arange(k, dtype=t.dtype, device=t.device) - (k - 1) / 2
        g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        t = F.conv2d(F.pad(t, (k // 2, k // 2, 0, 0), mode="replicate"),
                     g.view(1, 1, 1, k))
        t = F.conv2d(F.pad(t, (0, 0, k // 2, k // 2), mode="replicate"),
                     g.view(1, 1, k, 1))
        return t

    z1 = dots.sum(dim=(1, 2, 3)).clamp(min=1e-6)
    p1 = pred.sum(dim=(1, 2, 3)).clamp(min=1e-6)
    a = blur(dots) / z1.view(-1, 1, 1, 1)
    b = pred / p1.view(-1, 1, 1, 1)
    return ((a - b) ** 2).sum(dim=(1, 2, 3)).mean() * z1.mean()


def ot_loss(pred, target, grid=(32, 32), eps=0.01, iters=50):
    """L_OT（DM-Count 的最优传输项）。

    说明：论文未公布 OT 的实现细节（原版用 batch 内像素级 Sinkhorn，代价极高）。
    这里在 grid×grid 粗网格上做 Sinkhorn，属于**可复现的近似实现**，开启后请注意
    与论文数值的差异来源。默认不在 baseline 阶段启用。
    """
    B = pred.shape[0]
    p = F.adaptive_avg_pool2d(pred, grid).flatten(1)
    t = F.adaptive_avg_pool2d(target, grid).flatten(1)

    gh, gw = grid
    ys = torch.linspace(0, 1, gh, device=pred.device)
    xs = torch.linspace(0, 1, gw, device=pred.device)
    yy, xx = torch.meshgrid(ys, xs)
    coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)      # N,2
    C = torch.cdist(coords, coords) ** 2                               # N,N

    p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-6)
    t = t / t.sum(dim=1, keepdim=True).clamp(min=1e-6)
    K = torch.exp(-C / eps)

    u = torch.ones_like(p) / p.shape[1]
    v = torch.ones_like(t) / t.shape[1]
    for _ in range(iters):
        u = p / (K @ v).clamp(min=1e-9)
        v = t / (K.t() @ u).clamp(min=1e-9)
    plan = u[:, :, None] * K[None] * v[:, None, :]
    return (plan * C[None]).sum(dim=(1, 2)).mean() * float(gh * gw)


def counting_loss(pred, target, dots=None, lam1=0.01, lam2=0.1,
                  use_ot=False, use_tv=True, sigma=4.0):
    """论文式(13)的计数损失；dots 缺失时自动跳过 L_TV。"""
    loss = count_loss(pred, target)
    parts = {"L_C": float(loss.detach())}
    if use_ot:
        lo = ot_loss(pred, target)
        loss = loss + lam1 * lo
        parts["L_OT"] = float(lo.detach())
    if use_tv and dots is not None:
        lt = tv_loss(pred, dots, sigma=sigma)
        loss = loss + lam2 * lt
        parts["L_TV"] = float(lt.detach())
    return loss, parts


def seg_loss(logits, target, n_cls=3, eps=1e-6):
    """论文式(14)-(16)：L_seg = L_ce + L_dice。

    logits: (B,3,H,W) 未经 softmax；target: (B,H,W) long，取值 0/1/2。
    Dice 按式(15)：对每张图每类算 2ŷy/(ŷ+y) 再对 N、K 求平均。
    """
    ce = F.cross_entropy(logits, target)
    prob = logits.softmax(dim=1)
    onehot = F.one_hot(target, n_cls).permute(0, 3, 1, 2).float()   # B,K,H,W
    inter = (prob * onehot).sum(dim=(2, 3))
    denom = prob.sum(dim=(2, 3)) + onehot.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * inter / denom.clamp(min=eps)).mean()
    return ce + dice, {"ce": float(ce.detach()), "dice": float(dice.detach())}


def build_loss(name, seg_weight=1.0, **kw):
    """name: mse(基线默认) | dmcount(论文主方法，需要点图 z) | mse+seg | dmcount+seg

    seg_weight: 分割损失 L_seg 的额外权重。DM-Count 的 L_C（L1 总数误差）损失值量级
    被 --density-scale(默认100) 放大到 O(数千)，会把 L_seg(O(2)) 的梯度淹没，导致分割
    分支学崩、mIoU 骤降。论文 Eq.17 写的是两项等权，但那是基于未缩放密度图；本项目为
    MSE 稳定性对密度图 ×100，平移到 DM-Count 后必须显式平衡，故加 seg_weight（默认1）。
    """
    base, has_seg = (name.split("+", 1) + [None])[:2] if "+" in name else (name, None)

    if base == "mse":
        def f(pred, y, dots=None, seg=None, seg_logits=None):
            v = mse_loss(pred, y)
            parts = {"mse": float(v.detach())}
            total = v
            if has_seg and seg is not None:
                ls, sp = seg_loss(seg_logits, seg)
                ls = ls * seg_weight
                total = total + ls
                parts.update({k: v * seg_weight for k, v in sp.items()})
            return total, parts
        return f
    if base == "dmcount":
        def f(pred, y, dots=None, seg=None, seg_logits=None):
            total, parts = counting_loss(pred, y, dots=dots, **kw)
            if has_seg and seg is not None:
                ls, sp = seg_loss(seg_logits, seg)
                ls = ls * seg_weight
                total = total + ls
                parts.update({k: v * seg_weight for k, v in sp.items()})
            return total, parts
        return f
    raise ValueError("未知损失: %s" % name)

# -*- coding: utf-8 -*-
"""载入训练好的权重，在 val/test 上评测并可视化密度图预测。

用法：
    python predict.py --ckpt runs/baseline/best.pth --split test --out pred_check.png
"""
import os
import sys
import argparse

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import ARISDataset                    # noqa: E402
from src.models import UNetResNet18                 # noqa: E402
from src.metrics import mae_rmse                    # noqa: E402


def heat(d, vmax, gamma=0.5):
    """密度图 -> JET 伪彩（BGR）。稀疏密度峰值很小，用 gamma<1 提亮便于肉眼对比。"""
    n = np.clip(d / max(vmax, 1e-9), 0, 1) ** gamma
    return cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_JET)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data-root", default="D:/project/source/ARIS")
    a.add_argument("--ckpt", default="runs/baseline/best.pth")
    a.add_argument("--split", default="test", choices=["train", "val", "test"])
    a.add_argument("--density-scale", type=float, default=100.0)
    a.add_argument("--out", default="pred_check.png")
    a.add_argument("--n", type=int, default=6)
    args = a.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResNet18(pretrained=False).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    print("载入 %s (epoch %s, val MAE %.3f)" % (args.ckpt, ck.get("epoch"), ck.get("val_mae", -1)))

    ds = ARISDataset(args.data_root, args.split, crop=None, train=False)
    preds, gts, raw = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out = model(x[None].to(device))
                if isinstance(out, tuple):
                    out = out[0]          # 开启分割分支后 forward 返回 (密度, seg_logits)
            out = out.float() / args.density_scale
            preds.append(out.sum().item())
            gts.append(y.sum().item())
            if len(raw) < 400:
                raw.append((x.numpy(), y.numpy()[0], out[0, 0].cpu().numpy()))

    mae, rmse = mae_rmse(preds, gts)
    print("%s: MAE %.3f  RMSE %.3f   (n=%d)" % (args.split, mae, rmse, len(ds)))

    # 按真值计数分位数抽样做可视化
    order = np.argsort(gts)
    idx = [int(order[int(p * (len(order) - 1))]) for p in np.linspace(0, 1, args.n)]
    cell = 220
    pad = 8
    canvas = np.full((len(idx) * (cell + 26) + 30, 3 * (cell + pad) + pad, 3), 255, np.uint8)
    for j, name in enumerate(["Input", "GT density", "Pred density"]):
        cv2.putText(canvas, name, (pad + j * (cell + pad), 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    vmax = max(float(np.percentile(r[1], 99.9)) for r in raw) if raw else 1.0
    for k, i in enumerate(idx):
        x, gt, pr = raw[i]
        img = np.clip(x.transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) +
                      np.array([0.485, 0.456, 0.406]), 0, 1)
        yy = 30 + k * (cell + 26)
        img_u8 = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        tiles = [img_u8, heat(gt, vmax), heat(pr, vmax)]
        for j, t in enumerate(tiles):
            t = cv2.resize(t, (cell, cell), interpolation=cv2.INTER_NEAREST)
            canvas[yy:yy + cell, pad + j * (cell + pad):pad + j * (cell + pad) + cell] = t
        cv2.putText(canvas, "GT %d | Pred %.1f" % (gts[i], preds[i]),
                    (pad, yy + cell + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(args.out, canvas)
    print("已保存可视化:", args.out)


if __name__ == "__main__":
    main()

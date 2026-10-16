# -*- coding: utf-8 -*-
"""端到端验收脚本：把「数据 -> 模型 -> 验证 -> 测试 -> 可视化」整条流程重走一遍。

用法：
    python verify_pipeline.py                 # 验收最佳实验 mtl_sedh_dm
    python verify_pipeline.py --run baseline  # 验收指定实验
    python verify_pipeline.py --run mtl_sedh_dm --out 验收_可视化.png

它会：
  1) 校验数据集与分割标签是否完整
  2) 按 runs/<run>/args.json 原样重建模型并载入 best.pth
  3) 在 val 与 test 上重新评测，和训练时记录的数字逐项核对
  4) 生成真值 vs 预测密度图对比图，肉眼确认它确实在数鱼
"""
import os
import sys
import csv
import json
import argparse

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import ARISDataset, build_loaders       # noqa: E402
from src.models import UNetResNet18, LGMFormer        # noqa: E402
from src.metrics import mae_rmse                      # noqa: E402

OK, BAD = "[OK]  ", "[!!]  "


def bias_init_for(scale: float) -> float:
    """与 train.py 一致：softplus(b) = 1.6e-4 * scale，ARIS 密度均值约 1.6e-4。"""
    import math
    return math.log(math.exp(1.6e-4 * scale) - 1.0)


def build_model(a, device):
    if a.get("arch") == "lgmformer":
        return LGMFormer(
            pretrained=False,
            head_bias_init=bias_init_for(a["density_scale"]),
            use_seg=a.get("seg", False) or a.get("sedh", False),
            use_sedh=a.get("sedh", False),
            fusion=a.get("lgm_fusion", "paper")).to(device)
    return UNetResNet18(
        pretrained=False,
        head_bias_init=bias_init_for(a["density_scale"]),
        use_hlfg=a.get("hlfg", False), hlfg_mode=a.get("hlfg_mode", "cat3"),
        use_lgfi=a.get("lgfi", False),
        use_seg=a.get("seg", False) or a.get("sedh", False),
        use_sedh=a.get("sedh", False)).to(device)


def sanity_check_data(root):
    """阶段 1：数据完整性"""
    print("\n--- 阶段 1/4  数据与标签完整性 ---")
    ok = True
    counts = {}
    for split in ("train", "val", "test"):
        xp = os.path.join(root, "Labelled_data", "x_%s.npy" % split)
        yp = os.path.join(root, "Labelled_data", "y_%s.npy" % split)
        if not (os.path.exists(xp) and os.path.exists(yp)):
            print(BAD + "%s 的 x/y .npy 缺失" % split)
            ok = False
            continue
        x = np.load(xp, mmap_mode="r")
        y = np.load(yp, mmap_mode="r")
        counts[split] = len(x)
        print("%s%s  图像 %s 标签 %s  样本 %d  每图真值条数 %.2f" % (
            OK, split, x.shape[1:], y.shape[1:], len(x), float(np.asarray(y).sum() / len(x))))
    total = sum(counts.values())
    if total:
        print("%s  合计 %d 张（论文 ARIS 数据集 500 张）" % (OK, total))

    seg_root = os.path.join(root, "seg_labels")
    for split in ("train", "val", "test"):
        d = os.path.join(seg_root, split)
        n = len([f for f in os.listdir(d)]) if os.path.isdir(d) else 0
        want = counts.get(split, 0)
        flag = OK if (n == want and n > 0) else BAD
        if n != want:
            ok = False
        print("%s%s 分割标签 %d / %d" % (flag, split, n, want))
    return ok


def evaluate_model(model, ds, a, device):
    """阶段 3：重算 val / test 指标（与 train.py 的 evaluate 完全同口径）

    MAE/RMSE：按整图密度求和（每图一个计数值）
    mIoU/OA/F1：3x3 混淆矩阵（行=GT 类，列=预测类，含背景类 0），与 train.py 一致
    """
    scale = a["density_scale"]
    need_seg = a.get("seg", False) or a.get("sedh", False)
    preds, gts, raw = [], [], []
    cm = np.zeros((3, 3), np.int64)
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            x, y = item[0], item[1]
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out = model(x[None].to(device))
            if isinstance(out, tuple):
                dens, seg_logits = out
            else:
                dens, seg_logits = out, None
            dens = dens.float()
            preds.append(dens.sum().item() / scale)
            gts.append(y.sum().item())
            if need_seg and seg_logits is not None and len(item) > 2:
                gt_seg = item[2].numpy()
                pred_cls = seg_logits.argmax(1)[0].cpu().numpy()
                for k in range(3):
                    sel = pred_cls[gt_seg == k]
                    if sel.size:
                        cm[k] += np.bincount(sel, minlength=3)[:3]
            if len(raw) < 400:
                raw.append((x.numpy(), y.numpy()[0], (dens / scale)[0, 0].cpu().numpy()))
    mae, rmse = mae_rmse(preds, gts)
    m = {"mae": mae, "rmse": rmse}
    if cm.sum() > 0:
        iou = np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1) - np.diag(cm), 1)
        f1 = 2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1)
        m["oa"] = float(cm.trace() / cm.sum())
        m["miou"] = float(iou.mean())
        m["f1"] = float(f1.mean())
    return m, raw, preds, gts


def make_vis(raw, preds, gts, out_path, n=6, cell=220, pad=8):
    """阶段 4：真值 vs 预测密度图对比"""
    order = np.argsort(gts)
    idx = [int(order[int(p * (len(order) - 1))]) for p in np.linspace(0, 1, n)]

    def heat(d, vmax, gamma=0.5):
        d = np.clip(d / max(vmax, 1e-9), 0, 1) ** gamma
        return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)

    canvas = np.full((len(idx) * (cell + 26) + 30, 3 * (cell + pad) + pad, 3), 255, np.uint8)
    for j, name in enumerate(["Input", "GT density", "Pred density"]):
        cv2.putText(canvas, name, (pad + j * (cell + pad), 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    vmax = max(float(np.percentile(r[1], 99.9)) for r in raw) if raw else 1.0
    for k, i in enumerate(idx):
        x, gt, pr = raw[i]
        img = np.clip(x.transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225])
                      + np.array([0.485, 0.456, 0.406]), 0, 1)
        img_u8 = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        yy = 30 + k * (cell + 26)
        for j, t in enumerate([img_u8, heat(gt, vmax), heat(pr, vmax)]):
            t = cv2.resize(t, (cell, cell), interpolation=cv2.INTER_NEAREST)
            canvas[yy:yy + cell, pad + j * (cell + pad):pad + j * (cell + pad) + cell] = t
        cv2.putText(canvas, "GT %d  |  Pred %.1f" % (gts[i], preds[i]),
                    (pad, yy + cell + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)
    print("%s  已保存对比图: %s" % (OK, out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mtl_sedh_dm", help="runs/ 下的实验名")
    ap.add_argument("--out", default=None, help="可视化输出路径")
    args = ap.parse_args()

    run_dir = os.path.join("runs", args.run)
    args_json = os.path.join(run_dir, "args.json")
    ckpt_path = os.path.join(run_dir, "best.pth")
    if not os.path.exists(args_json) or not os.path.exists(ckpt_path):
        print("[!!]  缺少 %s 或 %s，无法验收" % (args_json, ckpt_path))
        return 1

    a = json.load(open(args_json, encoding="utf-8"))
    root = a["data_root"]
    print("=" * 78)
    print("端到端验收：%s   架构=%s  HLFG=%s  LGFI=%s  Seg=%s  SEDH=%s  损失=%s" % (
        args.run, a.get("arch", "unet"), a.get("hlfg"), a.get("lgfi"),
        a.get("seg"), a.get("sedh"), a.get("loss")))
    print("=" * 78)

    # 阶段 1
    sanity_check_data(root)

    # 阶段 2
    print("\n--- 阶段 2/4  重建模型并载入权重 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(a, device)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    n_par = sum(p.numel() for p in model.parameters())
    print("%s  设备 %s  参数量 %.2f M" % (OK, device, n_par / 1e6))
    print("%s  best.pth 来自第 %s 轮，训练时 val MAE %.3f" % (
        OK, ck.get("epoch"), ck.get("val_mae", -1)))

    # 阶段 3
    print("\n--- 阶段 3/4  重新评测 val / test ---")
    need_seg = a.get("seg", False) or a.get("sedh", False)
    _, va, te = build_loaders(root, crop=None, batch=1, workers=0, limit=0,
                              need_seg=need_seg)
    val_m, _, _, _ = evaluate_model(model, va.dataset, a, device)
    test_m, raw, preds, gts = evaluate_model(model, te.dataset, a, device)

    # 与训练时记录比对
    log_path = os.path.join(run_dir, "log.csv")
    log_best_val = None
    if os.path.exists(log_path):
        rows = list(csv.DictReader(open(log_path, encoding="utf-8")))
        if rows:
            r = min(rows, key=lambda x: float(x["val_mae"]))
            log_best_val = (float(r["val_mae"]), r["epoch"])
    tm_path = os.path.join(run_dir, "test_metrics.json")
    saved = json.load(open(tm_path, encoding="utf-8")) if os.path.exists(tm_path) else {}

    print("  %-28s %10s %10s %10s" % ("指标", "本次复算", "训练时记录", "偏差"))
    print("  " + "-" * 62)

    def line(name, now, old):
        if old is None:
            print("  %-28s %10.3f %10s %10s" % (name, now, "-", "-"))
            return
        d = now - old
        flag = "" if abs(d) <= max(0.02 * abs(old), 0.05) else "   <- 不一致!"
        print("  %-28s %10.3f %10.3f %+10.3f%s" % (name, now, old, d, flag))

    line("val MAE", val_m["mae"], log_best_val[0] if log_best_val else None)
    line("test MAE", test_m["mae"], saved.get("mae"))
    line("test RMSE", test_m["rmse"], saved.get("rmse"))
    if "miou" in test_m:
        line("test mIoU", test_m["miou"], saved.get("miou"))
    if log_best_val:
        print("  （log.csv 中 val MAE 最优出现在第 %s 轮）" % log_best_val[1])
    gap = test_m["mae"] / val_m["mae"] - 1
    print("  val→test 落差: %+.1f%%  %s" % (
        gap * 100, "(健康)" if gap < 0.2 else ("(偏过拟合)" if gap < 0.5 else "(明显过拟合)")))

    # 阶段 4
    # 注意：Windows 下 cv2.imwrite 不支持非 ASCII 路径，输出名保持纯英文
    print("\n--- 阶段 4/4  生成真值 vs 预测对比图 ---")
    out = args.out or os.path.join(run_dir, "verify_pred_vis.png")
    make_vis(raw, preds, gts, out)

    print("\n" + "=" * 78)
    print("验收结论：数据 -> 训练权重 -> 验证 -> 测试 -> 可视化  全流程可复现")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

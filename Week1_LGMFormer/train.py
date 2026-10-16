# -*- coding: utf-8 -*-
"""阶段 0：U-Net + ResNet18 纯密度回归基线（对应论文消融表 Table VI 第一行 Baseline）。

针对 4GB 显存：用梯度累积把有效 batch 拉到接近论文的 35，并开启 AMP。
用法示例：
    python train.py --data-root D:/project/source/ARIS --epochs 100 --batch 4 --accum 9
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import build_loaders                                   # noqa: E402
from src.models import UNetResNet18, LGMFormer                          # noqa: E402
from src.losses import build_loss                                     # noqa: E402
from src.metrics import mae_rmse, AverageMeter                        # noqa: E402
from src.utils import set_seed, CSVLogger, density_to_dots            # noqa: E402


def bias_init_for(scale, mean_density=1.6e-4):
    """反解 softplus 偏置，使初始输出 = 密度均值 × 缩放系数。"""
    t = max(mean_density * scale, 1e-12)
    return float(np.log(np.expm1(t)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="D:/project/source/ARIS")
    p.add_argument("--out", default="runs/exp",
                   help="实验输出目录。每次实验请换新名字，否则会覆盖之前的结果")
    p.add_argument("--epochs", type=int, default=300,
                   help="默认 300 轮（cosine 调度按 300 轮设计，少于 300 轮学习率不会退火完，结果会明显偏差）")
    p.add_argument("--batch", type=int, default=4, help="实际显存批次")
    p.add_argument("--accum", type=int, default=9, help="梯度累积步数(4x9≈论文batch35)")
    p.add_argument("--crop", type=int, nargs=2, default=(256, 256))
    p.add_argument("--lr", type=float, default=6e-4, help="解码器/主干学习率(论文值)")
    p.add_argument("--lr-encoder", type=float, default=6e-5, help="编码器学习率(论文值)")
    p.add_argument("--wd", type=float, default=1e-4, help="weight decay(论文未公布,取常用值)")
    p.add_argument("--loss", default="mse", choices=["mse", "dmcount"])
    p.add_argument("--density-scale", type=float, default=100.0,
                   help="密度图目标缩放系数。ARIS 密度均值仅 1.6e-4，直接做 MSE 会得到 "
                        "~1e-8 的梯度，低于 Adam 的 eps=1e-8 导致优化停摆；缩放后梯度回到正常量级。"
                        "推理时把预测除以该系数即得真实密度。")
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help=">0 时只用前 N 张(冒烟测试)")
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--hlfg", action="store_true",
                   help="启用 HLFG 模块(论文 §III-B, 3 个深层融合点)，对应消融表 Baseline+HLFG")
    p.add_argument("--hlfg-mode", default="cat3", choices=["cat3", "add", "replace"],
                   help="cat3=cat[x_up,skip,hf] 三路拼接(默认,基线通路原样保留); "
                        "add/replace 为早期两种融合方式，实测均劣于基线，仅作消融对照")
    p.add_argument("--lgfi", action="store_true",
                   help="启用 3 个 LGFI 模块(论文 §III-C)，对应消融表 Baseline+LGFI 系列行")
    p.add_argument("--seg", action="store_true",
                   help="启用分割分支(式8, MTL)，对应消融表 +分割头 行；需要 ARIS/seg_labels")
    p.add_argument("--sedh", action="store_true",
                   help="启用 SEDH(式9-11)，对应完整 LGMFormer；隐含 --seg")
    p.add_argument("--seg-weight", type=float, default=1.0,
                   help="分割损失 L_seg 的额外权重。DM-Count 的 L_C 量级被密度缩放放大到 O(数千)，"
                        "会淹没 L_seg(O(2))，导致分割学崩；调大(如20)可恢复 mIoU 并让 SEDH 生效。")
    p.add_argument("--arch", default="unet", choices=["unet", "lgmformer"],
                   help="unet=U-Net 解码器适配版(消融实验用); lgmformer=论文原装 transformer "
                        "解码器(Fig.2: LGFI 堆叠 + HLFG 融合, 与 --seg/--sedh/--loss 组合)")
    p.add_argument("--lgm-fusion", default="paper", choices=["paper", "cat"],
                   help="lgmformer 每级融合方式: paper=严格 Eq.3(实测 MAE 7.08); "
                        "cat=cat[up,skip,hlfg] 保留 raw skip 通路(应用 U-Net cat3 教训)")
    p.add_argument("--aug", action="store_true",
                   help="训练数据增强(亮度/对比度抖动+随机擦除)，治 val/test 泛化 gap")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, amp, scale=1.0):
    model.eval()
    loss_m = AverageMeter()
    preds, gts = [], []
    crit = nn.MSELoss()
    cm = np.zeros((3, 3), np.int64)                    # 分割混淆矩阵(行=GT,列=预测)
    for batch in loader:
        x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        seg = batch[2].to(device) if len(batch) > 2 else None
        with torch.cuda.amp.autocast(enabled=bool(amp)):
            out, seg_logits = model(x)
            loss = crit(out.float(), y * scale)
        preds.append(out.float().sum(dim=(1, 2, 3)).div(scale).cpu().numpy())
        gts.append(y.sum(dim=(1, 2, 3)).cpu().numpy())
        loss_m.update(loss.item(), x.size(0))
        if seg is not None and seg_logits is not None:
            pred_cls = seg_logits.argmax(1)
            for k in range(3):
                cm[k] += np.bincount(pred_cls[seg == k].cpu().numpy(), minlength=3)[:3]
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    mae, rmse = mae_rmse(preds, gts)
    res = dict(mse=loss_m.avg, mae=mae, rmse=rmse)
    if cm.sum() > 0:
        oa = cm.trace() / cm.sum()
        iou = np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1) - np.diag(cm), 1)
        f1 = 2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1)
        res.update(oa=float(oa), miou=float(iou.mean()), f1=float(f1.mean()))
    return res


def main():
    a = parse_args()
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    print("设备:", device, "| torch", torch.__version__)

    tr, va, te = build_loaders(a.data_root, crop=tuple(a.crop), batch=a.batch,
                               workers=a.workers, limit=a.limit, need_seg=a.seg,
                               aug=a.aug)
    print("数据: train %d / val %d / test %d%s" % (
        len(tr.dataset), len(va.dataset), len(te.dataset),
        " | 增强: 亮度/对比度抖动+随机擦除" if a.aug else ""))

    if a.arch == "lgmformer":
        model = LGMFormer(pretrained=not a.no_pretrained,
                          head_bias_init=bias_init_for(a.density_scale),
                          use_seg=a.seg or a.sedh, use_sedh=a.sedh,
                          fusion=a.lgm_fusion).to(device)
        print("架构: LGMFormer(transformer 解码器, Fig.2, fusion=%s) | Seg:" % a.lgm_fusion,
              a.seg or a.sedh, "| SEDH:", a.sedh)
    else:
        model = UNetResNet18(pretrained=not a.no_pretrained,
                             head_bias_init=bias_init_for(a.density_scale),
                             use_hlfg=a.hlfg, hlfg_mode=a.hlfg_mode,
                             use_lgfi=a.lgfi, use_seg=a.seg or a.sedh,
                             use_sedh=a.sedh).to(device)
        print("架构: U-Net 适配版 | 编码器初始化:", model.encoder_init,
              "| HLFG:", a.hlfg, a.hlfg_mode if a.hlfg else "",
              "| LGFI:", a.lgfi, "| Seg:", a.seg or a.sedh, "| SEDH:", a.sedh)
    n_par = sum(p.numel() for p in model.parameters())
    print("参数量: %.2f M  密度缩放系数: %g" % (n_par / 1e6, a.density_scale))

    crit = build_loss(a.loss + ("+seg" if (a.seg or a.sedh) else ""),
                       seg_weight=a.seg_weight)
    opt = torch.optim.AdamW(model.param_groups(a.lr, a.lr_encoder), weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(a.amp))

    log = CSVLogger(os.path.join(a.out, "log.csv"),
                    ["epoch", "train_loss", "lr", "val_mse", "val_mae", "val_rmse",
                     "val_oa", "val_miou", "val_f1", "sec", "best"])
    best_mae, best_path = 1e9, os.path.join(a.out, "best.pth")
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)
    print("有效 batch = %d x %d = %d" % (a.batch, a.accum, a.batch * a.accum))

    for ep in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        loss_m = AverageMeter()
        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(tr):
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            seg = batch[2].to(device) if len(batch) > 2 else None
            dots = None
            if a.loss == "dmcount":
                dots = torch.stack([torch.from_numpy(density_to_dots(yy[0].cpu().numpy()))
                                    for yy in y]).unsqueeze(1).to(device)
            with torch.cuda.amp.autocast(enabled=bool(a.amp)):
                out, seg_logits = model(x)
                loss, _ = crit(out, y * a.density_scale, dots, seg, seg_logits)
                loss = loss / a.accum
            scaler.scale(loss).backward()
            if (it + 1) % a.accum == 0 or (it + 1) == len(tr):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            loss_m.update(loss.item() * a.accum, x.size(0))

        sched.step()
        msg = "[%3d/%3d] loss %.4f  lr %.2e  %.1fs" % (
            ep, a.epochs, loss_m.avg, opt.param_groups[0]["lr"], time.time() - t0)

        if ep % a.val_every == 0 or ep == a.epochs:
            m = evaluate(model, va, device, a.amp, a.density_scale)
            msg += "  | val MAE %.3f  RMSE %.3f" % (m["mae"], m["rmse"])
            if "oa" in m:
                msg += "  OA %.4f  mIoU %.4f  F1 %.4f" % (m["oa"], m["miou"], m["f1"])
            is_best = m["mae"] < best_mae
            if is_best:
                best_mae = m["mae"]
                torch.save({"model": model.state_dict(), "epoch": ep, "val_mae": best_mae},
                           best_path)
                msg += "  *best"
            log.log(epoch=ep, train_loss=round(loss_m.avg, 5), lr="%.3e" % opt.param_groups[0]["lr"],
                    val_mse=round(m["mse"], 5), val_mae=round(m["mae"], 4),
                    val_rmse=round(m["rmse"], 4),
                    val_oa=round(m.get("oa", 0), 4), val_miou=round(m.get("miou", 0), 4),
                    val_f1=round(m.get("f1", 0), 4), sec=round(time.time() - t0, 1),
                    best=int(is_best))
        print(msg, flush=True)

    log.close()

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    t = evaluate(model, te, device, a.amp, a.density_scale)
    print("\n=== 测试集结果（best val 权重）===")
    print("MAE  = %.3f   (论文 LGMFormer 3.22 / 基线见 Table VI)" % t["mae"])
    print("RMSE = %.3f   (论文 LGMFormer 6.05)" % t["rmse"])
    if "oa" in t:
        print("Seg: OA %.4f  mIoU %.4f  F1 %.4f   (论文完整模型 mIoU 90.2%%)"
              % (t["oa"], t["miou"], t["f1"]))
    json.dump(t, open(os.path.join(a.out, "test_metrics.json"), "w"), indent=2)

    if a.hlfg:
        from src.models import HLFG
        for n, m in model.named_modules():
            if isinstance(m, HLFG):
                w = torch.softmax(m.w, 0).detach().cpu().numpy()
                print("  %s 聚合系数: alpha(Xlg)=%.3f beta(Xhf)=%.3f gamma(Xres)=%.3f"
                      % (n, w[0], w[1], w[2]))


if __name__ == "__main__":
    main()

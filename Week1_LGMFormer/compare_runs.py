# -*- coding: utf-8 -*-
"""汇总 runs/ 下所有训练结果，生成消融对照表（对比论文 Table VI 的趋势）。"""
import csv
import json
import os

RUNS = [
    ("baseline", "Baseline"),
    ("hlfg", "Baseline + HLFG (replace, 严格Eq.3)"),
    ("hlfg_add", "Baseline + HLFG (add, 已中止)"),
    ("hlfg_cat3", "Baseline + HLFG (cat3, 本文复现采用)"),
    ("lgfi", "Baseline + LGFI"),
    ("hlfg_cat3_lgfi", "Baseline + HLFG + LGFI"),
    ("mtl", "+ 分割头 (MTL)"),
    ("mtl_sedh_dm", "完整 LGMFormer 核心 (MTL+SEDH+DM-Count, 本文复现采用)"),
    ("mtl_sedh_dm_aug", "+ 数据增强 (已证伪,仅记录)"),
    ("lgm_base", "LGMFormer transformer 纯密度 (paper融合)"),
    ("lgm_full", "LGMFormer transformer 完整 (scale=100)"),
    ("lgm_full_s1", "LGMFormer transformer 完整 (scale=1)"),
]

# 论文 Table VI 的相对提升（§IV-F 文字）：+HLFG: MAE 12.3%, RMSE 3.7%
# +LGFI: MAE 12.4%, RMSE 11.3%；完整模型测试集 MAE 3.22 / RMSE 6.05 / mIoU 90.2%
PAPER = {
    "Baseline + HLFG (replace, 严格Eq.3)": (12.3, 3.7),
    "Baseline + LGFI": (12.4, 11.3),
}

rows = []
for run, name in RUNS:
    log = os.path.join("runs", run, "log.csv")
    tm = os.path.join("runs", run, "test_metrics.json")
    if not os.path.exists(log):
        continue
    rs = list(csv.DictReader(open(log, encoding="utf-8")))
    best = min(rs, key=lambda r: float(r["val_mae"]))
    d = dict(run=run, name=name, ep=rs[-1]["epoch"],
             val_mae=float(best["val_mae"]), best_ep=best["epoch"])
    if os.path.exists(tm):
        t = json.load(open(tm, encoding="utf-8"))
        d.update(test_mae=t["mae"], test_rmse=t["rmse"],
                 miou=t.get("miou"), oa=t.get("oa"), f1=t.get("f1"))
    rows.append(d)

base = next((r for r in rows if r["run"] == "baseline"), None)
print("%-38s %5s %9s %9s %10s %10s %8s" %
      ("配置", "轮数", "val MAE", "测试MAE", "测试RMSE", "mIoU", "vs基线MAE"))
print("-" * 95)
for r in rows:
    dm = " %+.1f%%" % (100 * (r["test_mae"] / base["test_mae"] - 1)) if base and "test_mae" in r else ""
    print("%-38s %5s %9.3f %9s %10s %10s %8s" % (
        r["name"], r["ep"], r["val_mae"],
        "%.3f" % r["test_mae"] if "test_mae" in r else "-",
        "%.3f" % r["test_rmse"] if "test_rmse" in r else "-",
        "%.3f" % r["miou"] if r.get("miou") else "-",
        dm or "-"))
print()
print("论文参考：完整模型 MAE 3.22 / RMSE 6.05 / mIoU 90.2%；"
      "+HLFG: MAE ↓12.3%；+LGFI: MAE ↓12.4%、RMSE ↓11.3%")

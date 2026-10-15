# -*- coding: utf-8 -*-
"""Baselines Fixed — 重跑修复后的 IncLOF / HST（仅这两个方法，14 数据集）。

目的：复现 reviewer 质疑的 IncLOF/HST 低分问题，用修复版重新评分。
关键保证：
  1) 数据划分与 baselines.py 完全一致（same split）：复用其 make_split、
     以 global_rng=RandomState(SEED) 按同顺序遍历 DATASETS。
  2) 不修改任何既有程序：修复类通过 importlib 从 *_fixed.py 直接加载。
  3) checkpoint 容灾：每个数据集完成后写 CSV。

产出：BaselinesFixed_checkpoint.csv / BaselinesFixed.csv
"""
import numpy as np, os, sys, time, warnings, importlib.util as iu, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from baselines import load_ds, make_split, compute_auc, DATASETS, SEED
from sklearn.metrics import f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_FILE = os.path.join(OUT_DIR, "BaselinesFixed_checkpoint.csv")
OUT_FILE = os.path.join(OUT_DIR, "BaselinesFixed.csv")


def _load(modname, fpath):
    spec = iu.spec_from_file_location(modname, fpath)
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def compute_f1(scores, y, AR):
    thr = np.percentile(scores, 100 * (1 - AR))
    pred = (scores > thr).astype(int)
    return f1_score(y, pred, zero_division=0)


def build_fixed():
    hst = _load("hst_fix", os.path.join(BASE, "src", "detectors", "baselines",
                                         "half_space_trees_fixed.py"))
    lof = _load("lof_fix", os.path.join(BASE, "src", "detectors", "baselines",
                                        "incremental_lof_fixed.py"))
    return {
        "IncLOF_fix": lambda: lof.IncrementalLOFFixed(window_size=128, n_neighbors=10),
        "HST_fix": lambda: hst.HalfSpaceTreesFixed(n_trees=25, max_depth=10),
    }


def load_ckpt():
    if os.path.exists(CKPT_FILE):
        try:
            return pd.read_csv(CKPT_FILE)
        except Exception:
            pass
    return None


def main():
    ckpt = load_ckpt()
    all_rows = [] if ckpt is None else ckpt.to_dict("records")
    done = {r["dataset"] for r in all_rows} if all_rows else set()
    global_rng = np.random.RandomState(SEED)
    baselines = build_fixed()

    for name, sub, fname, tier in DATASETS:
        if name in done:
            print(f"[{name}] SKIP (done)")
            continue
        print(f"\n=== {name} ({tier}) ===")
        N, d, AR, U_size, x_size, n_eval, U, eval_X, eval_y, p1_size = \
            make_split(name, sub, fname, tier, global_rng)
        row = {"dataset": name, "N": N, "d": d, "AR": AR, "tier": tier,
               "U": U_size, "X0": x_size, "n_eval": n_eval}
        for bname, bfn in baselines.items():
            try:
                det = bfn()
                t0 = time.perf_counter()
                det.fit(U)
                scores = np.array([det.predict(eval_X[i:i + 1])[0]
                                   for i in range(len(eval_X))])
                t_ms = (time.perf_counter() - t0) * 1000 / max(1, len(eval_X))
                auc, apr = compute_auc(scores, eval_y)
                f1v = compute_f1(scores, eval_y, AR)
                print(f"    {bname:<12s} AUC={auc:.4f}  APR={apr:.4f}  F1={f1v:.4f}  t={t_ms:.2f}ms/pt")
            except Exception as e:
                import traceback; traceback.print_exc()
                auc, apr, f1v, t_ms = np.nan, np.nan, np.nan, np.nan
                print(f"    {bname:<12s} ERROR: {e}")
            row[f"AUC_{bname}"] = auc
            row[f"AP_{bname}"] = apr
            row[f"F1_{bname}"] = f1v
            row[f"t_{bname}_ms"] = t_ms

        all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(CKPT_FILE, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_FILE, index=False)
    print(f"\nDONE.  checkpoint → {CKPT_FILE}\n  final    → {OUT_FILE}")
    print(df[["dataset", "AUC_IncLOF_fix", "AP_IncLOF_fix", "F1_IncLOF_fix",
              "AUC_HST_fix", "AP_HST_fix", "F1_HST_fix"]].to_string(index=False))


if __name__ == "__main__":
    main()
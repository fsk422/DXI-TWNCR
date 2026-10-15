"""λ=2.5 补充实验 —— 补全 λ 消融网格至 6 档。

复用 ablation_lambda.py 的 RQ5b 逻辑与数据划分（SEED=42 固定），
仅跑 λ=2.5 一个档位，保证与已有 λ∈{0.5,1.0,1.5,2.0,3.0} 数据同构可比。
输出：Lambda25_checkpoint.csv
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablation_lambda as rq5

CKPT = os.path.join(rq5.OUT_DIR, "Lambda25_checkpoint.csv")
LEVEL_NAME = "λ=2.5"
LEVEL = {"name": LEVEL_NAME, "lambda_val": 2.5, "p1x_sep": True, "K_val": 1}


def main():
    rows = []
    if os.path.exists(CKPT):
        rows = pd.read_csv(CKPT).to_dict("records")
    done = {(r["dataset"], r["level"]) for r in rows}

    for ds_name, sub, fname, tier in rq5.DATASETS:
        if (ds_name, LEVEL_NAME) in done:
            print(f"[{ds_name}|{LEVEL_NAME}] SKIP")
            continue
        print(f"[{ds_name}|{LEVEL_NAME}] RUNNING...", flush=True)
        try:
            r = rq5.run_one_level("RQ5b", ds_name, sub, fname, tier, LEVEL)
            rows.append(r)
            pd.DataFrame(rows).to_csv(CKPT, index=False)
        except Exception as e:
            print(f"ERROR {ds_name}: {e}")
            import traceback; traceback.print_exc()

    df = pd.DataFrame(rows)
    print(f"\nDone. {len(df)} runs. Output: {CKPT}")
    for _, row in df.iterrows():
        print(f"  {row['dataset']:<12s} λ=2.5  AUC={row['auc_roc']:.4f}  AP={row['auc_pr']:.4f}")


if __name__ == "__main__":
    main()
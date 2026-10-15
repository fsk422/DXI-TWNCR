"""Simple launcher for drift experiment - no stdout tricks."""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drift as de
import numpy as np, pandas as pd, time

all_rows = []
ckpt_file = de.CKPT_FILE
if os.path.exists(ckpt_file):
    ckpt = pd.read_csv(ckpt_file)
    for _, r in ckpt.iterrows(): all_rows.append(r.to_dict())
    done = set(ckpt["dataset"].unique())
    print(f"Resuming, done: {done}")
else:
    done = set()

global_rng = np.random.RandomState(de.SEED)
for name, sub, fname, tier in de.DATASETS:
    if name in done:
        print(f"SKIP {name}")
        continue
    print(f"Running {name}...", flush=True)
    t0 = time.perf_counter()
    try:
        rows = de.run_one(name, sub, fname, tier, global_rng)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(ckpt_file, index=False)
        print(f"DONE {name}: {len(rows)} rows in {time.perf_counter()-t0:.0f}s")
    except Exception as e:
        print(f"ERROR {name}: {e}")
        traceback.print_exc()

df = pd.DataFrame(all_rows)
out = os.path.join(de.OUT_DIR, "DRIFT_Drift.xlsx")
df.to_excel(out, index=False)
print(f"\nXLSX -> {out}")

for ds in [d[0] for d in de.DATASETS]:
    print(f"\n{ds}")
    for lvl in ["L0","L1","L2","L3"]:
        line = f"  {lvl:<4s}"
        for m in ["Incr1","Incr1-NR","SiForest","BWOAIF"]:
            sub = df[(df.dataset==ds)&(df.level==lvl)&(df.method==m)]
            v = sub["sw_post"].values[0] if len(sub)>0 else None
            line += f"  {v:.4f}" if v is not None else "     ---"
        print(line)
    print("  Phase C: Incr1 - Incr1-NR")
    for lvl in ["L0","L1","L2","L3"]:
        i1 = df[(df.dataset==ds)&(df.level==lvl)&(df.method=="Incr1")]
        nr = df[(df.dataset==ds)&(df.level==lvl)&(df.method=="Incr1-NR")]
        if len(i1)>0 and len(nr)>0:
            print(f"    {lvl}: {i1['sw_post'].values[0]-nr['sw_post'].values[0]:+.4f}")
print("\nALL DONE")

"""analysis/mission_plots.py -- figures and tables for Missions 1, 2 and 4.
    python -m analysis.mission_plots      (run from the top folder, after train.py)"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from app.model import FEATURES, BatteryNet
from app.main import layer_table

df = pd.read_csv("data/team_10_train.csv")
cols = FEATURES + ["days_left"]
# Mission 1: histograms (raw, glitches included -- that is how you spot them)
fig, ax = plt.subplots(2, 3, figsize=(13, 7))
for a, c in zip(ax.flat, cols): a.hist(df[c], bins=60); a.set_title(c)
fig.suptitle("Team 10: raw histograms (note the lone spikes at 0 V, -127 °C, 0 dBm)"); fig.tight_layout()
fig.savefig("analysis/m1_histograms.png", dpi=110)
clean = df[(df.battery_v >= 2.5) & (df.temp_c >= -55) & (df.wifi_rssi_dbm < 0)]
fig, ax = plt.subplots(1, 5, figsize=(20, 4))
for a, c in zip(ax, FEATURES): a.scatter(clean[c], clean.days_left, s=3, alpha=.4); a.set_xlabel(c); a.set_ylabel("days_left")
fig.suptitle("Team 10: days_left against each input (glitches removed)"); fig.tight_layout()
fig.savefig("analysis/m1_scatter.png", dpi=110)

# Mission 1: linear baseline, predicted vs true (same split as train.py)
rng = np.random.default_rng(0); idx = rng.permutation(len(clean)); n_val = len(clean) // 5
val, tr = clean.iloc[idx[:n_val]], clean.iloc[idx[n_val:]]
A = lambda d: np.hstack([d[FEATURES].to_numpy(), np.ones((len(d), 1))])
w, *_ = np.linalg.lstsq(A(tr), tr.days_left.to_numpy(), rcond=None)
p = A(val) @ w
plt.figure(figsize=(5, 5)); plt.scatter(val.days_left, p, s=5); plt.plot([0, 300], [0, 300], "k--")
plt.xlabel("true days_left"); plt.ylabel("linear prediction"); plt.title("Baseline 2: linear regression")
plt.tight_layout(); plt.savefig("analysis/m1_linear_pred_vs_true.png", dpi=110)
print("linear coefficients:", dict(zip(FEATURES + ["intercept"], np.round(w, 3))))

# Mission 2: validation MAE vs params (numbers from train.py runs, seed 0, 1500 epochs)
sweep = {"one hidden": {29: 12.76, 57: 15.66, 113: 15.61, 225: 10.25, 449: 15.55},
         "two hidden": {49: 10.28, 129: 6.41, 385: 5.29, 1281: 5.49, 4609: 5.21}}
plt.figure(figsize=(6, 4))
for k, d in sweep.items(): plt.plot(list(d), list(d.values()), "o-", label=k)
plt.xscale("log"); plt.xlabel("parameters"); plt.ylabel("validation MAE (days)"); plt.legend()
plt.title("Widths 4, 8, 16, 32, 64"); plt.tight_layout(); plt.savefig("analysis/m2_mae_vs_params.png", dpi=110)

# Mission 4: same function the API uses
print("\n| w | params | float32 B | int8 B | fits 300KB f32 | fits 300KB int8 | fits 8MB PSRAM f32 | MACs |")
print("|---|---|---|---|---|---|---|---|")
for wd in [32, 128, 256, 512, 2048]:
    t = layer_table(BatteryNet((wd, wd)))
    print(f"| {wd} | {t['total_params']:,} | {t['total_bytes_f32']:,} | {t['total_bytes_int8']:,} | "
          f"{t['fits_300kb_sram_f32']} | {t['fits_300kb_sram_int8']} | {t['total_bytes_f32'] <= 8*1024*1024} | {t['total_macs']:,} |")
for budget, name in [(300 * 1024, "300 KiB = 307,200 B"), (300_000, "300,000 B")]:
    for bpp, fmt in [(4, "f32"), (1, "int8")]:
        wd = 1
        while bpp * (wd * wd + 8 * wd + 1) <= budget: wd += 1
        print(f"budget {name}, {fmt}: width {wd-1} fits, {wd} does not")

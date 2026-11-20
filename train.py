"""
train.py -- STARTER. Clean the data, fit baselines, train the network, save it for the server.

    python train.py --team 7 --hidden 32,32
Writes model/model.pt and model/config.json, which app/main.py loads.

Look for TODO. The training loop is given; the decisions are yours.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from app.model import FEATURES, BatteryNet

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
ap.add_argument("--hidden", default="32,32")
ap.add_argument("--epochs", type=int, default=1500)
ap.add_argument("--no-norm", action="store_true", help="skip normalisation (Mission 2 comparison)")
a = ap.parse_args()
hidden = tuple(int(h) for h in a.hidden.split(","))

df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")
# TODO Mission 1: find the sensor glitches (plot every column first!) and write a rule that
# marks them. Say in your report how many you found and why each rule is safe.
#
# Three kinds of impossible reading were found in team 10's data (2000 rows):
#  1. battery_v == 0.0 V   (8 rows). A Li-ion cell that is powering the doorbell can't be at 0 V;
#     below ~2.5 V the protection circuit cuts it off. 0.0 means "ADC read failed". The real
#     readings all sit between 3.35 and 4.2 V, so the cut at 2.5 V is far from any real value.
#  2. temp_c == -127 °C    (10 rows). -127 is the DS18B20 sensor's "disconnected" error code.
#     Nothing on Earth's surface is below -90 °C, and the sensor's rated floor is -55 °C.
#  3. wifi_rssi_dbm == 0   (6 rows). RSSI is received power in dBm and is always negative for a
#     real Wi-Fi link (a router touching the antenna is still around -10 to -20). 0 = "no value".
# The three sets don't overlap, so 24 rows are dropped.
glitch_battery = df.battery_v < 2.5
glitch_temp = df.temp_c < -55
glitch_rssi = df.wifi_rssi_dbm >= 0
bad = (glitch_battery | glitch_temp | glitch_rssi).to_numpy()
print(f"glitches: battery {glitch_battery.sum()}, temp {glitch_temp.sum()}, rssi {glitch_rssi.sum()}")
print(f"{len(df)} rows, {bad.sum()} glitch rows dropped")
df = df[~bad]

rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
X, y = tr[FEATURES].to_numpy(np.float32), tr.days_left.to_numpy(np.float32)
Xv, yv = val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32)
# TODO Mission 2: normalisation. Why must mu and sd come from the TRAINING rows only?
# Answer: the validation rows stand in for readings we have never seen. If their values leak into
# mu/sd, the validation score is slightly optimistic. The server must also use these SAME numbers,
# which is why they are saved to config.json.
# Without this, battery_v (~0.3 V spread) and rssi (~13 dBm spread) differ by ~40x in scale, so
# Adam with one learning rate struggles. Pass --no-norm to compare.
if a.no_norm:
    mu, sd = np.zeros(X.shape[1], np.float32), np.ones(X.shape[1], np.float32)
else:
    mu, sd = X.mean(axis=0).astype(np.float32), X.std(axis=0).astype(np.float32)

mae = lambda p, t: float(np.mean(np.abs(p - t)))
print(f"baseline (predict the mean): {mae(np.full_like(yv, y.mean()), yv):.2f} days")
# TODO Mission 1: a straight-line model (linear regression) as the second baseline.
# Use np.linalg.lstsq on [X, 1]. Print its validation MAE the same way as the line above.
A = np.hstack([X, np.ones((len(X), 1), np.float32)])      # the column of 1s gives the intercept
w, *_ = np.linalg.lstsq(A, y, rcond=None)
lin_pred = np.hstack([Xv, np.ones((len(Xv), 1), np.float32)]) @ w
print(f"baseline (linear regression): {mae(lin_pred, yv):.2f} days")

torch.manual_seed(0)
net = BatteryNet(hidden)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
Xvt = torch.tensor((Xv - mu) / sd)
for ep in range(a.epochs):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 128):
        b = perm[i:i + 128]
        loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 300 == 0 or ep == a.epochs - 1:
        with torch.no_grad():
            print(f"  epoch {ep:4d}  val MAE {mae(net(Xvt).numpy(), yv):.2f} days")
with torch.no_grad():
    val_mae = mae(net(Xvt).numpy(), yv)
    train_mae = mae(net(Xt).numpy(), y)
n_params = sum(p.numel() for p in net.parameters())
print(f"network {hidden}: {n_params} params, train MAE {train_mae:.2f}, val MAE {val_mae:.2f} days")

os.makedirs("model", exist_ok=True)
torch.save(net.state_dict(), "model/model.pt")
json.dump(dict(team=a.team, hidden=list(hidden), mu=mu.tolist(), sd=sd.tolist(),
               features=FEATURES, val_mae=round(val_mae, 2)), open("model/config.json", "w"), indent=2)
print("saved model/model.pt and model/config.json")

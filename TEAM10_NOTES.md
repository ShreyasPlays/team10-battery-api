# Team 10 (Brute Force): working notes for Missions 1 to 4

All numbers come from `data/team_10_train.csv`, seed 0, 1500 epochs, unless marked otherwise.
**Laptop and Codespace numbers (Mission 3) still need to be measured by you.** The ones below
were measured on a local test machine against `localhost`, so they only show what the
Codespace-internal run should look like.

Re-make the figures: `python -m analysis.mission_plots` (from the top folder).

---

## Mission 1: Meet the data and set the bar

**Prediction for "always guess the mean":** `days_left` has std ≈ 61 days. For a roughly normal
spread the mean absolute deviation is about 0.8 × std ≈ 49 days. Measured: **47.20 days**.

**What the histograms show** (`analysis/m1_histograms.png`, `m1_scatter.png`):
- `battery_v` is the strongest input, and it is **not a straight line**. Below about 3.6 V,
  days_left is near zero. Above that it rises steeply and spreads out: a knee or exponential shape,
  just like a Li-ion discharge curve.
- `rings_per_day` falls off as roughly 1/x (more rings, fewer days), with a spread that shrinks.
- `wifi_rssi_dbm`: weaker signal (more negative) means fewer days. The upper edge rises with signal,
  because the radio spends more power on retries.
- `temp_c` has a weak hump-shaped effect: the best values are in the 20s, and both heat and cold
  cost days. `age_months` has a weak downward trend.
- Effects **multiply** rather than add. A high voltage only gives many days if rings are low and
  Wi-Fi is good. A straight line can't express this, and that is why the network wins.

**Glitch rules** (`train.py`), 24 of 2000 rows removed. No row has two glitches.

| Glitch | Rule | Rows | Why it's safe |
|---|---|---|---|
| battery_v = 0.0 V | `battery_v < 2.5` | 8 | A running doorbell can't be below Li-ion cut-off (~2.5–3.0 V). Real values are 3.35–4.2 V |
| temp_c = −127 °C | `temp_c < -55` | 10 | −127 is the DS18B20 "sensor disconnected" code. The sensor's floor is −55 °C. Real values are 8–46 °C |
| wifi_rssi_dbm = 0 | `wifi_rssi_dbm >= 0` | 6 | RSSI of a real link is always negative. 0 means "no value". Real values are −88 to −45 |

**Baselines** (validation rows = 395):

| Model | Val MAE (days) |
|---|---|
| Baseline 1: predict the mean | 47.20 |
| Baseline 2: linear regression (`lstsq`) | 20.57 |
| Network 5→32→32→1, starter (no normalisation) | 14.76 |
| Network 5→32→32→1, normalised (final) | **5.49** |

**Where is the linear model most wrong?** (`m1_linear_pred_vs_true.png`)
- At the **low end** it predicts *negative* days (down to about −60) for nearly flat batteries.
- At the **high end** (true > 200 days) it under-predicts, flattening out around 150–180.

Both errors come from forcing a curved, multiplicative relationship into a straight line.

---

## Mission 2: A small network, opened up

**By hand, 5 → 32 → 32 → 1** (Linear(a,b) = a·b weights + b biases; a·b MACs):

| Layer | in→out | Params | Bytes f32 | MACs |
|---|---|---|---|---|
| linear1 | 5→32 | 5·32+32 = 192 | 768 | 160 |
| linear2 | 32→32 | 32·32+32 = 1056 | 4224 | 1024 |
| linear3 | 32→1 | 32+1 = 33 | 132 | 32 |
| **Total** | | **1281** | **5124** (int8: 1281) | **1216** |

`/model_info` computes this from `net.net` and returns exactly these numbers.

**Normalisation:** with the same model and epochs, val MAE goes from **14.76 → 5.49 days**.
battery_v spreads over about 0.24 V while rssi spreads over about 12 dBm. Without scaling, one
learning rate is far too small for the voltage weights, and voltage is the most important input.
mu and sd come from **training rows only**: validation must imitate unseen data, and the server has
to apply the same mu/sd (saved in `config.json`).

**Width sweep** (`analysis/m2_mae_vs_params.png`):

| Hidden | Params | Train MAE | Val MAE |
|---|---|---|---|
| 4 | 29 | 12.82 | 12.76 |
| 8 | 57 | 15.24 | 15.66 |
| 16 | 113 | 15.22 | 15.61 |
| 32 | 225 | 10.04 | 10.25 |
| 64 | 449 | 15.20 | 15.55 |
| 4,4 | 49 | 10.03 | 10.28 |
| 8,8 | 129 | 5.76 | 6.41 |
| 16,16 | 385 | 4.67 | 5.29 |
| **32,32** | **1281** | **4.24** | **5.49** |
| 64,64 | 4609 | 3.55 | 5.21 |

Observations to discuss as a team (the Skeptic's job):
- Every one-hidden-layer net is stuck at 10–16 days, and widening it doesn't help. That makes this
  an optimisation or shape problem, not a capacity problem. Two layers can build products of
  features (multiplicative effects) much more easily.
- Among two-layer nets, val MAE flattens at about 5.2–5.5 from 16,16 upwards. Differences of ±0.2
  are within seed-to-seed noise: test it with another `torch.manual_seed`.

**Overfitting?** For 32,32, train MAE is 4.24 and val MAE is 5.49. That is a gap of about 1.25 days
(roughly 30%): mild overfitting, not severe. Validation stopped improving after epoch ~600 and
didn't get worse. 64,64 has a bigger gap (3.55 vs 5.21) for almost no gain.

**Final model:** 5→32→32→1. **Expected MAE on the company's 400 unseen readings: about 5.5 days**
(say 5–6.5, since 395 validation rows give a noisy estimate).

---

## Mission 3: Serve it, and time every stage

**LLM question before Mission 3** ("which stage of a web request will be the slowest?"):
ask it yourselves and log the answer. The typical answer is "model inference". Our measurement
says otherwise; see below.

**Input ranges** (`app/main.py`, `Reading`): these are based on physics and sensors, not on our min/max.

| Field | Allowed | Reason |
|---|---|---|
| battery_v | 2.5 – 4.5 V | Li-ion cut-off up to full charge plus ADC margin |
| temp_c | −40 – 70 °C | Outdoor electronics range. Refuses −127 and 85 (DS18B20 error and power-on codes) |
| rings_per_day | 0 – 500 | A count, so never negative |
| wifi_rssi_dbm | −110 – −10 dBm | Always negative. Refuses the 0 glitch |
| age_months | 0 – 120 | Never negative; 10 years is the cap |

NaN and Infinity are refused too. Tested, all → **422**: battery 0.0, temp −127, temp 85,
rssi 0, rings −3, age −1, a missing field, and a string value. A real edge reading
(3.35 V, 5 °C, −90 dBm, 36 months) still gets an answer.

**Local run** (`bench.py --n 200` against `localhost`, keep-alive on):

| Stage | p50 ms | p95 ms | p99 ms |
|---|---|---|---|
| round_trip | 1.04 | 1.66 | 2.96 |
| parse | 0.021 | 0.042 | 0.054 |
| normalise | 0.021 | 0.047 | 0.066 |
| inference | 0.110 | 0.184 | 0.241 |
| build | 0.012 | 0.024 | 0.057 |
| server_total | 0.177 | 0.284 | 0.384 |
| network + framework | 0.87 | 1.37 | 1.97 |

The first request took 2.4 ms. Even on localhost, about 80% of the time is outside our code
(HTTP parsing, FastAPI routing, JSON, sockets). Inference is about 0.1 ms. **From your laptop to
the Codespace, expect a round trip of tens to hundreds of ms**, so the model's share falls below 1%.
The LLM's "inference is slowest" is wrong for a model this small.

**Still to do from your laptop:** `--n 200`, `--n 200 --no-keepalive`, and the in-Codespace run.
Expect `--no-keepalive` to add a TCP and TLS handshake to every request, which can be several
round trips (+50–300 ms).

**400 singles vs 1 batch** (localhost): 400 × `/predict` took **465 ms**, while one
`/predict_batch` of 400 took **3.9 ms** (server_total 1.65 ms). That is about 120× faster,
because each request pays the fixed HTTP/framework cost once, and one matrix multiply on 400 rows
costs about the same as on 1 row. The batch answers equal the 50/400 single answers exactly.

---

## Mission 4: Would it fit inside the doorbell?

**Our model:** weights + biases = 5,124 B float32, 1,281 B int8. **Largest activation memory
at once:** while computing linear2 you hold its 32 inputs and 32 outputs = 64 values = 256 B (f32)
or 64 B (int8). Total is well under 300 KB, so it fits easily.

**5 → w → w → 1** (params = w² + 8w + 1), produced by the same `layer_table()` the API uses.
"300 KB" is taken as 300 KiB = 307,200 B, and 8 MB as 8 MiB.

| w | Params | float32 B | int8 B | Fits 300KB f32 | Fits 300KB int8 | Fits 8MB PSRAM (f32) | MACs |
|---|---|---|---|---|---|---|---|
| 32 | 1,281 | 5,124 | 1,281 | yes | yes | yes | 1,216 |
| 128 | 17,409 | 69,636 | 17,409 | yes | yes | yes | 17,152 |
| 256 | 67,585 | 270,340 | 67,585 | yes | yes | yes | 67,072 |
| 512 | 266,241 | 1,064,964 | 266,241 | **no** | yes | yes | 265,216 |
| 2048 | 4,210,689 | 16,842,756 | 4,210,689 | no | no | **no** (int8 does fit) | 4,206,592 |

**Exact width where weights stop fitting in 300 KiB:** float32 → w = 273 fits, **274 doesn't**.
int8 → w = 550 fits, **551 doesn't**. If you read 300 KB as 300,000 B, the answers are 269/270
and 543/544. Say which definition you used.

**Time per prediction on the board** (an assumption, state it as one): the ESP32-S3 is a 240 MHz
dual-core Xtensa LX7 with a single-precision FPU. If one float32 MAC (with loads) takes about
4 cycles, that is about 60 M MAC/s, so 1,216 MACs take **about 20 µs**, plus a little for ReLU and
loops. int8 with Espressif's ESP-NN SIMD kernels would be several times faster. Either way it is
microseconds, against roughly 100+ ms for a Wi-Fi round trip to the server.

**Server or on-device?** On-device wins for this model:
- **Latency:** about 20 µs on the board, against 100 ms or more over Wi-Fi plus the internet.
- **Wi-Fi down:** on-device still works. The server version shows nothing exactly when it matters.
- **Power:** a Wi-Fi transmit burst costs far more battery than 1.2k MACs. Calling a server to
  predict battery life drains the battery. (Weak RSSI makes it worse, as Mission 1 shows.)
- **Privacy:** usage patterns (rings per day) stay in the house.
- **Cost:** no server to run for millions of doorbells.

The case for the server: you can update the model centrally and collect fleet data. A good
compromise is to run on-device, and sync logs and model updates occasionally while charging or
on strong Wi-Fi.

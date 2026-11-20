"""
app/main.py -- STARTER battery API. Look for TODO.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

GET  /health          -> {"status": "ok", "team": 7, "params": 1281, "val_mae": 3.4}
GET  /model_info      -> the per-layer table: params, bytes, MACs, and whether it fits the doorbell
POST /predict         one reading  -> {"days_left": 52.1, "timings_ms": {...}}
POST /predict_batch   {"rows": [reading, ...]} -> {"days_left": [...], "timings_ms": {...}}
A reading that is out of range gets HTTP 422 with the reason, never a guess.
"""
import json
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.model import FEATURES, BatteryNet

cfg = json.load(open("model/config.json"))
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load("model/model.pt", weights_only=True))
net.eval()
torch.set_num_threads(1)
MU, SD = np.array(cfg["mu"], np.float32), np.array(cfg["sd"], np.float32)
N_PARAMS = sum(p.numel() for p in net.parameters())


class Reading(BaseModel):
    # TODO Mission 3: a doorbell will send you glitches like the ones you found in Mission 1.
    # Give every field a sensible range, e.g. Field(ge=..., le=...), so a glitch is refused
    # with HTTP 422 instead of being answered with a made-up number. Base the limits on what a
    # real sensor can report, not on your data's min and max: real readings must still get answers.
    #
    # battery_v: a single Li-ion cell. Below ~2.5 V the protection circuit cuts power, so the
    #   doorbell can't be running; above ~4.35 V no cell is charged. 4.5 leaves room for ADC noise.
    # temp_c: outdoor electronics are rated -40..+85 °C. We stop at 70 because 85 is the DS18B20's
    #   power-on "no reading yet" value, and -127 (its "disconnected" code) is far below -40.
    # rings_per_day: a count, so never negative. 500/day (one every ~3 min, all day) is a generous cap.
    # wifi_rssi_dbm: always negative for a real link. -110 is below any Wi-Fi radio's sensitivity,
    #   -10 is the antenna touching the router. 0 is the "no value" glitch.
    # age_months: can't be negative; 120 months (10 years) is past any battery's life.
    # allow_inf_nan=False refuses NaN/Infinity, which would otherwise slip through ge/le checks.
    battery_v: float = Field(ge=2.5, le=4.5, allow_inf_nan=False)
    temp_c: float = Field(ge=-40, le=70, allow_inf_nan=False)
    rings_per_day: float = Field(ge=0, le=500, allow_inf_nan=False)
    wifi_rssi_dbm: float = Field(ge=-110, le=-10, allow_inf_nan=False)
    age_months: float = Field(ge=0, le=120, allow_inf_nan=False)


class Batch(BaseModel):
    rows: list[Reading] = Field(min_length=1, max_length=10000)


app = FastAPI(title="Doorbell battery API")
ms = lambda t0: round((time.perf_counter() - t0) * 1000, 3)


@app.get("/model_info")
def model_info():
    # TODO Mission 2 + 4: return your per-layer table, worked out FROM THE NETWORK (net.net),
    # not typed in by hand, so it stays right if you change the layer sizes. Exactly these keys:
    # {"layers": [{"name": "linear1", "in": 5, "out": 32, "params": ..., "bytes_f32": ..., "macs": ...}, ...],
    #  "total_params": ..., "total_bytes_f32": ..., "total_bytes_int8": ..., "total_macs": ...,
    #  "fits_300kb_sram_f32": true/false, "fits_300kb_sram_int8": true/false}
    # Count only the Linear layers (ReLU has no parameters). Clue: a Linear layer m has
    # m.in_features and m.out_features.
    return layer_table(net)


SRAM_BUDGET = 300 * 1024      # "about 300 KB free" on the ESP32-S3, taken as 300 KiB = 307,200 bytes


def layer_table(model):
    """Per-layer params / bytes / MACs, read from the network itself so it can't go stale.
    Linear(a, b): a*b weights + b biases; a*b multiply-accumulates per prediction."""
    layers, i = [], 0
    for m in model.net:
        if isinstance(m, torch.nn.Linear):       # ReLU has no parameters and ~no MACs
            i += 1
            a, b = m.in_features, m.out_features
            params = a * b + b
            layers.append({"name": f"linear{i}", "in": a, "out": b, "params": params,
                           "bytes_f32": params * 4, "macs": a * b})
    total_params = sum(L["params"] for L in layers)
    total_f32, total_int8 = total_params * 4, total_params * 1
    return {"layers": layers,
            "total_params": total_params,
            "total_bytes_f32": total_f32,
            "total_bytes_int8": total_int8,
            "total_macs": sum(L["macs"] for L in layers),
            "fits_300kb_sram_f32": total_f32 <= SRAM_BUDGET,
            "fits_300kb_sram_int8": total_int8 <= SRAM_BUDGET}


@app.get("/health")
def health():
    return {"status": "ok", "team": cfg["team"], "params": N_PARAMS, "val_mae": cfg["val_mae"]}


async def run(request: Request, schema):
    t_all = time.perf_counter(); t = {}
    t0 = time.perf_counter()
    try:
        data = schema.model_validate_json(await request.body())
    except ValidationError as e:
        return JSONResponse(status_code=422, content={"error": "bad reading", "detail": json.loads(e.json())})
    t["parse"] = ms(t0)
    rows = data.rows if isinstance(data, Batch) else [data]
    # TODO Mission 3: time each of the three stages below the same way "parse" is timed above,
    # storing them as t["normalise"], t["inference"] and t["build"].
    t0 = time.perf_counter()
    x = (np.array([[getattr(r, f) for f in FEATURES] for r in rows], np.float32) - MU) / SD
    t["normalise"] = ms(t0)
    t0 = time.perf_counter()
    with torch.no_grad():
        y = net(torch.from_numpy(x)).numpy()
    t["inference"] = ms(t0)
    t0 = time.perf_counter()
    days = [round(float(v), 2) for v in np.maximum(y, 0)]
    body = {"days_left": days if isinstance(data, Batch) else days[0]}
    t["build"] = ms(t0)
    t["server_total"] = ms(t_all)
    body["timings_ms"] = t
    return body


@app.post("/predict")
async def predict(request: Request):
    return await run(request, Reading)


@app.post("/predict_batch")
async def predict_batch(request: Request):
    return await run(request, Batch)

"""
FastAPI WebSocket server streaming synthetic options quotes for a vol surface demo.

Stock at $50, strikes $35-$60, expiries 1-5 weeks.
2wk has an earnings event: ATM IV ~60%. Others ~30-35%.
Streams a full snapshot (all strikes × expiries) every 500ms.
"""

import asyncio
import json
import math
import random
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Black-76 pricer (pure Python) ----------

def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x * math.sqrt(0.5))


def black76_price(F: float, K: float, r: float, sigma: float, T: float, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return intrinsic
    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = math.exp(-r * T)
    if is_call:
        return df * (F * _ncdf(d1) - K * _ncdf(d2))
    return df * (K * _ncdf(-d2) - F * _ncdf(-d1))


# ---------- Market parameters ----------

SPOT = 50.0
RATE = 0.05
STRIKES = list(range(35, 61))  # 35..60 inclusive, 26 strikes

# (weeks, atm_iv, smile_curvature, skew)
EXPIRY_PARAMS = [
    {"label": "1wk", "weeks": 1, "atm_iv": 0.30, "curvature": 0.8, "skew": -0.10},
    {"label": "2wk", "weeks": 2, "atm_iv": 0.60, "curvature": 0.4, "skew": -0.05},  # earnings
    {"label": "3wk", "weeks": 3, "atm_iv": 0.35, "curvature": 0.6, "skew": -0.10},
    {"label": "4wk", "weeks": 4, "atm_iv": 0.32, "curvature": 0.5, "skew": -0.08},
    {"label": "5wk", "weeks": 5, "atm_iv": 0.30, "curvature": 0.4, "skew": -0.07},
]


def skewed_iv(F: float, K: float, T: float, atm_iv: float, curvature: float, skew: float) -> float:
    """ATM vol + quadratic smile + linear skew in log-moneyness space."""
    x = math.log(K / F)
    # Scale smile steepness by 1/sqrt(T) so short-dated smiles are steeper
    scale = 1.0 / math.sqrt(max(T, 1 / 365))
    iv = atm_iv + skew * x * scale + curvature * x ** 2 * scale
    return max(iv, 0.05)


def generate_snapshot() -> dict:
    spot = SPOT + random.gauss(0, 0.04)
    F = spot  # approximate forward (no carry for demo)

    quotes = []
    for exp in EXPIRY_PARAMS:
        T = exp["weeks"] * 7 / 365
        # Small stochastic IV perturbation per snapshot
        vol_shock = random.gauss(0, 0.003)
        half_spread_base = 0.012

        for K in STRIKES:
            iv = skewed_iv(F, K, T, exp["atm_iv"] + vol_shock, exp["curvature"], exp["skew"])
            # Wider spreads for deep OTM options
            otm_factor = abs(math.log(K / F))
            half_spread = half_spread_base * (1 + 3 * otm_factor)

            call_mid = black76_price(F, K, RATE, iv, T, True)
            put_mid = black76_price(F, K, RATE, iv, T, False)

            quotes.append({
                "expiry": exp["label"],
                "t_years": round(T, 6),
                "strike": K,
                "call_bid": round(max(call_mid - half_spread, 0.01), 3),
                "call_ask": round(call_mid + half_spread, 3),
                "put_bid": round(max(put_mid - half_spread, 0.01), 3),
                "put_ask": round(put_mid + half_spread, 3),
            })

    return {"spot": round(spot, 3), "timestamp": time.time(), "quotes": quotes}


# ---------- WebSocket endpoint ----------

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            payload = generate_snapshot()
            await ws.send_text(json.dumps(payload))
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, Exception):
        pass

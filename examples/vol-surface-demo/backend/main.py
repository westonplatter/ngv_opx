"""
FastAPI WebSocket server streaming synthetic options trades for a vol surface demo.

Stock at $50, strikes $35-$60, expiries 1-5 weeks.
2wk has an earnings event: ATM IV ~60%. Others ~30-35%.

Protocol (two message types):
  1. On connect, one "snapshot" with the full chain (all strikes × expiries) so the
     client has a complete starting surface.
  2. Then a stream of "trades" messages — each a small burst of individual contracts
     that just "traded", at a price near fair value. Only these move the surface.
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


def _half_spread(F: float, K: float) -> float:
    """Bid/ask half-spread — wider for deep-OTM strikes."""
    otm_factor = abs(math.log(K / F))
    return 0.012 * (1 + 3 * otm_factor)


def generate_snapshot(spot: float = SPOT) -> dict:
    """Full option chain (every strike × expiry) priced off the fair vol surface.

    Sent once on connect so the client has a complete starting surface.
    """
    F = spot  # approximate forward (no carry for demo)
    quotes = []
    for exp in EXPIRY_PARAMS:
        T = exp["weeks"] * 7 / 365
        for K in STRIKES:
            iv = skewed_iv(F, K, T, exp["atm_iv"], exp["curvature"], exp["skew"])
            half_spread = _half_spread(F, K)
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

    return {"type": "snapshot", "spot": round(spot, 3), "timestamp": time.time(), "quotes": quotes}


def simulate_trade(spot: float) -> dict:
    """One contract trading at a price near its fair value.

    Picks a random strike/expiry, trades the OTM option (call if K >= spot, else put)
    to match the smile convention, and fills at mid ± a fraction of the spread to
    mimic someone lifting the ask or hitting the bid.
    """
    exp = random.choice(EXPIRY_PARAMS)
    K = random.choice(STRIKES)
    T = exp["weeks"] * 7 / 365
    is_call = K >= spot

    iv = skewed_iv(spot, K, T, exp["atm_iv"], exp["curvature"], exp["skew"])
    mid = black76_price(spot, K, RATE, iv, T, is_call)
    price = mid + random.choice([-1, 1]) * random.uniform(0, _half_spread(spot, K))

    return {
        "expiry": exp["label"],
        "t_years": round(T, 6),
        "strike": K,
        "is_call": is_call,
        "price": round(max(price, 0.01), 3),
        "size": random.randint(1, 50),
    }


# ---------- WebSocket endpoint ----------

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    spot = SPOT
    try:
        # 1) full snapshot so the client starts with a complete surface
        await ws.send_text(json.dumps(generate_snapshot(spot)))

        # 2) stream small bursts of individual trades
        while True:
            spot += random.gauss(0, 0.02)  # slow spot drift
            trades = [simulate_trade(spot) for _ in range(random.randint(3, 8))]
            await ws.send_text(json.dumps({
                "type": "trades",
                "spot": round(spot, 3),
                "timestamp": time.time(),
                "trades": trades,
            }))
            await asyncio.sleep(0.4)
    except (WebSocketDisconnect, Exception):
        pass

"""
Legendary Pocket Option AI Scanner (single-file, OTC-aware, Actions-ready)

What this does:
- OTC support: maps Pocket Option-like symbols (e.g., EURUSD_otc) to public Yahoo Finance symbols
  and emulates OTC continuity (weekends + sparse liquidity) to maintain continuous analysis.
- Auto mode selection per signal (Quantum, Momentum, Breakout, Mean Reversion)
- Multi-timeframe top-down alignment (1y, 1M, 2M, 1w, 3w, 1d, 8h, 4h)
- Indicator/strategy alignment gating (>=4)
- Institutional filters (sessions + placeholder high-impact events)
- Sentiment integration (TextBlob placeholder)
- AI fusion with high-confidence gating
- Telegram alerts-only (no broker)
- Daily self-training hooks (optional)
- RUN_ONCE mode for GitHub Actions scheduled scans

Notes:
- Public, official OTC feeds for Pocket Option are not provided. This scanner uses Yahoo Finance
  OHLCV for underlying instruments and applies an "OTC emulator" to synthesize weekend/illiquid
  periods, ensuring continuous analysis. Unofficial Pocket Option API repos exist, but they are
  community-driven, not official.

Run:
- Local: python app.py (configure .env)
- GitHub Actions: workflow injects env vars and runs single-scan mode
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import threading
import warnings
warnings.filterwarnings('ignore')

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Config from env
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "False") == "True"
CONF_GATE = int(os.getenv("CONF_GATE", "85"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "20"))
DAILY_TRAIN_HOUR_UTC = int(os.getenv("DAILY_TRAIN_HOUR_UTC", "3"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "60"))
RUN_ONCE = os.getenv("RUN_ONCE", "False") == "True"

# Universe: include OTC aliases (mapped to Yahoo symbols)
ASSETS = [
    # OTC aliases mapped to Yahoo symbols
    "EURUSD_otc", "GBPUSD_otc", "AUDUSD_otc", "USDJPY_otc", "XAUUSD_otc",
    # Direct Yahoo symbols
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
    "GC=F", "SI=F", "CL=F", "BZ=F"  # Gold, Silver, WTI, Brent
]
EXPIRIES = [5, 15, 30, 60, 120, 240]  # minutes
TIMEFRAMES = ["1y", "1M", "2M", "1w", "3w", "1d", "8h", "4h"]
TF_WEIGHTS = {"1y": 4, "1M": 4, "2M": 3, "1w": 3, "3w": 3, "1d": 2, "8h": 2, "4h": 1}

# Optional libs
try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

# ------------------------------------------------------------------------------------
# AI Core: MANDATORY TensorFlow/Keras imports (TFK=True implicitly)
# The failure to import these will now stop the script early, forcing installation.
# ------------------------------------------------------------------------------------
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, BatchNormalization,
                                    Conv1D, MaxPooling1D, MultiHeadAttention, LayerNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

TFK = True
# ------------------------------------------------------------------------------------

# Telegram
try:
    from telegram.ext import Updater, CommandHandler
    TELEGRAM_AVAILABLE = True
except Exception:
    TELEGRAM_AVAILABLE = False

# ------------------------------------------------------------------------------------
# Indicator utilities
# ------------------------------------------------------------------------------------
def ema(series: pd.Series, period: int) -> np.ndarray:
    if len(series) == 0: return np.array([])
    alpha = 2 / (period + 1)
    ema_vals = [series.iloc[0]]
    for v in series.iloc[1:]:
        ema_vals.append(alpha * v + (1 - alpha) * ema_vals[-1])
    return np.array(ema_vals)

def rsi(series: pd.Series, period: int = 14) -> np.ndarray:
    if len(series) < period + 1: return np.array([50.0] * len(series))
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / (down.replace(0, np.nan))
    rsi_vals = 100 - (100 / (1 + rs))
    return rsi_vals.fillna(50.0).values

def macd(series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    ema_fast = pd.Series(ema(series, fast))
    ema_slow = pd.Series(ema(series, slow))
    macd_line = ema_fast - ema_slow
    macd_sig = pd.Series(ema(macd_line, sig))
    return macd_line.values, macd_sig.values

def bollinger_bands(series: pd.Series, period: int = 20, dev: float = 2.0) -> Dict[str, pd.Series]:
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + dev * std
    lower = ma - dev * std
    width = upper - lower
    return {"upper": upper, "lower": lower, "ma": ma, "width": width}

def bollinger_status(series: pd.Series, period=20, dev=2.0) -> Tuple[bool, bool, float]:
    bb = bollinger_bands(series, period, dev)
    upper_break = series.iloc[-1] > bb["upper"].iloc[-1] if pd.notna(bb["upper"].iloc[-1]) else False
    lower_break = series.iloc[-1] < bb["lower"].iloc[-1] if pd.notna(bb["lower"].iloc[-1]) else False
    width = bb["width"].iloc[-1] if pd.notna(bb["width"].iloc[-1]) else 0.0
    return upper_break, lower_break, width

def normalize(val: float, base: Optional[pd.Series] = None) -> float:
    try:
        if base is not None and len(base) > 0:
            return float(min(100.0, max(0.0, 100.0 * abs(val) / (abs(base.iloc[-1]) + 1e-9))))
        return float(min(100.0, max(0.0, val)))
    except Exception:
        return 0.0

def big_body(df: pd.DataFrame, idx: int = -1, ratio: float = 0.6) -> bool:
    high, low = df["high"].iloc[idx], df["low"].iloc[idx]
    body = abs(df["close"].iloc[idx] - df["open"].iloc[idx])
    spread = max(1e-9, high - low)
    return (body / spread) >= ratio

def wick_rejection_ratio(row: pd.Series) -> Dict[str, float]:
    total = max(1e-9, row["high"] - row["low"])
    upper = row["high"] - max(row["open"], row["close"])
    lower = min(row["open"], row["close"]) - row["low"]
    return {"upper": upper / total, "lower": lower / total}

def rolling_range(high: pd.Series, low: pd.Series, win: int = 20) -> Dict[str, float]:
    return {"high": high.rolling(win).max().iloc[-1], "low": low.rolling(win).min().iloc[-1]}

def volatility(df: pd.DataFrame, period: int = 14) -> float:
    tr = (df["high"] - df["low"]).rolling(period).mean()
    return float(tr.iloc[-1] if pd.notna(tr.iloc[-1]) else 0.0)

def trend_strength_score(fast: np.ndarray, slow: np.ndarray, macd_line: Optional[np.ndarray] = None) -> float:
    if len(fast) < 2 or len(slow) < 2: return 0.0
    slope = (fast[-1] - fast[-2]) - (slow[-1] - slow[-2])
    score = normalize(slope)
    if macd_line is not None and len(macd_line) >= 2:
        score += normalize(macd_line[-1] - macd_line[-2])
    return min(100.0, score)

# ------------------------------------------------------------------------------------
# OTC-aware Real Data Provider (Yahoo Finance + Emulator)
# ------------------------------------------------------------------------------------
class SymbolMapper:
    MAP = {
        "EURUSD_otc": ("EURUSD=X", True),
        "GBPUSD_otc": ("GBPUSD=X", True),
        "AUDUSD_otc": ("AUDUSD=X", True),
        "USDJPY_otc": ("USDJPY=X", True),
        "XAUUSD_otc": ("GC=F", True),
    }

    @staticmethod
    def resolve(symbol: str) -> Tuple[str, bool]:
        if symbol in SymbolMapper.MAP:
            return SymbolMapper.MAP[symbol]
        return symbol, False  # direct Yahoo

class OTCEmulator:
    def __init__(self, seed: int = 123):
        self.rng = np.random.default_rng(seed)

    def synthesize(self, df: pd.DataFrame, tf: str, limit: int) -> pd.DataFrame:
        if df is None or df.empty:
            return self._generate_series(tf, limit, base_price=1.0)
        produced = df.copy()
        last_ts = produced["timestamp"].iloc[-1]
        step = self._freq_for_tf(tf)
        now = datetime.utcnow()
        timestamps = []
        t = last_ts + step
        while len(timestamps) < max(0, limit - len(produced)):
            timestamps.append(t)
            t += step
            if t > now: break
        if timestamps:
            base = float(produced["close"].iloc[-1])
            synth = self._random_walk_series(base, len(timestamps))
            extra = pd.DataFrame({
                "timestamp": timestamps,
                "open": synth["open"],
                "high": synth["high"],
                "low": synth["low"],
                "close": synth["close"],
                "volume": synth["volume"]
            })
            produced = pd.concat([produced, extra], ignore_index=True)
        produced = self._add_noise(produced, scale=0.0005)
        return produced.tail(limit)

    def _freq_for_tf(self, tf: str) -> timedelta:
        return {
            "4h": timedelta(hours=4),
            "8h": timedelta(hours=8),
            "1d": timedelta(days=1),
            "3w": timedelta(weeks=3),
            "1w": timedelta(weeks=1),
            "2M": timedelta(days=60),
            "1M": timedelta(days=30),
            "1y": timedelta(days=365)
        }.get(tf, timedelta(hours=1))

    def _random_walk_series(self, base: float, n: int) -> Dict[str, np.ndarray]:
        steps = self.rng.normal(loc=0.0, scale=base * 0.0008, size=n)
        closes = base + np.cumsum(steps)
        opens = np.concatenate([[base], closes[:-1]])
        highs = closes + np.abs(self.rng.normal(0.0, base * 0.0005, size=n))
        lows = closes - np.abs(self.rng.normal(0.0, base * 0.0005, size=n))
        vols = self.rng.integers(80, 800, size=n)
        return {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols}

    def _generate_series(self, tf: str, limit: int, base_price: float) -> pd.DataFrame:
        step = self._freq_for_tf(tf)
        end = datetime.utcnow()
        idx = [end - step * (limit - i) for i in range(limit)]
        rw = self._random_walk_series(base_price, limit)
        return pd.DataFrame({
            "timestamp": idx,
            "open": rw["open"],
            "high": rw["high"],
            "low": rw["low"],
            "close": rw["close"],
            "volume": rw["volume"]
        })

    def _add_noise(self, df: pd.DataFrame, scale: float = 0.0003) -> pd.DataFrame:
        noisy = df.copy()
        noise = self.rng.normal(0.0, scale, size=len(noisy))
        noisy["close"] = noisy["close"] * (1.0 + noise)
        noisy["high"] = np.maximum(noisy["high"], noisy["close"])
        noisy["low"] = np.minimum(noisy["low"], noisy["close"])
        return noisy

class RealDataProvider:
    """
    Yahoo Finance OHLCV with OTC emulation layer.
    """
    def __init__(self):
        import yfinance as yf
        self.yf = yf
        self.otc = OTCEmulator(seed=321)

    def fetch_ohlcv(self, symbol: str, timeframe: str = "4h", limit: int = 1000) -> Optional[pd.DataFrame]:
        base_symbol, is_otc = SymbolMapper.resolve(symbol)
        interval = self._tf_to_yf_interval(timeframe)
        period = self._default_period_for_interval(interval)

        try:
            df = self.yf.download(base_symbol, period=period, interval=interval, progress=False)
            if df is None or df.empty:
                return None
            df = df.rename(columns={"Open": "open","High":"high","Low":"low","Close":"close","Volume":"volume"})
            df.reset_index(inplace=True)
            ts_col = "Datetime" if "Datetime" in df.columns else "Date"
            df["timestamp"] = pd.to_datetime(df[ts_col])
            df = df[["timestamp","open","high","low","close","volume"]]
            df = df.tail(limit)
        except Exception:
            return None

        if is_otc:
            df = self.otc.synthesize(df, timeframe, limit)
        return df

    def _tf_to_yf_interval(self, tf: str) -> str:
        return {
            "4h": "1h",
            "8h": "1h",
            "1d": "1d",
            "1w": "1d",
            "3w": "1d",
            "1M": "1d",
            "2M": "1d",
            "1y": "1d",
        }.get(tf, "1h")

    def _default_period_for_interval(self, interval: str) -> str:
        return {"1h": "3mo", "1d": "2y"}.get(interval, "6mo")

# ------------------------------------------------------------------------------------
# Institutional timetable + sentiment
# ------------------------------------------------------------------------------------
SESSIONS = [
    {"name": "Tokyo", "start": (0, 0), "end": (9, 0)},
    {"name": "London", "start": (7, 0), "end": (16, 0)},
    {"name": "New York", "start": (13, 0), "end": (22, 0)}
]

def current_sessions(now_utc: datetime) -> List[str]:
    active = []
    for s in SESSIONS:
        start = now_utc.replace(hour=s["start"][0], minute=s["start"][1], second=0, microsecond=0)
        end = now_utc.replace(hour=s["end"][0], minute=s["end"][1], second=0, microsecond=0)
        if start <= now_utc <= end:
            active.append(s["name"])
    return active

def fetch_economic_calendar(now_utc: datetime) -> List[Dict]:
    return []

def institutional_block(calendar_events: List[Dict], now_utc: datetime) -> bool:
    return any(abs((ev["time"] - now_utc).total_seconds()) <= 1800 and ev.get("impact") == "High" for ev in calendar_events)

def fetch_headlines(asset: str) -> List[str]:
    return []

def sentiment_score(headlines: List[str]) -> float:
    if TextBlob is None or not headlines: return 0.0
    vals = []
    for h in headlines:
        try:
            vals.append(TextBlob(h).sentiment.polarity)
        except Exception:
            pass
    return sum(vals)/len(vals) if vals else 0.0

# ------------------------------------------------------------------------------------
# Strategy modules and engines
# ------------------------------------------------------------------------------------
def find_swings(high: pd.Series, low: pd.Series, lookback: int = 20) -> Dict:
    piv_high = high.rolling(lookback).max().iloc[-1]
    piv_low = low.rolling(lookback).min().iloc[-1]
    return {"swing_high": piv_high, "swing_low": piv_low}

def locate_ob_zones(df: pd.DataFrame, min_body_ratio: float = 0.5) -> List[Dict]:
    zones = []
    window = df.iloc[-50:]
    for i in range(2, len(window)):
        body = abs(window["close"].iloc[i] - window["open"].iloc[i])
        spread = max(1e-9, window["high"].iloc[i] - window["low"].iloc[i])
        if body / spread >= min_body_ratio:
            direction = "demand" if window["close"].iloc[i] > window["open"].iloc[i] else "supply"
            zones.append({"index": i, "type": direction, "price": float(window["open"].iloc[i])})
    return zones

def ob_scoring(df: pd.DataFrame, zones: List[Dict]) -> Tuple[float, str]:
    if not zones: return 0.0, "NEUTRAL"
    last = zones[-1]
    price = df["close"].iloc[-1]
    dist = abs(price - last["price"])
    score = 100.0 - normalize(dist, df["close"])
    direction = "BUY" if last["type"] == "demand" else "SELL"
    return score, direction

def detect_order_blocks(df) -> Dict:
    zones = locate_ob_zones(df)
    score, direction = ob_scoring(df, zones)
    return {"signal": direction, "score": score, "reason": "OB"}

def recent_swing_high_low(df: pd.DataFrame, window: int = 50) -> Tuple[float, float]:
    return float(df["high"].iloc[-window:].max()), float(df["low"].iloc[-window:].min())

def detect_bos(df) -> Dict:
    sh, sl = recent_swing_high_low(df, 50)
    up = df["close"].iloc[-1] > sh and big_body(df, -1)
    down = df["close"].iloc[-1] < sl and big_body(df, -1)
    signal = "BUY" if up else ("SELL" if down else "NEUTRAL")
    score = normalize(abs(df["close"].iloc[-1] - (sh if up else sl)))
    return {"signal": signal, "score": score, "reason": "BOS"}

def find_fvg(df: pd.DataFrame) -> Tuple[bool, bool, float]:
    if len(df) < 3: return False, False, 0.0
    n = len(df) - 1
    up = df["low"].iloc[n] > df["high"].iloc[n-2]
    down = df["high"].iloc[n] < df["low"].iloc[n-2]
    size = max(0.0, (df["low"].iloc[n] - df["high"].iloc[n-2])) if up else max(0.0, (df["low"].iloc[n-2] - df["high"].iloc[n])) if down else 0.0
    return up, down, float(size)

def detect_fvg(df) -> Dict:
    up, down, size = find_fvg(df)
    signal = "BUY" if up else ("SELL" if down else "NEUTRAL")
    score = normalize(size, df["close"])
    return {"signal": signal, "score": score, "reason": "FVG"}

def ema_macd_signal(df, fast=12, slow=26, sig=9) -> Dict:
    macd_line, macd_sig = macd(df["close"], fast, slow, sig)
    ef = ema(df["close"], fast); es = ema(df["close"], slow)
    if len(ef) < 2 or len(es) < 2 or len(macd_line) < 1 or len(macd_sig) < 1:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA+MACD"}
    cross_up = ef[-1] > es[-1] and ef[-2] <= es[-2]
    cross_down = ef[-1] < es[-1] and ef[-2] >= es[-2]
    macd_up = macd_line[-1] > macd_sig[-1]
    macd_down = macd_line[-1] < macd_sig[-1]
    signal = "BUY" if (cross_up and macd_up) else ("SELL" if (cross_down and macd_down) else "NEUTRAL")
    score = trend_strength_score(ef, es, macd_line)
    return {"signal": signal, "score": score, "reason": "EMA+MACD"}

def supertrend(df: pd.DataFrame, atr_period=10, multiplier=3.0) -> str:
    tr = (df["high"] - df["low"]).rolling(atr_period).mean()
    mid = (df["high"] + df["low"]) / 2.0
    band_up = mid + multiplier * tr
    band_down = mid - multiplier * tr
    close = df["close"].iloc[-1]
    if pd.notna(band_up.iloc[-1]) and close > band_up.iloc[-1]: return "up"
    if pd.notna(band_down.iloc[-1]) and close < band_down.iloc[-1]: return "down"
    return "flat"

def supertrend_bollinger_signal(df) -> Dict:
    st = supertrend(df, 10, 3.0)
    bb_up, bb_down, width = bollinger_status(df["close"], 20, 2.0)
    signal = "BUY" if (st == "up" and bb_up) else ("SELL" if (st == "down" and bb_down) else "NEUTRAL")
    score = normalize(width) + (10 if st != "flat" else 0)
    return {"signal": signal, "score": score, "reason": "ST+BB"}

def volume_smart_money(df) -> Dict:
    vol = df["volume"]; v_avg = vol.rolling(50).mean()
    v_spike = vol.iloc[-1] > 1.8 * (v_avg.iloc[-1] if pd.notna(v_avg.iloc[-1]) else 0.0)
    spread = (df["high"] - df["low"]).iloc[-1]
    body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
    accum = v_spike and (body / max(1e-9, spread) < 0.3) and df["close"].iloc[-1] > df["open"].iloc[-1]
    distrib = v_spike and (body / max(1e-9, spread) < 0.3) and df["close"].iloc[-1] < df["open"].iloc[-1]
    signal = "BUY" if accum else ("SELL" if distrib else "NEUTRAL")
    score = normalize(volatility(df)) + (20 if v_spike else 0)
    return {"signal": signal, "score": score, "reason": "Vol+Smart"}

def momentum_break(df) -> Dict:
    rng = rolling_range(df["high"], df["low"], 20)
    up = df["close"].iloc[-1] > rng["high"] and big_body(df, -1)
    down = df["close"].iloc[-1] < rng["low"] and big_body(df, -1)
    signal = "BUY" if up else ("SELL" if down else "NEUTRAL")
    score = normalize(abs(df["close"].iloc[-1] - (rng["high"] if up else rng["low"])))
    return {"signal": signal, "score": score, "reason": "Momentum"}

def volume_spike(df) -> Dict:
    vol = df["volume"]
    spike = vol.iloc[-1] > 2.0 * vol.rolling(50).mean().iloc[-1]
    up = spike and df["close"].iloc[-1] > df["open"].iloc[-1]
    down = spike and df["close"].iloc[-1] < df["open"].iloc[-1]
    signal = "BUY" if up else ("SELL" if down else "NEUTRAL")
    score = normalize(vol.iloc[-1])
    return {"signal": signal, "score": score, "reason": "VolSpike"}

def rsi_oversold(df) -> Dict:
    rv = rsi(df["close"], 14)
    if len(rv) == 0: return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI"}
    signal = "BUY" if rv[-1] < 30 else ("SELL" if rv[-1] > 70 else "NEUTRAL")
    score = 100 - abs(rv[-1] - 50)
    return {"signal": signal, "score": score, "reason": "RSI"}

def ema_golden_cross(df) -> Dict:
    e50 = ema(df["close"], 50); e200 = ema(df["close"], 200)
    if len(e50) < 2 or len(e200) < 2:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA50/200"}
    up = e50[-1] > e200[-1] and e50[-2] <= e200[-2]
    down = e50[-1] < e200[-1] and e50[-2] >= e200[-2]
    signal = "BUY" if up else ("SELL" if down else "NEUTRAL")
    score = trend_strength_score(e50, e200)
    return {"signal": signal, "score": score, "reason": "EMA50/200"}

def support_resistance_levels(df: pd.DataFrame, lookback: int = 200) -> Dict:
    window = df.iloc[-lookback:]
    return {"support": [float(window["low"].min())], "resistance": [float(window["high"].max())]}

def sr_breakout(df) -> Dict:
    levels = support_resistance_levels(df, 200)
    up = df["close"].iloc[-1] > levels["resistance"][-1]
    down = df["close"].iloc[-1] < levels["support"][-1]
    signal = "BUY" if up else ("SELL" if down else "NEUTRAL")
    score = normalize(abs(df["close"].iloc[-1] - (levels["resistance"][-1] if up else levels["support"][-1])))
    return {"signal": signal, "score": score, "reason": "SRBreakout"}

def volume_confirmation(df) -> Dict:
    vol = df["volume"]
    conf = vol.iloc[-1] > vol.rolling(30).mean().iloc[-1]
    up = conf and df["close"].iloc[-1] > df["open"].iloc[-1]
    down = conf and df["close"].iloc[-1] < df["open"].iloc[-1]
    signal = "BUY" if up else ("SELL" if down else "NEUTRAL")
    score = normalize(vol.iloc[-1])
    return {"signal": signal, "score": score, "reason": "VolConfirm"}

def bollinger_breakout(df) -> Dict:
    up, down, width = bollinger_status(df["close"], 20, 2.0)
    signal = "BUY" if up else ("SELL" if down else "NEUTRAL")
    score = normalize(width)
    return {"signal": signal, "score": score, "reason": "BBreakout"}

def rsi_mean_reversion(df) -> Dict:
    rv = rsi(df["close"], 14)
    if len(rv) == 0: return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI MR"}
    signal = "BUY" if rv[-1] < 30 else ("SELL" if rv[-1] > 70 else "NEUTRAL")
    score = abs(50 - rv[-1])
    return {"signal": signal, "score": score, "reason": "RSI MR"}

def bollinger_touch(df) -> Dict:
    bb = bollinger_bands(df["close"])
    u = bb["upper"].iloc[-1]; l = bb["lower"].iloc[-1]
    touch_low = df["low"].iloc[-1] <= l if pd.notna(l) else False
    touch_high = df["high"].iloc[-1] >= u if pd.notna(u) else False
    signal = "BUY" if touch_low else ("SELL" if touch_high else "NEUTRAL")
    score = normalize(bb["width"].iloc[-1] if pd.notna(bb["width"].iloc[-1]) else 0.0)
    return {"signal": signal, "score": score, "reason": "BB Touch"}

def detect_volume_divergence(close: pd.Series, vol: pd.Series) -> Dict:
    last_high = close.iloc[-50:].max(); last_low = close.iloc[-50:].min()
    vol_high = vol.iloc[-50:].max()
    bearish = close.iloc[-1] >= last_high and vol.iloc[-1] < 0.8 * vol_high
    bullish = close.iloc[-1] <= last_low and vol.iloc[-1] < 0.8 * vol_high
    strength = 70.0 if (bearish or bullish) else 0.0
    return {"bearish": bearish, "bullish": bullish, "strength": strength}

def volume_divergence(df) -> Dict:
    div = detect_volume_divergence(df["close"], df["volume"])
    signal = "SELL" if div["bearish"] else ("BUY" if div["bullish"] else "NEUTRAL")
    score = div["strength"]
    return {"signal": signal, "score": score, "reason": "VolDiv"}

ENGINE_MODULES = {
    "quantum": [
        ("strategy", detect_order_blocks),
        ("strategy", detect_bos),
        ("strategy", detect_fvg),
        ("indicator", ema_macd_signal),
        ("indicator", supertrend_bollinger_signal),
        ("indicator", volume_smart_money),
    ],
    "momentum": [
        ("strategy", momentum_break),
        ("indicator", volume_spike),
        ("indicator", rsi_oversold),
        ("indicator", ema_golden_cross),
    ],
    "breakout": [
        ("strategy", sr_breakout),
        ("indicator", volume_confirmation),
        ("indicator", bollinger_breakout),
    ],
    "meanreversion": [
        ("strategy", rsi_mean_reversion),
        ("indicator", bollinger_touch),
        ("indicator", volume_divergence),
    ],
}

def run_engine_modules(df: pd.DataFrame, engine: str) -> List[Dict]:
    results = []
    for tag, fn in ENGINE_MODULES.get(engine, []):
        try:
            out = fn(df)
            out["tag"] = tag
            results.append(out)
        except Exception:
            pass
    return results

def choose_engine(df: pd.DataFrame) -> str:
    try:
        if detect_order_blocks(df)["signal"] != "NEUTRAL" or detect_bos(df)["signal"] != "NEUTRAL":
            return "quantum"
        if momentum_break(df)["signal"] != "NEUTRAL" or volume_spike(df)["signal"] != "NEUTRAL":
            return "momentum"
        if sr_breakout(df)["signal"] != "NEUTRAL" or bollinger_breakout(df)["signal"] != "NEUTRAL":
            return "breakout"
        if rsi_mean_reversion(df)["signal"] != "NEUTRAL" or bollinger_touch(df)["signal"] != "NEUTRAL":
            return "meanreversion"
    except Exception:
        pass
    return "quantum"

def align_by_tag(module_results: List[Dict], tag: str, min_align: int = 4) -> Dict:
    votes = {"BUY": [], "SELL": []}
    for r in module_results:
        if r.get("tag") == tag and r["signal"] in votes:
            votes[r["signal"]].append(r["score"])
    buy_count, sell_count = len(votes["BUY"]), len(votes["SELL"])
    if buy_count >= min_align and buy_count >= sell_count:
        return {"signal": "BUY", "count": buy_count, "avg_score": sum(votes["BUY"]) / buy_count}
    if sell_count >= min_align and sell_count > buy_count:
        return {"signal": "SELL", "count": sell_count, "avg_score": sum(votes["SELL"]) / sell_count}
    return {"signal": "NEUTRAL", "count": max(buy_count, sell_count), "avg_score": 0.0}

def topdown_alignment(tf_signals: Dict[str, Dict], min_tf_align: int = 4) -> Dict:
    s_accum = {"BUY": 0.0, "SELL": 0.0}
    c_accum = {"BUY": 0, "SELL": 0}
    for tf, res in tf_signals.items():
        if res["signal"] in s_accum:
            s_accum[res["signal"]] += TF_WEIGHTS.get(tf, 1) * (res["confidence"] / 100.0)
            c_accum[res["signal"]] += 1
    best_dir = max(s_accum, key=s_accum.get)
    if c_accum[best_dir] >= min_tf_align and s_accum[best_dir] > 0:
        conf = min(100.0, 100.0 * s_accum[best_dir] / sum(TF_WEIGHTS.values()))
        return {"signal": best_dir, "tf_count": c_accum[best_dir], "confidence": conf}
    return {"signal": "NEUTRAL", "tf_count": 0, "confidence": 0.0}

# ------------------------------------------------------------------------------------
# AI core (lightweight; TF required now)
# ------------------------------------------------------------------------------------
class AdvancedKerasAI:
    def __init__(self, lookback: int = 100, expiry_minutes: int = 60):
        self.lookback = lookback
        self.expiry_minutes = expiry_minutes
        self.model = None
        self.is_trained = False
        self.scaler = StandardScaler()

    def build_model(self, input_shape: Tuple[int, int]) -> Optional[Model]:
        # TFK is implicitly True now. No need for the check, but keeping it concise.
        inputs = Input(shape=input_shape)
        x = Conv1D(64, 3, activation="relu", padding="same")(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        attn = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
        x = LayerNormalization()(x + attn)
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.2)(x)
        direction = Dense(3, activation="softmax", name="direction")(x)
        confidence = Dense(1, activation="sigmoid", name="confidence")(x)
        regime = Dense(5, activation="softmax", name="regime")(x)
        model = Model(inputs=inputs, outputs=[direction, confidence, regime])
        model.compile(
            optimizer=Adam(1e-4),
            loss={"direction": "categorical_crossentropy", "confidence": "mse", "regime": "categorical_crossentropy"},
            loss_weights={"direction": 2.0, "confidence": 1.0, "regime": 0.7},
            metrics={"direction": "accuracy", "regime": "accuracy"},
        )
        return model

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        feats = []
        for i in range(len(df)):
            row = df.iloc[i]
            basic = [row["open"], row["high"], row["low"], row["close"], row.get("volume", 0)]
            rv = rsi(df["close"].iloc[:i+1], 14)
            m_line, m_sig = macd(df["close"].iloc[:i+1])
            bb = bollinger_bands(df["close"].iloc[:i+1])
            bw = bb["width"].iloc[-1] if len(bb["width"]) and pd.notna(bb["width"].iloc[-1]) else 0.0
            feats.append(basic + [
                rv[-1] if len(rv) else 50.0,
                m_line[-1] if len(m_line) else 0.0,
                m_sig[-1] if len(m_sig) else 0.0,
                bw
            ])
        X = np.array(feats)
        # Scaler is now guaranteed to exist
        X = self.scaler.fit_transform(X)
        return X

    def train(self, df: pd.DataFrame, epochs: int = 20, validation_split: float = 0.2) -> bool:
        if len(df) < self.lookback + 10: return False
        X_all = self.prepare_features(df)
        X, y_dir, y_conf, y_reg = [], [], [], []
        for i in range(self.lookback, len(df)-1):
            X.append(X_all[i-self.lookback:i])
            chg = (df["close"].iloc[i+1] - df["close"].iloc[i]) / max(1e-9, df["close"].iloc[i]) * 100
            if chg > 0.05: dir_label = [0, 0, 1]
            elif chg < -0.05: dir_label = [1, 0, 0]
            else: dir_label = [0, 1, 0]
            y_dir.append(dir_label)
            y_conf.append([min(abs(chg)/2.0, 1.0)])
            y_reg.append([0,0,1,0,0])  # placeholder regime
        X = np.array(X)
        y_dir, y_conf, y_reg = np.array(y_dir), np.array(y_conf), np.array(y_reg)
        if self.model is None:
            self.model = self.build_model((X.shape[1], X.shape[2]))
        if self.model is None: return False
        early = EarlyStopping(monitor="val_direction_accuracy", patience=6, restore_best_weights=True, mode="max")
        reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0)
        self.model.fit(X, {"direction": y_dir, "confidence": y_conf, "regime": y_reg},
                       epochs=epochs, batch_size=32, validation_split=validation_split,
                       callbacks=[early, reduce], verbose=0)
        self.is_trained = True
        return True

    def predict(self, df: pd.DataFrame, expiry_minutes: Optional[int] = None) -> Dict:
        if not self.is_trained or self.model is None or len(df) < self.lookback:
            # Fallback is still needed, but AI training/loading should succeed now
            macd_line, macd_sig = macd(df["close"])
            dir_prob = 0.5
            if len(macd_line) and len(macd_sig):
                dir_prob = 0.5 + 0.5 * np.tanh((macd_line[-1] - macd_sig[-1]) * 50)
            signal = "CALL" if dir_prob > 0.6 else "PUT" if dir_prob < 0.4 else "NEUTRAL"
            conf = float(abs(dir_prob - 0.5) * 200)
            return {"signal": signal, "confidence": conf, "ai_score": conf,
                    "market_regime": "UNKNOWN", "regime_confidence": 0.0,
                    "probabilities": {"PUT": (1-dir_prob)*100, "NEUTRAL": (1-abs(dir_prob-0.5)*2)*100, "CALL": dir_prob*100}}
        feats = self.prepare_features(df)
        X = feats[-self.lookback:].reshape(1, self.lookback, -1)
        preds = self.model.predict(X, verbose=0)
        dir_probs = preds[0][0]; conf = float(preds[1][0][0]); reg_probs = preds[2][0]
        idx = int(np.argmax(dir_probs))
        directions = ["PUT", "NEUTRAL", "CALL"]
        signal = directions[idx]
        return {
            "signal": signal,
            "confidence": float(conf*100),
            "ai_score": float(dir_probs[idx]*conf*100),
            "market_regime": "UNKNOWN",
            "regime_confidence": float(np.max(reg_probs)*100) if len(reg_probs)==5 else 0.0,
            "probabilities": {"PUT": float(dir_probs[0]*100), "NEUTRAL": float(dir_probs[1]*100), "CALL": float(dir_probs[2]*100)}
        }

    def save_model(self, path: str):
        if self.model is not None:
            self.model.save(path)

    def load_model(self, path: str) -> bool:
        try:
            self.model = load_model(path)
            self.is_trained = True
            return True
        except Exception:
            return False

class RobustSignalGenerator:
    def __init__(self):
        self.ai_models: Dict[int, AdvancedKerasAI] = {}

    def ensure_model(self, expiry: int):
        if expiry not in self.ai_models:
            self.ai_models[expiry] = AdvancedKerasAI(lookback=100, expiry_minutes=expiry)

    def train_multi_expiry(self, df: pd.DataFrame, expiries: List[int]):
        for e in expiries:
            self.ensure_model(e)
            self.ai_models[e].train(df, epochs=20, validation_split=0.2)

    def predict(self, df: pd.DataFrame, expiry: int) -> Dict:
        self.ensure_model(expiry)
        return self.ai_models[expiry].predict(df, expiry_minutes=expiry)

# ------------------------------------------------------------------------------------
# Multi-timeframe orchestration
# ------------------------------------------------------------------------------------
class MultiTFData:
    def __init__(self, provider):
        self.dp = provider

    def fetch_asset_all_tfs(self, symbol: str, limit_map: Optional[Dict[str, int]] = None) -> Dict[str, Optional[pd.DataFrame]]:
        data = {}
        for tf in TIMEFRAMES:
            try:
                df = self.dp.fetch_ohlcv(symbol, timeframe=tf, limit=(limit_map or {}).get(tf, 1000))
                data[tf] = df
            except Exception:
                data[tf] = None
        return data

    def valid(self, tf_data: Dict[str, Optional[pd.DataFrame]]) -> bool:
        return all(tf in tf_data and tf_data[tf] is not None and len(tf_data[tf]) >= 200 for tf in TIMEFRAMES)

def analyze_asset(asset: str,
                  tf_data: Dict[str, pd.DataFrame],
                  rsg: RobustSignalGenerator,
                  expiry: int,
                  min_ind_align: int = 4,
                  min_strat_align: int = 4,
                  min_tf_align: int = 4,
                  conf_gate: int = 85) -> Dict:

    tf_signals: Dict[str, Dict] = {}
    for tf in TIMEFRAMES:
        df = tf_data.get(tf)
        if df is None or len(df) < 200:
            tf_signals[tf] = {"signal": "NEUTRAL", "confidence": 0.0}
            continue

        engine = choose_engine(df)
        mods = run_engine_modules(df, engine)
        ind_align = align_by_tag(mods, "indicator", min_align=min_ind_align)
        strat_align = align_by_tag(mods, "strategy", min_align=min_strat_align)

        if ind_align["signal"] == strat_align["signal"] and ind_align["signal"] != "NEUTRAL":
            conf = (ind_align["avg_score"] + strat_align["avg_score"]) / 2
            tf_signals[tf] = {"signal": ind_align["signal"], "confidence": float(conf)}
        else:
            tf_signals[tf] = {"signal": "NEUTRAL", "confidence": 0.0}

    td = topdown_alignment(tf_signals, min_tf_align=min_tf_align)
    base_df = tf_data.get("4h") if tf_data.get("4h") is not None else next((d for d in tf_data.values() if d is not None), None)
    ai_res = rsg.predict(base_df, expiry) if base_df is not None else {"signal": "NEUTRAL", "confidence": 0.0, "ai_score": 0.0}
    ai_map = {"CALL": "BUY", "PUT": "SELL", "NEUTRAL": "NEUTRAL"}
    ai_dir = ai_map.get(ai_res["signal"], "NEUTRAL")

    now_utc = datetime.utcnow()
    sessions = current_sessions(now_utc)
    calendar = fetch_economic_calendar(now_utc)
    headlines = fetch_headlines(asset)
    sent = sentiment_score(headlines)

    agree = (td["signal"] == ai_dir and td["signal"] != "NEUTRAL")
    combined_conf = min(100.0, 0.5 * td["confidence"] + 0.5 * ai_res["confidence"])

    # Legendary filters
    if any(ev.get("impact") == "High" and abs((ev["time"] - now_utc).total_seconds()) <= 1800 for ev in calendar):
        return _neutral_result(asset, expiry, td, ai_res, tf_signals, sessions, "High-impact event window")

    if td["signal"] == "BUY" and sent < -0.2:
        return _neutral_result(asset, expiry, td, ai_res, tf_signals, sessions, "Negative sentiment vs BUY")
    if td["signal"] == "SELL" and sent > 0.2:
        return _neutral_result(asset, expiry, td, ai_res, tf_signals, sessions, "Positive sentiment vs SELL")

    if not (("London" in sessions) or ("New York" in sessions)):
        return _neutral_result(asset, expiry, td, ai_res, tf_signals, sessions, "Inactive session")

    if td["confidence"] < 70:
        return _neutral_result(asset, expiry, td, ai_res, tf_signals, sessions, "Low volatility/weak TD")

    if agree and combined_conf >= conf_gate and ai_res.get("confidence", 0.0) >= 85.0:
        reason = f"Legendary alignment | Sentiment={sent:.2f}"
        return {
            "asset": asset, "signal": td["signal"], "confidence": float(combined_conf),
            "expiry": expiry, "topdown": td, "ai": ai_res, "tf_signals": tf_signals,
            "sessions": sessions, "reason": reason
        }

    return _neutral_result(asset, expiry, td, ai_res, tf_signals, sessions, "No alignment")

def _neutral_result(asset, expiry, td, ai_res, tf_signals, sessions, reason):
    return {
        "asset": asset, "signal": "NEUTRAL", "confidence": 0.0, "expiry": expiry,
        "topdown": td, "ai": ai_res, "tf_signals": tf_signals, "sessions": sessions, "reason": reason
    }

def format_alert(res: Dict) -> str:
    return (
        f"🔥 Legendary Pocket Option Signal\n"
        f"• Asset: {res['asset']}\n"
        f"• Direction: {res['signal']}\n"
        f"• Expiry: {res['expiry']}m\n"
        f"• Confidence: {res['confidence']:.1f}%\n"
        f"• Top-Down: {res['topdown']['signal']} ({res['topdown']['tf_count']} TFs aligned)\n"
        f"• AI: {res['ai'].get('signal', 'NEUTRAL')} ({res['ai'].get('confidence', 0):.1f}%)\n"
        f"• Sessions: {', '.join(res.get('sessions', [])) or 'None'}\n"
        f"• Reason: {res.get('reason', '')}"
    )

# ------------------------------------------------------------------------------------
# Telegram service
# ------------------------------------------------------------------------------------
class TelegramService:
    def __init__(self, token: str, chat_id: str):
        self.available = TELEGRAM_AVAILABLE and bool(token) and bool(chat_id)
        self.token = token
        self.chat_id = chat_id
        self.updater = None
        if self.available:
            self.updater = Updater(token, use_context=True)
            dp = self.updater.dispatcher
            dp.add_handler(CommandHandler("start", self._start))
            dp.add_handler(CommandHandler("status", self._status))
            dp.add_handler(CommandHandler("train", self._train))
            dp.add_handler(CommandHandler("signal", self._signal))
        else:
            print("Telegram not available or missing token/chat_id. Alerts will print to stdout.")
        self.orchestrator = None

    def start(self):
        if not self.available: return
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        self.updater.start_polling()
        self.updater.idle()

    def send(self, text: str):
        if self.available:
            try:
                self.updater.bot.send_message(chat_id=self.chat_id, text=text)
            except Exception as e:
                print("Telegram send error:", e)
        else:
            print(text)

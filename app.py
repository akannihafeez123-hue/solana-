"""
🔮 ULTIMATE LEGENDARY Pocket Option AI Scanner — V3.3
- Async Telegram bot (python-telegram-bot v20+): /start, /status, /threshold
- Runtime install for TensorFlow & scikit-learn to avoid import errors
- Auto mode selection based on market regime
- Per-asset thresholds for volatility and filter sensitivity
- Alerts only when confidence ≥ threshold (global or per-asset via /threshold)
"""

import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import threading
import warnings
from flask import Flask, jsonify
from collections import Counter

warnings.filterwarnings('ignore')

# -------------------------------
# Runtime installer for TensorFlow & scikit-learn
# -------------------------------
def ensure_packages():
    pkgs = [
        ("tensorflow", "tensorflow==2.14.0"),
        ("sklearn", "scikit-learn==1.3.2")
    ]
    for mod_name, pkg in pkgs:
        try:
            __import__(mod_name)
        except ImportError:
            print(f"📦 Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

ensure_packages()

try:
    import tensorflow as tf  # noqa: F401
    from sklearn.preprocessing import StandardScaler  # noqa: F401
    TFK = True
except Exception:
    TFK = False

# -------------------------------
# Env, secrets, config
# -------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
ADMIN_IDS = []
_admin_env = os.getenv("ADMIN_IDS", "")
if _admin_env:
    for x in _admin_env.split(","):
        x = x.strip()
        if x.isdigit():
            ADMIN_IDS.append(int(x))
if not ADMIN_IDS and CHAT_ID and CHAT_ID.isdigit():
    ADMIN_IDS = [int(CHAT_ID)]

LEGENDARY_GATE = int(os.getenv("LEGENDARY_GATE", "95"))
GLOBAL_THRESHOLD = LEGENDARY_GATE
ASSET_THRESHOLD_OVERRIDE: Dict[str, float] = {}

COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "15"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "45"))
RUN_ONCE = os.getenv("RUN_ONCE", "False") == "True"
PORT = int(os.getenv("PORT") or os.getenv("port") or "8080")

ASSETS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "GC=F"]
EXPIRIES = [5, 15, 30, 60]
ALL_TIMEFRAMES = ["1M", "1w", "1d", "4h", "1h"]

SIGNAL_HISTORY: Dict[str, Dict] = {}
LAST_LEGENDARY_ALERT: Dict[str, datetime] = {}

app = Flask(__name__)

# -------------------------------
# Telegram bot (v20+ async Application API)
# -------------------------------
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

class LegendaryTelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.app: Optional[Application] = None

    def start(self):
        if not self.token or not self.chat_id:
            print("⚠️ Telegram not configured (missing TELEGRAM_TOKEN or CHAT_ID)")
            return
        self.app = Application.builder().token(self.token).build()
        self.app.add_handler(CommandHandler("start", self._start))
        self.app.add_handler(CommandHandler("status", self._status))
        self.app.add_handler(CommandHandler("threshold", self._threshold))
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        print("🚀 Telegram bot polling started")
        self.app.run_polling()

    async def _start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not is_admin(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized")
            return
        await update.message.reply_text("🔮 ULTIMATE Legendary Scanner Active")

    async def _status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not is_admin(update.effective_user.id):
            return
        overrides = "\n".join([f"{k}: {v:.1f}%" for k, v in ASSET_THRESHOLD_OVERRIDE.items()]) or "None"
        await update.message.reply_text(
            f"✅ Active\nMode: {EngineState.current_mode}\nGlobal threshold: {GLOBAL_THRESHOLD:.1f}%\nPer-asset overrides:\n{overrides}"
        )

    async def _threshold(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not is_admin(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized")
            return

        args = context.args or []
        global GLOBAL_THRESHOLD, ASSET_THRESHOLD_OVERRIDE

        if not args:
            await update.message.reply_text("Usage: /threshold <value|reset> or /threshold <ASSET> <value|reset>")
            return

        # single argument
        if len(args) == 1:
            token = args[0].strip().lower()
            if token == "reset":
                GLOBAL_THRESHOLD = LEGENDARY_GATE
                await update.message.reply_text(f"🔁 Global threshold reset to default {GLOBAL_THRESHOLD}%")
                return
            try:
                val = float(args[0])
                if val <= 0 or val > 100:
                    raise ValueError
                GLOBAL_THRESHOLD = val
                await update.message.reply_text(f"✅ Global threshold set to {GLOBAL_THRESHOLD:.1f}%")
                return
            except Exception:
                await update.message.reply_text("Invalid value. Provide a number between 1 and 100, or 'reset'.")
                return

        # two arguments: asset + value/reset
        if len(args) >= 2:
            asset = args[0].strip()
            token = args[1].strip().lower()
            if asset not in ASSETS:
                await update.message.reply_text(f"Unknown asset: {asset}. Supported: {', '.join(ASSETS)}")
                return
            if token == "reset":
                if asset in ASSET_THRESHOLD_OVERRIDE:
                    del ASSET_THRESHOLD_OVERRIDE[asset]
                    await update.message.reply_text(f"🔁 Threshold override removed for {asset}. Now uses global {GLOBAL_THRESHOLD:.1f}%")
                else:
                    await update.message.reply_text(f"No override set for {asset}. Global {GLOBAL_THRESHOLD:.1f}% applies.")
                return
            try:
                val = float(args[1])
                if val <= 0 or val > 100:
                    raise ValueError
                ASSET_THRESHOLD_OVERRIDE[asset] = val
                await update.message.reply_text(f"✅ Threshold for {asset} set to {val:.1f}% (alerts ≥ {val:.1f}%)")
                return
            except Exception:
                await update.message.reply_text("Invalid value. Provide a number between 1 and 100, or 'reset'.")
                return

# -------------------------------
# Math helpers & indicators
# -------------------------------
def ema(series: pd.Series, period: int) -> np.ndarray:
    if series is None or len(series) == 0:
        return np.array([])
    alpha = 2 / (period + 1)
    ema_vals = [series.iloc[0]]
    for v in series.iloc[1:]:
        ema_vals.append(alpha * v + (1 - alpha) * ema_vals[-1])
    return np.array(ema_vals)

def rsi(series: pd.Series, period: int = 14) -> np.ndarray:
    if series is None or len(series) < 2:
        return np.array([])
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / (down.replace(0, np.nan))
    rsi_vals = 100 - (100 / (1 + rs))
    return rsi_vals.fillna(50.0).values

def macd(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    if series is None or len(series) == 0:
        return np.array([]), np.array([])
    ema_fast = pd.Series(ema(series, 12))
    ema_slow = pd.Series(ema(series, 26))
    macd_line = ema_fast - ema_slow
    macd_sig = pd.Series(ema(macd_line, 9))
    return macd_line.values, macd_sig.values

def bollinger_bands(series: pd.Series, period: int = 20) -> Dict:
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return {"upper": upper, "lower": lower, "ma": ma}

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df is None or len(df) < 2:
        return pd.Series(dtype=float)
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    if df is None or len(df) < period:
        return pd.Series(dtype=float)
    _atr = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upperband = hl2 + (multiplier * _atr)
    lowerband = hl2 - (multiplier * _atr)
    st = pd.Series(index=df.index, dtype=float)
    trend = 1
    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = lowerband.iloc[i]
            continue
        if df["close"].iloc[i] > upperband.iloc[i-1]:
            trend = 1
        elif df["close"].iloc[i] < lowerband.iloc[i-1]:
            trend = -1
        st.iloc[i] = lowerband.iloc[i] if trend == 1 else upperband.iloc[i]
    return st

def vwap(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_pv = (tp * df["volume"]).cumsum()
    return cum_pv / (cum_vol.replace(0, np.nan))

def simple_volume_profile(df: pd.DataFrame, bins: int = 24) -> Dict[str, float]:
    if df is None or len(df) < 2:
        return {"POC": 0.0, "VAH": 0.0, "VAL": 0.0}
    price = df["close"].values
    vol = df["volume"].values
    hist, edges = np.histogram(price, bins=bins, weights=vol)
    poc_idx = int(np.argmax(hist))
    poc = (edges[poc_idx] + edges[poc_idx+1]) / 2
    total_vol = hist.sum() + 1e-9
    cum = np.cumsum(hist) / total_vol
    lower_idx = np.searchsorted(cum, 0.15)
    upper_idx = np.searchsorted(cum, 0.85)
    vah = (edges[upper_idx] + edges[upper_idx+1]) / 2 if upper_idx < len(edges)-1 else float(edges[-1])
    val = (edges[lower_idx] + edges[lower_idx+1]) / 2 if lower_idx < len(edges)-1 else float(edges[0])
    return {"POC": float(poc), "VAH": float(vah), "VAL": float(val)}

# -------------------------------
# Institutional & SMC filters (per-asset)
# -------------------------------
def smart_money_filter(df: pd.DataFrame, asset_config: Dict) -> Dict:
    if df is None or len(df) < asset_config.get("min_data_points", 30):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "SmartMoney", "type": "filter"}
    recent = df.tail(10)
    sh = df["high"].rolling(20).max().iloc[-2] if len(df) >= 22 else df["high"].max()
    sl = df["low"].rolling(20).min().iloc[-2] if len(df) >= 22 else df["low"].min()
    grab_up = (recent["high"].max() > sh) and (recent["close"].iloc[-1] < sh)
    grab_down = (recent["low"].min() < sl) and (recent["close"].iloc[-1] > sl)
    if grab_up and not grab_down:
        return {"signal": "SELL", "score": 88.0, "reason": "SmartMoney", "type": "filter"}
    if grab_down and not grab_up:
        return {"signal": "BUY", "score": 88.0, "reason": "SmartMoney", "type": "filter"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "SmartMoney", "type": "filter"}

def vwap_deviation_filter(df: pd.DataFrame, asset_config: Dict) -> Dict:
    if df is None or len(df) < asset_config.get("min_data_points", 50):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "VWAP", "type": "filter"}
    v = vwap(df)
    if v.empty:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "VWAP", "type": "filter"}
    price = df["close"].iloc[-1]
    dev = (price - v.iloc[-1]) / (v.iloc[-1] + 1e-9)
    threshold = asset_config.get("vwap_threshold", 0.01)
    if dev > threshold:
        return {"signal": "SELL", "score": 80.0, "reason": "VWAP", "type": "filter"}
    elif dev < -threshold:
        return {"signal": "BUY", "score": 80.0, "reason": "VWAP", "type": "filter"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "VWAP", "type": "filter"}

def volume_profile_filter(df: pd.DataFrame, asset_config: Dict) -> Dict:
    if df is None or len(df) < asset_config.get("min_data_points", 60):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "VolProfile", "type": "filter"}
    vp = simple_volume_profile(df)
    price = df["close"].iloc[-1]
    sensitivity = asset_config.get("volume_profile_sensitivity", 0.005)
    dist_vah = (price - vp["VAH"]) / (vp["VAH"] + 1e-9)
    dist_val = (price - vp["VAL"]) / (vp["VAL"] + 1e-9)
    if dist_val < -sensitivity:
        return {"signal": "BUY", "score": 78.0, "reason": "VolProfile", "type": "filter"}
    if dist_vah > sensitivity:
        return {"signal": "SELL", "score": 78.0, "reason": "VolProfile", "type": "filter"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "VolProfile", "type": "filter"}

# -------------------------------
# Strategies & indicators
# -------------------------------
def fibonacci_vortex_hidden(df: pd.DataFrame) -> Dict:
    if df is None or len(df) < 50:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "FibVortex", "type": "strategy"}
    swing_high = df["high"].rolling(20).max().iloc[-1]
    swing_low = df["low"].rolling(20).min().iloc[-1]
    swing_range = swing_high - swing_low
    golden_level = swing_high - 0.618 * swing_range
    current_price = df["close"].iloc[-1]
    at_golden = abs(current_price - golden_level) / (current_price + 1e-9) < 0.005
    if at_golden:
        return {"signal": "BUY", "score": 96.0, "reason": "FibVortex", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "FibVortex", "type": "strategy"}

def quantum_entanglement_hidden(df: pd.DataFrame) -> Dict:
    if df is None or len(df) < 30:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "QuantumEnt", "type": "strategy"}
    prices = df["close"].values[-30:]
    mean_price = np.mean(prices)
    std_price = np.std(prices)
    z_score = (prices[-1] - mean_price) / (std_price + 1e-9)
    if z_score < -1.5:
        return {"signal": "BUY", "score": 94.0, "reason": "QuantumEnt", "type": "strategy"}
    elif z_score > 1.5:
        return {"signal": "SELL", "score": 94.0, "reason": "QuantumEnt", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "QuantumEnt", "type": "strategy"}

def dark_pool_institutional_hidden(df: pd.DataFrame) -> Dict:
    if df is None or len(df) < 50:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "DarkPool", "type": "strategy"}
    vol = df["volume"]
    vol_ma = vol.rolling(50).mean()
    vol_spike = vol.iloc[-1] > (vol_ma.iloc[-1] * 2.0 if not np.isnan(vol_ma.iloc[-1]) else 0)
    if vol_spike:
        signal = "BUY" if df["close"].iloc[-1] > df["open"].iloc[-1] else "SELL"
        return {"signal": signal, "score": 95.0, "reason": "DarkPool", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "DarkPool", "type": "strategy"}

def order_block_detection(df: pd.DataFrame) -> Dict:
    if df is None or len(df) < 20:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "OrderBlock", "type": "strategy"}
    recent = df.tail(10)
    bull_engulf = (recent["close"].iloc[-1] > recent["open"].iloc[-1]) and (recent["close"].max() > recent["high"].shift(1).max())
    bear_engulf = (recent["close"].iloc[-1] < recent["open"].iloc[-1]) and (recent["close"].min() < recent["low"].shift(1).min())
    if bull_engulf:
        return {"signal": "BUY", "score": 88.0, "reason": "OrderBlock", "type": "strategy"}
    if bear_engulf:
        return {"signal": "SELL", "score": 88.0, "reason": "OrderBlock", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "OrderBlock", "type": "strategy"}

def break_of_structure(df: pd.DataFrame) -> Dict:
    if df is None or len(df) < 50:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "BOS", "type": "strategy"}
    sh = df["high"].rolling(20).max().iloc[-2]
    sl = df["low"].rolling(20).min().iloc[-2]
    if df["close"].iloc[-1] > sh:
        return {"signal": "BUY", "score": 86.0, "reason": "BOS", "type": "strategy"}
    elif df["close"].iloc[-1] < sl:
        return {"signal": "SELL", "score": 86.0, "reason": "BOS", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "BOS", "type": "strategy"}

def fair_value_gap(df: pd.DataFrame) -> Dict:
    if df is None or len(df) < 3:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "FVG", "type": "strategy"}
    if df["low"].iloc[-1] > df["high"].iloc[-3]:
        return {"signal": "BUY", "score": 89.0, "reason": "FVG", "type": "strategy"}
    elif df["high"].iloc[-1] < df["low"].iloc[-3]:
        return {"signal": "SELL", "score": 89.0, "reason": "FVG", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "FVG", "type": "strategy"}

def ema_macd_indicator(df: pd.DataFrame) -> Dict:
    if df is None or len(df) < 50:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "indicator"}
    ema_fast = ema(df["close"], 12)
    ema_slow = ema(df["close"], 26)
    macd_line, macd_sig = macd(df["close"])
    if len(ema_fast) < 2 or len(macd_line) < 1:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "indicator"}
    if ema_fast[-1] > ema_slow[-1] and macd_line[-1] > macd_sig[-1]:
        return {"signal": "BUY", "score": 82.0, "reason": "EMA_MACD", "type": "indicator"}
    elif ema_fast[-1] < ema_slow[-1] and macd_line[-1] < macd_sig[-1]:
        return {"signal": "SELL", "score": 82.0, "reason": "EMA_MACD", "type": "indicator"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "indicator"}

def rsi_indicator(df: pd.DataFrame) -> Dict:
    if df is None or len(df) < 20:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI", "type": "indicator"}
    rsi_val = rsi(df["close"])[-1]
    if rsi_val < 30:
        return {"signal": "BUY", "score": 75.0, "reason": "RSI", "type": "indicator"}
    elif rsi_val > 70:
        return {"signal": "SELL", "score": 75.0, "reason": "RSI", "type": "indicator"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI", "type": "indicator"}

def bollinger_indicator(df: pd.DataFrame) -> Dict:
    if df is None or len(df) < 30:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Bollinger", "type": "indicator"}
    bb = bollinger_bands(df["close"])
    if df["close"].iloc[-1] < bb["lower"].iloc[-1]:
        return {"signal": "BUY", "score": 76.0, "reason": "Bollinger", "type": "indicator"}
    elif df["close"].iloc[-1] > bb["upper"].iloc[-1]:
        return {"signal": "SELL", "score": 76.0, "reason": "Bollinger", "type": "indicator"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "Bollinger", "type": "indicator"}

def volume_indicator(df: pd.DataFrame, asset_config: Dict = None) -> Dict:
    if df is None or len(df) < (asset_config.get("min_data_points", 50) if asset_config else 50):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Volume", "type": "indicator"}
    multiplier = asset_config.get("momentum_volume_multiplier", 2.0) if asset_config else 2.0
    vol_spike = df["volume"].iloc[-1] > df["volume"].rolling(50).mean().iloc[-1] * multiplier
    if vol_spike and df["close"].iloc[-1] > df["open"].iloc[-1]:
        return {"signal": "BUY", "score": 86.0, "reason": "Volume", "type": "indicator"}
    elif vol_spike and df["close"].iloc[-1] < df["open"].iloc[-1]:
        return {"signal": "SELL", "score": 86.0, "reason": "Volume", "type": "indicator"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "Volume", "type": "indicator"}

def supertrend_bollinger_indicator(df: pd.DataFrame) -> Dict:
    if df is None or len(df) < 50:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "SuperTrend+Boll", "type": "indicator"}
    st = supertrend(df)
    bb = bollinger_bands(df["close"])
    close = df["close"].iloc[-1]
    bull = (not st.empty and close > st.iloc[-1]) and (close > bb["ma"].iloc[-1])
    bear = (not st.empty and close < st.iloc[-1]) and (close < bb["ma"].iloc[-1])
    if bull:
        return {"signal": "BUY", "score": 84.0, "reason": "SuperTrend+Boll", "type": "indicator"}
    if bear:
        return {"signal": "SELL", "score": 84.0, "reason": "SuperTrend+Boll", "type": "indicator"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "SuperTrend+Boll", "type": "indicator"}

# -------------------------------
# Engines composition
# -------------------------------
ALL_STRATEGIES = [
    fibonacci_vortex_hidden, quantum_entanglement_hidden, dark_pool_institutional_hidden,
    order_block_detection, break_of_structure, fair_value_gap
]
ALL_INDICATORS = [
    ema_macd_indicator, rsi_indicator, bollinger_indicator, volume_indicator, supertrend_bollinger_indicator
]

class Engines:
    QUANTUM = {
        "name": "Quantum Engine V2.0",
        "strategies": [order_block_detection, break_of_structure, fair_value_gap, fibonacci_vortex_hidden, dark_pool_institutional_hidden],
        "indicators": [ema_macd_indicator, supertrend_bollinger_indicator, volume_indicator, bollinger_indicator],
        "filters": [smart_money_filter, vwap_deviation_filter, volume_profile_filter]
    }
    MOMENTUM = {
        "name": "Momentum Scalper V1.0",
        "strategies": [break_of_structure],
        "indicators": [volume_indicator, rsi_indicator, ema_macd_indicator],
        "filters": [vwap_deviation_filter]
    }
    BREAKOUT = {
        "name": "Breakout Hunter V1.0",
        "strategies": [break_of_structure, fair_value_gap],
        "indicators": [bollinger_indicator, volume_indicator],
        "filters": [volume_profile_filter]
    }
    MEANREVERSION = {
        "name": "Mean Reversion V1.0",
        "strategies": [fibonacci_vortex_hidden],
        "indicators": [rsi_indicator, bollinger_indicator],
        "filters": [volume_profile_filter]
    }

class EngineState:
    current_mode: str = "quantum"

# -------------------------------
# Market regime detection
# -------------------------------
def compute_market_conditions(asset: str, df_map: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    asset_config = ASSET_CONFIG.get(asset, {})
    refs = [tf for tf in ["4h", "1h", "1d"] if tf in df_map]
    if not refs:
        return {"volatility": 0.0, "trend_strength": 0.0, "momentum_spike": False, "breakout_detected": False, "range_bound": True}
    df = df_map[refs[0]]
    if df is None or len(df) < asset_config.get("min_data_points", 50):
        return {"volatility": 0.0, "trend_strength": 0.0, "momentum_spike": False, "breakout_detected": False, "range_bound": True}
    _atr_series = atr(df)
    _atr = _atr_series.iloc[-1] if not _atr_series.empty else 0.0
    price = df["close"].iloc[-1]
    volatility = float((_atr / (price + 1e-9)) * 100)
    ema_fast = pd.Series(ema(df["close"], 12))
    ema_slow = pd.Series(ema(df["close"], 26))
    ema_slope = 0.0
    if len(ema_fast) >= 6:
        ema_slope = float((ema_fast.iloc[-1] - ema_fast.iloc[-5]) / (abs(ema_fast.iloc[-5]) + 1e-9))
    st = supertrend(df)
    trend_up = (not st.empty and df["close"].iloc[-1] > st.iloc[-1]) and (ema_fast.iloc[-1] > ema_slow.iloc[-1]) if not st.empty else False
    trend_down = (not st.empty and df["close"].iloc[-1] < st.iloc[-1]) and (ema_fast.iloc[-1] < ema_slow.iloc[-1]) if not st.empty else False
    trend_strength = float(abs(ema_slope)) * asset_config.get("trend_strength_multiplier", 1.0) * (1.5 if (trend_up or trend_down) else 0.8)
    vol_mult = asset_config.get("momentum_volume_multiplier", 1.8)
    vol_spike = df["volume"].iloc[-1] > df["volume"].rolling(30).mean().iloc[-1] * vol_mult if len(df) >= 31 else False
    price_change = abs(df["close"].iloc[-1] - df["close"].iloc[-5]) / (df["close"].iloc[-5] + 1e-9) if len(df) >= 6 else 0.0
    momentum_spike = bool(vol_spike and price_change > 0.01)
    bb = bollinger_bands(df["close"])
    breakout_detected = bool(df["close"].iloc[-1] > bb["upper"].iloc[-1] or df["close"].iloc[-1] < bb["lower"].iloc[-1]) if not bb["upper"].isna().all() else False
    range_bound = bool(volatility < asset_config.get("volatility_threshold", 0.5) and abs(df["close"].iloc[-1] - bb["ma"].iloc[-1]) / (bb["ma"].iloc[-1] + 1e-9) < 0.005) if not bb["ma"].isna().all() else False
    return {
        "volatility": volatility,
        "trend_strength": trend_strength,
        "momentum_spike": momentum_spike,
        "breakout_detected": breakout_detected,
        "range_bound": range_bound
    }

def select_mode(market_conditions: Dict[str, float]) -> str:
    if market_conditions["momentum_spike"]:
        return "momentum"
    if market_conditions["breakout_detected"]:
        return "breakout"
    if market_conditions["range_bound"]:
        return "meanreversion"
    return "quantum"

def get_engine(mode: str) -> Dict:
    if mode == "momentum":
        return Engines.MOMENTUM
    if mode == "breakout":
        return Engines.BREAKOUT
    if mode == "meanreversion":
        return Engines.MEANREVERSION
    return Engines.QUANTUM

# -------------------------------
# Data provider
# -------------------------------
class RealDataProvider:
    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except Exception:
            self.yf = None

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 300) -> Optional[pd.DataFrame]:
        if self.yf is None:
            return None
        interval = {"1h": "1h", "4h": "1h", "1d": "1d", "1w": "1d", "1M": "1d"}.get(timeframe, "1d")
        period = {"1h": "1mo", "1d": "1y"}.get(interval, "1y")
        try:
            df = self.yf.download(symbol, period=period, interval=interval, progress=False)
            if df is None or df.empty:
                return None
            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            df.reset_index(inplace=True)
            ts_col = "Datetime" if "Datetime" in df.columns else "Date"
            df["timestamp"] = pd.to_datetime(df[ts_col])
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].tail(limit)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna().reset_index(drop=True)
            return df
        except Exception:
            return None

# -------------------------------
# Selection helpers
# -------------------------------
def select_best(items: List[Dict], item_type: str, min_score: float, limit: int = 4) -> List[Dict]:
    picked = [r for r in items if r.get("type") == item_type and r.get("score", 0.0) >= min_score]
    picked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return picked[:limit]

# -------------------------------
# Core analysis per engine
# -------------------------------
def engine_ai_analysis(asset: str, tf_data: Dict[str, pd.DataFrame], engine: Dict) -> Dict:
    asset_config = ASSET_CONFIG.get(asset, {})
    all_strategy_results = []
    all_indicator_results = []
    all_filter_results = []

    for tf, df in tf_data.items():
        if df is None or len(df) < asset_config.get("min_data_points", 50):
            continue
        for strategy_fn in engine["strategies"]:
            try:
                result = strategy_fn(df)
                if result and result.get("signal") != "NEUTRAL":
                    result["timeframe"] = tf
                    all_strategy_results.append(result)
            except Exception:
                pass
        for indicator_fn in engine["indicators"]:
            try:
                if indicator_fn == volume_indicator:
                    result = indicator_fn(df, asset_config)
                else:
                    result = indicator_fn(df)
                if result and result.get("signal") != "NEUTRAL":
                    result["timeframe"] = tf
                    all_indicator_results.append(result)
            except Exception:
                pass
        for filt_fn in engine.get("filters", []):
            try:
                result = filt_fn(df, asset_config)
                if result and result.get("signal") != "NEUTRAL":
                    result["timeframe"] = tf
                    all_filter_results.append(result)
            except Exception:
                pass

    best_strategies = select_best(all_strategy_results, "strategy", min_score=85.0)
    best_indicators = select_best(all_indicator_results, "indicator", min_score=75.0)
    best_filters = select_best(all_filter_results, "filter", min_score=75.0)

    all_signals = [r["signal"] for r in best_strategies + best_indicators]
    signal_counts = Counter(all_signals)

    if len(best_strategies) >= 3 and len(best_indicators) >= 3:
        if signal_counts.get("BUY", 0) >= 5:
            final_signal = "BUY"
            confidence = sum(r["score"] for r in best_strategies + best_indicators if r["signal"] == "BUY") / (signal_counts["BUY"] + 1e-9)
        elif signal_counts.get("SELL", 0) >= 5:
            final_signal = "SELL"
            confidence = sum(r["score"] for r in best_strategies + best_indicators if r["signal"] == "SELL") / (signal_counts["SELL"] + 1e-9)
        else:
            final_signal = "NEUTRAL"
            confidence = 0.0
    else:
        final_signal = "NEUTRAL"
        confidence = 0.0

    filter_bias = Counter([f["signal"] for f in best_filters])
    if final_signal in ["BUY", "SELL"] and best_filters:
        agree = filter_bias.get(final_signal, 0)
        disagree = sum(filter_bias.values()) - agree
        if agree > disagree:
            confidence = min(100.0, confidence + 3.0)
        elif disagree > agree:
            confidence = max(0.0, confidence - 5.0)

    return {
        "signal": final_signal,
        "confidence": float(confidence),
        "best_strategies": best_strategies,
        "best_indicators": best_indicators,
        "best_filters": best_filters,
        "num_strategies_aligned": len([r for r in best_strategies if r["signal"] == final_signal]),
        "num_indicators_aligned": len([r for r in best_indicators if r["signal"] == final_signal]),
        "num_filters_aligned": len([r for r in best_filters if r["signal"] == final_signal]),
    }

# -------------------------------
# Alert formatting
# -------------------------------
def format_ultimate_alert(result: Dict, engine_name: str) -> str:
    strategies_list = "\n".join([f"  • {s['reason']} ({s['score']:.0f}%)" for s in result.get('best_strategies', [])[:6]]) or "  • None"
    indicators_list = "\n".join([f"  • {i['reason']} ({i['score']:.0f}%)" for i in result.get('best_indicators', [])[:6]]) or "  • None"
    filters_list = "\n".join([f"  • {f['reason']} ({f['score']:.0f}%)" for f in result.get('best_filters', [])[:4]]) or "  • None"

    return f"""🔮 <b>ULTIMATE LEGENDARY SIGNAL</b>  <i>[{engine_name}]</i>

<b>Asset:</b> {result.get('asset', 'N/A')}
<b>Direction:</b> {result.get('signal', 'NEUTRAL')}
<b>Expiry:</b> {result.get('expiry', 'N/A')}m
<b>Confidence:</b> {result.get('confidence', 0.0):.1f}%

<b>Strategies:</b>
{strategies_list}

<b>Indicators:</b>
{indicators_list}

<b>Institutional filters:</b>
{filters_list}

<b>Time:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"""

# -------------------------------
# API routes
# -------------------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "legendary", "timestamp": datetime.utcnow().isoformat(), "tensorflow": TFK, "mode": EngineState.current_mode}), 200

# -------------------------------
# Main loop
# -------------------------------
def main_scan_loop():
    print("🔮 Initializing ULTIMATE Legendary Scanner (Async Telegram + Runtime TF/Sklearn)...")
    print(f"⚡ Engines: Quantum / Momentum / Breakout / MeanReversion")
    print(f"📊 Base Pools — Strategies: {len(ALL_STRATEGIES)} | Indicators: {len(ALL_INDICATORS)}")
    print(f"🧠 TensorFlow: {'ENABLED' if TFK else 'FALLBACK MODE'}")
    print(f"🤖 Telegram: {'configured' if (TELEGRAM_TOKEN and CHAT_ID) else 'not configured'}")

    provider = RealDataProvider()
    telegram = LegendaryTelegramBot(TELEGRAM_TOKEN, CHAT_ID)
    telegram.start()

    print("✅ Scanner ready!")

    while True:
        try:
            now_utc = datetime.utcnow()
            print(f"\n🔍 Scan: {now_utc.strftime('%H:%M:%S')}")

            for asset in ASSETS:
                tf_data = {}
                for tf in ALL_TIMEFRAMES[:4]:
                    try:
                        df = provider.fetch_ohlcv(asset, tf, limit=300)
                        if df is not None and len(df) >= ASSET_CONFIG.get(asset, {}).get("min_data_points", 50):
                            tf_data[tf] = df
                    except Exception:
                        pass

                if len(tf_data) < 3:
                    continue

                market_conditions = compute_market_conditions(asset, tf_data)
                EngineState.current_mode = select_mode(market_conditions)
                engine = get_engine(EngineState.current_mode)

                for expiry in EXPIRIES:
                    cooldown_key = f"{asset}_{expiry}"
                    if cooldown_key in LAST_LEGENDARY_ALERT:
                        elapsed = (now_utc - LAST_LEGENDARY_ALERT[cooldown_key]).total_seconds() / 60
                        if elapsed < COOLDOWN_MINUTES:
                            continue

                    result = engine_ai_analysis(asset, tf_data, engine)
                    threshold = ASSET_THRESHOLD_OVERRIDE.get(asset, GLOBAL_THRESHOLD)

                    if (result['signal'] != "NEUTRAL" and
                        result['confidence'] >= threshold and
                        result['num_strategies_aligned'] >= 3 and
                        result['num_indicators_aligned'] >= 3):

                        alert_data = {
                            'asset': asset,
                            'signal': result['signal'],
                            'confidence': result['confidence'],
                            'expiry': expiry,
                            'best_strategies': result['best_strategies'],
                            'best_indicators': result['best_indicators'],
                            'best_filters': result.get('best_filters', []),
                            'num_strategies_aligned': result['num_strategies_aligned'],
                            'num_indicators_aligned': result['num_indicators_aligned'],
                            'num_filters_aligned': result['num_filters_aligned'],
                            'timestamp': now_utc.isoformat()
                        }

                        signal_key = f"{asset}_{expiry}_{now_utc.strftime('%Y%m%d_%H%M')}"
                        SIGNAL_HISTORY[signal_key] = alert_data
                        LAST_LEGENDARY_ALERT[cooldown_key] = now_utc

                        alert_msg = format_ultimate_alert(alert_data, engine["name"])

                        # Send Telegram alert if app is ready
                        if telegram.app:
                            try:
                                telegram.app.bot.send_message(chat_id=CHAT_ID, text=alert_msg, parse_mode="HTML")
                            except Exception as e:
                                print(f"Telegram send error: {e}")
                        print(f"  🔥 [{engine['name']}] {asset} {result['signal']} @ {result['confidence']:.1f}% (threshold {threshold:.1f}%)")

                    time.sleep(0.15)

            print(f"✅ Scan complete. Next in {SCAN_INTERVAL_SEC}s...")

            if RUN_ONCE:
                break
            time.sleep(SCAN_INTERVAL_SEC)

        except KeyboardInterrupt:
            print("\n🛑 Stopped")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(8)

# -------------------------------
# Entrypoint
# -------------------------------
if __name__ == "__main__":
    print(f"🚀 Starting on port {PORT}")
    scanner_thread = threading.Thread(target=main_scan_loop, daemon=True)
    scanner_thread.start()
    app.run(host='0.0.0.0', port=PORT, debug=False)

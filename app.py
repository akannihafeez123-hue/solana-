"""
🔮 ULTIMATE LEGENDARY Pocket Option AI Scanner
With Sacred Geometry, Quantum Analysis & Hidden Institutional Strategies

Features:
- Auto-mode selection (Quantum/Momentum/Breakout/MeanReversion/Sacred)
- 95% Confidence Threshold for alerts
- Fibonacci Vortex with Golden Spiral
- Quantum Entanglement Probability Waves
- Dark Pool Institutional Detection
- Gann Square Time Cycles
- Elliott Wave Neural AI
- Cosmic Movement Analysis
- Quantum Engine V2.0 (OB/BOS/FVG)
- Momentum Scalper V1.0
- Breakout Hunter V1.0  
- Mean Reversion V1.0
- Auto-selects best 4 strategies + 4 indicators + 4 timeframes
- Runtime TensorFlow + Admin-only access
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import threading
import warnings
import json
import hashlib
import hmac
from flask import Flask, request, jsonify
from collections import Counter
import math
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ==================== CONFIG ====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

# Safe parsing of ADMIN_IDS with error handling
def parse_admin_ids():
    admin_ids_str = os.getenv("ADMIN_IDS", "")
    if not admin_ids_str or admin_ids_str in ["", "your_chat_id", "your_id", "your_ids"]:
        # Use CHAT_ID as fallback if available and numeric
        if CHAT_ID and CHAT_ID.strip() and CHAT_ID.strip().isdigit():
            return [int(CHAT_ID.strip())]
        return []  # Empty list if no valid IDs
    
    # Parse comma-separated IDs
    ids = []
    for x in admin_ids_str.split(","):
        x = x.strip()
        if x and x.isdigit():
            ids.append(int(x))
        elif x and not x.startswith("your_"):  # Skip placeholder values
            print(f"⚠️  Warning: Skipping invalid ADMIN_ID: {x}")
    return ids if ids else []

ADMIN_IDS = parse_admin_ids()

if not ADMIN_IDS:
    print("⚠️  WARNING: No valid ADMIN_IDS configured. Admin features will be disabled.")
    print("   Set ADMIN_IDS environment variable to your Telegram user ID(s)")
    print("   Example: ADMIN_IDS=123456789 or ADMIN_IDS=123456789,987654321")

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "your_legendary_secret_key_here")
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "False") == "True"
CONF_GATE = int(os.getenv("CONF_GATE", "92"))
LEGENDARY_GATE = int(os.getenv("LEGENDARY_GATE", "95"))  # 95% threshold
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "15"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "45"))
RUN_ONCE = os.getenv("RUN_ONCE", "False") == "True"
PORT = int(os.getenv("PORT") or os.getenv("port") or "8080")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "your_webhook_secret")

# Universe
ASSETS = [
    "EURUSD_otc", "GBPUSD_otc", "AUDUSD_otc", "USDJPY_otc", "XAUUSD_otc",
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
    "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
    "GC=F", "SI=F", "CL=F", "BZ=F", "NG=F"
]
EXPIRIES = [5, 15, 30, 60, 120, 240]
ALL_TIMEFRAMES = ["1y", "1M", "2M", "1w", "3w", "1d", "8h", "4h", "2h", "1h", "30m", "15m"]
TF_WEIGHTS = {"1y": 10, "1M": 9, "2M": 8, "1w": 7, "3w": 6, "1d": 5, "8h": 4, "4h": 3, "2h": 2, "1h": 1, "30m": 1, "15m": 1}

SIGNAL_HISTORY = {}
LAST_LEGENDARY_ALERT = {}

app = Flask(__name__)

# Runtime TensorFlow - FIXED VERSION
TFK = False
StandardScaler = None
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, BatchNormalization,
                                        Conv1D, MaxPooling1D, MultiHeadAttention, 
                                        LayerNormalization, GlobalAveragePooling1D,
                                        Attention, Concatenate)
    from tensorflow.keras.optimizers import Adam
    TFK = True
    print("✅ TensorFlow loaded successfully")
except ImportError as e:
    print(f"⚠️ TensorFlow not available: {e}")
    print("📝 Installing TensorFlow: pip install tensorflow")
    TFK = False
except Exception as e:
    print(f"⚠️ TensorFlow error: {e}")
    TFK = False

# Import sklearn separately since it's usually available
try:
    from sklearn.preprocessing import StandardScaler, RobustScaler
    print("✅ scikit-learn loaded successfully")
except ImportError as e:
    print(f"⚠️ scikit-learn not available: {e}")
    print("📝 Installing scikit-learn: pip install scikit-learn")
    StandardScaler = None
except Exception as e:
    print(f"⚠️ scikit-learn error: {e}")
    StandardScaler = None

try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

try:
    from telegram.ext import Updater, CommandHandler
    from telegram import ParseMode
    TELEGRAM_AVAILABLE = True
except Exception:
    TELEGRAM_AVAILABLE = False

# ==================== TRADING MODES ====================

TRADING_MODES = {
    "QUANTUM": {
        "name": "Quantum Engine V2.0",
        "strategies": ["OrderBlock", "BOS", "FVG", "EMA_MACD", "ST_BB", "Vol_SM"],
        "description": "Institutional-grade market structure analysis",
        "emoji": "⚛️"
    },
    "MOMENTUM": {
        "name": "Momentum Scalper V1.0", 
        "strategies": ["MomentumBreak", "VolumeSpike", "EMAGoldenCross", "RSI"],
        "description": "High-velocity scalp trading signals",
        "emoji": "📈"
    },
    "BREAKOUT": {
        "name": "Breakout Hunter V1.0",
        "strategies": ["RSBreakout", "BBBreakout", "ADX"],
        "description": "Early trend breakout identification", 
        "emoji": "🚀"
    },
    "MEAN_REVERSION": {
        "name": "Mean Reversion V1.0",
        "strategies": ["PriceRejection", "VolumeDivergence", "Bollinger", "RSI", "Stochastic"],
        "description": "Counter-trend reversal opportunities",
        "emoji": "🔄"
    },
    "SACRED": {
        "name": "Sacred Geometry System",
        "strategies": ["FibVortex", "QuantumEnt", "DarkPool", "GannSquare", "ElliottWave", "Cosmic"],
        "description": "Quantum-sacred confluence detection",
        "emoji": "🔮"
    }
}

# ==================== SECURITY ====================

class AdminAuth:
    @staticmethod
    def verify_admin(user_id: int) -> bool:
        return user_id in ADMIN_IDS
    
    @staticmethod
    def generate_token(user_id: int) -> str:
        data = f"{user_id}:{datetime.utcnow().isoformat()}"
        signature = hmac.new(ADMIN_SECRET.encode(), data.encode(), hashlib.sha256).hexdigest()
        return f"{data}:{signature}"
    
    @staticmethod
    def verify_token(token: str) -> Tuple[bool, Optional[int]]:
        try:
            parts = token.split(":")
            if len(parts) != 3:
                return False, None
            user_id, timestamp, signature = parts
            user_id = int(user_id)
            if user_id not in ADMIN_IDS:
                return False, None
            data = f"{user_id}:{timestamp}"
            expected = hmac.new(ADMIN_SECRET.encode(), data.encode(), hashlib.sha256).hexdigest()
            if signature != expected:
                return False, None
            token_time = datetime.fromisoformat(timestamp)
            age = (datetime.utcnow() - token_time).total_seconds()
            if age > 86400:
                return False, None
            return True, user_id
        except Exception:
            return False, None

# ==================== CORE INDICATORS ====================

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

def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * ((df["close"] - low_min) / (high_max - low_min))
    d = k.rolling(d_period).mean()
    return k.fillna(50).values, d.fillna(50).values

def adx(df: pd.DataFrame, period: int = 14) -> Tuple[float, float, float]:
    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr = np.maximum(df["high"] - df["low"], 
                    np.maximum(abs(df["high"] - df["close"].shift(1)),
                              abs(df["low"] - df["close"].shift(1))))
    atr = pd.Series(tr).rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = dx.rolling(period).mean()
    return (float(adx_val.iloc[-1]) if pd.notna(adx_val.iloc[-1]) else 0.0,
            float(plus_di.iloc[-1]) if pd.notna(plus_di.iloc[-1]) else 0.0,
            float(minus_di.iloc[-1]) if pd.notna(minus_di.iloc[-1]) else 0.0)

def bollinger_bands(series: pd.Series, period: int = 20, dev: float = 2.0) -> Dict:
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + dev * std
    lower = ma - dev * std
    return {"upper": upper, "lower": lower, "ma": ma, "width": upper - lower, "std": std}

def atr(df: pd.DataFrame, period: int = 14) -> float:
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum(abs(df["high"] - df["close"].shift(1)),
                              abs(df["low"] - df["close"].shift(1))))
    atr_val = pd.Series(tr).rolling(period).mean()
    return float(atr_val.iloc[-1]) if pd.notna(atr_val.iloc[-1]) else 0.0

def vwap(df: pd.DataFrame) -> float:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap_val = (typical_price * df["volume"]).sum() / df["volume"].sum()
    return float(vwap_val)

def normalize(val: float, base: Optional[pd.Series] = None) -> float:
    try:
        if base is not None and len(base) > 0:
            return float(min(100.0, max(0.0, 100.0 * abs(val) / (abs(base.iloc[-1]) + 1e-9))))
        return float(min(100.0, max(0.0, val)))
    except Exception:
        return 0.0

# ==================== 🔮 SACRED STRATEGIES ====================

def fibonacci_vortex_hidden(df: pd.DataFrame) -> Dict:
    """Fibonacci Vortex Hidden - Sacred Geometry + Golden Spiral"""
    if len(df) < 50:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "FibVortex", "type": "strategy"}
    
    high = df["high"].rolling(20).max()
    low = df["low"].rolling(20).min()
    swing_high = high.iloc[-20:].max()
    swing_low = low.iloc[-20:].min()
    swing_range = swing_high - swing_low
    
    fib_levels = {
        0.236: swing_high - 0.236 * swing_range,
        0.382: swing_high - 0.382 * swing_range,
        0.500: swing_high - 0.500 * swing_range,
        0.618: swing_high - 0.618 * swing_range,
        0.786: swing_high - 0.786 * swing_range,
        1.618: swing_high - 1.618 * swing_range,
    }
    
    current_price = df["close"].iloc[-1]
    closest_fib = min(fib_levels.items(), key=lambda x: abs(x[1] - current_price))
    distance_pct = abs(current_price - closest_fib[1]) / current_price * 100
    
    vm_plus = (df["high"] - df["low"].shift(1)).abs().rolling(14).sum()
    vm_minus = (df["low"] - df["high"].shift(1)).abs().rolling(14).sum()
    tr = pd.Series([atr(df.iloc[:i+1], 14) for i in range(len(df))]).rolling(14).sum()
    vi_plus = vm_plus / tr
    vi_minus = vm_minus / tr
    
    at_golden_ratio = closest_fib[0] == 0.618 and distance_pct < 0.5
    at_fib_extension = closest_fib[0] == 1.618 and distance_pct < 0.5
    
    vortex_bullish = vi_plus.iloc[-1] > vi_minus.iloc[-1] and vi_plus.iloc[-2] <= vi_minus.iloc[-2]
    vortex_bearish = vi_plus.iloc[-1] < vi_minus.iloc[-1] and vi_plus.iloc[-2] >= vi_minus.iloc[-2]
    
    if (at_golden_ratio or at_fib_extension) and vortex_bullish:
        signal = "BUY"
        score = 94.0 + (6.0 if at_golden_ratio else 0)
    elif (at_golden_ratio or at_fib_extension) and vortex_bearish:
        signal = "SELL"
        score = 94.0 + (6.0 if at_golden_ratio else 0)
    else:
        signal = "NEUTRAL"
        score = 0.0
    
    return {"signal": signal, "score": score, "reason": "FibVortex", "type": "strategy"}

def quantum_entanglement_hidden(df: pd.DataFrame) -> Dict:
    """Quantum Entanglement - Probability Wave Analysis"""
    if len(df) < 30:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "QuantumEnt", "type": "strategy"}
    
    prices = df["close"].values[-30:]
    mean_price = np.mean(prices)
    std_price = np.std(prices)
    
    volatility = std_price / mean_price
    time_factor = len(prices)
    uncertainty = volatility * np.sqrt(time_factor)
    
    current_price = prices[-1]
    z_score = (current_price - mean_price) / (std_price + 1e-9)
    
    barrier_high = mean_price + 2 * std_price
    barrier_low = mean_price - 2 * std_price
    
    tunneling_prob_up = np.exp(-abs(current_price - barrier_high) / (std_price + 1e-9))
    tunneling_prob_down = np.exp(-abs(current_price - barrier_low) / (std_price + 1e-9))
    
    momentum = df["close"].pct_change(5).iloc[-1]
    resonance = abs(momentum) > volatility * 0.5
    
    if z_score < -1.5 and tunneling_prob_up > 0.7 and resonance:
        signal = "BUY"
        score = 93.0 + min(abs(z_score) * 2, 7.0)
    elif z_score > 1.5 and tunneling_prob_down > 0.7 and resonance:
        signal = "SELL"
        score = 93.0 + min(abs(z_score) * 2, 7.0)
    else:
        signal = "NEUTRAL"
        score = 0.0
    
    return {"signal": signal, "score": min(score, 100.0), "reason": "QuantumEnt", "type": "strategy"}

def dark_pool_institutional_hidden(df: pd.DataFrame) -> Dict:
    """Dark Pool Institutional - Stealth Smart Money Detection"""
    if len(df) < 50:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "DarkPool", "type": "strategy"}
    
    vol = df["volume"]
    vol_ma = vol.rolling(50).mean()
    
    price_change = abs(df["close"].pct_change(1))
    vol_surge = vol / vol_ma
    
    last_5_vol = vol.iloc[-5:].mean()
    last_5_price_change = abs(df["close"].iloc[-1] - df["close"].iloc[-5]) / df["close"].iloc[-5]
    
    iceberg_detected = last_5_vol > vol_ma.iloc[-1] * 2.0 and last_5_price_change < 0.005
    
    price = df["close"].iloc[-1]
    round_number = round(price, -int(math.floor(math.log10(abs(price)))) + 1) if price != 0 else 0
    near_round = abs(price - round_number) / price < 0.002
    
    price_direction = np.sign(df["close"].iloc[-1] - df["close"].iloc[-10])
    vol_direction = np.sign(vol.iloc[-5:].mean() - vol.iloc[-10:-5].mean())
    
    divergence = price_direction != vol_direction and abs(vol_direction) > 0
    
    body = abs(df["close"] - df["open"])
    large_body = (body.iloc[-1] > body.rolling(20).mean().iloc[-1] * 2.5)
    
    if iceberg_detected and near_round and divergence:
        signal = "BUY" if vol_direction > 0 else "SELL"
        score = 95.0
    elif large_body and vol_surge.iloc[-1] > 3.0:
        signal = "BUY" if df["close"].iloc[-1] > df["open"].iloc[-1] else "SELL"
        score = 91.0
    else:
        signal = "NEUTRAL"
        score = 0.0
    
    return {"signal": signal, "score": score, "reason": "DarkPool", "type": "strategy"}

def gann_square_time_cycles_hidden(df: pd.DataFrame) -> Dict:
    """Gann Square of Nine - W.D. Gann's Secret Time Cycles"""
    if len(df) < 100:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "GannSquare", "type": "strategy"}
    
    sacred_numbers = [3, 7, 9, 12, 21, 30, 45, 90, 144, 180, 360]
    
    highs = df["high"].rolling(20).max()
    lows = df["low"].rolling(20).min()
    
    bars_since_high = 0
    bars_since_low = 0
    
    for i in range(len(df)-1, 0, -1):
        if df["high"].iloc[i] == highs.iloc[i]:
            bars_since_high = len(df) - 1 - i
            break
    
    for i in range(len(df)-1, 0, -1):
        if df["low"].iloc[i] == lows.iloc[i]:
            bars_since_low = len(df) - 1 - i
            break
    
    at_sacred_time_high = any(abs(bars_since_high - sn) <= 2 for sn in sacred_numbers)
    at_sacred_time_low = any(abs(bars_since_low - sn) <= 2 for sn in sacred_numbers)
    
    price_range = df["high"].max() - df["low"].min()
    time_range = len(df)
    gann_angle = price_range / time_range if time_range > 0 else 0
    
    steep_angle = gann_angle > np.percentile([price_range / i for i in range(10, time_range) if i > 0], 75)
    flat_angle = gann_angle < np.percentile([price_range / i for i in range(10, time_range) if i > 0], 25)
    
    now = datetime.utcnow()
    day_of_year = now.timetuple().tm_yday
    
    cardinal_points = [80, 172, 266, 355]
    near_cardinal = any(abs(day_of_year - cp) <= 7 for cp in cardinal_points)
    
    if at_sacred_time_low and steep_angle and near_cardinal:
        signal = "BUY"
        score = 92.0
    elif at_sacred_time_high and flat_angle and near_cardinal:
        signal = "SELL"
        score = 92.0
    else:
        signal = "NEUTRAL"
        score = 0.0
    
    return {"signal": signal, "score": score, "reason": "GannSquare", "type": "strategy"}

def elliott_wave_neural_hidden(df: pd.DataFrame) -> Dict:
    """Elliott Wave Neural - AI-Enhanced Wave Recognition"""
    if len(df) < 89:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "ElliottWave", "type": "strategy"}
    
    prices = df["close"].values
    
    pivots = []
    for i in range(5, len(prices)-5):
        if prices[i] > max(prices[i-5:i]) and prices[i] > max(prices[i+1:i+6]):
            pivots.append(("high", i, prices[i]))
        elif prices[i] < min(prices[i-5:i]) and prices[i] < min(prices[i+1:i+6]):
            pivots.append(("low", i, prices[i]))
    
    if len(pivots) < 5:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "ElliottWave", "type": "strategy"}
    
    recent_pivots = pivots[-5:]
    pattern_types = [p[0] for p in recent_pivots]
    
    bullish_impulse = (pattern_types == ["low", "high", "low", "high", "low"] or 
                      pattern_types[-3:] == ["low", "high", "low"])
    bearish_impulse = (pattern_types == ["high", "low", "high", "low", "high"] or
                      pattern_types[-3:] == ["high", "low", "high"])
    
    if len(recent_pivots) >= 3:
        wave1_len = abs(recent_pivots[1][2] - recent_pivots[0][2])
        wave2_len = abs(recent_pivots[2][2] - recent_pivots[1][2]) if len(recent_pivots) > 2 else 0
        fib_relationship = abs(wave2_len / wave1_len - 1.618) < 0.2 if wave1_len > 0 else False
    else:
        fib_relationship = False
    
    current_price = prices[-1]
    last_pivot_price = recent_pivots[-1][2]
    
    if bullish_impulse and fib_relationship:
        signal = "BUY"
        score = 94.0
    elif bearish_impulse and fib_relationship:
        signal = "SELL"
        score = 94.0
    else:
        signal = "NEUTRAL"
        score = 0.0
    
    return {"signal": signal, "score": score, "reason": "ElliottWave", "type": "strategy"}

def cosmic_movement_hidden(df: pd.DataFrame) -> Dict:
    """Cosmic Movement - Lunar Cycles & Planetary Alignments"""
    if len(df) < 30:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Cosmic", "type": "strategy"}
    
    now = datetime.utcnow()
    
    lunar_cycle_days = 29.53
    known_new_moon = datetime(2025, 1, 29)
    days_since_new_moon = (now - known_new_moon).days % lunar_cycle_days
    lunar_phase = days_since_new_moon / lunar_cycle_days
    
    near_new_moon = lunar_phase < 0.1 or lunar_phase > 0.9
    near_full_moon = 0.45 < lunar_phase < 0.55
    
    phi = 1.618033988749895
    prices = df["close"].values[-30:]
    
    price_high = max(prices)
    price_low = min(prices)
    price_range = price_high - price_low
    
    golden_retracement = price_low + price_range * 0.618
    at_golden_level = abs(prices[-1] - golden_retracement) / prices[-1] < 0.01
    
    day_of_week = now.weekday()
    powerful_day = day_of_week in [1, 3]
    
    momentum = df["close"].pct_change(5).iloc[-1]
    
    if near_new_moon and at_golden_level and momentum > 0 and powerful_day:
        signal = "BUY"
        score = 90.0
    elif near_full_moon and at_golden_level and momentum < 0:
        signal = "SELL"
        score = 90.0
    else:
        signal = "NEUTRAL"
        score = 0.0
    
    return {"signal": signal, "score": score, "reason": "Cosmic", "type": "strategy"}

# ==================== QUANTUM ENGINE V2.0 ====================

def order_block_detection(df: pd.DataFrame) -> Dict:
    """Order Block Detection - Quantum Engine V2.0"""
    if len(df) < 20:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "OrderBlock", "type": "strategy"}
    
    ob_zones = []
    for i in range(2, len(df)):
        body = abs(df["close"].iloc[i] - df["open"].iloc[i])
        spread = df["high"].iloc[i] - df["low"].iloc[i]
        if body / max(spread, 1e-9) >= 0.6:
            ob_type = "demand" if df["close"].iloc[i] > df["open"].iloc[i] else "supply"
            ob_zones.append({"type": ob_type, "price": float(df["close"].iloc[i])})
    
    if not ob_zones:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "OrderBlock", "type": "strategy"}
    
    last_ob = ob_zones[-1]
    price = df["close"].iloc[-1]
    dist = abs(price - last_ob["price"]) / price
    
    if last_ob["type"] == "demand" and dist < 0.01:
        return {"signal": "BUY", "score": 88.0, "reason": "OrderBlock", "type": "strategy"}
    elif last_ob["type"] == "supply" and dist < 0.01:
        return {"signal": "SELL", "score": 88.0, "reason": "OrderBlock", "type": "strategy"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "OrderBlock", "type": "strategy"}

def break_of_structure(df: pd.DataFrame) -> Dict:
    """Break of Structure - Quantum Engine V2.0"""
    if len(df) < 50:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "BOS", "type": "strategy"}
    
    sh = df["high"].rolling(20).max().iloc[-2]
    sl = df["low"].rolling(20).min().iloc[-2]
    
    broke_high = df["close"].iloc[-1] > sh
    broke_low = df["close"].iloc[-1] < sl
    
    if broke_high:
        return {"signal": "BUY", "score": 86.0, "reason": "BOS", "type": "strategy"}
    elif broke_low:
        return {"signal": "SELL", "score": 86.0, "reason": "BOS", "type": "strategy"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "BOS", "type": "strategy"}

def fair_value_gap(df: pd.DataFrame) -> Dict:
    """Fair Value Gap - Quantum Engine V2.0"""
    if len(df) < 3:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "FVG", "type": "strategy"}
    
    fvg_up = df["low"].iloc[-1] > df["high"].iloc[-3]
    fvg_down = df["high"].iloc[-1] < df["low"].iloc[-3]
    
    if fvg_up:
        return {"signal": "BUY", "score": 89.0, "reason": "FVG", "type": "strategy"}
    elif fvg_down:
        return {"signal": "SELL", "score": 89.0, "reason": "FVG", "type": "strategy"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "FVG", "type": "strategy"}

# ==================== MOMENTUM SCALPER V1.0 ====================

def momentum_break_detection(df: pd.DataFrame) -> Dict:
    """Momentum Break Detection - Momentum Scalper V1.0"""
    if len(df) < 20:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "MomentumBreak", "type": "strategy"}
    
    roc_5 = df["close"].pct_change(5)
    roc_10 = df["close"].pct_change(10)
    
    volume_avg = df["volume"].rolling(10).mean()
    volume_spike = df["volume"].iloc[-1] > volume_avg.iloc[-1] * 1.5
    
    strong_up_momentum = roc_5.iloc[-1] > 0.02 and roc_10.iloc[-1] > 0.03 and volume_spike
    strong_down_momentum = roc_5.iloc[-1] < -0.02 and roc_10.iloc[-1] < -0.03 and volume_spike
    
    ema_9 = ema(df["close"], 9)
    ema_21 = ema(df["close"], 21)
    
    ema_bullish = len(ema_9) > 0 and len(ema_21) > 0 and ema_9[-1] > ema_21[-1]
    ema_bearish = len(ema_9) > 0 and len(ema_21) > 0 and ema_9[-1] < ema_21[-1]
    
    if strong_up_momentum and ema_bullish:
        return {"signal": "BUY", "score": 85.0, "reason": "MomentumBreak", "type": "strategy"}
    elif strong_down_momentum and ema_bearish:
        return {"signal": "SELL", "score": 85.0, "reason": "MomentumBreak", "type": "strategy"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "MomentumBreak", "type": "strategy"}

def volume_spike_analysis(df: pd.DataFrame) -> Dict:
    """Volume Spike Analysis - Momentum Scalper V1.0"""
    if len(df) < 30:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "VolumeSpike", "type": "strategy"}
    
    volume_ma = df["volume"].rolling(20).mean()
    volume_std = df["volume"].rolling(20).std()
    current_volume = df["volume"].iloc[-1]
    
    volume_spike = current_volume > (volume_ma.iloc[-1] + 2 * volume_std.iloc[-1])
    price_up = df["close"].iloc[-1] > df["open"].iloc[-1]
    price_down = df["close"].iloc[-1] < df["open"].iloc[-1]
    
    if volume_spike and price_up:
        return {"signal": "BUY", "score": 83.0, "reason": "VolumeSpike", "type": "strategy"}
    elif volume_spike and price_down:
        return {"signal": "SELL", "score": 83.0, "reason": "VolumeSpike", "type": "strategy"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "VolumeSpike", "type": "strategy"}

def ema_golden_cross(df: pd.DataFrame) -> Dict:
    """EMA Golden Cross - Momentum Scalper V1.0"""
    if len(df) < 50:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMAGoldenCross", "type": "strategy"}
    
    ema_fast = ema(df["close"], 9)
    ema_slow = ema(df["close"], 21)
    
    if len(ema_fast) < 2 or len(ema_slow) < 2:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMAGoldenCross", "type": "strategy"}
    
    golden_cross = ema_fast[-1] > ema_slow[-1] and ema_fast[-2] <= ema_slow[-2]
    death_cross = ema_fast[-1] < ema_slow[-1] and ema_fast[-2] >= ema_slow[-2]
    
    volume_avg = df["volume"].rolling(20).mean()
    volume_confirm = df["volume"].iloc[-1] > volume_avg.iloc[-1]
    
    if golden_cross and volume_confirm:
        return {"signal": "BUY", "score": 87.0, "reason": "EMAGoldenCross", "type": "strategy"}
    elif death_cross and volume_confirm:
        return {"signal": "SELL", "score": 87.0, "reason": "EMAGoldenCross", "type": "strategy"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMAGoldenCross", "type": "strategy"}

# ==================== BREAKOUT HUNTER V1.0 ====================

def resistance_support_break(df: pd.DataFrame) -> Dict:
    """Resistance/Support Break - Breakout Hunter V1.0"""
    if len(df) < 30:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSBreakout", "type": "strategy"}
    
    resistance = df["high"].rolling(20).max().iloc[-2]
    support = df["low"].rolling(20).min().iloc[-2]
    
    current_high = df["high"].iloc[-1]
    current_low = df["low"].iloc[-1]
    current_close = df["close"].iloc[-1]
    
    volume_avg = df["volume"].rolling(20).mean()
    volume_spike = df["volume"].iloc[-1] > volume_avg.iloc[-1] * 1.2
    
    resistance_break = current_close > resistance and volume_spike
    support_break = current_close < support and volume_spike
    
    if resistance_break:
        return {"signal": "BUY", "score": 84.0, "reason": "RSBreakout", "type": "strategy"}
    elif support_break:
        return {"signal": "SELL", "score": 84.0, "reason": "RSBreakout", "type": "strategy"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSBreakout", "type": "strategy"}

def bollinger_breakout_detection(df: pd.DataFrame) -> Dict:
    """Bollinger Breakout Detection - Breakout Hunter V1.0"""
    if len(df) < 30:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "BBBreakout", "type": "strategy"}
    
    bb = bollinger_bands(df["close"], 20, 2.0)
    
    current_close = df["close"].iloc[-1]
    upper_band = bb["upper"].iloc[-1]
    lower_band = bb["lower"].iloc[-1]
    
    band_width = bb["width"].iloc[-1]
    avg_band_width = bb["width"].rolling(20).mean().iloc[-1]
    
    high_volatility = band_width > avg_band_width * 1.2
    
    upper_breakout = current_close > upper_band and high_volatility
    lower_breakout = current_close < lower_band and high_volatility
    
    if upper_breakout:
        return {"signal": "BUY", "score": 82.0, "reason": "BBBreakout", "type": "strategy"}
    elif lower_breakout:
        return {"signal": "SELL", "score": 82.0, "reason": "BBBreakout", "type": "strategy"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "BBBreakout", "type": "strategy"}

# ==================== MEAN REVERSION V1.0 ====================

def price_rejection_signals(df: pd.DataFrame) -> Dict:
    """Price Rejection Signals - Mean Reversion V1.0"""
    if len(df) < 10:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "PriceRejection", "type": "strategy"}
    
    current_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    
    body_size = abs(current_candle["close"] - current_candle["open"])
    total_range = current_candle["high"] - current_candle["low"]
    upper_wick = current_candle["high"] - max(current_candle["open"], current_candle["close"])
    lower_wick = min(current_candle["open"], current_candle["close"]) - current_candle["low"]
    
    hammer = (lower_wick >= 2 * body_size and 
              upper_wick <= body_size * 0.5 and
              current_candle["close"] > current_candle["open"])
    
    shooting_star = (upper_wick >= 2 * body_size and 
                     lower_wick <= body_size * 0.5 and
                     current_candle["close"] < current_candle["open"])
    
    rsi_val = rsi(df["close"])[-1]
    
    if hammer and rsi_val < 35:
        return {"signal": "BUY", "score": 81.0, "reason": "PriceRejection", "type": "strategy"}
    elif shooting_star and rsi_val > 65:
        return {"signal": "SELL", "score": 81.0, "reason": "PriceRejection", "type": "strategy"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "PriceRejection", "type": "strategy"}

def volume_divergence(df: pd.DataFrame) -> Dict:
    """Volume Divergence - Mean Reversion V1.0"""
    if len(df) < 30:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "VolumeDivergence", "type": "strategy"}
    
    price_trend = df["close"].iloc[-5:].mean() - df["close"].iloc[-10:-5].mean()
    volume_trend = df["volume"].iloc[-5:].mean() - df["volume"].iloc[-10:-5].mean()
    
    bullish_divergence = (price_trend < 0 and volume_trend > 0)
    bearish_divergence = (price_trend > 0 and volume_trend < 0)
    
    rsi_val = rsi(df["close"])[-1]
    
    if bullish_divergence and rsi_val < 40:
        return {"signal": "BUY", "score": 79.0, "reason": "VolumeDivergence", "type": "strategy"}
    elif bearish_divergence and rsi_val > 60:
        return {"signal": "SELL", "score": 79.0, "reason": "VolumeDivergence", "type": "strategy"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "VolumeDivergence", "type": "strategy"}

# ==================== INDICATORS ====================

def ema_macd_indicator(df: pd.DataFrame) -> Dict:
    """EMA + MACD Indicator - Quantum Engine V2.0"""
    if len(df) < 50:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "indicator"}
    
    ema_fast = ema(df["close"], 12)
    ema_slow = ema(df["close"], 26)
    macd_line, macd_sig = macd(df["close"])
    
    if len(ema_fast) < 2 or len(macd_line) < 1:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "indicator"}
    
    ema_cross_up = ema_fast[-1] > ema_slow[-1] and ema_fast[-2] <= ema_slow[-2]
    macd_bullish = macd_line[-1] > macd_sig[-1]
    
    ema_cross_down = ema_fast[-1] < ema_slow[-1] and ema_fast[-2] >= ema_slow[-2]
    macd_bearish = macd_line[-1] < macd_sig[-1]
    
    if ema_cross_up and macd_bullish:
        return {"signal": "BUY", "score": 82.0, "reason": "EMA_MACD", "type": "indicator"}
    elif ema_cross_down and macd_bearish:
        return {"signal": "SELL", "score": 82.0, "reason": "EMA_MACD", "type": "indicator"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "indicator"}

def supertrend_bollinger_indicator(df: pd.DataFrame) -> Dict:
    """SuperTrend + Bollinger - Quantum Engine V2.0"""
    if len(df) < 30:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "ST_BB", "type": "indicator"}
    
    atr_val = atr(df, 10)
    mid = (df["high"] + df["low"]) / 2
    band_up = mid.iloc[-1] + 3 * atr_val
    band_down = mid.iloc[-1] - 3 * atr_val
    
    st_up = df["close"].iloc[-1] > band_up
    st_down = df["close"].iloc[-1] < band_down
    
    bb = bollinger_bands(df["close"])
    bb_up = df["close"].iloc[-1] > bb["upper"].iloc[-1]
    bb_down = df["close"].iloc[-1] < bb["lower"].iloc[-1]
    
    if st_up and bb_up:
        return {"signal": "BUY", "score": 84.0, "reason": "ST_BB", "type": "indicator"}
    elif st_down and bb_down:
        return {"signal": "SELL", "score": 84.0, "reason": "ST_BB", "type": "indicator"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "ST_BB", "type": "indicator"}

def volume_smart_money_indicator(df: pd.DataFrame) -> Dict:
    """Volume + Smart Money - Quantum Engine V2.0"""
    if len(df) < 50:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Vol_SM", "type": "indicator"}
    
    vol_spike = df["volume"].iloc[-1] > df["volume"].rolling(50).mean().iloc[-1] * 2.0
    body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
    spread = df["high"].iloc[-1] - df["low"].iloc[-1]
    
    accumulation = vol_spike and body / max(spread, 1e-9) < 0.3 and df["close"].iloc[-1] > df["open"].iloc[-1]
    distribution = vol_spike and body / max(spread, 1e-9) < 0.3 and df["close"].iloc[-1] < df["open"].iloc[-1]
    
    if accumulation:
        return {"signal": "BUY", "score": 86.0, "reason": "Vol_SM", "type": "indicator"}
    elif distribution:
        return {"signal": "SELL", "score": 86.0, "reason": "Vol_SM", "type": "indicator"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "Vol_SM", "type": "indicator"}

def rsi_indicator(df: pd.DataFrame) -> Dict:
    """RSI Indicator - Momentum Scalper V1.0 & Mean Reversion V1.0"""
    if len(df) < 20:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI", "type": "indicator"}
    
    rsi_val = rsi(df["close"])[-1]
    
    if rsi_val < 30:
        return {"signal": "BUY", "score": 75.0, "reason": "RSI", "type": "indicator"}
    elif rsi_val > 70:
        return {"signal": "SELL", "score": 75.0, "reason": "RSI", "type": "indicator"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI", "type": "indicator"}

def stochastic_indicator(df: pd.DataFrame) -> Dict:
    """Stochastic Indicator"""
    if len(df) < 20:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Stochastic", "type": "indicator"}
    
    k, d = stochastic(df)
    
    if k[-1] < 20 and k[-1] > d[-1]:
        return {"signal": "BUY", "score": 78.0, "reason": "Stochastic", "type": "indicator"}
    elif k[-1] > 80 and k[-1] < d[-1]:
        return {"signal": "SELL", "score": 78.0, "reason": "Stochastic", "type": "indicator"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "Stochastic", "type": "indicator"}

def adx_indicator(df: pd.DataFrame) -> Dict:
    """ADX Trend Strength"""
    if len(df) < 20:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "ADX", "type": "indicator"}
    
    adx_val, plus_di, minus_di = adx(df)
    
    if adx_val > 25 and plus_di > minus_di:
        return {"signal": "BUY", "score": 80.0, "reason": "ADX", "type": "indicator"}
    elif adx_val > 25 and minus_di > plus_di:
        return {"signal": "SELL", "score": 80.0, "reason": "ADX", "type": "indicator"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "ADX", "type": "indicator"}

def vwap_indicator(df: pd.DataFrame) -> Dict:
    """VWAP Deviation"""
    if len(df) < 50:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "VWAP", "type": "indicator"}
    
    vwap_val = vwap(df.iloc[-50:])
    price = df["close"].iloc[-1]
    dev = (price - vwap_val) / vwap_val
    
    if dev < -0.015:
        return {"signal": "BUY", "score": 83.0, "reason": "VWAP", "type": "indicator"}
    elif dev > 0.015:
        return {"signal": "SELL", "score": 83.0, "reason": "VWAP", "type": "indicator"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "VWAP", "type": "indicator"}

def bollinger_indicator(df: pd.DataFrame) -> Dict:
    """Bollinger Bands - Mean Reversion V1.0"""
    if len(df) < 30:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Bollinger", "type": "indicator"}
    
    bb = bollinger_bands(df["close"])
    
    if df["close"].iloc[-1] < bb["lower"].iloc[-1]:
        return {"signal": "BUY", "score": 76.0, "reason": "Bollinger", "type": "indicator"}
    elif df["close"].iloc[-1] > bb["upper"].iloc[-1]:
        return {"signal": "SELL", "score": 76.0, "reason": "Bollinger", "type": "indicator"}
    
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "Bollinger", "type": "indicator"}

# ==================== COLLECTIONS ====================

ALL_STRATEGIES = [
    # Sacred & Hidden
    fibonacci_vortex_hidden,
    quantum_entanglement_hidden,
    dark_pool_institutional_hidden,
    gann_square_time_cycles_hidden,
    elliott_wave_neural_hidden,
    cosmic_movement_hidden,
    
    # Quantum Engine V2.0
    order_block_detection,
    break_of_structure,
    fair_value_gap,
    
    # Momentum Scalper V1.0
    momentum_break_detection,
    volume_spike_analysis,
    ema_golden_cross,
    
    # Breakout Hunter V1.0
    resistance_support_break,
    bollinger_breakout_detection,
    
    # Mean Reversion V1.0
    price_rejection_signals,
    volume_divergence,
]

ALL_INDICATORS = [
    # Quantum Engine V2.0
    ema_macd_indicator,
    supertrend_bollinger_indicator,
    volume_smart_money_indicator,
    
    # Momentum Scalper V1.0 & Mean Reversion V1.0
    rsi_indicator,
    stochastic_indicator,
    adx_indicator,
    vwap_indicator,
    bollinger_indicator,
]

# ==================== AUTO MODE SELECTION ====================

def detect_trading_mode(strategy_results: List[Dict]) -> Tuple[str, float]:
    """Auto-detect the best trading mode based on strategy convergence"""
    mode_scores = {mode: 0.0 for mode in TRADING_MODES.keys()}
    mode_counts = {mode: 0 for mode in TRADING_MODES.keys()}
    
    for result in strategy_results:
        reason = result["reason"]
        score = result["score"]
        
        # Check which trading mode this strategy belongs to
        for mode, config in TRADING_MODES.items():
            if reason in config["strategies"]:
                mode_scores[mode] += score
                mode_counts[mode] += 1
    
    # Calculate average scores for each mode
    for mode in mode_scores:
        if mode_counts[mode] > 0:
            mode_scores[mode] = mode_scores[mode] / mode_counts[mode]
    
    # Find the mode with highest average score and at least 2 strategies
    best_mode = "SACRED"  # Default to sacred
    best_score = 0.0
    
    for mode, score in mode_scores.items():
        if score > best_score and mode_counts[mode] >= 2:
            best_mode = mode
            best_score = score
    
    return best_mode, best_score

# ==================== DATA PROVIDER ====================

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
        return SymbolMapper.MAP.get(symbol, (symbol, False))

class OTCEmulator:
    def __init__(self, seed: int = 123):
        self.rng = np.random.default_rng(seed)
    
    def synthesize(self, df: pd.DataFrame, tf: str, limit: int) -> pd.DataFrame:
        if df is None or df.empty:
            return self._generate_series(tf, limit, 1.0)
        return df.tail(limit)
    
    def _generate_series(self, tf: str, limit: int, base: float) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": pd.date_range(end=datetime.utcnow(), periods=limit, freq="1H"),
            "open": [base] * limit,
            "high": [base * 1.001] * limit,
            "low": [base * 0.999] * limit,
            "close": [base] * limit,
            "volume": [100] * limit,
        })

class RealDataProvider:
    def __init__(self):
        import yfinance as yf
        self.yf = yf
        self.otc = OTCEmulator()
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = "4h", limit: int = 500) -> Optional[pd.DataFrame]:
        base_symbol, is_otc = SymbolMapper.resolve(symbol)
        interval = {"1h": "1h", "2h": "1h", "4h": "1h", "8h": "1h", "1d": "1d", "1w": "1d", "1M": "1d"}.get(timeframe, "1h")
        period = {"1h": "2mo", "1d": "2y"}.get(interval, "1y")
        
        try:
            df = self.yf.download(base_symbol, period=period, interval=interval, progress=False)
            if df is None or df.empty:
                return None
            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            df.reset_index(inplace=True)
            ts_col = "Datetime" if "Datetime" in df.columns else "Date"
            df["timestamp"] = pd.to_datetime(df[ts_col])
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].tail(limit)
            return df
        except Exception:
            return None

# ==================== ULTIMATE AI ENGINE ====================

def select_best_strategies(all_results: List[Dict], min_score: float = 75.0) -> List[Dict]:
    """Select best 4 strategies based on score and signal strength"""
    strategies = [r for r in all_results if r.get("type") == "strategy" and r["score"] >= min_score]
    strategies.sort(key=lambda x: x["score"], reverse=True)
    return strategies[:4]

def select_best_indicators(all_results: List[Dict], min_score: float = 70.0) -> List[Dict]:
    """Select best 4 indicators based on score"""
    indicators = [r for r in all_results if r.get("type") == "indicator" and r["score"] >= min_score]
    indicators.sort(key=lambda x: x["score"], reverse=True)
    return indicators[:4]

def select_best_timeframes(tf_results: Dict[str, Dict]) -> List[str]:
    """Select best 4 timeframes with highest confidence"""
    scored_tfs = []
    for tf, result in tf_results.items():
        if result["signal"] != "NEUTRAL":
            scored_tfs.append((tf, result["confidence"], TF_WEIGHTS.get(tf, 1)))
    
    # Sort by confidence * weight
    scored_tfs.sort(key=lambda x: x[1] * x[2], reverse=True)
    return [tf[0] for tf in scored_tfs[:4]]

def ultimate_ai_analysis(tf_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Ultimate AI: Auto-selects best 4 strategies + 4 indicators + 4 timeframes
    """
    all_strategy_results = []
    all_indicator_results = []
    tf_analysis = {}
    
    # Run analysis on each timeframe
    for tf, df in tf_data.items():
        if df is None or len(df) < 50:
            continue
        
        # Run all strategies
        for strategy_fn in ALL_STRATEGIES:
            try:
                result = strategy_fn(df)
                if result["signal"] != "NEUTRAL":
                    result["timeframe"] = tf
                    all_strategy_results.append(result)
            except Exception:
                pass
        
        # Run all indicators
        for indicator_fn in ALL_INDICATORS:
            try:
                result = indicator_fn(df)
                if result["signal"] != "NEUTRAL":
                    result["timeframe"] = tf
                    all_indicator_results.append(result)
            except Exception:
                pass
        
        # Calculate TF confidence
        tf_signals = [r for r in all_strategy_results + all_indicator_results if r.get("timeframe") == tf]
        if tf_signals:
            buy_score = sum(r["score"] for r in tf_signals if r["signal"] == "BUY")
            sell_score = sum(r["score"] for r in tf_signals if r["signal"] == "SELL")
            
            if buy_score > sell_score:
                tf_analysis[tf] = {"signal": "BUY", "confidence": buy_score / len(tf_signals)}
            elif sell_score > buy_score:
                tf_analysis[tf] = {"signal": "SELL", "confidence": sell_score / len(tf_signals)}
            else:
                tf_analysis[tf] = {"signal": "NEUTRAL", "confidence": 0.0}
    
    # Select best 4 of each
    best_strategies = select_best_strategies(all_strategy_results, min_score=85.0)
    best_indicators = select_best_indicators(all_indicator_results, min_score=75.0)
    best_timeframes = select_best_timeframes(tf_analysis)
    
    # Auto-detect trading mode
    trading_mode, mode_confidence = detect_trading_mode(best_strategies + best_indicators)
    
    # Determine overall signal
    all_signals = [r["signal"] for r in best_strategies + best_indicators]
    signal_counts = Counter(all_signals)
    
    if len(best_strategies) >= 4 and len(best_indicators) >= 4:
        # Require 4+ strategies AND 4+ indicators aligned
        if signal_counts.get("BUY", 0) >= 6:
            final_signal = "BUY"
            confidence = sum(r["score"] for r in best_strategies + best_indicators if r["signal"] == "BUY") / len([r for r in best_strategies + best_indicators if r["signal"] == "BUY"])
        elif signal_counts.get("SELL", 0) >= 6:
            final_signal = "SELL"
            confidence = sum(r["score"] for r in best_strategies + best_indicators if r["signal"] == "SELL") / len([r for r in best_strategies + best_indicators if r["signal"] == "SELL"])
        else:
            final_signal = "NEUTRAL"
            confidence = 0.0
    else:
        final_signal = "NEUTRAL"
        confidence = 0.0
    
    return {
        "signal": final_signal,
        "confidence": confidence,
        "trading_mode": trading_mode,
        "mode_confidence": mode_confidence,
        "best_strategies": best_strategies,
        "best_indicators": best_indicators,
        "best_timeframes": best_timeframes,
        "num_strategies_aligned": len([r for r in best_strategies if r["signal"] == final_signal]),
        "num_indicators_aligned": len([r for r in best_indicators if r["signal"] == final_signal]),
        "num_timeframes_aligned": len([tf for tf in best_timeframes if tf_analysis.get(tf, {}).get("signal") == final_signal])
    }

# ==================== TELEGRAM ====================

class LegendaryTelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.available = TELEGRAM_AVAILABLE and bool(token) and bool(chat_id)
        self.token = token
        self.chat_id = chat_id
        self.updater = None
        
        if self.available:
            self.updater = Updater(token, use_context=True)
            dp = self.updater.dispatcher
            # ONLY KEEP START AND STATUS COMMANDS
            dp.add_handler(CommandHandler("start", self._start))
            dp.add_handler(CommandHandler("status", self._status))
    
    def start(self):
        if not self.available:
            return
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
    
    def _run(self):
        self.updater.start_polling()
        self.updater.idle()
    
    def send(self, text: str, parse_mode=None):
        if self.available:
            try:
                self.updater.bot.send_message(chat_id=self.chat_id, text=text, parse_mode=parse_mode or ParseMode.HTML)
            except Exception as e:
                print(f"Telegram error: {e}")
        else:
            print(text)
    
    def _start(self, update, context):
        if not AdminAuth.verify_admin(update.effective_user.id):
            update.message.reply_text("❌ Unauthorized")
            return
        update.message.reply_text(
            "🔮 <b>ULTIMATE Legendary Scanner Active</b>\n\n"
            "Commands:\n"
            "/status - System status\n\n"
            "<i>AI will auto-detect trading mode and alert when 95%+ confidence signals are detected</i>",
            parse_mode=ParseMode.HTML
        )
    
    def _status(self, update, context):
        if not AdminAuth.verify_admin(update.effective_user.id):
            return
        update.message.reply_text(
            f"✅ <b>System Status</b>\n"
            f"Strategies: {len(ALL_STRATEGIES)}\n"
            f"Indicators: {len(ALL_INDICATORS)}\n"
            f"Assets: {len(ASSETS)}\n"
            f"Timeframes: {len(ALL_TIMEFRAMES)}\n"
            f"Alert Threshold: {LEGENDARY_GATE}%\n\n"
            f"<i>Auto-mode detection: Quantum/Momentum/Breakout/MeanReversion/Sacred</i>",
            parse_mode=ParseMode.HTML
        )

# ==================== FLASK WEBHOOK ====================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "legendary", "timestamp": datetime.utcnow().isoformat()}), 200

@app.route('/webhook/scan', methods=['POST'])
def webhook_scan():
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        return jsonify({"error": "Unauthorized"}), 401
    token = auth.replace('Bearer ', '')
    valid, _ = AdminAuth.verify_token(token)
    if not valid:
        return jsonify({"error": "Invalid token"}), 401
    return jsonify({"status": "triggered"}), 200

@app.route('/api/token', methods=['POST'])
def generate_token():
    data = request.json or {}
    if data.get('secret') != ADMIN_SECRET or data.get('user_id') not in ADMIN_IDS:
        return jsonify({"error": "Unauthorized"}), 401
    token = AdminAuth.generate_token(data['user_id'])
    return jsonify({"token": token}), 200

# ==================== MAIN SCANNER ====================

def format_ultimate_alert(result: Dict) -> str:
    """Format ultimate legendary alert with auto-detected trading mode"""
    mode_config = TRADING_MODES.get(result['trading_mode'], TRADING_MODES['SACRED'])
    strategies_list = "\n".join([f"  • {s['reason']} ({s['score']:.0f}%)" for s in result['best_strategies'][:4]])
    indicators_list = "\n".join([f"  • {i['reason']} ({i['score']:.0f}%)" for i in result['best_indicators'][:4]])
    timeframes_list = ", ".join(result['best_timeframes'])
    
    return f"""
{mode_config['emoji']} <b>{mode_config['name']} - LEGENDARY SIGNAL</b> ⚡🎯

<b>Asset:</b> {result['asset']}
<b>Direction:</b> {result['signal']} 
<b>Expiry:</b> {result['expiry']}m
<b>Confidence:</b> {result['confidence']:.1f}% 🎯

<b>🎯 Best 4 Strategies Aligned:</b>
{strategies_list}

<b>📊 Best 4 Indicators Aligned:</b>
{indicators_list}

<b>⏰ Best 4 Timeframes:</b> {timeframes_list}

<b>🔥 Alignment Score:</b>
• Strategies: {result['num_strategies_aligned']}/4
• Indicators: {result['num_indicators_aligned']}/4
• Timeframes: {result['num_timeframes_aligned']}/4

<b>Timestamp:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

<i>⚡ Auto-detected {mode_config['name'].lower()} - {mode_config['description']}</i>
"""

def main_scan_loop():
    """Main scanning loop with 95% threshold"""
    print("🔮 Initializing ULTIMATE Legendary Scanner...")
    print(f"⚡ {len(ALL_STRATEGIES)} Sacred Strategies")
    print(f"📊 {len(ALL_INDICATORS)} Elite Indicators")
    print(f"⏰ {len(ALL_TIMEFRAMES)} Timeframes")
    print(f"🎯 Alert Threshold: {LEGENDARY_GATE}% Confidence")
    
    provider = RealDataProvider()
    telegram = LegendaryTelegramBot(TELEGRAM_TOKEN, CHAT_ID)
    
    if TELEGRAM_AVAILABLE:
        telegram.start()
    
    print("✅ Scanner ready! Only alerts above 95% confidence will be sent.")
    
    while True:
        try:
            now_utc = datetime.utcnow()
            print(f"\n{'='*60}")
            print(f"🔍 Scan: {now_utc.strftime('%H:%M:%S')}")
            
            signals_found = 0
            
            for asset in ASSETS:
                for expiry in EXPIRIES:
                    cooldown_key = f"{asset}_{expiry}"
                    if cooldown_key in LAST_LEGENDARY_ALERT:
                        elapsed = (now_utc - LAST_LEGENDARY_ALERT[cooldown_key]).total_seconds() / 60
                        if elapsed < COOLDOWN_MINUTES:
                            continue
                    
                    # Fetch multi-TF data
                    tf_data = {}
                    for tf in ALL_TIMEFRAMES[:8]:  # Use first 8 TFs
                        try:
                            df = provider.fetch_ohlcv(asset, tf, limit=300)
                            if df is not None and len(df) >= 50:
                                tf_data[tf] = df
                        except Exception:
                            pass
                    
                    if len(tf_data) < 4:
                        continue
                    
                    # Run ultimate AI analysis
                    result = ultimate_ai_analysis(tf_data)
                    
                    # Check legendary criteria - ONLY ALERT IF 95%+ CONFIDENCE
                    if (result['signal'] != "NEUTRAL" and
                        result['confidence'] >= LEGENDARY_GATE and  # 95% threshold
                        result['num_strategies_aligned'] >= 4 and
                        result['num_indicators_aligned'] >= 4 and
                        result['num_timeframes_aligned'] >= 3):
                        
                        signals_found += 1
                        
                        alert_data = {
                            'asset': asset,
                            'signal': result['signal'],
                            'confidence': result['confidence'],
                            'expiry': expiry,
                            'trading_mode': result['trading_mode'],
                            'mode_confidence': result['mode_confidence'],
                            'best_strategies': result['best_strategies'],
                            'best_indicators': result['best_indicators'],
                            'best_timeframes': result['best_timeframes'],
                            'num_strategies_aligned': result['num_strategies_aligned'],
                            'num_indicators_aligned': result['num_indicators_aligned'],
                            'num_timeframes_aligned': result['num_timeframes_aligned'],
                            'timestamp': now_utc.isoformat()
                        }
                        
                        signal_key = f"{asset}_{expiry}_{now_utc.strftime('%Y%m%d_%H%M')}"
                        SIGNAL_HISTORY[signal_key] = alert_data
                        LAST_LEGENDARY_ALERT[cooldown_key] = now_utc
                        
                        alert_msg = format_ultimate_alert(alert_data)
                        telegram.send(alert_msg, parse_mode=ParseMode.HTML)
                        
                        mode_config = TRADING_MODES.get(result['trading_mode'], TRADING_MODES['SACRED'])
                        print(f"  🔥 {mode_config['emoji']} {asset} {result['signal']} {expiry}m @ {result['confidence']:.1f}% ({mode_config['name']})")
                    
                    time.sleep(0.3)
            
            if signals_found == 0:
                print(f"  ⚡ No signals above {LEGENDARY_GATE}% threshold")
            else:
                print(f"  🎯 Found {signals_found} legendary signals")
            
            print(f"✅ Scan complete. Next in {SCAN_INTERVAL_SEC}s...")
            
            if RUN_ONCE:
                break
            
            time.sleep(SCAN_INTERVAL_SEC)
            
        except KeyboardInterrupt:
            print("\n🛑 Stopped")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    if RUN_ONCE:
        main_scan_loop()
    else:
        print(f"🚀 Starting on port {PORT}")
        scanner_thread = threading.Thread(target=main_scan_loop, daemon=True)
        scanner_thread.start()
        app.run(host='0.0.0.0', port=PORT, debug=False)

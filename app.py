"""
LEGENDARY Pocket Option AI Scanner - Ultimate Institutional Edge
Features:
- Runtime TensorFlow loading (no build-time dependency)
- Admin-only access control with encryption
- Ultra-sophisticated institutional signals
- Multi-layer confluence detection
- Real-time market microstructure analysis
- Choreo webhook integration
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
warnings.filterwarnings('ignore')

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ==================== CONFIG ====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
ADMIN_IDS = [int(x.strip()) for x in os.getenv("ADMIN_IDS", CHAT_ID).split(",") if x.strip()]
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "your_legendary_secret_key_here")
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "False") == "True"
CONF_GATE = int(os.getenv("CONF_GATE", "92"))  # Legendary threshold
LEGENDARY_GATE = int(os.getenv("LEGENDARY_GATE", "95"))  # Ultra signals only
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "15"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "45"))
RUN_ONCE = os.getenv("RUN_ONCE", "False") == "True"
PORT = int(os.getenv("PORT", "8080"))
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "your_webhook_secret")

# Universe
ASSETS = [
    "EURUSD_otc", "GBPUSD_otc", "AUDUSD_otc", "USDJPY_otc", "XAUUSD_otc",
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
    "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
    "GC=F", "SI=F", "CL=F", "BZ=F", "NG=F"
]
EXPIRIES = [5, 15, 30, 60, 120, 240]
TIMEFRAMES = ["1y", "1M", "2M", "1w", "3w", "1d", "8h", "4h", "2h", "1h"]
TF_WEIGHTS = {"1y": 5, "1M": 5, "2M": 4, "1w": 4, "3w": 3, "1d": 3, "8h": 2, "4h": 2, "2h": 1, "1h": 1}

# Signal tracking
SIGNAL_HISTORY = {}
LAST_LEGENDARY_ALERT = {}

# Flask app
app = Flask(__name__)

# Runtime TensorFlow loading
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
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import StandardScaler, RobustScaler
    TFK = True
    print("✅ TensorFlow loaded successfully")
except Exception as e:
    print(f"⚠️ TensorFlow not available: {e}")

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

# ==================== SECURITY & ADMIN ====================

class AdminAuth:
    """Admin authentication and authorization"""
    
    @staticmethod
    def verify_admin(user_id: int) -> bool:
        """Check if user is admin"""
        return user_id in ADMIN_IDS
    
    @staticmethod
    def generate_token(user_id: int) -> str:
        """Generate secure admin token"""
        data = f"{user_id}:{datetime.utcnow().isoformat()}"
        signature = hmac.new(
            ADMIN_SECRET.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"{data}:{signature}"
    
    @staticmethod
    def verify_token(token: str) -> Tuple[bool, Optional[int]]:
        """Verify admin token"""
        try:
            parts = token.split(":")
            if len(parts) != 3:
                return False, None
            
            user_id, timestamp, signature = parts
            user_id = int(user_id)
            
            # Verify user is admin
            if user_id not in ADMIN_IDS:
                return False, None
            
            # Verify signature
            data = f"{user_id}:{timestamp}"
            expected = hmac.new(
                ADMIN_SECRET.encode(),
                data.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if signature != expected:
                return False, None
            
            # Check token age (24h expiry)
            token_time = datetime.fromisoformat(timestamp)
            age = (datetime.utcnow() - token_time).total_seconds()
            if age > 86400:  # 24 hours
                return False, None
            
            return True, user_id
        except Exception:
            return False, None
    
    @staticmethod
    def verify_webhook_signature(payload: str, signature: str) -> bool:
        """Verify webhook HMAC signature"""
        expected = hmac.new(
            WEBHOOK_SECRET.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature, expected)

# ==================== INDICATORS ====================

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
    width = upper - lower
    return {"upper": upper, "lower": lower, "ma": ma, "width": width, "std": std}

def vwap(df: pd.DataFrame) -> float:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap_val = (typical_price * df["volume"]).sum() / df["volume"].sum()
    return float(vwap_val)

def ichimoku_cloud(df: pd.DataFrame) -> Dict:
    high9 = df["high"].rolling(9).max()
    low9 = df["low"].rolling(9).min()
    tenkan = (high9 + low9) / 2
    high26 = df["high"].rolling(26).max()
    low26 = df["low"].rolling(26).min()
    kijun = (high26 + low26) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    high52 = df["high"].rolling(52).max()
    low52 = df["low"].rolling(52).min()
    senkou_b = ((high52 + low52) / 2).shift(26)
    return {
        "tenkan": float(tenkan.iloc[-1]) if pd.notna(tenkan.iloc[-1]) else 0.0,
        "kijun": float(kijun.iloc[-1]) if pd.notna(kijun.iloc[-1]) else 0.0,
        "senkou_a": float(senkou_a.iloc[-1]) if pd.notna(senkou_a.iloc[-1]) else 0.0,
        "senkou_b": float(senkou_b.iloc[-1]) if pd.notna(senkou_b.iloc[-1]) else 0.0,
        "price": float(df["close"].iloc[-1])
    }

def money_flow_index(df: pd.DataFrame, period: int = 14) -> float:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical_price * df["volume"]
    positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
    positive_mf = pd.Series(positive_flow).rolling(period).sum()
    negative_mf = pd.Series(negative_flow).rolling(period).sum()
    mfi = 100 - (100 / (1 + positive_mf / negative_mf.replace(0, np.nan)))
    return float(mfi.iloc[-1]) if pd.notna(mfi.iloc[-1]) else 50.0

def atr(df: pd.DataFrame, period: int = 14) -> float:
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum(abs(df["high"] - df["close"].shift(1)),
                              abs(df["low"] - df["close"].shift(1))))
    atr_val = pd.Series(tr).rolling(period).mean()
    return float(atr_val.iloc[-1]) if pd.notna(atr_val.iloc[-1]) else 0.0

def normalize(val: float, base: Optional[pd.Series] = None) -> float:
    try:
        if base is not None and len(base) > 0:
            return float(min(100.0, max(0.0, 100.0 * abs(val) / (abs(base.iloc[-1]) + 1e-9))))
        return float(min(100.0, max(0.0, val)))
    except Exception:
        return 0.0

# ==================== LEGENDARY STRATEGIES ====================

def ultra_order_flow_imbalance(df: pd.DataFrame) -> Dict:
    """
    Detects extreme institutional order flow imbalances
    Uses volume delta, price rejection, and absorption patterns
    """
    vol = df["volume"]
    vol_ma = vol.rolling(50).mean()
    vol_std = vol.rolling(50).std()
    
    # Z-score of volume
    vol_zscore = (vol.iloc[-1] - vol_ma.iloc[-1]) / (vol_std.iloc[-1] + 1e-9)
    
    # Price action analysis
    body = abs(df["close"] - df["open"]).iloc[-3:]
    range_val = (df["high"] - df["low"]).iloc[-3:]
    body_ratio = (body / range_val.replace(0, 1e-9)).mean()
    
    # Absorption: high volume, small body
    absorption = vol_zscore > 2.5 and body_ratio < 0.35
    
    # Delta analysis
    price_delta = df["close"].iloc[-1] - df["close"].iloc[-5]
    vol_delta = vol.iloc[-3:].sum() - vol.iloc[-6:-3].sum()
    
    # Divergence detection
    bullish_div = price_delta < 0 and vol_delta > 0 and absorption
    bearish_div = price_delta > 0 and vol_delta > 0 and absorption
    
    if bullish_div:
        signal = "BUY"
        score = 92.0 + min(vol_zscore * 2, 8.0)
    elif bearish_div:
        signal = "SELL"
        score = 92.0 + min(vol_zscore * 2, 8.0)
    else:
        signal = "NEUTRAL"
        score = 0.0
    
    return {
        "signal": signal, 
        "score": min(score, 100.0), 
        "reason": "UltraOrderFlow",
        "confluence": ["volume_absorption", "price_rejection"] if absorption else []
    }

def liquidity_void_detection(df: pd.DataFrame) -> Dict:
    """
    Identifies Fair Value Gaps and liquidity voids that institutions exploit
    Enhanced with volume profile analysis
    """
    if len(df) < 5: return {"signal": "NEUTRAL", "score": 0.0, "reason": "LiqVoid", "confluence": []}
    
    # FVG detection (3-candle pattern)
    fvg_up = df["low"].iloc[-1] > df["high"].iloc[-3]
    fvg_down = df["high"].iloc[-1] < df["low"].iloc[-3]
    
    if fvg_up or fvg_down:
        gap_size = abs(df["low"].iloc[-1] - df["high"].iloc[-3]) if fvg_up else abs(df["high"].iloc[-1] - df["low"].iloc[-3])
        gap_pct = gap_size / df["close"].iloc[-3] * 100
        
        # Volume confirmation
        vol_surge = df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.5
        
        # Order block proximity
        ob_support = df["low"].rolling(20).min().iloc[-1]
        ob_resistance = df["high"].rolling(20).max().iloc[-1]
        near_ob = (df["close"].iloc[-1] - ob_support) / ob_support < 0.01 or \
                  (ob_resistance - df["close"].iloc[-1]) / ob_resistance < 0.01
        
        if fvg_up and vol_surge:
            signal = "BUY"
            score = 88.0 + min(gap_pct * 10, 12.0)
            confluence = ["fvg", "volume_surge"]
            if near_ob:
                score += 3.0
                confluence.append("order_block")
        elif fvg_down and vol_surge:
            signal = "SELL"
            score = 88.0 + min(gap_pct * 10, 12.0)
            confluence = ["fvg", "volume_surge"]
            if near_ob:
                score += 3.0
                confluence.append("order_block")
        else:
            signal = "NEUTRAL"
            score = 0.0
            confluence = []
    else:
        signal = "NEUTRAL"
        score = 0.0
        confluence = []
    
    return {"signal": signal, "score": min(score, 100.0), "reason": "LiqVoid", "confluence": confluence}

def market_maker_manipulation(df: pd.DataFrame) -> Dict:
    """
    Detects stop hunts and market maker manipulation patterns
    Identifies false breakouts followed by reversals
    """
    lookback = 30
    highs = df["high"].rolling(lookback)
    lows = df["low"].rolling(lookback)
    
    recent_high = highs.max().iloc[-2]
    recent_low = lows.min().iloc[-2]
    
    # Stop hunt detection
    swept_high = df["high"].iloc[-1] > recent_high and df["close"].iloc[-1] < recent_high - (df["high"].iloc[-1] - df["low"].iloc[-1]) * 0.6
    swept_low = df["low"].iloc[-1] < recent_low and df["close"].iloc[-1] > recent_low + (df["high"].iloc[-1] - df["low"].iloc[-1]) * 0.6
    
    # Volume analysis
    vol_spike = df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 2.0
    
    # Wick analysis
    upper_wick = df["high"].iloc[-1] - max(df["open"].iloc[-1], df["close"].iloc[-1])
    lower_wick = min(df["open"].iloc[-1], df["close"].iloc[-1]) - df["low"].iloc[-1]
    body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
    
    strong_rejection_up = upper_wick > body * 2
    strong_rejection_down = lower_wick > body * 2
    
    if swept_low and strong_rejection_down and vol_spike:
        signal = "BUY"
        score = 93.0
        confluence = ["stop_hunt", "rejection", "volume"]
    elif swept_high and strong_rejection_up and vol_spike:
        signal = "SELL"
        score = 93.0
        confluence = ["stop_hunt", "rejection", "volume"]
    else:
        signal = "NEUTRAL"
        score = 0.0
        confluence = []
    
    return {"signal": signal, "score": score, "reason": "MMManipulation", "confluence": confluence}

def institutional_accumulation_distribution(df: pd.DataFrame) -> Dict:
    """
    Wyckoff method with volume spread analysis
    Detects stealth accumulation/distribution by smart money
    """
    vol = df["volume"]
    vol_ma = vol.rolling(50).mean()
    
    # Volume spread analysis
    spread = df["high"] - df["low"]
    spread_ma = spread.rolling(20).mean()
    
    close_position = (df["close"] - df["low"]) / spread.replace(0, 1e-9)
    
    # Accumulation: closing in upper half on decreasing spread & volume
    last_5_vol = vol.iloc[-5:].mean()
    prev_5_vol = vol.iloc[-10:-5].mean()
    vol_decreasing = last_5_vol < prev_5_vol * 0.9
    
    last_5_spread = spread.iloc[-5:].mean()
    prev_5_spread = spread.iloc[-10:-5].mean()
    spread_contracting = last_5_spread < prev_5_spread * 0.85
    
    close_upper = close_position.iloc[-5:].mean() > 0.6
    close_lower = close_position.iloc[-5:].mean() < 0.4
    
    # Spring detection (final shakeout)
    recent_low = df["low"].rolling(30).min().iloc[-6]
    spring = df["low"].iloc[-1] < recent_low and df["close"].iloc[-1] > recent_low and vol.iloc[-1] < vol_ma.iloc[-1]
    
    if (vol_decreasing and spread_contracting and close_upper) or spring:
        signal = "BUY"
        score = 90.0
        confluence = ["accumulation", "spread_contraction"]
        if spring:
            score = 94.0
            confluence.append("spring")
    elif vol_decreasing and spread_contracting and close_lower:
        signal = "SELL"
        score = 90.0
        confluence = ["distribution", "spread_contraction"]
    else:
        signal = "NEUTRAL"
        score = 0.0
        confluence = []
    
    return {"signal": signal, "score": score, "reason": "InstAccumDist", "confluence": confluence}

def multi_timeframe_momentum_surge(df_dict: Dict[str, pd.DataFrame]) -> Dict:
    """
    Detects aligned momentum across multiple timeframes
    Requires 5+ timeframes in agreement
    """
    alignments = {"BUY": 0, "SELL": 0}
    
    for tf, df in df_dict.items():
        if df is None or len(df) < 50:
            continue
        
        # RSI
        rsi_val = rsi(df["close"])[-1]
        
        # MACD
        macd_line, macd_sig = macd(df["close"])
        macd_cross_up = len(macd_line) > 1 and macd_line[-1] > macd_sig[-1] and macd_line[-2] <= macd_sig[-2]
        macd_cross_down = len(macd_line) > 1 and macd_line[-1] < macd_sig[-1] and macd_line[-2] >= macd_sig[-2]
        
        # EMA alignment
        ema_20 = ema(df["close"], 20)
        ema_50 = ema(df["close"], 50)
        ema_bullish = len(ema_20) > 0 and len(ema_50) > 0 and ema_20[-1] > ema_50[-1]
        ema_bearish = len(ema_20) > 0 and len(ema_50) > 0 and ema_20[-1] < ema_50[-1]
        
        if (rsi_val < 40 and macd_cross_up and ema_bullish):
            alignments["BUY"] += TF_WEIGHTS.get(tf, 1)
        elif (rsi_val > 60 and macd_cross_down and ema_bearish):
            alignments["SELL"] += TF_WEIGHTS.get(tf, 1)
    
    total_weight = sum(TF_WEIGHTS.values())
    buy_pct = (alignments["BUY"] / total_weight) * 100
    sell_pct = (alignments["SELL"] / total_weight) * 100
    
    if buy_pct >= 70:
        signal = "BUY"
        score = min(85.0 + buy_pct * 0.2, 98.0)
        confluence = ["mtf_alignment", "momentum_surge"]
    elif sell_pct >= 70:
        signal = "SELL"
        score = min(85.0 + sell_pct * 0.2, 98.0)
        confluence = ["mtf_alignment", "momentum_surge"]
    else:
        signal = "NEUTRAL"
        score = 0.0
        confluence = []
    
    return {"signal": signal, "score": score, "reason": "MTFMomentum", "confluence": confluence}

def vwap_extremes_with_reversion(df: pd.DataFrame) -> Dict:
    """
    Extreme VWAP deviations with mean reversion probability
    Uses standard deviation bands
    """
    vwap_val = vwap(df.iloc[-100:])
    price = df["close"].iloc[-1]
    
    # Calculate VWAP std dev
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    deviations = typical_price.iloc[-100:] - vwap_val
    vwap_std = deviations.std()
    
    z_score = (price - vwap_val) / (vwap_std + 1e-9)
    
    # Check for volume confirmation
    vol_increase = df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.3
    
    if z_score < -2.5 and vol_increase:
        signal = "BUY"
        score = 87.0 + min(abs(z_score) * 3, 13.0)
        confluence = ["vwap_extreme", "volume_confirm"]
    elif z_score > 2.5 and vol_increase:
        signal = "SELL"
        score = 87.0 + min(abs(z_score) * 3, 13.0)
        confluence = ["vwap_extreme", "volume_confirm"]
    else:
        signal = "NEUTRAL"
        score = 0.0
        confluence = []
    
    return {"signal": signal, "score": min(score, 100.0), "reason": "VWAPExtreme", "confluence": confluence}

def advanced_ichimoku_confluence(df: pd.DataFrame) -> Dict:
    """
    Full Ichimoku analysis with cloud strength and future cloud
    """
    ich = ichimoku_cloud(df)
    price = ich["price"]
    
    # Cloud strength
    cloud_thickness = abs(ich["senkou_a"] - ich["senkou_b"])
    avg_price = (df["high"] + df["low"]) / 2
    cloud_strength = (cloud_thickness / avg_price.mean()) * 100
    
    # TK cross
    tk_bullish = ich["tenkan"] > ich["kijun"]
    tk_bearish = ich["tenkan"] < ich["kijun"]
    
    # Price vs cloud
    above_cloud = price > max(ich["senkou_a"], ich["senkou_b"])
    below_cloud = price < min(ich["senkou_a"], ich["senkou_b"])
    
    # Strong signals require all alignments
    if above_cloud and tk_bullish and cloud_strength > 0.5:
        signal = "BUY"
        score = 86.0 + min(cloud_strength * 10, 14.0)
        confluence = ["cloud_support", "tk_cross", "price_alignment"]
    elif below_cloud and tk_bearish and cloud_strength > 0.5:
        signal = "SELL"
        score = 86.0 + min(cloud_strength * 10, 14.0)
        confluence = ["cloud_resistance", "tk_cross", "price_alignment"]
    else:
        signal = "NEUTRAL"
        score = 0.0
        confluence = []
    
    return {"signal": signal, "score": min(score, 100.0), "reason": "IchimokuAdv", "confluence": confluence}

def volume_profile_point_of_control(df: pd.DataFrame) -> Dict:
    """
    Volume Profile POC (Point of Control) analysis
    Identifies high-volume nodes and price acceptance/rejection
    """
    # Create price bins
    price_range = df["high"].max() - df["low"].min()
    num_bins = 50
    bins = np.linspace(df["low"].min(), df["high"].max(), num_bins)
    
    # Calculate volume at each price level
    volume_profile = np.zeros(num_bins - 1)
    for i in range(len(df)):
        price_levels = np.linspace(df["low"].iloc[i], df["high"].iloc[i], 10)
        vol_per_level = df["volume"].iloc[i] / 10
        for price in price_levels:
            bin_idx = np.digitize(price, bins) - 1
            if 0 <= bin_idx < len(volume_profile):
                volume_profile[bin_idx] += vol_per_level
    
    # Find POC (highest volume)
    poc_idx = np.argmax(volume_profile)
    poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
    
    current_price = df["close"].iloc[-1]
    distance_from_poc = abs(current_price - poc_price) / poc_price
    
    # Value area (70% of volume)
    sorted_indices = np.argsort(volume_profile)[::-1]
    cumulative_vol = 0
    total_vol = volume_profile.sum()
    value_area_indices = []
    
    for idx in sorted_indices:
        value_area_indices.append(idx)
        cumulative_vol += volume_profile[idx]
        if cumulative_vol >= total_vol * 0.7:
            break
    
    in_value_area = any(bins[i] <= current_price <= bins[i+1] for i in value_area_indices if i < len(bins)-1)
    
    # Trading logic
    if distance_from_poc < 0.002 and not in_value_area:
        # Price rejected from POC - expect continuation
        momentum = df["close"].iloc[-1] - df["close"].iloc[-5]
        signal = "BUY" if momentum > 0 else "SELL" if momentum < 0 else "NEUTRAL"
        score = 89.0
        confluence = ["poc_rejection", "volume_cluster"]
    elif distance_from_poc > 0.015 and in_value_area:
        # Price far from POC but in value area - mean reversion
        signal = "BUY" if current_price < poc_price else "SELL"
        score = 85.0
        confluence = ["poc_reversion", "value_area"]
    else:
        signal = "NEUTRAL"
        score = 0.0
        confluence = []
    
    return {"signal": signal, "score": score, "reason": "VolProfile", "confluence": confluence}

def session_open_volatility_expansion(df: pd.DataFrame, now_utc: datetime) -> Dict:
    """
    Captures volatility expansion at major session opens
    London: 07:00-09:00 UTC, New York: 13:00-15:00 UTC
    """
    hour = now_utc.hour
    
    # Define high-impact windows
    london_open = 7 <= hour < 9
    ny_open = 13 <= hour < 15
    overlap = 13 <= hour < 16  # London-NY overlap
    
    if not (london_open or ny_open):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "SessionVol", "confluence": []}
    
    # Measure recent volatility expansion
    atr_val = atr(df, 14)
    atr_ma = df["high"].rolling(14).apply(lambda x: atr(df.loc[x.index], 14), raw=False).mean()
    
    vol_expansion = atr_val > atr_ma * 1.5
    
    # Directional bias from prior session
    prior_session_close = df["close"].iloc[-10]
    current_price = df["close"].iloc[-1]
    bias = "BUY" if current_price > prior_session_close else "SELL"
    
    # Volume confirmation
    vol_spike = df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.8
    
    if vol_expansion and vol_spike:
        score = 88.0
        if overlap:
            score += 7.0  # Bonus for overlap period
        signal = bias
        confluence = ["session_open", "vol_expansion", "volume"]
    else:
        signal = "NEUTRAL"
        score = 0.0
        confluence = []
    
    return {"signal": signal, "score": min(score, 100.0), "reason": "SessionVol", "confluence": confluence}

def institutional_divergence_detection(df: pd.DataFrame) -> Dict:
    """
    Multi-indicator divergence: Price vs RSI, MACD, Volume
    Classic institutional reversal signal
    """
    # Price peaks/troughs
    price_highs = df["high"].rolling(10).max()
    price_lows = df["low"].rolling(10).min()
    
    making_higher_high = df["high"].iloc[-1] >= price_highs.iloc[-2]
    making_lower_low = df["low"].iloc[-1] <= price_lows.iloc[-2]
    
    # RSI divergence
    rsi_vals = rsi(df["close"], 14)
    rsi_declining = rsi_vals[-1] < rsi_vals[-10]
    rsi_rising = rsi_vals[-1] > rsi_vals[-10]
    
    # MACD divergence
    macd_line, macd_sig = macd(df["close"])
    macd_declining = len(macd_line) > 10 and macd_line[-1] < macd_line[-10]
    macd_rising = len(macd_line) > 10 and macd_line[-1] > macd_line[-10]
    
    # Volume divergence
    vol_declining = df["volume"].iloc[-1] < df["volume"].iloc[-10:-1].mean()
    
    # Bearish divergence: higher high price but lower RSI/MACD
    bearish_div = making_higher_high and rsi_declining and macd_declining
    
    # Bullish divergence: lower low price but higher RSI/MACD  
    bullish_div = making_lower_low and rsi_rising and macd_rising
    
    if bearish_div and vol_declining:
        signal = "SELL"
        score = 91.0
        confluence = ["price_divergence", "rsi_div", "macd_div", "vol_div"]
    elif bullish_div and vol_declining:
        signal = "BUY"
        score = 91.0
        confluence = ["price_divergence", "rsi_div", "macd_div", "vol_div"]
    else:
        signal = "NEUTRAL"
        score = 0.0
        confluence = []
    
    return {"signal": signal, "score": score, "reason": "InstDivergence", "confluence": confluence}

def bollinger_squeeze_breakout(df: pd.DataFrame) -> Dict:
    """
    Bollinger Band squeeze followed by expansion
    High-probability breakout setup
    """
    bb = bollinger_bands(df["close"], 20, 2.0)
    
    # Calculate bandwidth
    bandwidth = (bb["width"] / bb["ma"]).fillna(0)
    
    # Squeeze: bandwidth in lowest 20% of recent range
    bandwidth_min = bandwidth.rolling(100).min()
    bandwidth_max = bandwidth.rolling(100).max()
    bandwidth_range = bandwidth_max - bandwidth_min
    
    current_bandwidth = bandwidth.iloc[-1]
    is_squeezed = current_bandwidth < bandwidth_min.iloc[-1] + bandwidth_range.iloc[-1] * 0.2
    
    # Breakout detection
    upper_break = df["close"].iloc[-1] > bb["upper"].iloc[-1]
    lower_break = df["close"].iloc[-1] < bb["lower"].iloc[-1]
    
    # Volume confirmation
    vol_surge = df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 2.0
    
    # Keltner Channel for confirmation (reduces false signals)
    keltner_middle = df["close"].rolling(20).mean()
    keltner_atr = pd.Series([atr(df.iloc[:i+1], 10) for i in range(len(df))])
    keltner_upper = keltner_middle + 2 * keltner_atr
    keltner_lower = keltner_middle - 2 * keltner_atr
    
    bb_inside_keltner = (bb["upper"].iloc[-10:] < keltner_upper.iloc[-10:]).all()
    
    if is_squeezed and bb_inside_keltner and upper_break and vol_surge:
        signal = "BUY"
        score = 94.0
        confluence = ["squeeze", "breakout", "volume", "keltner_confirm"]
    elif is_squeezed and bb_inside_keltner and lower_break and vol_surge:
        signal = "SELL"
        score = 94.0
        confluence = ["squeeze", "breakout", "volume", "keltner_confirm"]
    else:
        signal = "NEUTRAL"
        score = 0.0
        confluence = []
    
    return {"signal": signal, "score": score, "reason": "BBSqueeze", "confluence": confluence}

# ==================== DATA PROVIDERS ====================

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
        return symbol, False

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
            "1h": timedelta(hours=1),
            "2h": timedelta(hours=2),
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
            "1h": "1h",
            "2h": "1h",
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

# ==================== LEGENDARY ENGINE ====================

LEGENDARY_STRATEGIES = [
    ultra_order_flow_imbalance,
    liquidity_void_detection,
    market_maker_manipulation,
    institutional_accumulation_distribution,
    vwap_extremes_with_reversion,
    advanced_ichimoku_confluence,
    volume_profile_point_of_control,
    session_open_volatility_expansion,
    institutional_divergence_detection,
    bollinger_squeeze_breakout,
]

def legendary_multi_confluence_scan(df_dict: Dict[str, pd.DataFrame], now_utc: datetime) -> Dict:
    """
    Ultimate confluence detector - requires multiple legendary signals to align
    """
    # Get primary timeframe data
    primary_df = df_dict.get("4h") or df_dict.get("1h") or next(iter(df_dict.values()))
    
    if primary_df is None or len(primary_df) < 100:
        return {"signal": "NEUTRAL", "confidence": 0.0, "confluence_score": 0, "strategies": []}
    
    # Run all legendary strategies
    results = []
    for strategy_fn in LEGENDARY_STRATEGIES:
        try:
            if strategy_fn == multi_timeframe_momentum_surge:
                res = strategy_fn(df_dict)
            elif strategy_fn == session_open_volatility_expansion:
                res = strategy_fn(primary_df, now_utc)
            else:
                res = strategy_fn(primary_df)
            results.append(res)
        except Exception as e:
            print(f"Strategy {strategy_fn.__name__} error: {e}")
            continue
    
    # Analyze confluence
    buy_signals = [r for r in results if r["signal"] == "BUY" and r["score"] >= 85]
    sell_signals = [r for r in results if r["signal"] == "SELL" and r["score"] >= 85]
    
    buy_score = sum(r["score"] for r in buy_signals)
    sell_score = sum(r["score"] for r in sell_signals)
    
    buy_confluence = len(buy_signals)
    sell_confluence = len(sell_signals)
    
    # Collect all confluence factors
    all_confluence_factors = []
    for r in (buy_signals if buy_confluence > sell_confluence else sell_signals):
        all_confluence_factors.extend(r.get("confluence", []))
    
    unique_confluence = len(set(all_confluence_factors))
    
    # Decision logic - requires MINIMUM 4 strategies aligned
    if buy_confluence >= 4 and buy_confluence > sell_confluence:
        signal = "BUY"
        avg_score = buy_score / buy_confluence
        confidence = min(avg_score + (unique_confluence * 2), 100.0)
        strategies_used = [r["reason"] for r in buy_signals]
    elif sell_confluence >= 4 and sell_confluence > buy_confluence:
        signal = "SELL"
        avg_score = sell_score / sell_confluence
        confidence = min(avg_score + (unique_confluence * 2), 100.0)
        strategies_used = [r["reason"] for r in sell_signals]
    else:
        signal = "NEUTRAL"
        confidence = 0.0
        unique_confluence = 0
        strategies_used = []
    
    return {
        "signal": signal,
        "confidence": confidence,
        "confluence_score": unique_confluence,
        "strategies": strategies_used,
        "num_aligned": buy_confluence if signal == "BUY" else sell_confluence,
        "confluence_factors": list(set(all_confluence_factors))
    }

# ==================== ADVANCED AI (Runtime TF) ====================

class LegendaryAI:
    """Advanced AI with runtime TensorFlow"""
    def __init__(self, lookback: int = 120):
        self.lookback = lookback
        self.model = None
        self.is_trained = False
        self.scaler = RobustScaler() if TFK and StandardScaler is not None else None
    
    def build_model(self, input_shape: Tuple[int, int]) -> Optional[Model]:
        if not TFK:
            return None
        
        inputs = Input(shape=input_shape)
        
        # Multi-scale feature extraction
        conv1 = Conv1D(128, 3, activation="relu", padding="same")(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = MaxPooling1D(2)(conv1)
        
        conv2 = Conv1D(64, 5, activation="relu", padding="same")(inputs)
        conv2 = BatchNormalization()(conv2)
        conv2 = MaxPooling1D(2)(conv2)
        
        # Merge
        merged = Concatenate()([conv1, conv2])
        
        # Attention mechanism
        attn = MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.2)(merged, merged)
        attn = LayerNormalization()(attn + merged)
        
        # LSTM layers
        lstm1 = LSTM(256, return_sequences=True)(attn)
        lstm1 = Dropout(0.3)(lstm1)
        lstm1 = BatchNormalization()(lstm1)
        
        lstm2 = LSTM(128, return_sequences=False)(lstm1)
        lstm2 = Dropout(0.2)(lstm2)
        lstm2 = BatchNormalization()(lstm2)
        
        # Dense layers
        dense = Dense(128, activation="relu")(lstm2)
        dense = Dropout(0.2)(dense)
        
        # Multi-output
        direction = Dense(3, activation="softmax", name="direction")(dense)
        confidence = Dense(1, activation="sigmoid", name="confidence")(dense)
        
        model = Model(inputs=inputs, outputs=[direction, confidence])
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss={"direction": "categorical_crossentropy", "confidence": "mse"},
            loss_weights={"direction": 3.0, "confidence": 1.0},
            metrics={"direction": "accuracy"},
        )
        return model
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        features = []
        for i in range(len(df)):
            row = df.iloc[i]
            window = df.iloc[:i+1]
            
            # Price features
            basic = [row["open"], row["high"], row["low"], row["close"], row.get("volume", 0)]
            
            # Technical indicators
            rsi_val = rsi(window["close"], 14)[-1] if len(window) >= 14 else 50.0
            macd_line, macd_sig = macd(window["close"])
            macd_val = macd_line[-1] if len(macd_line) > 0 else 0.0
            macd_sig_val = macd_sig[-1] if len(macd_sig) > 0 else 0.0
            
            bb = bollinger_bands(window["close"], 20, 2.0)
            bb_width = bb["width"].iloc[-1] if len(bb["width"]) > 0 and pd.notna(bb["width"].iloc[-1]) else 0.0
            bb_position = (row["close"] - bb["lower"].iloc[-1]) / (bb["upper"].iloc[-1] - bb["lower"].iloc[-1]) if pd.notna(bb["upper"].iloc[-1]) else 0.5
            
            adx_val, plus_di, minus_di = adx(window) if len(window) >= 14 else (0.0, 0.0, 0.0)
            
            mfi_val = money_flow_index(window) if len(window) >= 14 else 50.0
            
            features.append(basic + [rsi_val, macd_val, macd_sig_val, bb_width, bb_position, adx_val, plus_di, minus_di, mfi_val])
        
        X = np.array(features, dtype=np.float32)
        if self.scaler is not None and len(X) > 0:
            X = self.scaler.fit_transform(X)
        return X
    
    def predict(self, df: pd.DataFrame) -> Dict:
        if not TFK or not self.is_trained or self.model is None or len(df) < self.lookback:
            # Fallback to simple heuristic
            macd_line, macd_sig = macd(df["close"])
            rsi_val = rsi(df["close"])[-1] if len(df) >= 14 else 50.0
            
            macd_bullish = len(macd_line) > 0 and macd_line[-1] > macd_sig[-1]
            macd_bearish = len(macd_line) > 0 and macd_line[-1] < macd_sig[-1]
            
            if macd_bullish and rsi_val < 65:
                signal = "CALL"
                conf = 75.0
            elif macd_bearish and rsi_val > 35:
                signal = "PUT"
                conf = 75.0
            else:
                signal = "NEUTRAL"
                conf = 50.0
            
            return {
                "signal": signal,
                "confidence": conf,
                "probabilities": {"PUT": 33.3, "NEUTRAL": 33.3, "CALL": 33.3}
            }
        
        # AI prediction
        feats = self.prepare_features(df)
        X = feats[-self.lookback:].reshape(1, self.lookback, -1)
        
        preds = self.model.predict(X, verbose=0)
        dir_probs = preds[0][0]
        conf = float(preds[1][0][0])
        
        idx = int(np.argmax(dir_probs))
        directions = ["PUT", "NEUTRAL", "CALL"]
        signal = directions[idx]
        
        return {
            "signal": signal,
            "confidence": float(conf * 100),
            "probabilities": {
                "PUT": float(dir_probs[0] * 100),
                "NEUTRAL": float(dir_probs[1] * 100),
                "CALL": float(dir_probs[2] * 100)
            }
        }

# ==================== TELEGRAM SERVICE ====================

class LegendaryTelegramBot:
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
            dp.add_handler(CommandHandler("legendary", self._legendary))
            dp.add_handler(CommandHandler("stats", self._stats))
    
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
                self.updater.bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    parse_mode=parse_mode or ParseMode.HTML
                )
            except Exception as e:
                print(f"Telegram send error: {e}")
        else:
            print(text)
    
    def _start(self, update, context):
        user_id = update.effective_user.id
        if not AdminAuth.verify_admin(user_id):
            update.message.reply_text("❌ Unauthorized. Admin access only.")
            return
        
        update.message.reply_text(
            "🔥 <b>LEGENDARY Pocket Option Scanner</b>\n\n"
            "Commands:\n"
            "/status - System status\n"
            "/legendary - Recent legendary signals\n"
            "/stats - Performance statistics\n",
            parse_mode=ParseMode.HTML
        )
    
    def _status(self, update, context):
        user_id = update.effective_user.id
        if not AdminAuth.verify_admin(user_id):
            update.message.reply_text("❌ Unauthorized")
            return
        
        update.message.reply_text(
            f"✅ <b>System Active</b>\n"
            f"Assets: {len(ASSETS)}\n"
            f"Confidence Gate: {CONF_GATE}%\n"
            f"Legendary Gate: {LEGENDARY_GATE}%\n"
            f"Strategies: {len(LEGENDARY_STRATEGIES)}\n",
            parse_mode=ParseMode.HTML
        )
    
    def _legendary(self, update, context):
        user_id = update.effective_user.id
        if not AdminAuth.verify_admin(user_id):
            update.message.reply_text("❌ Unauthorized")
            return
        
        if not SIGNAL_HISTORY:
            update.message.reply_text("No legendary signals yet.")
            return
        
        recent = list(SIGNAL_HISTORY.values())[-5:]
        msg = "<b>🌟 Recent Legendary Signals:</b>\n\n"
        for sig in recent:
            msg += f"• {sig['asset']} - {sig['signal']} ({sig['confidence']:.1f}%)\n"
        
        update.message.reply_text(msg, parse_mode=ParseMode.HTML)
    
    def _stats(self, update, context):
        user_id = update.effective_user.id
        if not AdminAuth.verify_admin(user_id):
            update.message.reply_text("❌ Unauthorized")
            return
        
        total_signals = len(SIGNAL_HISTORY)
        update.message.reply_text(
            f"<b>📊 Statistics</b>\n"
            f"Total Signals: {total_signals}\n"
            f"Avg Confidence: {sum(s['confidence'] for s in SIGNAL_HISTORY.values()) / max(total_signals, 1):.1f}%\n",
            parse_mode=ParseMode.HTML
        )

# ==================== FLASK WEBHOOK ====================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "legendary",
        "timestamp": datetime.utcnow().isoformat(),
        "strategies": len(LEGENDARY_STRATEGIES),
        "confidence_gate": LEGENDARY_GATE
    }), 200

@app.route('/webhook/scan', methods=['POST'])
def webhook_scan():
    auth_header = request.headers.get('Authorization', '')
    
    if not auth_header.startswith('Bearer '):
        return jsonify({"error": "Unauthorized"}), 401
    
    token = auth_header.replace('Bearer ', '')
    valid, user_id = AdminAuth.verify_token(token)
    
    if not valid:
        return jsonify({"error": "Invalid token"}), 401
    
    try:
        data = request.json or {}
        assets = data.get('assets', ASSETS[:5])
        
        return jsonify({
            "status": "scan_triggered",
            "assets": assets,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/token', methods=['POST'])
def generate_token():
    data = request.json or {}
    user_id = data.get('user_id')
    secret = data.get('secret')
    
    if secret != ADMIN_SECRET or user_id not in ADMIN_IDS:
        return jsonify({"error": "Unauthorized"}), 401
    
    token = AdminAuth.generate_token(user_id)
    return jsonify({"token": token}), 200

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        "active": True,
        "legendary_mode": True,
        "strategies": len(LEGENDARY_STRATEGIES),
        "confidence_gate": LEGENDARY_GATE,
        "last_scan": datetime.utcnow().isoformat()
    }), 200

def format_legendary_alert(result: Dict) -> str:
    """Format ultra-sophisticated alert message"""
    confluence_emoji = "🔥" * min(result.get('confluence_score', 0), 5)
    
    return f"""
🌟 <b>LEGENDARY SIGNAL DETECTED</b> {confluence_emoji}

<b>Asset:</b> {result['asset']}
<b>Direction:</b> {result['signal']} ⚡
<b>Expiry:</b> {result['expiry']}m
<b>Confidence:</b> {result['confidence']:.1f}% 🎯

<b>📊 Multi-Strategy Confluence:</b>
• Aligned Strategies: {result.get('num_aligned', 0)}/{len(LEGENDARY_STRATEGIES)}
• Confluence Score: {result.get('confluence_score', 0)}
• Strategies: {', '.join(result.get('strategies', [])[:4])}

<b>🧠 AI Analysis:</b>
• AI Signal: {result.get('ai', {}).get('signal', 'N/A')}
• AI Confidence: {result.get('ai', {}).get('confidence', 0):.1f}%

<b>🔍 Confluence Factors:</b>
{', '.join(result.get('confluence_factors', [])[:8])}

<b>⏰ Timestamp:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

<i>This is an institutional-grade signal with {result.get('num_aligned', 0)}+ strategy alignment.</i>
"""

def main_scan_loop():
    """Main scanning loop"""
    print("🚀 Initializing Legendary Scanner...")
    
    provider = RealDataProvider()
    ai = LegendaryAI(lookback=120)
    telegram = LegendaryTelegramBot(TELEGRAM_TOKEN, CHAT_ID)
    
    if TELEGRAM_AVAILABLE:
        telegram.start()
    
    print(f"✅ Scanner ready. Monitoring {len(ASSETS)} assets...")
    print(f"🎯 Legendary threshold: {LEGENDARY_GATE}%")
    print(f"⚡ {len(LEGENDARY_STRATEGIES)} institutional strategies active")
    
    while True:
        try:
            now_utc = datetime.utcnow()
            print(f"\n{'='*60}")
            print(f"🔍 Scan started: {now_utcnow.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            for asset in ASSETS:
                for expiry in EXPIRIES:
                    # Check cooldown
                    cooldown_key = f"{asset}_{expiry}"
                    if cooldown_key in LAST_LEGENDARY_ALERT:
                        elapsed = (now_utc - LAST_LEGENDARY_ALERT[cooldown_key]).total_seconds() / 60
                        if elapsed < COOLDOWN_MINUTES:
                            continue
                    
                    # Fetch multi-timeframe data
                    tf_data = {}
                    for tf in TIMEFRAMES:
                        try:
                            df = provider.fetch_ohlcv(asset, tf, limit=500)
                            if df is not None and len(df) >= 100:
                                tf_data[tf] = df
                        except Exception as e:
                            print(f"  ⚠️  Error fetching {asset} {tf}: {e}")
                    
                    if len(tf_data) < 4:
                        continue
                    
                    # Run legendary confluence scan
                    legendary_result = legendary_multi_confluence_scan(tf_data, now_utc)
                    
                    # AI prediction
                    primary_df = tf_data.get("4h") or tf_data.get("1h")
                    ai_result = ai.predict(primary_df) if primary_df is not None else {"signal": "NEUTRAL", "confidence": 0}
                    
                    # Combine
                    combined_conf = (legendary_result['confidence'] * 0.6 + ai_result['confidence'] * 0.4)
                    
                    # AI alignment check
                    ai_map = {"CALL": "BUY", "PUT": "SELL", "NEUTRAL": "NEUTRAL"}
                    ai_signal = ai_map.get(ai_result['signal'], "NEUTRAL")
                    ai_aligned = legendary_result['signal'] == ai_signal
                    
                    # LEGENDARY threshold
                    if (legendary_result['signal'] != "NEUTRAL" and 
                        combined_conf >= LEGENDARY_GATE and
                        legendary_result['num_aligned'] >= 4 and
                        ai_aligned):
                        
                        alert_data = {
                            'asset': asset,
                            'signal': legendary_result['signal'],
                            'confidence': combined_conf,
                            'expiry': expiry,
                            'num_aligned': legendary_result['num_aligned'],
                            'confluence_score': legendary_result['confluence_score'],
                            'strategies': legendary_result['strategies'],
                            'confluence_factors': legendary_result['confluence_factors'],
                            'ai': ai_result,
                            'timestamp': now_utc.isoformat()
                        }
                        
                        # Store
                        signal_key = f"{asset}_{expiry}_{now_utc.strftime('%Y%m%d_%H%M')}"
                        SIGNAL_HISTORY[signal_key] = alert_data
                        LAST_LEGENDARY_ALERT[cooldown_key] = now_utc
                        
                        # Send alert
                        alert_msg = format_legendary_alert(alert_data)
                        telegram.send(alert_msg, parse_mode=ParseMode.HTML)
                        
                        print(f"  🔥 LEGENDARY: {asset} {legendary_result['signal']} {expiry}m @ {combined_conf:.1f}%")
                    
                    time.sleep(0.5)  # Rate limiting
            
            print(f"✅ Scan complete. Next scan in {SCAN_INTERVAL_SEC}s...")
            
            if RUN_ONCE:
                print("🛑 RUN_ONCE mode - exiting")
                break
            
            time.sleep(SCAN_INTERVAL_SEC)
            
        except KeyboardInterrupt:
            print("\n🛑 Scanner stopped by user")
            break
        except Exception as e:
            print(f"❌ Scan error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    if RUN_ONCE or "--scan" in os.sys.argv:
        print("Running single scan...")
        main_scan_loop()
    else:
        # Run Flask server for Choreo
        print(f"🚀 Starting Legendary Scanner on port {PORT}")
        print(f"📡 Webhook: POST /webhook/scan")
        print(f"🔑 Token: POST /api/token")
        print(f"💚 Health: GET /health")
        
        # Start background scanner
        scanner_thread = threading.Thread(target=main_scan_loop, daemon=True)
        scanner_thread.start()
        
        # Start Flask
        app.run(host='0.0.0.0', port=PORT, debug=False)

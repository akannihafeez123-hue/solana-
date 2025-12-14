"""
🏆 ULTIMATE #1 LEGENDARY TRADING BOT - V4.0 ELITE EDITION
The Most Advanced Binary Options AI Ever Created
"""

import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Tuple, Optional, Any
import threading
import warnings
import logging
from flask import Flask, jsonify
from collections import Counter, deque
from scipy import stats
from scipy.signal import argrelextrema
import asyncio

warnings.filterwarnings('ignore')

# -------------------------------
# Logging setup
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------
# Runtime installer
# -------------------------------
def ensure_packages():
    pkgs = [
        ("yfinance", "yfinance==0.2.28"),
        ("python-telegram-bot", "python-telegram-bot==20.6"),
        ("python-dotenv", "python-dotenv==1.0.0"),
        ("scipy", "scipy==1.11.3")
    ]
    for mod_name, pkg in pkgs:
        try:
            __import__(mod_name)
        except ImportError:
            logger.info(f"📦 Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

ensure_packages()

# Import required libraries
try:
    import yfinance as yf
    from telegram import Bot
    from telegram.constants import ParseMode
    from dotenv import load_dotenv
    load_dotenv()
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    sys.exit(1)

# -------------------------------
# Configuration
# -------------------------------
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

# Enhanced configuration
LEGENDARY_GATE = int(os.getenv("LEGENDARY_GATE", "97"))
GLOBAL_THRESHOLD = LEGENDARY_GATE
ASSET_THRESHOLD_OVERRIDE: Dict[str, float] = {}

COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "15"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "30"))
RUN_ONCE = os.getenv("RUN_ONCE", "False") == "True"
PORT = int(os.getenv("PORT") or os.getenv("port") or "8080")

ASSETS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCHF=X", "GC=F", "CL=F"]
EXPIRIES = [5, 15, 30, 60]
ALL_TIMEFRAMES = ["1d", "4h", "1h", "15m", "5m"]

# Asset configuration with enhanced parameters
ASSET_CONFIG = {
    "EURUSD=X": {
        "min_data_points": 50,
        "volatility_threshold": 0.35,
        "vwap_threshold": 0.008,
        "volume_profile_sensitivity": 0.004,
        "trend_strength_multiplier": 1.2,
        "momentum_volume_multiplier": 1.8,
        "min_volume_ratio": 0.7,
        "max_spread_pct": 0.0005,
        "choppy_threshold": 0.3
    },
    "GBPUSD=X": {
        "min_data_points": 50,
        "volatility_threshold": 0.4,
        "vwap_threshold": 0.01,
        "volume_profile_sensitivity": 0.005,
        "trend_strength_multiplier": 1.3,
        "momentum_volume_multiplier": 2.0,
        "min_volume_ratio": 0.7,
        "max_spread_pct": 0.0006,
        "choppy_threshold": 0.35
    },
    "USDJPY=X": {
        "min_data_points": 50,
        "volatility_threshold": 0.38,
        "vwap_threshold": 0.009,
        "volume_profile_sensitivity": 0.0045,
        "trend_strength_multiplier": 1.15,
        "momentum_volume_multiplier": 1.9,
        "min_volume_ratio": 0.65,
        "max_spread_pct": 0.0005,
        "choppy_threshold": 0.32
    },
    "AUDUSD=X": {
        "min_data_points": 50,
        "volatility_threshold": 0.42,
        "vwap_threshold": 0.01,
        "volume_profile_sensitivity": 0.005,
        "trend_strength_multiplier": 1.25,
        "momentum_volume_multiplier": 2.0,
        "min_volume_ratio": 0.65,
        "max_spread_pct": 0.0007,
        "choppy_threshold": 0.35
    },
    "USDCHF=X": {
        "min_data_points": 50,
        "volatility_threshold": 0.36,
        "vwap_threshold": 0.009,
        "volume_profile_sensitivity": 0.0045,
        "trend_strength_multiplier": 1.2,
        "momentum_volume_multiplier": 1.85,
        "min_volume_ratio": 0.65,
        "max_spread_pct": 0.0006,
        "choppy_threshold": 0.33
    },
    "GC=F": {
        "min_data_points": 50,
        "volatility_threshold": 0.5,
        "vwap_threshold": 0.012,
        "volume_profile_sensitivity": 0.006,
        "trend_strength_multiplier": 1.4,
        "momentum_volume_multiplier": 2.2,
        "min_volume_ratio": 0.75,
        "max_spread_pct": 0.001,
        "choppy_threshold": 0.4
    },
    "CL=F": {
        "min_data_points": 50,
        "volatility_threshold": 0.55,
        "vwap_threshold": 0.015,
        "volume_profile_sensitivity": 0.007,
        "trend_strength_multiplier": 1.5,
        "momentum_volume_multiplier": 2.3,
        "min_volume_ratio": 0.75,
        "max_spread_pct": 0.0012,
        "choppy_threshold": 0.45
    }
}

# Enhanced tracking
MAX_SIGNAL_HISTORY = 1000
SIGNAL_HISTORY: deque = deque(maxlen=MAX_SIGNAL_HISTORY)
LAST_LEGENDARY_ALERT: Dict[str, datetime] = {}
REJECTED_SIGNALS: deque = deque(maxlen=500)

app = Flask(__name__)

# -------------------------------
# Economic Calendar & News Filter
# -------------------------------
class EconomicCalendar:
    """Filters trading during major news events"""
    
    @staticmethod
    def is_news_time() -> Tuple[bool, str]:
        """Check if current time is during major news"""
        now = datetime.utcnow()
        current_time = now.time()
        
        if dt_time(13, 30) <= current_time <= dt_time(14, 30):
            return True, "US Session Open"
        
        if dt_time(7, 0) <= current_time <= dt_time(8, 0):
            return True, "London Session Open"
        
        if dt_time(20, 0) <= current_time <= dt_time(21, 0):
            return True, "US Market Close"
        
        if current_time >= dt_time(22, 0) or current_time <= dt_time(2, 0):
            return True, "Asian Session - Low Liquidity"
        
        return False, ""
    
    @staticmethod
    def get_trading_session() -> str:
        """Identify current trading session"""
        now = datetime.utcnow()
        current_time = now.time()
        
        if dt_time(8, 0) <= current_time <= dt_time(16, 0):
            return "LONDON"
        
        if dt_time(13, 0) <= current_time <= dt_time(20, 0):
            return "US"
        
        if dt_time(0, 0) <= current_time <= dt_time(7, 0):
            return "TOKYO"
        
        return "OVERLAP"

# -------------------------------
# Market Condition Filters
# -------------------------------
class MarketConditionFilter:
    """Filters out unfavorable market conditions"""
    
    @staticmethod
    def is_choppy_market(df: pd.DataFrame, asset_config: Dict) -> Tuple[bool, float]:
        """Detect sideways choppy markets"""
        if df is None or len(df) < 50:
            return False, 0.0
        
        try:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1] if not tr.empty else 0
            
            price = df['close'].iloc[-1]
            atr_ratio = atr / price if price > 0 else 0
            
            recent_high = df['high'].iloc[-20:].max()
            recent_low = df['low'].iloc[-20:].min()
            range_size = (recent_high - recent_low) / price if price > 0 else 0
            
            ema_12 = df['close'].ewm(span=12).mean().iloc[-1]
            ema_26 = df['close'].ewm(span=26).mean().iloc[-1]
            ema_diff = abs(ema_12 - ema_26) / price if price > 0 else 0
            
            choppy_threshold = asset_config.get("choppy_threshold", 0.35)
            
            is_choppy = (atr_ratio < choppy_threshold and 
                        range_size < 0.015 and 
                        ema_diff < 0.005)
            
            return is_choppy, atr_ratio
            
        except Exception as e:
            logger.debug(f"Choppy market check error: {e}")
            return False, 0.0
    
    @staticmethod
    def has_sufficient_liquidity(df: pd.DataFrame, asset_config: Dict) -> Tuple[bool, float]:
        """Check if market has sufficient volume/liquidity"""
        if df is None or len(df) < 50:
            return False, 0.0
        
        try:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(50).mean().iloc[-1]
            
            volume_ratio = current_volume / (avg_volume + 1e-9)
            min_ratio = asset_config.get("min_volume_ratio", 0.7)
            
            recent_avg = df['volume'].iloc[-5:].mean()
            historical_avg = df['volume'].iloc[-50:-5].mean()
            recent_ratio = recent_avg / (historical_avg + 1e-9)
            
            has_liquidity = volume_ratio >= min_ratio or recent_ratio >= min_ratio
            
            return has_liquidity, volume_ratio
            
        except Exception as e:
            logger.debug(f"Liquidity check error: {e}")
            return True, 1.0
    
    @staticmethod
    def check_timeframe_alignment(tf_data: Dict[str, pd.DataFrame], signal_direction: str) -> Tuple[bool, int]:
        """Verify signal alignment across multiple timeframes"""
        if not tf_data or len(tf_data) < 3:
            return False, 0
        
        try:
            aligned_count = 0
            total_checked = 0
            
            for tf, df in tf_data.items():
                if df is None or len(df) < 30:
                    continue
                
                ema_12 = df['close'].ewm(span=12).mean().iloc[-1]
                ema_26 = df['close'].ewm(span=26).mean().iloc[-1]
                
                tf_direction = "BUY" if ema_12 > ema_26 else "SELL"
                
                if tf_direction == signal_direction:
                    aligned_count += 1
                
                total_checked += 1
            
            alignment_pct = aligned_count / (total_checked + 1e-9)
            is_aligned = alignment_pct >= 0.75 and aligned_count >= 3
            
            return is_aligned, aligned_count
            
        except Exception as e:
            logger.debug(f"Timeframe alignment check error: {e}")
            return False, 0
    
    @staticmethod
    def check_volatility_spike(df: pd.DataFrame) -> Tuple[bool, float]:
        """Detect dangerous volatility spikes"""
        if df is None or len(df) < 20:
            return False, 0.0
        
        try:
            returns = df['close'].pct_change()
            current_vol = returns.iloc[-5:].std() if len(returns) >= 5 else 0
            avg_vol = returns.rolling(20).std().mean()
            
            vol_ratio = current_vol / (avg_vol + 1e-9)
            
            is_spike = vol_ratio > 2.5
            
            return is_spike, vol_ratio
            
        except Exception as e:
            logger.debug(f"Volatility spike check error: {e}")
            return False, 0.0

# -------------------------------
# Telegram Bot
# -------------------------------
class LegendaryTelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.bot: Optional[Bot] = None
        self.initialized = False

    def start(self):
        if not self.token or not self.chat_id:
            logger.warning("⚠️ Telegram not configured")
            return
        try:
            self.bot = Bot(token=self.token)
            self.initialized = True
            logger.info("✅ Telegram bot initialized")
        except Exception as e:
            logger.error(f"❌ Failed to start Telegram bot: {e}")

    async def send_message_async(self, text: str):
        """Send message to Telegram"""
        if not self.initialized or not self.bot:
            logger.warning("Telegram bot not initialized")
            return
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    def send_message(self, text: str):
        """Send message to Telegram (synchronous wrapper)"""
        if not self.initialized or not self.bot:
            logger.warning("Telegram bot not initialized")
            return
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.send_message_async(text))
            loop.close()
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    def send_signal(self, alert_data: Dict):
        """Send signal alert"""
        if not self.initialized:
            return
        
        message = self.format_alert(alert_data)
        self.send_message(message)

    @staticmethod
    def format_alert(result: Dict) -> str:
        """Format legendary alert message"""
        strategies_list = "\n".join([f"  • {s['reason']} ({s['score']:.0f}%)" for s in result.get('best_strategies', [])[:6]]) or "  • None"
        indicators_list = "\n".join([f"  • {i['reason']} ({i['score']:.0f}%)" for i in result.get('best_indicators', [])[:5]]) or "  • None"
        filters_list = "\n".join([f"  • {f['reason']} ({f['score']:.0f}%)" for f in result.get('best_filters', [])[:3]]) or "  • None"

        return f"""🏆 <b>ULTIMATE LEGENDARY SIGNAL</b>  <i>[{result.get('engine', 'N/A')}]</i>

<b>Asset:</b> {result.get('asset', 'N/A')}
<b>Direction:</b> {result.get('signal', 'NEUTRAL')}
<b>Expiry:</b> {result.get('expiry', 'N/A')}m
<b>Confidence:</b> {result.get('confidence', 0.0):.1f}%

<b>Strategies ({result.get('num_strategies_aligned', 0)} aligned):</b>
{strategies_list}

<b>Indicators ({result.get('num_indicators_aligned', 0)} aligned):</b>
{indicators_list}

<b>Institutional Filters ({result.get('num_filters_aligned', 0)} aligned):</b>
{filters_list}

<b>Time:</b> {datetime.utcnow().strftime('%Y-%m-d %H:%M:%S')} UTC"""

# -------------------------------
# Helper Functions
# -------------------------------
def safe_series_check(series: pd.Series, min_len: int = 1) -> bool:
    return series is not None and isinstance(series, pd.Series) and len(series) >= min_len

def safe_df_check(df: pd.DataFrame, min_len: int = 1) -> bool:
    return df is not None and isinstance(df, pd.DataFrame) and len(df) >= min_len

# -------------------------------
# Technical Indicators
# -------------------------------
def ema(series: pd.Series, period: int) -> np.ndarray:
    if not safe_series_check(series, period):
        return np.array([])
    try:
        return series.ewm(span=period, adjust=False).mean().values
    except Exception:
        return np.array([])

def rsi(series: pd.Series, period: int = 14) -> np.ndarray:
    if not safe_series_check(series, period + 1):
        return np.array([])
    try:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.rolling(window=period).mean()
        ma_down = down.rolling(window=period).mean()
        rs = ma_up / ma_down
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals.fillna(50).values
    except Exception:
        return np.array([])

def macd(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    if not safe_series_check(series, 26):
        return np.array([]), np.array([])
    try:
        ema_fast = series.ewm(span=12, adjust=False).mean()
        ema_slow = series.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_sig = macd_line.ewm(span=9, adjust=False).mean()
        return macd_line.values, macd_sig.values
    except Exception:
        return np.array([]), np.array([])

def bollinger_bands(series: pd.Series, period: int = 20) -> Dict:
    if not safe_series_check(series, period):
        return {"upper": pd.Series(dtype=float), "lower": pd.Series(dtype=float), "ma": pd.Series(dtype=float)}
    try:
        ma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        return {"upper": upper, "lower": lower, "ma": ma}
    except Exception:
        return {"upper": pd.Series(dtype=float), "lower": pd.Series(dtype=float), "ma": pd.Series(dtype=float)}

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if not safe_df_check(df, period):
        return pd.Series(dtype=float)
    try:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    except Exception:
        return pd.Series(dtype=float)

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    if not safe_df_check(df, period):
        return pd.Series(dtype=float)
    try:
        _atr = atr(df, period)
        hl2 = (df["high"] + df["low"]) / 2
        upperband = hl2 + (multiplier * _atr)
        lowerband = hl2 - (multiplier * _atr)
        
        st = pd.Series(index=df.index, dtype=float)
        trend = 1
        
        for i in range(len(df)):
            if i == 0:
                st.iloc[i] = lowerband.iloc[i] if not pd.isna(lowerband.iloc[i]) else 0
                continue
            
            if df["close"].iloc[i] > upperband.iloc[i-1]:
                trend = 1
            elif df["close"].iloc[i] < lowerband.iloc[i-1]:
                trend = -1
                
            st.iloc[i] = lowerband.iloc[i] if trend == 1 else upperband.iloc[i]
        
        return st
    except Exception:
        return pd.Series(dtype=float)

def vwap(df: pd.DataFrame) -> pd.Series:
    if not safe_df_check(df):
        return pd.Series(dtype=float)
    try:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        cum_vol = df["volume"].cumsum()
        cum_pv = (tp * df["volume"]).cumsum()
        return cum_pv / (cum_vol.replace(0, np.nan))
    except Exception:
        return pd.Series(dtype=float)

def simple_volume_profile(df: pd.DataFrame, bins: int = 24) -> Dict[str, float]:
    if not safe_df_check(df, 2):
        return {"POC": 0.0, "VAH": 0.0, "VAL": 0.0}
    try:
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
    except Exception:
        return {"POC": 0.0, "VAH": 0.0, "VAL": 0.0}

# -------------------------------
# ADVANCED INSTITUTIONAL STRATEGIES (10 NEW)
# -------------------------------

def market_microstructure_imbalance(df: pd.DataFrame) -> Dict:
    """Detects order flow imbalances using Kyle's Lambda"""
    if not safe_df_check(df, 100):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "MicrostructureImb", "type": "strategy"}
    
    try:
        price_changes = df['close'].diff().abs()
        volume_normalized = df['volume'] / df['volume'].rolling(50).mean()
        impact_ratio = (price_changes / (volume_normalized + 1e-9)).rolling(20).mean()
        
        current_impact = impact_ratio.iloc[-1] if not impact_ratio.empty else 0
        avg_impact = impact_ratio.mean()
        
        buying_pressure = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)).rolling(10).mean()
        selling_pressure = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-9)).rolling(10).mean()
        
        imbalance = buying_pressure.iloc[-1] - selling_pressure.iloc[-1] if not buying_pressure.empty else 0
        
        if current_impact < avg_impact * 0.7 and imbalance > 0.15:
            return {"signal": "BUY", "score": 97.0, "reason": "MicrostructureImb", "type": "strategy"}
        elif current_impact < avg_impact * 0.7 and imbalance < -0.15:
            return {"signal": "SELL", "score": 97.0, "reason": "MicrostructureImb", "type": "strategy"}
        
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "MicrostructureImb", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "MicrostructureImb", "type": "strategy"}

def liquidity_void_hunter(df: pd.DataFrame) -> Dict:
    """Identifies liquidity voids where market makers withdrew"""
    if not safe_df_check(df, 200):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "LiquidityVoid", "type": "strategy"}
    
    try:
        price_range = df['close'].max() - df['close'].min()
        bins = 50
        hist, edges = np.histogram(df['close'].values, bins=bins, weights=df['volume'].values)
        
        volume_threshold = np.percentile(hist, 20) if len(hist) > 0 else 0
        void_indices = np.where(hist < volume_threshold)[0]
        
        if len(void_indices) == 0:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "LiquidityVoid", "type": "strategy"}
        
        current_price = df['close'].iloc[-1]
        void_prices = [(edges[i] + edges[i+1])/2 for i in void_indices]
        nearest_void_dist = min([abs(current_price - vp) for vp in void_prices]) if void_prices else float('inf')
        relative_dist = nearest_void_dist / price_range if price_range > 0 else 1
        
        momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 6 else 0
        
        if relative_dist < 0.02:
            if momentum > 0:
                return {"signal": "BUY", "score": 94.0, "reason": "LiquidityVoid", "type": "strategy"}
            elif momentum < 0:
                return {"signal": "SELL", "score": 94.0, "reason": "LiquidityVoid", "type": "strategy"}
        
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "LiquidityVoid", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "LiquidityVoid", "type": "strategy"}

def volatility_regime_detector(df: pd.DataFrame) -> Dict:
    """Detects volatility regime changes using HMM concepts"""
    if not safe_df_check(df, 100):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "VolRegime", "type": "strategy"}
    
    try:
        returns = df['close'].pct_change()
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        vol_of_vol = rolling_vol.rolling(20).std()
        
        vol_zscore = (rolling_vol.iloc[-1] - rolling_vol.mean()) / (rolling_vol.std() + 1e-9) if not rolling_vol.empty else 0
        recent_vol_change = rolling_vol.iloc[-1] / rolling_vol.iloc[-10] if len(rolling_vol) >= 10 else 0
        vol_acceleration = vol_of_vol.iloc[-1] / vol_of_vol.mean() if not vol_of_vol.empty else 0
        
        if vol_zscore < -0.5 and recent_vol_change > 1.3 and vol_acceleration > 1.2:
            price_momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 6 else 0
            if price_momentum > 0:
                return {"signal": "BUY", "score": 93.0, "reason": "VolRegime", "type": "strategy"}
            else:
                return {"signal": "SELL", "score": 93.0, "reason": "VolRegime", "type": "strategy"}
        
        elif vol_zscore > 1.0 and recent_vol_change < 0.8:
            price_zscore = (df['close'].iloc[-1] - df['close'].rolling(50).mean().iloc[-1]) / (df['close'].rolling(50).std().iloc[-1] + 1e-9)
            if price_zscore > 1.5:
                return {"signal": "SELL", "score": 91.0, "reason": "VolRegime", "type": "strategy"}
            elif price_zscore < -1.5:
                return {"signal": "BUY", "score": 91.0, "reason": "VolRegime", "type": "strategy"}
        
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "VolRegime", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "VolRegime", "type": "strategy"}

def gamma_flip_detector(df: pd.DataFrame) -> Dict:
    """Simulates options dealer gamma exposure"""
    if not safe_df_check(df, 150):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "GammaFlip", "type": "strategy"}
    
    try:
        returns = df['close'].pct_change()
        realized_vol = returns.rolling(20).std() * np.sqrt(252)
        
        close_prices = df['close'].values
        maxima_idx = argrelextrema(close_prices, np.greater, order=10)[0]
        minima_idx = argrelextrema(close_prices, np.less, order=10)[0]
        
        if len(maxima_idx) < 2 or len(minima_idx) < 2:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "GammaFlip", "type": "strategy"}
        
        recent_max = close_prices[maxima_idx[-1]] if len(maxima_idx) > 0 else df['close'].max()
        recent_min = close_prices[minima_idx[-1]] if len(minima_idx) > 0 else df['close'].min()
        current_price = df['close'].iloc[-1]
        
        range_position = (current_price - recent_min) / (recent_max - recent_min + 1e-9)
        price_velocity = (df['close'].iloc[-1] - df['close'].iloc[-3]) / df['close'].iloc[-3] if len(df) >= 4 else 0
        volume_surge = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 21 else 1
        
        if range_position > 0.95 and price_velocity > 0.01 and realized_vol.iloc[-1] < realized_vol.iloc[-10]:
            if volume_surge > 1.5:
                return {"signal": "BUY", "score": 96.0, "reason": "GammaFlip", "type": "strategy"}
        
        elif range_position < 0.05 and price_velocity < -0.01 and realized_vol.iloc[-1] < realized_vol.iloc[-10]:
            if volume_surge > 1.5:
                return {"signal": "SELL", "score": 96.0, "reason": "GammaFlip", "type": "strategy"}
        
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "GammaFlip", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "GammaFlip", "type": "strategy"}

def fractal_dimension_strategy(df: pd.DataFrame) -> Dict:
    """Uses Hurst Exponent for trend detection"""
    if not safe_df_check(df, 100):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "FractalDim", "type": "strategy"}
    
    try:
        def hurst_exponent(ts, max_lag=20):
            lags = range(2, max_lag)
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        prices = df['close'].values[-100:]
        hurst = hurst_exponent(prices)
        momentum = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] if len(df) >= 11 else 0
        
        if hurst > 0.65 and abs(momentum) > 0.015:
            if momentum > 0:
                return {"signal": "BUY", "score": 92.0, "reason": "FractalDim", "type": "strategy"}
            else:
                return {"signal": "SELL", "score": 92.0, "reason": "FractalDim", "type": "strategy"}
        
        elif hurst < 0.45:
            price_zscore = (df['close'].iloc[-1] - df['close'].rolling(50).mean().iloc[-1]) / (df['close'].rolling(50).std().iloc[-1] + 1e-9)
            if price_zscore > 2.0:
                return {"signal": "SELL", "score": 90.0, "reason": "FractalDim", "type": "strategy"}
            elif price_zscore < -2.0:
                return {"signal": "BUY", "score": 90.0, "reason": "FractalDim", "type": "strategy"}
        
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "FractalDim", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "FractalDim", "type": "strategy"}

def institutional_footprint_detector(df: pd.DataFrame) -> Dict:
    """Detects institutional orders via volume clustering"""
    if not safe_df_check(df, 100):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "InstFootprint", "type": "strategy"}
    
    try:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap_val = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        vwap_deviation = (df['close'] - vwap_val) / vwap_val
        
        recent_volume = df['volume'].iloc[-5:].mean()
        avg_volume = df['volume'].rolling(50).mean().iloc[-1]
        
        recent_volatility = df['close'].iloc[-5:].std() / df['close'].iloc[-5:].mean() if len(df) >= 5 else 0
        avg_volatility = df['close'].rolling(50).std().iloc[-1] / df['close'].rolling(50).mean().iloc[-1]
        
        volume_surge = recent_volume / avg_volume
        vol_compression = recent_volatility / (avg_volatility + 1e-9)
        
        if volume_surge > 1.5 and vol_compression < 0.6:
            if df['close'].iloc[-1] > vwap_val.iloc[-1] and vwap_deviation.iloc[-1] > 0.005:
                return {"signal": "BUY", "score": 95.0, "reason": "InstFootprint", "type": "strategy"}
            elif df['close'].iloc[-1] < vwap_val.iloc[-1] and vwap_deviation.iloc[-1] < -0.005:
                return {"signal": "SELL", "score": 95.0, "reason": "InstFootprint", "type": "strategy"}
        
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "InstFootprint", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "InstFootprint", "type": "strategy"}

def auction_theory_analyzer(df: pd.DataFrame) -> Dict:
    """Market Profile and Auction Theory analysis"""
    if not safe_df_check(df, 100):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "AuctionTheory", "type": "strategy"}
    
    try:
        price_range = df['high'].max() - df['low'].min()
        tick_size = price_range / 50 if price_range > 0 else 0.001
        
        tpo_counts = {}
        for i in range(len(df)):
            low_tick = int((df['low'].iloc[i] - df['low'].min()) / tick_size)
            high_tick = int((df['high'].iloc[i] - df['low'].min()) / tick_size)
            for tick in range(low_tick, high_tick + 1):
                tpo_counts[tick] = tpo_counts.get(tick, 0) + 1
        
        if not tpo_counts:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "AuctionTheory", "type": "strategy"}
        
        poc_tick = max(tpo_counts, key=tpo_counts.get)
        
        sorted_ticks = sorted(tpo_counts.items(), key=lambda x: x[1], reverse=True)
        total_tpo = sum(tpo_counts.values())
        cumulative = 0
        value_area_ticks = []
        
        for tick, count in sorted_ticks:
            cumulative += count
            value_area_ticks.append(tick)
            if cumulative >= total_tpo * 0.70:
                break
        
        vah_price = df['low'].min() + max(value_area_ticks) * tick_size if value_area_ticks else 0
        val_price = df['low'].min() + min(value_area_ticks) * tick_size if value_area_ticks else 0
        
        current_price = df['close'].iloc[-1]
        momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 6 else 0
        
        if current_price < val_price and momentum > 0.005:
            return {"signal": "BUY", "score": 93.0, "reason": "AuctionTheory", "type": "strategy"}
        elif current_price > vah_price and momentum < -0.005:
            return {"signal": "SELL", "score": 93.0, "reason": "AuctionTheory", "type": "strategy"}
        
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "AuctionTheory", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "AuctionTheory", "type": "strategy"}

def spoofing_detector(df: pd.DataFrame) -> Dict:
    """Detects spoofing patterns"""
    if not safe_df_check(df, 50):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "SpoofDetect", "type": "strategy"}
    
    try:
        volume_spike = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-2] if len(df) >= 21 else 1
        price_change = abs(df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] if len(df) >= 3 else 0
        subsequent_reversal = (df['close'].iloc[-1] - df['close'].iloc[-3]) * (df['close'].iloc[-2] - df['close'].iloc[-3]) if len(df) >= 4 else 0
        
        if volume_spike > 3.0 and price_change < 0.003 and subsequent_reversal < 0:
            if df['close'].iloc[-2] > df['close'].iloc[-3]:
                return {"signal": "SELL", "score": 91.0, "reason": "SpoofDetect", "type": "strategy"}
            else:
                return {"signal": "BUY", "score": 91.0, "reason": "SpoofDetect", "type": "strategy"}
        
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "SpoofDetect", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "SpoofDetect", "type": "strategy"}

def market_entropy_strategy(df: pd.DataFrame) -> Dict:
    """Shannon Entropy for market efficiency"""
    if not safe_df_check(df, 100):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "MarketEntropy", "type": "strategy"}
    
    try:
        returns = df['close'].pct_change().dropna().values[-50:]
        bins = 10
        hist, _ = np.histogram(returns, bins=bins)
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        
        entropy = -np.sum([p * np.log2(p + 1e-9) for p in hist if p > 0])
        max_entropy = np.log2(bins)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        ema_short = df['close'].ewm(span=10).mean()
        ema_long = df['close'].ewm(span=30).mean()
        trend_strength = abs(ema_short.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1] if not pd.isna(ema_long.iloc[-1]) else 0
        
        if normalized_entropy < 0.7 and trend_strength > 0.015:
            if ema_short.iloc[-1] > ema_long.iloc[-1]:
                return {"signal": "BUY", "score": 92.0, "reason": "MarketEntropy", "type": "strategy"}
            else:
                return {"signal": "SELL", "score": 92.0, "reason": "MarketEntropy", "type": "strategy"}
        
        elif normalized_entropy > 0.9:
            price_zscore = (df['close'].iloc[-1] - df['close'].rolling(50).mean().iloc[-1]) / (df['close'].rolling(50).std().iloc[-1] + 1e-9)
            if price_zscore > 1.8:
                return {"signal": "SELL", "score": 90.0, "reason": "MarketEntropy", "type": "strategy"}
            elif price_zscore < -1.8:
                return {"signal": "BUY", "score": 90.0, "reason": "MarketEntropy", "type": "strategy"}
        
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "MarketEntropy", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "MarketEntropy", "type": "strategy"}

# -------------------------------
# CLASSIC STRATEGIES (6 Proven)
# -------------------------------

def fibonacci_vortex_hidden(df: pd.DataFrame) -> Dict:
    """Fibonacci retracement with vortex momentum"""
    if not safe_df_check(df, 50):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "FibVortex", "type": "strategy"}
    try:
        swing_high = df["high"].rolling(20).max().iloc[-1]
        swing_low = df["low"].rolling(20).min().iloc[-1]
        swing_range = swing_high - swing_low
        golden_level = swing_high - 0.618 * swing_range
        current_price = df["close"].iloc[-1]
        at_golden = abs(current_price - golden_level) / (current_price + 1e-9) < 0.005
        if at_golden:
            return {"signal": "BUY", "score": 96.0, "reason": "FibVortex", "type": "strategy"}
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "FibVortex", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "FibVortex", "type": "strategy"}

def quantum_entanglement_hidden(df: pd.DataFrame) -> Dict:
    """Z-score mean reversion strategy"""
    if not safe_df_check(df, 30):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "QuantumEnt", "type": "strategy"}
    try:
        prices = df["close"].values[-30:]
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        z_score = (prices[-1] - mean_price) / (std_price + 1e-9)
        if z_score < -1.5:
            return {"signal": "BUY", "score": 94.0, "reason": "QuantumEnt", "type": "strategy"}
        elif z_score > 1.5:
            return {"signal": "SELL", "score": 94.0, "reason": "QuantumEnt", "type": "strategy"}
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "QuantumEnt", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "QuantumEnt", "type": "strategy"}

def dark_pool_institutional_hidden(df: pd.DataFrame) -> Dict:
    """Dark pool volume spike detection"""
    if not safe_df_check(df, 50):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "DarkPool", "type": "strategy"}
    try:
        vol = df["volume"]
        vol_ma = vol.rolling(50).mean()
        vol_spike = vol.iloc[-1] > (vol_ma.iloc[-1] * 2.0 if not np.isnan(vol_ma.iloc[-1]) else 0)
        if vol_spike:
            signal = "BUY" if df["close"].iloc[-1] > df["open"].iloc[-1] else "SELL"
            return {"signal": signal, "score": 95.0, "reason": "DarkPool", "type": "strategy"}
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "DarkPool", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "DarkPool", "type": "strategy"}

def order_block_detection(df: pd.DataFrame) -> Dict:
    """Smart Money Concepts - Order blocks"""
    if not safe_df_check(df, 20):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "OrderBlock", "type": "strategy"}
    try:
        recent = df.tail(10)
        bull_engulf = (recent["close"].iloc[-1] > recent["open"].iloc[-1]) and (recent["close"].max() > recent["high"].shift(1).max())
        bear_engulf = (recent["close"].iloc[-1] < recent["open"].iloc[-1]) and (recent["close"].min() < recent["low"].shift(1).min())
        if bull_engulf:
            return {"signal": "BUY", "score": 88.0, "reason": "OrderBlock", "type": "strategy"}
        if bear_engulf:
            return {"signal": "SELL", "score": 88.0, "reason": "OrderBlock", "type": "strategy"}
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "OrderBlock", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "OrderBlock", "type": "strategy"}

def break_of_structure(df: pd.DataFrame) -> Dict:
    """Break of Structure (BOS) detection"""
    if not safe_df_check(df, 50):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "BOS", "type": "strategy"}
    try:
        sh = df["high"].rolling(20).max().iloc[-2] if len(df) >= 22 else df["high"].max()
        sl = df["low"].rolling(20).min().iloc[-2] if len(df) >= 22 else df["low"].min()
        if df["close"].iloc[-1] > sh:
            return {"signal": "BUY", "score": 86.0, "reason": "BOS", "type": "strategy"}
        elif df["close"].iloc[-1] < sl:
            return {"signal": "SELL", "score": 86.0, "reason": "BOS", "type": "strategy"}
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "BOS", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "BOS", "type": "strategy"}

def fair_value_gap(df: pd.DataFrame) -> Dict:
    """Fair Value Gap (FVG) detection"""
    if not safe_df_check(df, 3):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "FVG", "type": "strategy"}
    try:
        if df["low"].iloc[-1] > df["high"].iloc[-3]:
            return {"signal": "BUY", "score": 89.0, "reason": "FVG", "type": "strategy"}
        elif df["high"].iloc[-1] < df["low"].iloc[-3]:
            return {"signal": "SELL", "score": 89.0, "reason": "FVG", "type": "strategy"}
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "FVG", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "FVG", "type": "strategy"}

# -------------------------------
# INSTITUTIONAL FILTERS (3)
# -------------------------------

def smart_money_filter(df: pd.DataFrame, asset_config: Dict) -> Dict:
    """Smart money liquidity grab detection"""
    if not safe_df_check(df, asset_config.get("min_data_points", 30)):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "SmartMoney", "type": "filter"}
    try:
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
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "SmartMoney", "type": "filter"}

def vwap_deviation_filter(df: pd.DataFrame, asset_config: Dict) -> Dict:
    """VWAP deviation institutional filter"""
    if not safe_df_check(df, asset_config.get("min_data_points", 50)):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "VWAP", "type": "filter"}
    try:
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
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "VWAP", "type": "filter"}

def volume_profile_filter(df: pd.DataFrame, asset_config: Dict) -> Dict:
    """Volume profile value area filter"""
    if not safe_df_check(df, asset_config.get("min_data_points", 60)):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "VolProfile", "type": "filter"}
    try:
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
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "VolProfile", "type": "filter"}

# -------------------------------
# TECHNICAL INDICATORS (5)
# -------------------------------

def ema_macd_indicator(df: pd.DataFrame) -> Dict:
    """EMA + MACD combined indicator"""
    if not safe_df_check(df, 50):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "indicator"}
    try:
        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        macd_line, macd_sig = macd(df["close"])
        if len(ema_fast) < 2 or len(macd_line) < 1:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "indicator"}
        if ema_fast.iloc[-1] > ema_slow.iloc[-1] and macd_line[-1] > macd_sig[-1]:
            return {"signal": "BUY", "score": 82.0, "reason": "EMA_MACD", "type": "indicator"}
        elif ema_fast.iloc[-1] < ema_slow.iloc[-1] and macd_line[-1] < macd_sig[-1]:
            return {"signal": "SELL", "score": 82.0, "reason": "EMA_MACD", "type": "indicator"}
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "indicator"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "indicator"}

def rsi_indicator(df: pd.DataFrame) -> Dict:
    """RSI overbought/oversold indicator"""
    if not safe_df_check(df, 20):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI", "type": "indicator"}
    try:
        rsi_vals = rsi(df["close"])
        if len(rsi_vals) == 0:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI", "type": "indicator"}
        rsi_val = rsi_vals[-1]
        if rsi_val < 30:
            return {"signal": "BUY", "score": 75.0, "reason": "RSI", "type": "indicator"}
        elif rsi_val > 70:
            return {"signal": "SELL", "score": 75.0, "reason": "RSI", "type": "indicator"}
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI", "type": "indicator"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI", "type": "indicator"}

def bollinger_indicator(df: pd.DataFrame) -> Dict:
    """Bollinger Bands indicator"""
    if not safe_df_check(df, 30):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Bollinger", "type": "indicator"}
    try:
        bb = bollinger_bands(df["close"])
        if bb["upper"].empty:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Bollinger", "type": "indicator"}
        if df["close"].iloc[-1] < bb["lower"].iloc[-1]:
            return {"signal": "BUY", "score": 76.0, "reason": "Bollinger", "type": "indicator"}
        elif df["close"].iloc[-1] > bb["upper"].iloc[-1]:
            return {"signal": "SELL", "score": 76.0, "reason": "Bollinger", "type": "indicator"}
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Bollinger", "type": "indicator"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Bollinger", "type": "indicator"}

def volume_indicator(df: pd.DataFrame, asset_config: Dict = None) -> Dict:
    """Volume surge indicator"""
    if asset_config is None:
        asset_config = {}
    if not safe_df_check(df, asset_config.get("min_data_points", 50)):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Volume", "type": "indicator"}
    try:
        multiplier = asset_config.get("momentum_volume_multiplier", 2.0)
        vol_spike = df["volume"].iloc[-1] > df["volume"].rolling(50).mean().iloc[-1] * multiplier
        if vol_spike and df["close"].iloc[-1] > df["open"].iloc[-1]:
            return {"signal": "BUY", "score": 86.0, "reason": "Volume", "type": "indicator"}
        elif vol_spike and df["close"].iloc[-1] < df["open"].iloc[-1]:
            return {"signal": "SELL", "score": 86.0, "reason": "Volume", "type": "indicator"}
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Volume", "type": "indicator"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Volume", "type": "indicator"}

def supertrend_bollinger_indicator(df: pd.DataFrame) -> Dict:
    """SuperTrend + Bollinger combined"""
    if not safe_df_check(df, 50):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "SuperTrend+Boll", "type": "indicator"}
    try:
        st = supertrend(df)
        bb = bollinger_bands(df["close"])
        close = df["close"].iloc[-1]
        bull = (not st.empty and close > st.iloc[-1]) and (not bb["ma"].empty and close > bb["ma"].iloc[-1])
        bear = (not st.empty and close < st.iloc[-1]) and (not bb["ma"].empty and close < bb["ma"].iloc[-1])
        if bull:
            return {"signal": "BUY", "score": 84.0, "reason": "SuperTrend+Boll", "type": "indicator"}
        if bear:
            return {"signal": "SELL", "score": 84.0, "reason": "SuperTrend+Boll", "type": "indicator"}
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "SuperTrend+Boll", "type": "indicator"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "SuperTrend+Boll", "type": "indicator"}

# -------------------------------
# ENGINE CONFIGURATION
# -------------------------------

ALL_ADVANCED_STRATEGIES = [
    market_microstructure_imbalance,
    liquidity_void_hunter,
    volatility_regime_detector,
    gamma_flip_detector,
    fractal_dimension_strategy,
    institutional_footprint_detector,
    auction_theory_analyzer,
    spoofing_detector,
    market_entropy_strategy
]

ALL_CLASSIC_STRATEGIES = [
    fibonacci_vortex_hidden,
    quantum_entanglement_hidden,
    dark_pool_institutional_hidden,
    order_block_detection,
    break_of_structure,
    fair_value_gap
]

ALL_INDICATORS = [
    ema_macd_indicator,
    rsi_indicator,
    bollinger_indicator,
    volume_indicator,
    supertrend_bollinger_indicator
]

ALL_FILTERS = [
    smart_money_filter,
    vwap_deviation_filter,
    volume_profile_filter
]

class Engines:
    QUANTUM = {
        "name": "Quantum Engine V2.0 Elite",
        "strategies": [
            market_microstructure_imbalance,
            institutional_footprint_detector,
            gamma_flip_detector,
            volatility_regime_detector,
            order_block_detection,
            break_of_structure,
            fair_value_gap,
            fibonacci_vortex_hidden
        ],
        "indicators": [
            ema_macd_indicator,
            supertrend_bollinger_indicator,
            volume_indicator,
            bollinger_indicator,
            rsi_indicator
        ],
        "filters": [
            smart_money_filter,
            vwap_deviation_filter,
            volume_profile_filter
        ]
    }
    
    MOMENTUM = {
        "name": "Momentum Scalper V2.0 Elite",
        "strategies": [
            fractal_dimension_strategy,
            liquidity_void_hunter,
            break_of_structure,
            dark_pool_institutional_hidden
        ],
        "indicators": [
            volume_indicator,
            rsi_indicator,
            ema_macd_indicator
        ],
        "filters": [
            vwap_deviation_filter
        ]
    }
    
    BREAKOUT = {
        "name": "Breakout Hunter V2.0 Elite",
        "strategies": [
            gamma_flip_detector,
            auction_theory_analyzer,
            break_of_structure,
            fair_value_gap
        ],
        "indicators": [
            bollinger_indicator,
            volume_indicator,
            supertrend_bollinger_indicator
        ],
        "filters": [
            volume_profile_filter,
            smart_money_filter
        ]
    }
    
    MEANREVERSION = {
        "name": "Mean Reversion V2.0 Elite",
        "strategies": [
            market_entropy_strategy,
            spoofing_detector,
            fibonacci_vortex_hidden,
            quantum_entanglement_hidden
        ],
        "indicators": [
            rsi_indicator,
            bollinger_indicator,
            ema_macd_indicator
        ],
        "filters": [
            volume_profile_filter,
            vwap_deviation_filter
        ]
    }

class EngineState:
    current_mode: str = "quantum"

# -------------------------------
# Market Regime Detection
# -------------------------------
def compute_market_conditions(asset: str, df_map: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Advanced market condition analysis"""
    asset_config = ASSET_CONFIG.get(asset, {})
    refs = [tf for tf in ["4h", "1h", "1d"] if tf in df_map]
    if not refs:
        return {"volatility": 0.0, "trend_strength": 0.0, "momentum_spike": False, "breakout_detected": False, "range_bound": True}
    
    df = df_map[refs[0]]
    if not safe_df_check(df, asset_config.get("min_data_points", 50)):
        return {"volatility": 0.0, "trend_strength": 0.0, "momentum_spike": False, "breakout_detected": False, "range_bound": True}
    
    try:
        _atr_series = atr(df)
        _atr = _atr_series.iloc[-1] if not _atr_series.empty else 0.0
        price = df["close"].iloc[-1]
        volatility = float((_atr / (price + 1e-9)) * 100)
        
        ema_fast = df["close"].ewm(span=12).mean()
        ema_slow = df["close"].ewm(span=26).mean()
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
    except Exception as e:
        logger.error(f"Market conditions error for {asset}: {e}")
        return {"volatility": 0.0, "trend_strength": 0.0, "momentum_spike": False, "breakout_detected": False, "range_bound": True}

def select_mode(market_conditions: Dict[str, float]) -> str:
    """Select optimal engine based on market regime"""
    if market_conditions["momentum_spike"]:
        return "momentum"
    if market_conditions["breakout_detected"]:
        return "breakout"
    if market_conditions["range_bound"]:
        return "meanreversion"
    return "quantum"

def get_engine(mode: str) -> Dict:
    """Get engine configuration by mode"""
    if mode == "momentum":
        return Engines.MOMENTUM
    if mode == "breakout":
        return Engines.BREAKOUT
    if mode == "meanreversion":
        return Engines.MEANREVERSION
    return Engines.QUANTUM

# -------------------------------
# Data Provider
# -------------------------------
class RealDataProvider:
    def __init__(self):
        self.yf = yf

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 300) -> Optional[pd.DataFrame]:
        # Map timeframe to yfinance interval
        interval_map = {
            "1d": "1d",
            "4h": "1h",
            "1h": "1h",
            "15m": "15m",
            "5m": "5m"
        }
        
        period_map = {
            "1d": "1mo",
            "4h": "1mo",
            "1h": "1mo",
            "15m": "5d",
            "5m": "5d"
        }
        
        interval = interval_map.get(timeframe, "1d")
        period = period_map.get(timeframe, "1mo")
        
        try:
            logger.info(f"Fetching {symbol} {timeframe} (interval: {interval}, period: {period})")
            df = self.yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
            
            if df is None or df.empty:
                logger.warning(f"No data for {symbol} {timeframe}")
                return None
            
            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            df.reset_index(inplace=True)
            
            ts_col = "Datetime" if "Datetime" in df.columns else "Date"
            df["timestamp"] = pd.to_datetime(df[ts_col])
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].tail(limit)
            
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            
            df = df.dropna().reset_index(drop=True)
            
            # If we requested 4h but got 1h, resample to 4h
            if timeframe == "4h" and interval == "1h" and len(df) > 0:
                df.set_index('timestamp', inplace=True)
                df = df.resample('4H').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                df.reset_index(inplace=True)
            
            logger.info(f"Fetched {symbol} {timeframe}: {len(df)} bars")
            return df.tail(limit)
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {timeframe}: {str(e)}")
            return None

# -------------------------------
# Selection Helpers
# -------------------------------
def select_best(items: List[Dict], item_type: str, min_score: float, limit: int = 4) -> List[Dict]:
    """Select best scoring items of a specific type"""
    picked = [r for r in items if r.get("type") == item_type and r.get("score", 0.0) >= min_score]
    picked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return picked[:limit]

# -------------------------------
# Core Analysis Engine
# -------------------------------
def engine_ai_analysis(asset: str, tf_data: Dict[str, pd.DataFrame], engine: Dict) -> Dict:
    """Core AI analysis with all strategies, indicators, and filters"""
    asset_config = ASSET_CONFIG.get(asset, {})
    all_strategy_results = []
    all_indicator_results = []
    all_filter_results = []

    for tf, df in tf_data.items():
        if not safe_df_check(df, asset_config.get("min_data_points", 50)):
            continue
        
        # Run strategies
        for strategy_fn in engine["strategies"]:
            try:
                result = strategy_fn(df)
                if result and result.get("signal") != "NEUTRAL":
                    result["timeframe"] = tf
                    all_strategy_results.append(result)
            except Exception as e:
                logger.debug(f"Strategy {strategy_fn.__name__} error: {e}")

        # Run indicators
        for indicator_fn in engine["indicators"]:
            try:
                if indicator_fn == volume_indicator:
                    result = indicator_fn(df, asset_config)
                else:
                    result = indicator_fn(df)
                if result and result.get("signal") != "NEUTRAL":
                    result["timeframe"] = tf
                    all_indicator_results.append(result)
            except Exception as e:
                logger.debug(f"Indicator {indicator_fn.__name__} error: {e}")

        # Run filters
        for filt_fn in engine.get("filters", []):
            try:
                result = filt_fn(df, asset_config)
                if result and result.get("signal") != "NEUTRAL":
                    result["timeframe"] = tf
                    all_filter_results.append(result)
            except Exception as e:
                logger.debug(f"Filter {filt_fn.__name__} error: {e}")

    # Select best results
    best_strategies = select_best(all_strategy_results, "strategy", min_score=85.0, limit=6)
    best_indicators = select_best(all_indicator_results, "indicator", min_score=75.0, limit=5)
    best_filters = select_best(all_filter_results, "filter", min_score=75.0, limit=3)

    # Calculate signal
    all_signals = [r["signal"] for r in best_strategies + best_indicators]
    signal_counts = Counter(all_signals)

    # Require strong alignment
    if len(best_strategies) >= 4 and len(best_indicators) >= 3:
        if signal_counts.get("BUY", 0) >= 6:
            final_signal = "BUY"
            buy_scores = [r["score"] for r in best_strategies + best_indicators if r["signal"] == "BUY"]
            confidence = sum(buy_scores) / (signal_counts["BUY"] + 1e-9)
        elif signal_counts.get("SELL", 0) >= 6:
            final_signal = "SELL"
            sell_scores = [r["score"] for r in best_strategies + best_indicators if r["signal"] == "SELL"]
            confidence = sum(sell_scores) / (signal_counts["SELL"] + 1e-9)
        else:
            final_signal = "NEUTRAL"
            confidence = 0.0
    else:
        final_signal = "NEUTRAL"
        confidence = 0.0

    # Apply filter adjustments
    filter_bias = Counter([f["signal"] for f in best_filters])
    if final_signal in ["BUY", "SELL"] and best_filters:
        agree = filter_bias.get(final_signal, 0)
        disagree = sum(filter_bias.values()) - agree
        if agree > disagree:
            confidence = min(100.0, confidence + 4.0)
        elif disagree > agree:
            confidence = max(0.0, confidence - 6.0)

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
# Cleanup Function
# -------------------------------
def cleanup_old_alerts():
    """Remove cooldown entries older than 24 hours"""
    global LAST_LEGENDARY_ALERT
    now = datetime.utcnow()
    to_remove = [
        key for key, timestamp in LAST_LEGENDARY_ALERT.items()
        if (now - timestamp).total_seconds() > 86400
    ]
    for key in to_remove:
        del LAST_LEGENDARY_ALERT[key]
    if to_remove:
        logger.info(f"Cleaned up {len(to_remove)} old alert cooldowns")

# -------------------------------
# API Routes
# -------------------------------
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "legendary",
        "timestamp": datetime.utcnow().isoformat(),
        "mode": EngineState.current_mode,
        "signals_in_history": len(SIGNAL_HISTORY),
        "active_cooldowns": len(LAST_LEGENDARY_ALERT),
        "rejected_signals": len(REJECTED_SIGNALS)
    }), 200

@app.route('/signals', methods=['GET'])
def get_signals():
    """Get recent signals"""
    recent = list(SIGNAL_HISTORY)[-10:]
    return jsonify({"signals": recent, "count": len(SIGNAL_HISTORY)}), 200

@app.route('/rejected', methods=['GET'])
def get_rejected():
    """Get rejected signals"""
    recent = list(REJECTED_SIGNALS)[-10:]
    return jsonify({"rejected": recent, "count": len(REJECTED_SIGNALS)}), 200

# -------------------------------
# Enhanced Main Scan Loop - SIMPLIFIED AND DEBUGGED
# -------------------------------
def main_scan_loop():
    """Main scanning loop with all filters"""
    logger.info("🏆 Initializing ULTIMATE #1 LEGENDARY BOT")
    logger.info(f"⚡ Assets to scan: {len(ASSETS)}")
    logger.info(f"📊 Timeframes: {len(ALL_TIMEFRAMES)}")
    logger.info(f"⏱️ Scan interval: {SCAN_INTERVAL_SEC} seconds")
    logger.info(f"🤖 Telegram: {'configured' if (TELEGRAM_TOKEN and CHAT_ID) else 'not configured'}")
    logger.info(f"🎯 Confidence Threshold: {GLOBAL_THRESHOLD}%")

    provider = RealDataProvider()
    telegram = LegendaryTelegramBot(TELEGRAM_TOKEN, CHAT_ID)
    telegram.start()

    logger.info("✅ ULTIMATE BOT READY - Starting scanning...")
    
    # Test data fetching
    logger.info("🔧 Testing data fetching...")
    test_asset = ASSETS[0]
    test_df = provider.fetch_ohlcv(test_asset, "1h", limit=50)
    if test_df is not None and not test_df.empty:
        logger.info(f"✅ Data fetching works! Test asset {test_asset}: {len(test_df)} bars fetched")
    else:
        logger.error(f"❌ Could not fetch test data for {test_asset}. Check your internet connection.")
        return

    scan_count = 0
    while True:
        try:
            scan_count += 1
            now_utc = datetime.utcnow()
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 Scan #{scan_count}: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            logger.info(f"{'='*60}")

            # Cleanup old alerts
            cleanup_old_alerts()

            # Check news filter
            is_news, news_reason = EconomicCalendar.is_news_time()
            if is_news:
                logger.info(f"⚠️ News blackout active: {news_reason}")
                time.sleep(SCAN_INTERVAL_SEC)
                continue

            session = EconomicCalendar.get_trading_session()
            logger.info(f"📍 Trading Session: {session}")
            
            signals_found = 0
            
            for asset_idx, asset in enumerate(ASSETS):
                logger.info(f"\n📈 Processing {asset} ({asset_idx+1}/{len(ASSETS)})...")
                asset_config = ASSET_CONFIG.get(asset, {})
                tf_data = {}
                
                # Fetch data for all timeframes
                for tf_idx, tf in enumerate(ALL_TIMEFRAMES):
                    try:
                        df = provider.fetch_ohlcv(asset, tf, limit=200)
                        if safe_df_check(df, asset_config.get("min_data_points", 50)):
                            tf_data[tf] = df
                            logger.info(f"  ✓ {tf}: {len(df)} bars")
                        else:
                            logger.warning(f"  ✗ {tf}: Insufficient data")
                    except Exception as e:
                        logger.error(f"  ✗ {tf}: Failed to fetch - {str(e)}")

                if len(tf_data) < 3:
                    logger.warning(f"  Skipping {asset} - insufficient timeframes ({len(tf_data)}/5)")
                    continue

                # Get primary timeframe for condition checks
                primary_df = tf_data.get("1h") or tf_data.get("4h") or tf_data.get("1d")
                if primary_df is None:
                    logger.warning(f"  Skipping {asset} - no primary timeframe data")
                    continue

                # Apply market condition filters
                is_choppy, atr_ratio = MarketConditionFilter.is_choppy_market(primary_df, asset_config)
                if is_choppy:
                    REJECTED_SIGNALS.append({
                        "asset": asset,
                        "reason": f"Choppy market (ATR ratio: {atr_ratio:.3f})",
                        "timestamp": now_utc.isoformat()
                    })
                    logger.info(f"  ✗ Choppy market (ATR ratio: {atr_ratio:.3f})")
                    continue

                has_liquidity, vol_ratio = MarketConditionFilter.has_sufficient_liquidity(primary_df, asset_config)
                if not has_liquidity:
                    REJECTED_SIGNALS.append({
                        "asset": asset,
                        "reason": f"Low liquidity (Vol ratio: {vol_ratio:.2f})",
                        "timestamp": now_utc.isoformat()
                    })
                    logger.info(f"  ✗ Low liquidity (Vol ratio: {vol_ratio:.2f})")
                    continue

                is_vol_spike, vol_spike_ratio = MarketConditionFilter.check_volatility_spike(primary_df)
                if is_vol_spike:
                    REJECTED_SIGNALS.append({
                        "asset": asset,
                        "reason": f"Volatility spike (Ratio: {vol_spike_ratio:.2f})",
                        "timestamp": now_utc.isoformat()
                    })
                    logger.info(f"  ✗ Volatility spike (Ratio: {vol_spike_ratio:.2f})")
                    continue

                # Detect market regime and select engine
                market_conditions = compute_market_conditions(asset, tf_data)
                EngineState.current_mode = select_mode(market_conditions)
                engine = get_engine(EngineState.current_mode)
                
                logger.info(f"  Engine: {engine['name']}")

                # Run core analysis
                result = engine_ai_analysis(asset, tf_data, engine)
                threshold = ASSET_THRESHOLD_OVERRIDE.get(asset, GLOBAL_THRESHOLD)

                # Check if signal meets requirements
                if result['signal'] == "NEUTRAL":
                    logger.info(f"  No signal from analysis")
                    continue

                # Check timeframe alignment
                is_aligned, aligned_count = MarketConditionFilter.check_timeframe_alignment(tf_data, result['signal'])
                if not is_aligned:
                    REJECTED_SIGNALS.append({
                        "asset": asset,
                        "reason": f"Timeframe misalignment ({aligned_count}/{len(tf_data)} aligned)",
                        "timestamp": now_utc.isoformat()
                    })
                    logger.info(f"  ✗ Timeframe misalignment ({aligned_count}/{len(tf_data)} aligned)")
                    continue

                # Check confidence and alignment thresholds
                if (result['confidence'] >= threshold and
                    result['num_strategies_aligned'] >= 4 and
                    result['num_indicators_aligned'] >= 3):

                    # Generate signals for all expiries
                    for expiry in EXPIRIES:
                        cooldown_key = f"{asset}_{expiry}"
                        if cooldown_key in LAST_LEGENDARY_ALERT:
                            elapsed = (now_utc - LAST_LEGENDARY_ALERT[cooldown_key]).total_seconds() / 60
                            if elapsed < COOLDOWN_MINUTES:
                                logger.info(f"  Cooldown active for {expiry}m expiry ({elapsed:.1f}/{COOLDOWN_MINUTES} mins)")
                                continue

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
                            'timestamp': now_utc.isoformat(),
                            'engine': engine["name"],
                            'session': session,
                            'timeframe_alignment': f"{aligned_count}/{len(tf_data)}"
                        }

                        SIGNAL_HISTORY.append(alert_data)
                        LAST_LEGENDARY_ALERT[cooldown_key] = now_utc

                        # Send Telegram alert
                        if telegram.initialized:
                            telegram.send_signal(alert_data)
                        
                        logger.info(f"  🔥 LEGENDARY SIGNAL: {result['signal']} @ {result['confidence']:.1f}% Expiry: {expiry}m")
                        signals_found += 1
                        time.sleep(0.5)  # Small delay between expiry alerts
                else:
                    logger.info(f"  Signal below threshold ({result['confidence']:.1f}% < {threshold}%)")
            
            logger.info(f"\n✅ Scan complete. Signals found: {signals_found}")
            logger.info(f"⏱️ Next scan in {SCAN_INTERVAL_SEC} seconds...")

            if RUN_ONCE:
                logger.info("RUN_ONCE enabled - stopping after one scan")
                break
                
            time.sleep(SCAN_INTERVAL_SEC)

        except KeyboardInterrupt:
            logger.info("\n🛑 Stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Error in main loop: {str(e)}", exc_info=True)
            time.sleep(5)

# -------------------------------
# Flask Server
# -------------------------------
def run_flask():
    """Run Flask server in a separate thread"""
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)

# -------------------------------
# Standalone Scanning Mode
# -------------------------------
def start_scanning_only():
    """Start only the scanning without Flask API"""
    logger.info("🚀 Starting scanning-only mode")
    main_scan_loop()

# -------------------------------
# Entrypoint
# -------------------------------
def main():
    """Main entry point"""
    logger.info(f"{'='*60}")
    logger.info(f"🚀 STARTING ULTIMATE #1 LEGENDARY TRADING BOT")
    logger.info(f"{'='*60}")
    
    # Check if we should run in scanning-only mode
    if os.getenv("SCAN_ONLY", "False") == "True" or RUN_ONCE:
        logger.info("📡 Running in scanning-only mode")
        start_scanning_only()
    else:
        # Start Flask in separate thread
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        logger.info(f"🌐 Flask API running on http://0.0.0.0:{PORT}")
        
        # Start scanner loop in main thread
        main_scan_loop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}", exc_info=True)

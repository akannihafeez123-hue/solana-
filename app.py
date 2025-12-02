"""
📮 ULTIMATE LEGENDARY Pocket Option AI Scanner – V4.5
- Telegram bot starts immediately at app startup
- Scanner loop runs in background thread
- Flask API runs in main thread
- Full strategies, indicators, and filters included (no placeholders)
- Debug logging of every incoming Telegram update
"""

import os
import sys
import time
import threading
import logging
import asyncio
import warnings
from typing import Dict, List, Tuple, Optional
from collections import deque, Counter
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, jsonify

warnings.filterwarnings("ignore")

# -------------------------------
# Logging setup
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -------------------------------
# Environment config
# -------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
ADMIN_IDS: List[int] = []
_admin_env = os.getenv("ADMIN_IDS", "")
if _admin_env:
    for x in _admin_env.split(","):
        x = x.strip()
        if x.isdigit():
            ADMIN_IDS.append(int(x))
if not ADMIN_IDS and CHAT_ID and CHAT_ID.isdigit():
    ADMIN_IDS = [int(CHAT_ID)]

LEGENDARY_GATE = float(os.getenv("LEGENDARY_GATE", "95"))
GLOBAL_THRESHOLD = LEGENDARY_GATE
ASSET_THRESHOLD_OVERRIDE: Dict[str, float] = {}

COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "15"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "45"))
RUN_ONCE = os.getenv("RUN_ONCE", "False").lower() in ("true", "1", "yes")
PORT = int(os.getenv("PORT") or os.getenv("port") or "8080")

ASSETS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "GC=F"]
EXPIRIES = [5, 15, 30, 60]
ALL_TIMEFRAMES = ["1M", "1w", "1d", "4h", "1h"]

ASSET_CONFIG = {
    "EURUSD=X": {"min_data_points": 50, "volatility_threshold": 0.4, "vwap_threshold": 0.008, "volume_profile_sensitivity": 0.004, "trend_strength_multiplier": 1.2, "momentum_volume_multiplier": 1.8},
    "GBPUSD=X": {"min_data_points": 50, "volatility_threshold": 0.5, "vwap_threshold": 0.01, "volume_profile_sensitivity": 0.005, "trend_strength_multiplier": 1.3, "momentum_volume_multiplier": 2.0},
    "USDJPY=X": {"min_data_points": 50, "volatility_threshold": 0.45, "vwap_threshold": 0.009, "volume_profile_sensitivity": 0.0045, "trend_strength_multiplier": 1.15, "momentum_volume_multiplier": 1.9},
    "AUDUSD=X": {"min_data_points": 50, "volatility_threshold": 0.5, "vwap_threshold": 0.01, "volume_profile_sensitivity": 0.005, "trend_strength_multiplier": 1.25, "momentum_volume_multiplier": 2.0},
    "GC=F": {"min_data_points": 50, "volatility_threshold": 0.6, "vwap_threshold": 0.012, "volume_profile_sensitivity": 0.006, "trend_strength_multiplier": 1.4, "momentum_volume_multiplier": 2.2}
}

MAX_SIGNAL_HISTORY = 1000
SIGNAL_HISTORY: deque = deque(maxlen=MAX_SIGNAL_HISTORY)
LAST_LEGENDARY_ALERT: Dict[str, datetime] = {}

app = Flask(__name__)

# -------------------------------
# Telegram bot
# -------------------------------
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
    TELEGRAM_AVAILABLE = True
except Exception:
    TELEGRAM_AVAILABLE = False
    Application = None
    logger.warning("python-telegram-bot not available; Telegram features disabled")

def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

class LegendaryTelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.app: Optional[Application] = None

    def start(self):
        if not TELEGRAM_AVAILABLE:
            logger.warning("Telegram lib not available; skipping bot start")
            return
        if not self.token or not self.chat_id:
            logger.warning("Telegram not configured (missing TELEGRAM_TOKEN or CHAT_ID)")
            return
        try:
            self.app = Application.builder().token(self.token).build()

            # Debug: log every incoming message
            async def debug_log(update: Update, context: ContextTypes.DEFAULT_TYPE):
                text = update.message.text if update.message else None
                uid = update.effective_user.id if update.effective_user else None
                logger.info(f"Incoming update: user_id={uid}, text={text}")

            self.app.add_handler(MessageHandler(filters.ALL, debug_log))
            self.app.add_handler(CommandHandler("start", self._start))
            self.app.add_handler(CommandHandler("status", self._status))
            self.app.add_handler(CommandHandler("history", self._history))
            self.app.add_handler(CommandHandler("threshold", self._threshold))

            threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")

    async def _run(self):
        try:
            logger.info("🚀 Telegram bot polling started")
            await self.app.run_polling()
        except Exception as e:
            logger.error(f"Telegram polling error: {e}")

    async def _start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user_id = update.effective_user.id
            if not is_admin(user_id):
                await update.message.reply_text("❌ Unauthorized")
                return
            await update.message.reply_text("📮 ULTIMATE Legendary Scanner Active")
        except Exception as e:
            logger.error(f"/start handler error: {e}")

    async def _status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not is_admin(update.effective_user.id):
                await update.message.reply_text("❌ Unauthorized")
                return
            overrides = "\n".join([f"{k}: {v:.1f}%" for k, v in ASSET_THRESHOLD_OVERRIDE.items()]) or "None"
            await update.message.reply_text(
                f"✅ Active\nMode: {EngineState.current_mode}\nGlobal threshold: {GLOBAL_THRESHOLD:.1f}%\n"
                f"Signals in history: {len(SIGNAL_HISTORY)}\nPer-asset overrides:\n{overrides}"
            )
        except Exception as e:
            logger.error(f"/status handler error: {e}")

    async def _history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not is_admin(update.effective_user.id):
                await update.message.reply_text("❌ Unauthorized")
                return
            if not SIGNAL_HISTORY:
                await update.message.reply_text("No signals in history")
                return
            recent = list(SIGNAL_HISTORY)[-5:]
            msg = "📊 Recent Signals:\n\n"
            for sig in recent:
                msg += f"{sig['asset']} {sig['signal']} @ {sig['confidence']:.1f}% ({sig['expiry']}m)\n"
            await update.message.reply_text(msg)
        except Exception as e:
            logger.error(f"/history handler error: {e}")

    async def _threshold(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not is_admin(update.effective_user.id):
                await update.message.reply_text("❌ Unauthorized")
                return
            args = context.args or []
            global GLOBAL_THRESHOLD, ASSET_THRESHOLD_OVERRIDE
            if not args:
                await update.message.reply_text("Usage: /threshold <value|reset> or /threshold <ASSET> <value|reset>")
                return
            if len(args) == 1:
                token = args[0].strip().lower()
                if token == "reset":
                    GLOBAL_THRESHOLD = LEGENDARY_GATE
                    await update.message.reply_text(f"🔄 Global threshold reset to {GLOBAL_THRESHOLD}%")
                    return
                try:
                    val = float(args[0])
                    if val <= 0 or val > 100:
                        raise ValueError
                    GLOBAL_THRESHOLD = val
                    await update.message.reply_text(f"✅ Global threshold set to {GLOBAL_THRESHOLD:.1f}%")
                    return
                except Exception:
                    await update.message.reply_text("Invalid number between 1 and 100, or 'reset'.")
                    return
            if len(args) >= 2:
                asset = args[0].strip()
                token = args[1].strip().lower()
                if asset not in ASSETS:
                    await update.message.reply_text(f"Unknown asset: {asset}. Supported: {', '.join(ASSETS)}")
                    return
                if token == "reset":
                    if asset in ASSET_THRESHOLD_OVERRIDE:
                        del ASSET_THRESHOLD_OVERRIDE[asset]
                        await update.message.reply_text(f"🔄 Threshold override removed for {asset}")
                    else:
                        await update.message.reply_text(f"No override set for {asset}")
                    return
                try:
                    val = float(args[1])
                    if val <= 0 or val > 100:
                        raise ValueError
                    ASSET_THRESHOLD_OVERRIDE[asset] = val
                    await update.message.reply_text(f"✅ Threshold for {asset} set to {val:.1f}%")
                    return
                except Exception:
                    await update.message.reply_text("Invalid number between 1 and 100, or 'reset'.")
                    return
        except Exception as e:
            logger.error(f"/threshold handler error: {e}")

    def send_alert(self, message: str):
        if not TELEGRAM_AVAILABLE or not self.app:
            return
        try:
            asyncio.run(self.app.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="HTML"))
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

# -------------------------------
# Safe validation helpers
# -------------------------------
def safe_series_check(series: pd.Series, min_len: int = 1) -> bool:
    return series is not None and isinstance(series, pd.Series) and len(series) >= min_len

def safe_df_check(df: pd.DataFrame, min_len: int = 1) -> bool:
    return df is not None and isinstance(df, pd.DataFrame) and len(df) >= min_len

# -------------------------------
# Math helpers & indicators
# -------------------------------
def ema(series: pd.Series, period: int) -> np.ndarray:
    if not safe_series_check(series, period):
        return np.array([])
    alpha = 2 / (period + 1)
    ema_vals = [series.iloc[0]]
    for v in series.iloc[1:]:
        ema_vals.append(alpha * v + (1 - alpha) * ema_vals[-1])
    return np.array(ema_vals)

def rsi(series: pd.Series, period: int = 14) -> np.ndarray:
    if not safe_series_check(series, period + 1):
        return np.array([])
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / (down.replace(0, np.nan))
    rsi_vals = 100 - (100 / (1 + rs))
    return rsi_vals.fillna(50.0).values

def macd(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    if not safe_series_check(series, 26):
        return np.array([]), np.array([])
    ema_fast = pd.Series(ema(series, 12))
    ema_slow = pd.Series(ema(series, 26))
    macd_line = ema_fast - ema_slow
    macd_sig = pd.Series(ema(macd_line, 9))
    return macd_line.values, macd_sig.values

def bollinger_bands(series: pd.Series, period: int = 20) -> Dict:
    if not safe_series_check(series, period):
        return {"upper": pd.Series(dtype=float), "lower": pd.Series(dtype=float), "ma": pd.Series(dtype=float)}
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return {"upper": upper, "lower": lower, "ma": ma}

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if not safe_df_check(df, period):
        return pd.Series(dtype=float)
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    if not safe_df_check(df, period):
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
    if not safe_df_check(df):
        return pd.Series(dtype=float)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_pv = (tp * df["volume"]).cumsum()
    return cum_pv / (cum_vol.replace(0, np.nan))

def simple_volume_profile(df: pd.DataFrame, bins: int = 24) -> Dict[str, float]:
    if not safe_df_check(df, 2):
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
# Filters
# -------------------------------
def smart_money_filter(df: pd.DataFrame, asset_config: Dict) -> Dict:
    if not safe_df_check(df, asset_config.get("min_data_points", 30)):
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
    if not safe_df_check(df, asset_config.get("min_data_points", 50)):
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
    if not safe_df_check(df, asset_config.get("min_data_points", 60)):
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
# Core strategies
# -------------------------------
def fibonacci_vortex_hidden(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 50):
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
    if not safe_df_check(df, 30):
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
    if not safe_df_check(df, 50):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "DarkPool", "type": "strategy"}
    vol_ma = df["volume"].rolling(50).mean().iloc[-1]
    vol_spike = df["volume"].iloc[-1] > (vol_ma * 2.0 if not np.isnan(vol_ma) else 0)
    if vol_spike:
        signal = "BUY" if df["close"].iloc[-1] > df["open"].iloc[-1] else "SELL"
        return {"signal": signal, "score": 95.0, "reason": "DarkPool", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "DarkPool", "type": "strategy"}

def order_block_detection(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 20):
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
    if not safe_df_check(df, 50):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "BOS", "type": "strategy"}
    sh = df["high"].rolling(20).max().iloc[-2]
    sl = df["low"].rolling(20).min().iloc[-2]
    if df["close"].iloc[-1] > sh:
        return {"signal": "BUY", "score": 86.0, "reason": "BOS", "type": "strategy"}
    elif df["close"].iloc[-1] < sl:
        return {"signal": "SELL", "score": 86.0, "reason": "BOS", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "BOS", "type": "strategy"}

def fair_value_gap(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 3):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "FVG", "type": "strategy"}
    if df["low"].iloc[-1] > df["high"].iloc[-3]:
        return {"signal": "BUY", "score": 89.0, "reason": "FVG", "type": "strategy"}
    elif df["high"].iloc[-1] < df["low"].iloc[-3]:
        return {"signal": "SELL", "score": 89.0, "reason": "FVG", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "FVG", "type": "strategy"}

# -------------------------------
# Advanced institutional strategies
# -------------------------------
def market_microstructure_imbalance(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 100):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "MicrostructureImbalance", "type": "strategy"}
    price_changes = df['close'].diff().abs()
    volume_normalized = df['volume'] / (df['volume'].rolling(50).mean() + 1e-9)
    impact_ratio = (price_changes / (volume_normalized + 1e-9)).rolling(20).mean()
    current_impact = impact_ratio.iloc[-1]
    avg_impact = impact_ratio.mean()
    buying_pressure = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)).rolling(10).mean()
    selling_pressure = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-9)).rolling(10).mean()
    imbalance = (buying_pressure.iloc[-1] if not np.isnan(buying_pressure.iloc[-1]) else 0.0) - (selling_pressure.iloc[-1] if not np.isnan(selling_pressure.iloc[-1]) else 0.0)
    if current_impact < avg_impact * 0.7 and imbalance > 0.15:
        return {"signal": "BUY", "score": 97.0, "reason": "MicrostructureImbalance", "type": "strategy"}
    elif current_impact < avg_impact * 0.7 and imbalance < -0.15:
        return {"signal": "SELL", "score": 97.0, "reason": "MicrostructureImbalance", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "MicrostructureImbalance", "type": "strategy"}

def liquidity_void_hunter(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 200):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "LiquidityVoid", "type": "strategy"}
    price_range = df['close'].max() - df['close'].min()
    bins = 50
    hist, edges = np.histogram(df['close'].values, bins=bins, weights=df['volume'].values)
    volume_threshold = np.percentile(hist, 20)
    void_indices = np.where(hist < volume_threshold)[0]
    if len(void_indices) == 0:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "LiquidityVoid", "type": "strategy"}
    current_price = df['close'].iloc[-1]
    void_prices = [(edges[i] + edges[i+1])/2 for i in void_indices]
    nearest_void_dist = min([abs(current_price - vp) for vp in void_prices])
    relative_dist = nearest_void_dist / (price_range + 1e-9)
    momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / (df['close'].iloc[-5] + 1e-9)
    if relative_dist < 0.02:
        if momentum > 0:
            return {"signal": "BUY", "score": 94.0, "reason": "LiquidityVoid", "type": "strategy"}
        elif momentum < 0:
            return {"signal": "SELL", "score": 94.0, "reason": "LiquidityVoid", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "LiquidityVoid", "type": "strategy"}

def volatility_regime_detector(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 100):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "VolRegime", "type": "strategy"}
    returns = df['close'].pct_change()
    rolling_vol = returns.rolling(20).std() * np.sqrt(252)
    vol_of_vol = rolling_vol.rolling(20).std()
    vol_zscore = (rolling_vol.iloc[-1] - rolling_vol.mean()) / (rolling_vol.std() + 1e-9)
    recent_vol_change = rolling_vol.iloc[-1] / (rolling_vol.iloc[-10] + 1e-9)
    vol_acceleration = vol_of_vol.iloc[-1] / (vol_of_vol.mean() + 1e-9)
    if vol_zscore < -0.5 and recent_vol_change > 1.3 and vol_acceleration > 1.2:
        price_momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / (df['close'].iloc[-5] + 1e-9)
        return {"signal": "BUY" if price_momentum > 0 else "SELL", "score": 93.0, "reason": "VolRegime", "type": "strategy"}
    elif vol_zscore > 1.0 and recent_vol_change < 0.8:
        price_zscore = (df['close'].iloc[-1] - df['close'].rolling(50).mean().iloc[-1]) / (df['close'].rolling(50).std().iloc[-1] + 1e-9)
        if price_zscore > 1.5:
            return {"signal": "SELL", "score": 91.0, "reason": "VolRegime", "type": "strategy"}
        elif price_zscore < -1.5:
            return {"signal": "BUY", "score": 91.0, "reason": "VolRegime", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "VolRegime", "type": "strategy"}

def gamma_flip_detector(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 150):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "GammaFlip", "type": "strategy"}
    returns = df['close'].pct_change()
    realized_vol = returns.rolling(20).std() * np.sqrt(252)
    close_prices = df['close'].values
    try:
        from scipy.signal import argrelextrema
        maxima_idx = argrelextrema(close_prices, np.greater, order=10)[0]
        minima_idx = argrelextrema(close_prices, np.less, order=10)[0]
    except Exception:
        maxima_idx = np.array([])
        minima_idx = np.array([])
    if len(maxima_idx) < 2 or len(minima_idx) < 2:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "GammaFlip", "type": "strategy"}
    recent_max = close_prices[maxima_idx[-1]] if len(maxima_idx) > 0 else df['close'].max()
    recent_min = close_prices[minima_idx[-1]] if len(minima_idx) > 0 else df['close'].min()
    current_price = df['close'].iloc[-1]
    range_position = (current_price - recent_min) / (recent_max - recent_min + 1e-9)
    price_velocity = (df['close'].iloc[-1] - df['close'].iloc[-3]) / (df['close'].iloc[-3] + 1e-9)
    volume_surge = df['volume'].iloc[-1] / (df['volume'].rolling(20).mean().iloc[-1] + 1e-9)
    if range_position > 0.95 and price_velocity > 0.01 and realized_vol.iloc[-1] < realized_vol.iloc[-10] and volume_surge > 1.5:
        return {"signal": "BUY", "score": 96.0, "reason": "GammaFlip", "type": "strategy"}
    elif range_position < 0.05 and price_velocity < -0.01 and realized_vol.iloc[-1] < realized_vol.iloc[-10] and volume_surge > 1.5:
        return {"signal": "SELL", "score": 96.0, "reason": "GammaFlip", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "GammaFlip", "type": "strategy"}

def fractal_dimension_strategy(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 100):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "FractalDim", "type": "strategy"}
    def hurst_exponent(ts, max_lag=20):
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    prices = df['close'].values[-100:]
    hurst = hurst_exponent(prices)
    momentum = (df['close'].iloc[-1] - df['close'].iloc[-10]) / (df['close'].iloc[-10] + 1e-9)
    if hurst > 0.65 and abs(momentum) > 0.015:
        return {"signal": "BUY" if momentum > 0 else "SELL", "score": 92.0, "reason": "FractalDim", "type": "strategy"}
    elif hurst < 0.45:
        price_zscore = (df['close'].iloc[-1] - df['close'].rolling(50).mean().iloc[-1]) / (df['close'].rolling(50).std().iloc[-1] + 1e-9)
        if price_zscore > 2.0:
            return {"signal": "SELL", "score": 90.0, "reason": "FractalDim", "type": "strategy"}
        elif price_zscore < -2.0:
            return {"signal": "BUY", "score": 90.0, "reason": "FractalDim", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "FractalDim", "type": "strategy"}

def institutional_footprint_detector(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 100):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "InstFootprint", "type": "strategy"}
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    _vwap = (typical_price * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-9)
    vwap_deviation = (df['close'] - _vwap) / (_vwap + 1e-9)
    recent_volume = df['volume'].iloc[-5:].mean()
    avg_volume = df['volume'].rolling(50).mean().iloc[-1]
    recent_volatility = df['close'].iloc[-5:].std() / (df['close'].iloc[-5:].mean() + 1e-9)
    avg_volatility = (df['close'].rolling(50).std().iloc[-1] / (df['close'].rolling(50).mean().iloc[-1] + 1e-9))
    volume_surge = recent_volume / (avg_volume + 1e-9)
    vol_compression = recent_volatility / (avg_volatility + 1e-9)
    if volume_surge > 1.5 and vol_compression < 0.6:
        if df['close'].iloc[-1] > _vwap.iloc[-1] and vwap_deviation.iloc[-1] > 0.005:
            return {"signal": "BUY", "score": 95.0, "reason": "InstFootprint", "type": "strategy"}
        elif df['close'].iloc[-1] < _vwap.iloc[-1] and vwap_deviation.iloc[-1] < -0.005:
            return {"signal": "SELL", "score": 95.0, "reason": "InstFootprint", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "InstFootprint", "type": "strategy"}

def auction_theory_analyzer(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 100):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "AuctionTheory", "type": "strategy"}
    price_range = df['high'].max() - df['low'].min()
    tick_size = price_range / 50 if price_range != 0 else 1.0
    tpo_counts = {}
    base_low = df['low'].min()
    for i in range(len(df)):
        low_tick = int((df['low'].iloc[i] - base_low) / tick_size)
        high_tick = int((df['high'].iloc[i] - base_low) / tick_size)
        for tick in range(low_tick, high_tick + 1):
            tpo_counts[tick] = tpo_counts.get(tick, 0) + 1
    if not tpo_counts:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "AuctionTheory", "type": "strategy"}
    sorted_ticks = sorted(tpo_counts.items(), key=lambda x: x[1], reverse=True)
    total_tpo = sum(tpo_counts.values())
    cumulative = 0
    value_area_ticks = []
    for tick, count in sorted_ticks:
        cumulative += count
        value_area_ticks.append(tick)
        if cumulative >= total_tpo * 0.70:
            break
    vah_price = base_low + max(value_area_ticks) * tick_size
    val_price = base_low + min(value_area_ticks) * tick_size
    current_price = df['close'].iloc[-1]
    momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / (df['close'].iloc[-5] + 1e-9)
    if current_price < val_price and momentum > 0.005:
        return {"signal": "BUY", "score": 93.0, "reason": "AuctionTheory", "type": "strategy"}
    elif current_price > vah_price and momentum < -0.005:
        return {"signal": "SELL", "score": 93.0, "reason": "AuctionTheory", "type": "strategy"}
    price_distance_from_vah = abs(current_price - vah_price) / (vah_price + 1e-9)
    price_distance_from_val = abs(current_price - val_price) / (val_price + 1e-9)
    if price_distance_from_vah < 0.002 and momentum < 0:
        return {"signal": "SELL", "score": 89.0, "reason": "AuctionTheory", "type": "strategy"}
    elif price_distance_from_val < 0.002 and momentum > 0:
        return {"signal": "BUY", "score": 89.0, "reason": "AuctionTheory", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "AuctionTheory", "type": "strategy"}

def spoofing_detector(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 50):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "SpoofDetect", "type": "strategy"}
    volume_spike = df['volume'].iloc[-1] / (df['volume'].rolling(20).mean().iloc[-2] + 1e-9)
    price_change = abs(df['close'].iloc[-1] - df['close'].iloc[-2]) / (df['close'].iloc[-2] + 1e-9)
    subsequent_reversal = (df['close'].iloc[-1] - df['close'].iloc[-3]) * (df['close'].iloc[-2] - df['close'].iloc[-3])
    if volume_spike > 3.0 and price_change < 0.003 and subsequent_reversal < 0:
        if df['close'].iloc[-2] > df['close'].iloc[-3]:
            return {"signal": "SELL", "score": 91.0, "reason": "SpoofDetect", "type": "strategy"}
        else:
            return {"signal": "BUY", "score": 91.0, "reason": "SpoofDetect", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "SpoofDetect", "type": "strategy"}

def market_entropy_strategy(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 100):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "MarketEntropy", "type": "strategy"}
    returns = df['close'].pct_change().dropna().values[-50:]
    bins = 10
    hist, _ = np.histogram(returns, bins=bins)
    if hist.sum() == 0:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "MarketEntropy", "type": "strategy"}
    hist = hist / hist.sum()
    entropy = -np.sum([p * np.log2(p + 1e-9) for p in hist if p > 0])
    max_entropy = np.log2(bins)
    normalized_entropy = entropy / (max_entropy + 1e-9)
    ema_short = df['close'].ewm(span=10).mean()
    ema_long = df['close'].ewm(span=30).mean()
    trend_strength = abs(ema_short.iloc[-1] - ema_long.iloc[-1]) / (ema_long.iloc[-1] + 1e-9)
    if normalized_entropy < 0.7 and trend_strength > 0.015:
        return {"signal": "BUY" if ema_short.iloc[-1] > ema_long.iloc[-1] else "SELL", "score": 92.0, "reason": "MarketEntropy", "type": "strategy"}
    elif normalized_entropy > 0.9:
        price_zscore = (df['close'].iloc[-1] - df['close'].rolling(50).mean().iloc[-1]) / (df['close'].rolling(50).std().iloc[-1] + 1e-9)
        if price_zscore > 1.8:
            return {"signal": "SELL", "score": 90.0, "reason": "MarketEntropy", "type": "strategy"}
        elif price_zscore < -1.8:
            return {"signal": "BUY", "score": 90.0, "reason": "MarketEntropy", "type": "strategy"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "MarketEntropy", "type": "strategy"}

# -------------------------------
# Indicators aggregation
# -------------------------------
def ema_macd_indicator(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 50):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "indicator"}
    ema_fast_vals = ema(df["close"], 12)
    ema_slow_vals = ema(df["close"], 26)
    macd_line, macd_sig = macd(df["close"])
    if len(ema_fast_vals) < 2 or len(macd_line) < 1 or len(macd_sig) < 1:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "indicator"}
    if ema_fast_vals[-1] > ema_slow_vals[-1] and macd_line[-1] > macd_sig[-1]:
        return {"signal": "BUY", "score": 82.0, "reason": "EMA_MACD", "type": "indicator"}
    elif ema_fast_vals[-1] < ema_slow_vals[-1] and macd_line[-1] < macd_sig[-1]:
        return {"signal": "SELL", "score": 82.0, "reason": "EMA_MACD", "type": "indicator"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "indicator"}

def rsi_indicator(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 20):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI", "type": "indicator"}
    rsi_vals = rsi(df["close"])
    if len(rsi_vals) == 0:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI", "type": "indicator"}
    rsi_val = rsi_vals[-1]
    if rsi_val < 30:
        return {"signal": "BUY", "score": 75.0, "reason": "RSI", "type": "indicator"}
    elif rsi_val > 70:
        return {"signal": "SELL", "score": 75.0, "reason": "RSI", "type": "indicator"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI", "type": "indicator"}

def bollinger_indicator(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 30):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Bollinger", "type": "indicator"}
    bb = bollinger_bands(df["close"])
    if bb["upper"].empty or bb["lower"].empty:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Bollinger", "type": "indicator"}
    if df["close"].iloc[-1] < bb["lower"].iloc[-1]:
        return {"signal": "BUY", "score": 76.0, "reason": "Bollinger", "type": "indicator"}
    elif df["close"].iloc[-1] > bb["upper"].iloc[-1]:
        return {"signal": "SELL", "score": 76.0, "reason": "Bollinger", "type": "indicator"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "Bollinger", "type": "indicator"}

def volume_indicator(df: pd.DataFrame, asset_config: Dict = None) -> Dict:
    if asset_config is None:
        asset_config = {}
    if not safe_df_check(df, asset_config.get("min_data_points", 50)):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Volume", "type": "indicator"}
    multiplier = asset_config.get("momentum_volume_multiplier", 2.0)
    vol_spike = df["volume"].iloc[-1] > df["volume"].rolling(50).mean().iloc[-1] * multiplier
    if vol_spike and df["close"].iloc[-1] > df["open"].iloc[-1]:
        return {"signal": "BUY", "score": 86.0, "reason": "Volume", "type": "indicator"}
    elif vol_spike and df["close"].iloc[-1] < df["open"].iloc[-1]:
        return {"signal": "SELL", "score": 86.0, "reason": "Volume", "type": "indicator"}
    return {"signal": "NEUTRAL", "score": 0.0, "reason": "Volume", "type": "indicator"}

def supertrend_bollinger_indicator(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 50):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "SuperTrend+Boll", "type": "indicator"}
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

# -------------------------------
# Engines composition
# -------------------------------
ALL_STRATEGIES = [
    order_block_detection, break_of_structure, fair_value_gap, fibonacci_vortex_hidden, dark_pool_institutional_hidden,
    market_microstructure_imbalance, institutional_footprint_detector, gamma_flip_detector,
    liquidity_void_hunter, auction_theory_analyzer, volatility_regime_detector, fractal_dimension_strategy,
    spoofing_detector, market_entropy_strategy, quantum_entanglement_hidden
]
ALL_INDICATORS = [
    ema_macd_indicator, supertrend_bollinger_indicator, volume_indicator, bollinger_indicator, rsi_indicator
]

class Engines:
    QUANTUM = {
        "name": "Quantum Engine V2.0",
        "strategies": [
            order_block_detection, break_of_structure, fair_value_gap,
            fibonacci_vortex_hidden, dark_pool_institutional_hidden,
            market_microstructure_imbalance, institutional_footprint_detector, gamma_flip_detector
        ],
        "indicators": [ema_macd_indicator, supertrend_bollinger_indicator, volume_indicator, bollinger_indicator],
        "filters": [smart_money_filter, vwap_deviation_filter, volume_profile_filter]
    }
    MOMENTUM = {
        "name": "Momentum Scalper V1.0",
        "strategies": [break_of_structure, volatility_regime_detector, fractal_dimension_strategy],
        "indicators": [volume_indicator, rsi_indicator, ema_macd_indicator],
        "filters": [vwap_deviation_filter]
    }
    BREAKOUT = {
        "name": "Breakout Hunter V1.0",
        "strategies": [break_of_structure, fair_value_gap, liquidity_void_hunter, auction_theory_analyzer],
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
    if not safe_df_check(df, asset_config.get("min_data_points", 50)):
        return {"volatility": 0.0, "trend_strength": 0.0, "momentum_spike": False, "breakout_detected": False, "range_bound": True}
    _atr_series = atr(df)
    _atr_val = _atr_series.iloc[-1] if not _atr_series.empty else 0.0
    price = df["close"].iloc[-1]
    volatility = float((_atr_val / (price + 1e-9)) * 100)
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
    vol_spike = df["volume"].iloc[-1] > (df["volume"].rolling(30).mean().iloc[-1] * vol_mult) if len(df) >= 31 else False
    price_change = abs(df["close"].iloc[-1] - df["close"].iloc[-5]) / (df["close"].iloc[-5] + 1e-9) if len(df) >= 6 else 0.0
    momentum_spike = bool(vol_spike and price_change > 0.01)
    bb = bollinger_bands(df["close"])
    breakout_detected = bool((not bb["upper"].isna().all()) and (df["close"].iloc[-1] > bb["upper"].iloc[-1] or df["close"].iloc[-1] < bb["lower"].iloc[-1]))
    range_bound = bool(volatility < asset_config.get("volatility_threshold", 0.5) and (not bb["ma"].isna().all()) and abs(df["close"].iloc[-1] - bb["ma"].iloc[-1]) / (bb["ma"].iloc[-1] + 1e-9) < 0.005)
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
# Data provider (yfinance)
# -------------------------------
class RealDataProvider:
    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except Exception:
            self.yf = None
            logger.warning("yfinance not available")

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
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
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
        if not safe_df_check(df, asset_config.get("min_data_points", 50)):
            continue

        for strategy_fn in engine["strategies"]:
            try:
                result = strategy_fn(df)
                if result and result.get("signal") != "NEUTRAL":
                    result["timeframe"] = tf
                    all_strategy_results.append(result)
            except Exception as e:
                logger.debug(f"Strategy {strategy_fn.__name__} error: {e}")

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

        for filt_fn in engine.get("filters", []):
            try:
                result = filt_fn(df, asset_config)
                if result and result.get("signal") != "NEUTRAL":
                    result["timeframe"] = tf
                    all_filter_results.append(result)
            except Exception as e:
                logger.debug(f"Filter {filt_fn.__name__} error: {e}")

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

    return f"""📮 <b>ULTIMATE LEGENDARY SIGNAL</b>  <i>[{engine_name}]</i>

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
# Cleanup old alerts
# -------------------------------
def cleanup_old_alerts():
    global LAST_LEGENDARY_ALERT
    now = datetime.utcnow()
    to_remove = [key for key, timestamp in LAST_LEGENDARY_ALERT.items() if (now - timestamp).total_seconds() > 86400]
    for key in to_remove:
        del LAST_LEGENDARY_ALERT[key]
    if to_remove:
        logger.info(f"Cleaned up {len(to_remove)} old alert cooldowns")

# -------------------------------
# API routes
# -------------------------------
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "legendary",
        "timestamp": datetime.utcnow().isoformat(),
        "mode": EngineState.current_mode,
        "signals_in_history": len(SIGNAL_HISTORY),
        "active_cooldowns": len(LAST_LEGENDARY_ALERT)
    }), 200

@app.route("/signals", methods=["GET"])
def get_signals():
    recent = list(SIGNAL_HISTORY)[-10:]
    return jsonify({"signals": recent, "count": len(SIGNAL_HISTORY)}), 200

# -------------------------------
# Main scanner loop
# -------------------------------
def main_scan_loop():
    logger.info("📮 Initializing ULTIMATE Legendary Scanner")
    logger.info(f"⚡ Engines: Quantum / Momentum / Breakout / MeanReversion")
    logger.info(f"📊 Pools – Strategies: {len(ALL_STRATEGIES)} | Indicators: {len(ALL_INDICATORS)}")

    provider = RealDataProvider()

    while True:
        try:
            now_utc = datetime.utcnow()
            logger.info(f"\n🔍 Scan: {now_utc.strftime('%H:%M:%S')}")
            cleanup_old_alerts()

            for asset in ASSETS:
                tf_data: Dict[str, pd.DataFrame] = {}
                for tf in ALL_TIMEFRAMES[:4]:
                    try:
                        df = provider.fetch_ohlcv(asset, tf, limit=300)
                        min_len = ASSET_CONFIG.get(asset, {}).get("min_data_points", 50)
                        if safe_df_check(df, min_len):
                            tf_data[tf] = df
                    except Exception as e:
                        logger.debug(f"Failed to fetch {asset} {tf}: {e}")

                if len(tf_data) < 3:
                    logger.debug(f"Insufficient data for {asset}")
                    continue

                market_conditions = compute_market_conditions(asset, tf_data)
                EngineState.current_mode = select_mode(market_conditions)
                engine = get_engine(EngineState.current_mode)

                for expiry in EXPIRIES:
                    cooldown_key = f"{asset}_{expiry}"
                    last_ts = LAST_LEGENDARY_ALERT.get(cooldown_key)
                    if last_ts:
                        elapsed = (now_utc - last_ts).total_seconds() / 60
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
                            'timestamp': now_utc.isoformat(),
                            'engine': engine["name"]
                        }

                        SIGNAL_HISTORY.append(alert_data)
                        LAST_LEGENDARY_ALERT[cooldown_key] = now_utc

                        alert_msg = format_ultimate_alert(alert_data, engine["name"])

                        # send via Telegram if configured
                        try:
                            if TELEGRAM_AVAILABLE and telegram_instance and telegram_instance.app:
                                asyncio.run(telegram_instance.app.bot.send_message(chat_id=CHAT_ID, text=alert_msg, parse_mode="HTML"))
                        except Exception as e:
                            logger.error(f"Telegram send error: {e}")

                        logger.info(f"  🔥 [{engine['name']}] {asset} {result['signal']} @ {result['confidence']:.1f}% (threshold {threshold:.1f}%)")

                    time.sleep(0.12)

            logger.info(f"✅ Scan complete. Next in {SCAN_INTERVAL_SEC}s...")
            if RUN_ONCE:
                break
            time.sleep(SCAN_INTERVAL_SEC)

        except KeyboardInterrupt:
            logger.info("\n🛑 Stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}", exc_info=True)
            time.sleep(8)

# -------------------------------
# Entrypoint
# -------------------------------
if __name__ == "__main__":
    logger.info(f"🚀 Starting on port {PORT}")

    # Start Telegram bot immediately
    telegram_instance = None
    try:
        telegram_instance = LegendaryTelegramBot(TELEGRAM_TOKEN, CHAT_ID)
        telegram_instance.start()
    except Exception as e:
        logger.error(f"Failed to init telegram bot: {e}")

    # Run scanner in background thread
    scanner_thread = threading.Thread(target=main_scan_loop, daemon=True)
    scanner_thread.start()

    # Run Flask API in main thread
    app.run(host="0.0.0.0", port=PORT, debug=False)

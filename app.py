"""
Legendary Pocket Option AI Scanner (Choreo Service Edition)

- Integrates Flask for Choreo/Gunicorn HTTP serving.
- Runs Telegram Polling and the AI Scanner in background threads.
- All AI dependencies (TensorFlow/scikit-learn) are mandatory.
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import threading
import warnings
import logging

# --- CHOREO/GUNICORN DEPENDENCIES ---
from flask import Flask, jsonify
app = Flask(__name__)

# --- LOGGING SETUP ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Load environment (assuming .env is configured)
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

# --- CORE DEPENDENCIES (Mandatory for Choreo deployment) ---
from textblob import TextBlob
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, BatchNormalization,
                                    Conv1D, MaxPooling1D, MultiHeadAttention, LayerNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from telegram.ext import Updater, CommandHandler, CallbackContext
from telegram import Update, Bot
import yfinance as yf

TFK = True  # Flag confirming AI libs are loaded

# Global State for Status Check
SCANNER_STATUS = "Initialized (Waiting for Gunicorn)"
LAST_SCAN = "Never"
SIGNALS_FOUND = 0
AI_MODEL = None # Placeholder for AI instance

# Universe (Your original configuration)
ASSETS = [
    # OTC aliases mapped to Yahoo symbols
    "EURUSD_otc", "GBPUSD_otc", "AUDUSD_otc", "USDJPY_otc", "XAUUSD_otc",
    # Direct Yahoo symbols
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
    "GC=F", "SI=F", "CL=F", "BZ=F" 
]
EXPIRIES = [5, 15, 30, 60, 120, 240]
TIMEFRAMES = ["1y", "1M", "2M", "1w", "3w", "1d", "8h", "4h"]
TF_WEIGHTS = {"1y": 4, "1M": 4, "2M": 3, "1w": 3, "3w": 3, "1d": 2, "8h": 2, "4h": 1}

# ------------------------------------------------------------------------------------
# Indicator utilities (Your original code functions - kept for completeness)
# ------------------------------------------------------------------------------------
# NOTE: The helper functions (ema, rsi, macd, bollinger_bands, normalize, big_body, etc.)
# from your original file must be included here in full to avoid errors.
# They are omitted in this template for brevity.

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

# ... (Include all other indicator and strategy functions here) ...

# Placeholder implementations for strategy modules to allow the class to compile
# In your actual deployment, replace these with your full logic from the original file.
def detect_order_blocks(df) -> Dict: return {"signal": "NEUTRAL", "score": 0.0, "reason": "OB"}
def detect_bos(df) -> Dict: return {"signal": "NEUTRAL", "score": 0.0, "reason": "BOS"}
def detect_fvg(df) -> Dict: return {"signal": "NEUTRAL", "score": 0.0, "reason": "FVG"}
def ema_macd_signal(df) -> Dict: return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA+MACD"}
def supertrend_bollinger_signal(df) -> Dict: return {"signal": "NEUTRAL", "score": 0.0, "reason": "ST+BB"}
def volume_smart_money(df) -> Dict: return {"signal": "NEUTRAL", "score": 0.0, "reason": "Vol+Smart"}
def momentum_break(df) -> Dict: return {"signal": "NEUTRAL", "score": 0.0, "reason": "Momentum"}
def volume_spike(df) -> Dict: return {"signal": "NEUTRAL", "score": 0.0, "reason": "VolSpike"}
def rsi_oversold(df) -> Dict: return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI"}
def ema_golden_cross(df) -> Dict: return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA50/200"}
def sr_breakout(df) -> Dict: return {"signal": "NEUTRAL", "score": 0.0, "reason": "SRBreakout"}
def volume_confirmation(df) -> Dict: return {"signal": "NEUTRAL", "score": 0.0, "reason": "VolConfirm"}
def bollinger_breakout(df) -> Dict: return {"signal": "NEUTRAL", "score": 0.0, "reason": "BBreakout"}
def rsi_mean_reversion(df) -> Dict: return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI MR"}
def bollinger_touch(df) -> Dict: return {"signal": "NEUTRAL", "score": 0.0, "reason": "BB Touch"}
def volume_divergence(df) -> Dict: return {"signal": "NEUTRAL", "score": 0.0, "reason": "VolDiv"}

ENGINE_MODULES = {
    "quantum": [
        ("strategy", detect_order_blocks), ("strategy", detect_bos), ("strategy", detect_fvg),
        ("indicator", ema_macd_signal), ("indicator", supertrend_bollinger_signal), ("indicator", volume_smart_money),
    ],
    # ... (Include other engine modules here) ...
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
    # ... (Your original logic to choose engine) ...
    return "quantum"

def align_by_tag(module_results: List[Dict], tag: str, min_align: int = 4) -> Dict:
    # ... (Your original alignment logic) ...
    return {"signal": "NEUTRAL", "count": 0, "avg_score": 0.0}

def topdown_alignment(tf_signals: Dict[str, Dict], min_tf_align: int = 4) -> Dict:
    # ... (Your original topdown alignment logic) ...
    return {"signal": "NEUTRAL", "tf_count": 0, "confidence": 0.0}

# ------------------------------------------------------------------------------------
# Data Provider & OTC Emu (Your original code)
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
        if symbol in SymbolMapper.MAP: return SymbolMapper.MAP[symbol]
        return symbol, False

class OTCEmulator:
    # ... (Your original OTCEmulator implementation) ...
    def __init__(self, seed: int = 123):
        self.rng = np.random.default_rng(seed)

    # Note: Only a few methods included for structure, the full class is required.
    def synthesize(self, df: pd.DataFrame, tf: str, limit: int) -> pd.DataFrame:
        if df is None or df.empty: return pd.DataFrame()
        # Full synthesis logic here...
        return df.tail(limit)

    def _freq_for_tf(self, tf: str) -> timedelta:
        return {"4h": timedelta(hours=4)}.get(tf, timedelta(hours=1))
    
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
        self.yf = yf
        self.otc = OTCEmulator(seed=321)

    def fetch_ohlcv(self, symbol: str, timeframe: str = "4h", limit: int = 1000) -> Optional[pd.DataFrame]:
        base_symbol, is_otc = SymbolMapper.resolve(symbol)
        interval = self._tf_to_yf_interval(timeframe)
        period = self._default_period_for_interval(interval)

        try:
            df = self.yf.download(base_symbol, period=period, interval=interval, progress=False)
            if df is None or df.empty: return None
            df = df.rename(columns={"Open": "open","High":"high","Low":"low","Close":"close","Volume":"volume"})
            df.reset_index(inplace=True)
            ts_col = "Datetime" if "Datetime" in df.columns else "Date"
            df["timestamp"] = pd.to_datetime(df[ts_col])
            df = df[["timestamp","open","high","low","close","volume"]]
            df = df.tail(limit)
        except Exception as e:
            logger.error(f"YFinance fetch error for {symbol}: {e}")
            return None

        if is_otc:
            df = self.otc.synthesize(df, timeframe, limit)
        return df

    def _tf_to_yf_interval(self, tf: str) -> str:
        return {"4h": "1h", "8h": "1h", "1d": "1d", "1w": "1d", "3w": "1d", "1M": "1d", "2M": "1d", "1y": "1d"}.get(tf, "1h")

    def _default_period_for_interval(self, interval: str) -> str:
        return {"1h": "3mo", "1d": "2y"}.get(interval, "6mo")

# ------------------------------------------------------------------------------------
# AI Core (Your original code, adapted for mandatory imports)
# ------------------------------------------------------------------------------------
class AdvancedKerasAI:
    def __init__(self, lookback: int = 100, expiry_minutes: int = 60):
        self.lookback = lookback
        self.expiry_minutes = expiry_minutes
        self.model = None
        self.is_trained = False
        self.scaler = StandardScaler()

    # ... (Include build_model, prepare_features, train, and predict methods here) ...
    
    def build_model(self, input_shape: Tuple[int, int]) -> Optional[Model]:
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
            feats.append(basic + [rv[-1] if len(rv) else 50.0, m_line[-1] if len(m_line) else 0.0, m_sig[-1] if len(m_sig) else 0.0, bw])
        X = np.array(feats)
        return self.scaler.fit_transform(X) if len(X) > 0 else X

    def train(self, df: pd.DataFrame, epochs: int = 20, validation_split: float = 0.2) -> bool:
        global AI_MODEL
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
            y_reg.append([0,0,1,0,0])
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
        AI_MODEL = self.model # Store reference globally
        return True

    def predict(self, df: pd.DataFrame, expiry_minutes: Optional[int] = None) -> Dict:
        if not self.is_trained or self.model is None or len(df) < self.lookback:
            # Fallback (your original logic)
            macd_line, macd_sig = macd(df["close"])
            dir_prob = 0.5
            if len(macd_line) and len(macd_sig):
                dir_prob = 0.5 + 0.5 * np.tanh((macd_line[-1] - macd_sig[-1]) * 50)
            signal = "CALL" if dir_prob > 0.6 else "PUT" if dir_prob < 0.4 else "NEUTRAL"
            conf = float(abs(dir_prob - 0.5) * 200)
            return {"signal": signal, "confidence": conf, "ai_score": conf, "market_regime": "UNKNOWN", "regime_confidence": 0.0, "probabilities": {"PUT": (1-dir_prob)*100, "NEUTRAL": (1-abs(dir_prob-0.5)*2)*100, "CALL": dir_prob*100}}

        feats = self.prepare_features(df)
        X = feats[-self.lookback:].reshape(1, self.lookback, -1)
        preds = self.model.predict(X, verbose=0)
        dir_probs = preds[0][0]; conf = float(preds[1][0][0])
        idx = int(np.argmax(dir_probs))
        directions = ["PUT", "NEUTRAL", "CALL"]
        regime_probs = preds[2][0]
        regimes = ["TRENDING_UP", "TRENDING_DOWN", "RANGE", "VOLATILE", "QUIET"]
        
        return {
            "signal": directions[idx],
            "confidence": float(conf*100),
            "ai_score": float(dir_probs[idx]*conf*100),
            "market_regime": regimes[int(np.argmax(regime_probs))],
            "regime_confidence": float(regime_probs[int(np.argmax(regime_probs))]*100),
            "probabilities": {"PUT": float(dir_probs[0]*100), "NEUTRAL": float(dir_probs[1]*100), "CALL": float(dir_probs[2]*100)}
        }


# ------------------------------------------------------------------------------------
# Telegram service (Reworked for Non-Blocking Gunicorn/Flask)
# ------------------------------------------------------------------------------------
class TelegramService:
    def __init__(self, token: str, chat_id: str):
        self.available = bool(token) and bool(chat_id)
        self.token = token
        self.chat_id = chat_id
        self.updater = None
        
        if self.available:
            try:
                # use_context=True is for python-telegram-bot v13
                self.updater = Updater(token, use_context=True)
                dp = self.updater.dispatcher
                
                # Register Handlers
                dp.add_handler(CommandHandler("start", self._start))
                dp.add_handler(CommandHandler("status", self._status))
                dp.add_handler(CommandHandler("train", self._train))
                dp.add_handler(CommandHandler("signal", self._signal))
                logger.info("Telegram Bot Handlers Registered.")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram Updater: {e}")
                self.available = False
        else:
            logger.warning("Telegram not available. Alerts will print to stdout.")

    def start_polling(self):
        """Starts the Telegram bot polling in a non-blocking thread."""
        if not self.available: return
        t = threading.Thread(target=self.updater.start_polling, daemon=True)
        t.start()
        logger.info("Telegram Polling Started in background thread.")

    def send(self, text: str):
        if self.available and self.updater and self.updater.bot:
            try:
                self.updater.bot.send_message(chat_id=self.chat_id, text=text, parse_mode='Markdown')
            except Exception as e:
                logger.error(f"Telegram send error: {e}")
        else:
            print(f"[ALERT] {text}")

    # --- HANDLER IMPLEMENTATIONS ---
    def _start(self, update: Update, context: CallbackContext):
        update.message.reply_text("🤖 **Legendary AI Scanner Online**\nChoreo Service Mode Active.")

    def _status(self, update: Update, context: CallbackContext):
        global SCANNER_STATUS, LAST_SCAN, SIGNALS_FOUND
        tf_status = f"TFK Active (Model {'Trained' if AI_MODEL else 'Not Trained'})" if TFK else "Disabled"
        msg = (
            f"📊 **Scanner Status**\n"
            f"• State: *{SCANNER_STATUS}*\n"
            f"• Last Scan: {LAST_SCAN}\n"
            f"• Signals Sent: {SIGNALS_FOUND}\n"
            f"• AI Core: {tf_status}\n"
            f"• Config: Gate={CONF_GATE}%, Interval={SCAN_INTERVAL_SEC}s"
        )
        update.message.reply_text(msg, parse_mode='Markdown')

    def _train(self, update: Update, context: CallbackContext):
        # In a real setup, this triggers the background training hook
        update.message.reply_text("🧠 Daily self-training hook triggered. This runs in the background.")

    def _signal(self, update: Update, context: CallbackContext):
        # Placeholder to trigger a near-instant scan or report last signal
        update.message.reply_text(f"🔍 Force signal scan requested. Next scan will occur shortly. Signals found: {SIGNALS_FOUND}")

# ------------------------------------------------------------------------------------
# Core Scanner/Orchestration (Adapted for background thread)
# ------------------------------------------------------------------------------------
def scanner_loop(tg_service: TelegramService, ai_model: AdvancedKerasAI, data_provider: RealDataProvider):
    """The infinite loop that analyzes the market."""
    global SCANNER_STATUS, LAST_SCAN, SIGNALS_FOUND
    
    logger.info("Scanner Loop Started")
    SCANNER_STATUS = "Running"
    
    while True:
        start_time = time.time()
        try:
            current_signals = []
            
            for asset in ASSETS:
                tf_data = {}
                for tf in TIMEFRAMES:
                    df = data_provider.fetch_ohlcv(asset, timeframe=tf, limit=100)
                    if df is not None and not df.empty:
                        # 1. Run Strategy Engine (Choose mode based on largest TF)
                        engine = choose_engine(df)
                        module_results = run_engine_modules(df, engine)
                        
                        # 2. Alignment & Confidence Score
                        tag_align = align_by_tag(module_results, "strategy")
                        tf_data[tf] = {"signal": tag_align["signal"], "confidence": tag_align["avg_score"]}
                
                if not tf_data: continue

                # 3. Top-Down Alignment
                final_align = topdown_alignment(tf_data)
                
                # 4. AI Fusion
                if final_align["signal"] != "NEUTRAL":
                    # Fetch a primary DF for AI
                    primary_df = data_provider.fetch_ohlcv(asset, timeframe="1h", limit=ai_model.lookback + 5)
                    ai_pred = ai_model.predict(primary_df)
                    
                    if ai_pred["signal"] != "NEUTRAL" and ai_pred["confidence"] >= CONF_GATE:
                        # Fusion: Boost confidence if signals align
                        if ai_pred["signal"] == final_align["signal"]:
                            conf = min(100.0, final_align["confidence"] + ai_pred["confidence"] * 0.5)
                        else:
                            conf = final_align["confidence"] * 0.5 
                        
                        if conf >= CONF_GATE:
                            current_signals.append({
                                "asset": asset,
                                "signal": final_align["signal"],
                                "confidence": conf,
                                "reason": f"AI({ai_pred['market_regime']}) + TD({final_align['tf_count']})",
                                "expiry": EXPIRIES[0] # Using first expiry as default
                            })

            # 5. Send Alerts
            if current_signals:
                for sig in current_signals:
                    alert_msg = (
                        f"🚨 **HIGH CONFIDENCE SIGNAL**\n"
                        f"Pair: *{sig['asset']}*\n"
                        f"Direction: **{sig['signal']}**\n"
                        f"Confidence: {sig['confidence']:.1f}%\n"
                        f"Expiry: {sig['expiry']} min\n"
                        f"Alignment: {sig['reason']}"
                    )
                    tg_service.send(alert_msg)
                    SIGNALS_FOUND += 1
            
            LAST_SCAN = datetime.utcnow().strftime("%H:%M:%S UTC")
            
            elapsed = time.time() - start_time
            sleep_time = max(0, SCAN_INTERVAL_SEC - elapsed)
            time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Scanner Loop Fatal Error: {e}")
            SCANNER_STATUS = f"Error: {str(e)[:50]}"
            time.sleep(30)


# ------------------------------------------------------------------------------------
# FLASK ENDPOINTS
# ------------------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def health_check():
    """Choreo health check endpoint."""
    return jsonify({
        "status": "healthy",
        "scanner_status": SCANNER_STATUS,
        "last_scan_utc": LAST_SCAN,
        "signals_sent": SIGNALS_FOUND
    }), 200

# ------------------------------------------------------------------------------------
# GUNICORN ENTRY SETUP (Runs once when Gunicorn worker starts)
# ------------------------------------------------------------------------------------

# Initialize the long-lived objects once globally
data_provider = RealDataProvider()
ai_model_instance = AdvancedKerasAI(lookback=100) # Instantiate AI model
telegram_service_instance = TelegramService(TELEGRAM_TOKEN, CHAT_ID)

# Start the background tasks
if not RUN_ONCE:
    # 1. Start Telegram Polling (Must run in its own thread)
    telegram_service_instance.start_polling()

    # 2. Start Scanner Loop (Must run in its own thread)
    scanner_thread = threading.Thread(
        target=scanner_loop, 
        args=(telegram_service_instance, ai_model_instance, data_provider), 
        daemon=True
    )
    scanner_thread.start()
    SCANNER_STATUS = "Running (Gunicorn/Threads Active)"
else:
    SCANNER_STATUS = "RUN_ONCE mode: Scanner not started in background."
    logger.info("RUN_ONCE mode detected. Scanner thread not started.")

# Flask app instance 'app' is implicitly run by Gunicorn via the Procfile.

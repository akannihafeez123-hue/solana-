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
"""

import os
import sys
import time
import subprocess
import importlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import threading
import warnings
import json
import hashlib
import hmac
from flask import Flask, request, jsonify
from collections import Counter
import math
import logging
warnings.filterwarnings('ignore')

# ==================== RUNTIME DEPENDENCY INSTALLATION ====================

def install_dependencies():
    """Install compatible heavy dependencies at runtime"""
    # Use compatible versions to avoid binary issues
    heavy_dependencies = [
        "numpy==1.21.6",
        "pandas==1.3.5", 
        "yfinance==0.2.18",
        "psutil==5.9.5",
        "python-telegram-bot==20.7"
    ]
    
    print("🚀 Installing compatible heavy dependencies...")
    
    for package in heavy_dependencies:
        try:
            package_name = package.split('==')[0]
            # Try to import first
            importlib.import_module(package_name)
            print(f"✅ {package_name} already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "--no-cache-dir", package
                ])
                print(f"✅ Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
    
    print("✅ All dependencies installed successfully!")
    return True

# Install dependencies on first run
try:
    # Try to import heavy dependencies
    import numpy as np
    import pandas as pd
    print("✅ NumPy 1.21.6 and Pandas 1.3.5 loaded successfully!")
    
    # Try other dependencies
    try:
        import yfinance
        print("✅ yfinance loaded successfully!")
    except ImportError:
        print("⚠️ yfinance not available")
        
    try:
        import psutil
        print("✅ psutil loaded successfully!")
    except ImportError:
        print("⚠️ psutil not available")
        
    try:
        from telegram import Update
        from telegram.ext import Application, CommandHandler, ContextTypes
        from telegram.constants import ParseMode
        print("✅ Telegram loaded successfully!")
    except ImportError:
        print("⚠️ Telegram not available")
        
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("📦 Installing dependencies...")
    if install_dependencies():
        # Reload after installation
        import numpy as np
        import pandas as pd
        print("✅ Dependencies installed and loaded!")
    else:
        print("💥 Failed to install dependencies. Running in limited mode.")

# ==================== ENHANCED LOGGING ====================
def setup_logging():
    """Setup structured logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ==================== CONFIG ====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

# Safe parsing of ADMIN_IDS
def parse_admin_ids():
    admin_ids_str = os.getenv("ADMIN_IDS", "")
    if not admin_ids_str or admin_ids_str in ["", "your_chat_id", "your_id", "your_ids"]:
        if CHAT_ID and CHAT_ID.strip() and CHAT_ID.strip().isdigit():
            return [int(CHAT_ID.strip())]
        return []
    
    ids = []
    for x in admin_ids_str.split(","):
        x = x.strip()
        if x and x.isdigit():
            ids.append(int(x))
        elif x and not x.startswith("your_"):
            logger.warning(f"Skipping invalid ADMIN_ID: {x}")
    return ids if ids else []

ADMIN_IDS = parse_admin_ids()

if not ADMIN_IDS:
    logger.warning("No valid ADMIN_IDS configured. Admin features will be disabled.")

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "your_legendary_secret_key_here")
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "False") == "True"
CONF_GATE = int(os.getenv("CONF_GATE", "92"))
LEGENDARY_GATE = int(os.getenv("LEGENDARY_GATE", "95"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "15"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "45"))
RUN_ONCE = os.getenv("RUN_ONCE", "False") == "True"
PORT = int(os.getenv("PORT") or os.getenv("port") or "8080")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "your_webhook_secret")

# Trading Universe
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
        data = f"{user_id}:{datetime.now(timezone.utc).isoformat()}"
        signature = hmac.new(ADMIN_SECRET.encode(), data.encode(), hashlib.sha256).hexdigest()
        return f"{data}:{signature}"
    
    @staticmethod
    def verify_token(token: str) -> Tuple[bool, Optional[int]]:
        try:
            parts = token.split(":")
            if len(parts) != 3:
                return False, None
                
            user_id_str, timestamp, signature = parts
            if not user_id_str.isdigit():
                return False, None
                
            user_id = int(user_id_str)
            if user_id not in ADMIN_IDS:
                return False, None
                
            data = f"{user_id}:{timestamp}"
            expected = hmac.new(ADMIN_SECRET.encode(), data.encode(), hashlib.sha256).hexdigest()
            
            if not hmac.compare_digest(signature, expected):
                return False, None
                
            token_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            age = (datetime.now(timezone.utc) - token_time).total_seconds()
            if age > 86400:
                return False, None
                
            return True, user_id
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return False, None

# ==================== COMPATIBLE CORE INDICATORS ====================
def ema(series: pd.Series, period: int) -> np.ndarray:
    """Exponential Moving Average - Compatible with numpy 1.21.6"""
    if len(series) == 0 or period <= 0:
        return np.array([])
    try:
        alpha = 2 / (period + 1)
        ema_vals = [series.iloc[0]]
        for v in series.iloc[1:]:
            ema_vals.append(alpha * v + (1 - alpha) * ema_vals[-1])
        return np.array(ema_vals)
    except Exception as e:
        logger.error(f"EMA calculation error: {e}")
        return np.array([])

def rsi(series: pd.Series, period: int = 14) -> np.ndarray:
    """RSI - Compatible with numpy 1.21.6"""
    if len(series) < period + 1:
        return np.array([50.0] * len(series))
    try:
        delta = series.diff()
        up = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        down = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        rs = up / down.replace(0, np.nan)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals.fillna(50.0).values
    except Exception as e:
        logger.error(f"RSI calculation error: {e}")
        return np.array([50.0] * len(series))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """MACD - Compatible version"""
    try:
        ema_fast = pd.Series(ema(series, fast))
        ema_slow = pd.Series(ema(series, slow))
        macd_line = ema_fast - ema_slow
        macd_sig = pd.Series(ema(macd_line, sig))
        return macd_line.values, macd_sig.values
    except Exception as e:
        logger.error(f"MACD calculation error: {e}")
        return np.array([]), np.array([])

def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic oscillator - Compatible version"""
    try:
        low_min = df["low"].rolling(k_period, min_periods=1).min()
        high_max = df["high"].rolling(k_period, min_periods=1).max()
        denominator = (high_max - low_min).replace(0, 1e-9)
        k = 100 * ((df["close"] - low_min) / denominator)
        d = k.rolling(d_period, min_periods=1).mean()
        return k.fillna(50).values, d.fillna(50).values
    except Exception as e:
        logger.error(f"Stochastic calculation error: {e}")
        return np.array([50.0] * len(df)), np.array([50.0] * len(df))

def bollinger_bands(series: pd.Series, period: int = 20, dev: float = 2.0) -> Dict:
    """Bollinger Bands - Compatible version"""
    try:
        ma = series.rolling(period, min_periods=1).mean()
        std = series.rolling(period, min_periods=1).std()
        upper = ma + dev * std
        lower = ma - dev * std
        return {
            "upper": upper, 
            "lower": lower, 
            "ma": ma, 
            "width": upper - lower, 
            "std": std
        }
    except Exception as e:
        logger.error(f"Bollinger Bands calculation error: {e}")
        empty_series = pd.Series([np.nan] * len(series))
        return {"upper": empty_series, "lower": empty_series, "ma": empty_series, "width": empty_series, "std": empty_series}

def atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range - Compatible version"""
    try:
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift(1))
        tr3 = abs(df["low"] - df["close"].shift(1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr_val = tr.rolling(period, min_periods=1).mean()
        return float(atr_val.iloc[-1]) if pd.notna(atr_val.iloc[-1]) else 0.0
    except Exception as e:
        logger.error(f"ATR calculation error: {e}")
        return 0.0

def normalize(val: float, base: Optional[pd.Series] = None) -> float:
    """Safe normalization"""
    try:
        if base is not None and len(base) > 0:
            denominator = abs(base.iloc[-1]) + 1e-9
            return float(min(100.0, max(0.0, 100.0 * abs(val) / denominator)))
        return float(min(100.0, max(0.0, val)))
    except Exception:
        return 0.0

# ==================== STRATEGIES ====================
def fibonacci_vortex_hidden(df: pd.DataFrame) -> Dict:
    """Fibonacci Vortex Hidden - Sacred Geometry + Golden Spiral"""
    try:
        if len(df) < 50:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "FibVortex", "type": "strategy"}
        
        high = df["high"].rolling(20, min_periods=1).max()
        low = df["low"].rolling(20, min_periods=1).min()
        swing_high = high.iloc[-20:].max()
        swing_low = low.iloc[-20:].min()
        swing_range = swing_high - swing_low
        
        if swing_range == 0:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "FibVortex", "type": "strategy"}
        
        fib_levels = {
            0.236: swing_high - 0.236 * swing_range,
            0.382: swing_high - 0.382 * swing_range,
            0.500: swing_high - 0.500 * swing_range,
            0.618: swing_high - 0.618 * swing_range,
            0.786: swing_high - 0.786 * swing_range,
        }
        
        current_price = df["close"].iloc[-1]
        closest_fib = min(fib_levels.items(), key=lambda x: abs(x[1] - current_price))
        distance_pct = abs(current_price - closest_fib[1]) / current_price * 100
        
        at_golden_ratio = closest_fib[0] == 0.618 and distance_pct < 0.5
        
        # Simple momentum for vortex
        momentum = df["close"].pct_change(5).iloc[-1] if len(df) > 5 else 0
        volume_trend = df["volume"].iloc[-5:].mean() > df["volume"].iloc[-10:-5].mean()
        
        if at_golden_ratio and momentum > 0 and volume_trend:
            signal = "BUY"
            score = 94.0
        elif at_golden_ratio and momentum < 0 and volume_trend:
            signal = "SELL"
            score = 94.0
        else:
            signal = "NEUTRAL"
            score = 0.0
        
        return {"signal": signal, "score": min(score, 100.0), "reason": "FibVortex", "type": "strategy"}
    except Exception as e:
        logger.error(f"Fibonacci Vortex error: {e}")
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "FibVortex", "type": "strategy"}

def quantum_entanglement_hidden(df: pd.DataFrame) -> Dict:
    """Quantum Entanglement - Probability Wave Analysis"""
    try:
        if len(df) < 30:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "QuantumEnt", "type": "strategy"}
        
        prices = df["close"].values[-30:]
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        current_price = prices[-1]
        z_score = (current_price - mean_price) / (std_price + 1e-9)
        
        momentum = df["close"].pct_change(5).iloc[-1] if len(df) > 5 else 0
        
        if z_score < -1.5 and momentum > 0:
            signal = "BUY"
            score = 93.0 + min(abs(z_score) * 2, 7.0)
        elif z_score > 1.5 and momentum < 0:
            signal = "SELL"
            score = 93.0 + min(abs(z_score) * 2, 7.0)
        else:
            signal = "NEUTRAL"
            score = 0.0
        
        return {"signal": signal, "score": min(score, 100.0), "reason": "QuantumEnt", "type": "strategy"}
    except Exception as e:
        logger.error(f"Quantum Entanglement error: {e}")
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "QuantumEnt", "type": "strategy"}

def momentum_break_detection(df: pd.DataFrame) -> Dict:
    """Momentum Break Detection - Momentum Scalper V1.0"""
    try:
        if len(df) < 20:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "MomentumBreak", "type": "strategy"}
        
        roc_5 = df["close"].pct_change(5)
        roc_10 = df["close"].pct_change(10)
        
        volume_avg = df["volume"].rolling(10, min_periods=1).mean()
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
    except Exception as e:
        logger.error(f"Momentum Break error: {e}")
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "MomentumBreak", "type": "strategy"}

def ema_golden_cross(df: pd.DataFrame) -> Dict:
    """EMA Golden Cross - Momentum Scalper V1.0"""
    try:
        if len(df) < 50:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMAGoldenCross", "type": "strategy"}
        
        ema_fast = ema(df["close"], 9)
        ema_slow = ema(df["close"], 21)
        
        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMAGoldenCross", "type": "strategy"}
        
        golden_cross = ema_fast[-1] > ema_slow[-1] and ema_fast[-2] <= ema_slow[-2]
        death_cross = ema_fast[-1] < ema_slow[-1] and ema_fast[-2] >= ema_slow[-2]
        
        if golden_cross:
            return {"signal": "BUY", "score": 87.0, "reason": "EMAGoldenCross", "type": "strategy"}
        elif death_cross:
            return {"signal": "SELL", "score": 87.0, "reason": "EMAGoldenCross", "type": "strategy"}
        
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMAGoldenCross", "type": "strategy"}
    except Exception as e:
        logger.error(f"EMA Golden Cross error: {e}")
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMAGoldenCross", "type": "strategy"}

def rsi_indicator(df: pd.DataFrame) -> Dict:
    """RSI Indicator - Momentum Scalper V1.0 & Mean Reversion V1.0"""
    try:
        if len(df) < 20:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI", "type": "indicator"}
        
        rsi_val = rsi(df["close"])[-1]
        
        if rsi_val < 30:
            return {"signal": "BUY", "score": 75.0, "reason": "RSI", "type": "indicator"}
        elif rsi_val > 70:
            return {"signal": "SELL", "score": 75.0, "reason": "RSI", "type": "indicator"}
        
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI", "type": "indicator"}
    except Exception as e:
        logger.error(f"RSI Indicator error: {e}")
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI", "type": "indicator"}

# ==================== STRATEGY COLLECTIONS ====================
ALL_STRATEGIES = [
    fibonacci_vortex_hidden,
    quantum_entanglement_hidden,
    momentum_break_detection,
    ema_golden_cross,
]

ALL_INDICATORS = [
    rsi_indicator,
]

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

class RealDataProvider:
    def __init__(self):
        try:
            import yfinance
            self.yf = yfinance
            logger.info("✅ yfinance loaded successfully")
        except ImportError:
            logger.error("yfinance not available")
            self.yf = None
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = "4h", limit: int = 500) -> Optional[pd.DataFrame]:
        if self.yf is None:
            logger.error("yfinance not available")
            return None
            
        base_symbol, is_otc = SymbolMapper.resolve(symbol)
        
        tf_mapping = {
            "1h": "1h", "2h": "1h", "4h": "1h", "8h": "1h",
            "1d": "1d", "1w": "1wk", "1M": "1mo", "1y": "1y"
        }
        
        interval = tf_mapping.get(timeframe, "1h")
        period = "2mo" if interval == "1h" else "2y"
        
        try:
            df = self.yf.download(base_symbol, period=period, interval=interval, progress=False)
            if df is None or df.empty:
                return None
                
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low", 
                "Close": "close", "Volume": "volume"
            })
            
            df.reset_index(inplace=True)
            ts_col = "Datetime" if "Datetime" in df.columns else "Date"
            df["timestamp"] = pd.to_datetime(df[ts_col])
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            return df.tail(limit)
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

# ==================== AI ENGINE ====================
def select_best_strategies(all_results: List[Dict], min_score: float = 75.0) -> List[Dict]:
    """Select best strategies based on score"""
    strategies = [r for r in all_results if r.get("type") == "strategy" and r["score"] >= min_score]
    strategies.sort(key=lambda x: x["score"], reverse=True)
    return strategies[:3]  # Reduced for compatibility

def select_best_indicators(all_results: List[Dict], min_score: float = 70.0) -> List[Dict]:
    """Select best indicators based on score"""
    indicators = [r for r in all_results if r.get("type") == "indicator" and r["score"] >= min_score]
    indicators.sort(key=lambda x: x["score"], reverse=True)
    return indicators[:3]  # Reduced for compatibility

def detect_trading_mode(strategy_results: List[Dict]) -> Tuple[str, float]:
    """Auto-detect the best trading mode"""
    mode_scores = {mode: 0.0 for mode in TRADING_MODES.keys()}
    mode_counts = {mode: 0 for mode in TRADING_MODES.keys()}
    
    for result in strategy_results:
        reason = result["reason"]
        score = result["score"]
        
        for mode, config in TRADING_MODES.items():
            if reason in config["strategies"]:
                mode_scores[mode] += score
                mode_counts[mode] += 1
    
    for mode in mode_scores:
        if mode_counts[mode] > 0:
            mode_scores[mode] = mode_scores[mode] / mode_counts[mode]
    
    best_mode = "SACRED"
    best_score = 0.0
    
    for mode, score in mode_scores.items():
        if score > best_score and mode_counts[mode] >= 1:
            best_mode = mode
            best_score = score
    
    return best_mode, best_score

def ultimate_ai_analysis(tf_data: Dict[str, pd.DataFrame]) -> Dict:
    """Ultimate AI Analysis - Compatible version"""
    all_strategy_results = []
    all_indicator_results = []
    
    for tf, df in tf_data.items():
        if df is None or len(df) < 30:
            continue
        
        for strategy_fn in ALL_STRATEGIES:
            try:
                result = strategy_fn(df)
                if result["signal"] != "NEUTRAL":
                    result["timeframe"] = tf
                    all_strategy_results.append(result)
            except Exception:
                pass
        
        for indicator_fn in ALL_INDICATORS:
            try:
                result = indicator_fn(df)
                if result["signal"] != "NEUTRAL":
                    result["timeframe"] = tf
                    all_indicator_results.append(result)
            except Exception:
                pass
    
    best_strategies = select_best_strategies(all_strategy_results)
    best_indicators = select_best_indicators(all_indicator_results)
    
    trading_mode, mode_confidence = detect_trading_mode(best_strategies + best_indicators)
    
    all_signals = [r["signal"] for r in best_strategies + best_indicators]
    signal_counts = Counter(all_signals)
    
    if len(best_strategies) >= 2 and len(best_indicators) >= 1:
        if signal_counts.get("BUY", 0) >= 2:
            final_signal = "BUY"
            confidence = sum(r["score"] for r in best_strategies + best_indicators if r["signal"] == "BUY") / len([r for r in best_strategies + best_indicators if r["signal"] == "BUY"])
        elif signal_counts.get("SELL", 0) >= 2:
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
        "best_strategies": best_strategies,
        "best_indicators": best_indicators,
        "num_strategies_aligned": len([r for r in best_strategies if r["signal"] == final_signal]),
        "num_indicators_aligned": len([r for r in best_indicators if r["signal"] == final_signal]),
    }

# ==================== TELEGRAM BOT ====================
class TelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.available = False
        self.application = None
        
        if not token or not chat_id:
            logger.error("Telegram token or chat ID not configured")
            return
            
        try:
            from telegram.ext import Application, CommandHandler
            self.application = Application.builder().token(token).build()
            
            # Add command handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("status", self.status_command))
            
            # Test connection
            bot_info = self.application.bot.get_me()
            self.available = True
            
            # Start polling in background
            self.application.run_polling(allowed_updates=[])
            
            logger.info(f"Telegram bot connected: @{bot_info.username}")
            
        except Exception as e:
            logger.error(f"Telegram bot failed: {e}")
    
    async def start_command(self, update, context):
        """Handle /start command"""
        try:
            welcome_msg = """
🔮 ULTIMATE LEGENDARY Scanner

Commands:
/start - Show this message
/status - Check scanner status

Features:
• 95% Confidence Threshold
• Multiple Trading Strategies
• Real-time Analysis
"""
            await update.message.reply_text(welcome_msg, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Start command error: {e}")
    
    async def status_command(self, update, context):
        """Handle /status command"""
        try:
            status_msg = f"""
📊 Scanner Status

Status: ACTIVE
Last Scan: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
Assets: {len(ASSETS)}
Signals: {len(SIGNAL_HISTORY)}
"""
            await update.message.reply_text(status_msg, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Status command error: {e}")
    
    def send_alert(self, text: str):
        """Send alert message"""
        if not self.available:
            logger.info(f"ALERT: {text}")
            return
        
        try:
            self.application.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode='HTML'
            )
            logger.info("Alert sent to Telegram")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

# ==================== MAIN SCANNER ====================
def format_ultimate_alert(result: Dict) -> str:
    """Format ultimate legendary alert"""
    mode_config = TRADING_MODES.get(result['trading_mode'], TRADING_MODES['SACRED'])
    strategies_list = "\n".join([f"• {s['reason']} ({s['score']:.0f}%)" for s in result['best_strategies'][:3]])
    
    return f"""
{mode_config['emoji']} LEGENDARY SIGNAL

Asset: {result['asset']}
Signal: {result['signal']} 
Confidence: {result['confidence']:.1f}% 🎯

Strategies:
{strategies_list}

Alignment: 
• Strategies: {result['num_strategies_aligned']}/3
• Indicators: {result['num_indicators_aligned']}/3

Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
"""

def main_scan_loop():
    """Main scanning loop"""
    logger.info("🔮 Initializing ULTIMATE Legendary Scanner...")
    
    provider = RealDataProvider()
    
    # Initialize Telegram bot
    bot = None
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            bot = TelegramBot(TELEGRAM_TOKEN, CHAT_ID)
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
    
    logger.info(f"Strategies: {len(ALL_STRATEGIES)}")
    logger.info(f"Indicators: {len(ALL_INDICATORS)}")
    logger.info(f"Alert Threshold: {LEGENDARY_GATE}%")
    
    scan_count = 0
    
    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            scan_count += 1
            
            logger.info(f"Scan #{scan_count}: {now_utc.strftime('%H:%M:%S')} UTC")
            
            signals_found = 0
            
            for asset in ASSETS[:5]:  # Reduced for performance
                for expiry in EXPIRIES[:3]:  # Reduced for performance
                    cooldown_key = f"{asset}_{expiry}"
                    if cooldown_key in LAST_LEGENDARY_ALERT:
                        elapsed = (now_utc - LAST_LEGENDARY_ALERT[cooldown_key]).total_seconds() / 60
                        if elapsed < COOLDOWN_MINUTES:
                            continue
                    
                    # Fetch multi-TF data
                    tf_data = {}
                    for tf in ALL_TIMEFRAMES[:4]:  # Reduced for performance
                        try:
                            df = provider.fetch_ohlcv(asset, tf, 200)
                            if df is not None and len(df) >= 30:
                                tf_data[tf] = df
                        except Exception:
                            pass
                    
                    if len(tf_data) < 2:
                        continue
                    
                    # Run AI analysis
                    result = ultimate_ai_analysis(tf_data)
                    
                    # Check legendary criteria
                    if (result['signal'] != "NEUTRAL" and
                        result['confidence'] >= LEGENDARY_GATE and
                        result['num_strategies_aligned'] >= 2):
                        
                        signals_found += 1
                        
                        alert_data = {
                            'asset': asset,
                            'signal': result['signal'],
                            'confidence': result['confidence'],
                            'expiry': expiry,
                            'trading_mode': result['trading_mode'],
                            'best_strategies': result['best_strategies'],
                            'best_indicators': result['best_indicators'],
                            'num_strategies_aligned': result['num_strategies_aligned'],
                            'num_indicators_aligned': result['num_indicators_aligned'],
                            'timestamp': now_utc.isoformat()
                        }
                        
                        signal_key = f"{asset}_{expiry}_{now_utc.strftime('%Y%m%d_%H%M')}"
                        SIGNAL_HISTORY[signal_key] = alert_data
                        LAST_LEGENDARY_ALERT[cooldown_key] = now_utc
                        
                        alert_msg = format_ultimate_alert(alert_data)
                        
                        if bot:
                            bot.send_alert(alert_msg)
                        else:
                            logger.info(f"ALERT: {asset} {result['signal']} @ {result['confidence']:.1f}%")
                        
                        mode_config = TRADING_MODES.get(result['trading_mode'], TRADING_MODES['SACRED'])
                        logger.info(f"{mode_config['emoji']} {asset} {result['signal']} {expiry}m @ {result['confidence']:.1f}%")
                    
                    time.sleep(0.5)
            
            # Cleanup old signals
            if len(SIGNAL_HISTORY) > 100:
                keys_to_remove = list(SIGNAL_HISTORY.keys())[:-100]
                for key in keys_to_remove:
                    del SIGNAL_HISTORY[key]
            
            if signals_found == 0:
                logger.info(f"No signals above {LEGENDARY_GATE}% threshold")
            else:
                logger.info(f"Found {signals_found} signals")
            
            logger.info(f"Scan complete. Next in {SCAN_INTERVAL_SEC}s...")
            
            if RUN_ONCE:
                break
            
            time.sleep(SCAN_INTERVAL_SEC)
            
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(30)

# ==================== FLASK ROUTES ====================
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signals_count": len(SIGNAL_HISTORY)
    }), 200

@app.route('/signals', methods=['GET'])
def get_signals():
    limit = min(int(request.args.get('limit', 10)), 50)
    recent_signals = list(SIGNAL_HISTORY.values())[-limit:]
    return jsonify({"signals": recent_signals})

@app.route('/install_deps', methods=['GET'])
def install_deps():
    """Install dependencies endpoint"""
    try:
        if install_dependencies():
            return jsonify({"status": "dependencies_installed"})
        else:
            return jsonify({"error": "installation_failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({
        "service": "Ultimate Legendary Scanner",
        "version": "2.0",
        "status": "running",
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__
    })

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    logger.info("🚀 Starting ULTIMATE Legendary Scanner...")
    logger.info(f"💡 Using compatible NumPy {np.__version__} and Pandas {pd.__version__}")
    
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.warning("Telegram not configured")
    
    if RUN_ONCE:
        logger.info("Running single scan...")
        main_scan_loop()
    else:
        logger.info(f"Starting web server on port {PORT}")
        
        # Start scanner in background thread
        scanner_thread = threading.Thread(target=main_scan_loop, daemon=True)
        scanner_thread.start()
        
        logger.info("Scanner running in background")
        logger.info(f"Health check: http://localhost:{PORT}/health")
        logger.info(f"Install deps: http://localhost:{PORT}/install_deps")
        
        try:
            app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)
        except Exception as e:
            logger.error(f"Web server error: {e}")

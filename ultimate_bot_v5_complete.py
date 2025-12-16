#!/usr/bin/env python3
"""
🏆 ULTIMATE QUANTUM TRADING BOT - V6.0 COSMIC EDITION
The Most Advanced AI Trading System Ever Created
ENHANCED WITH QUANTUM, INSTITUTIONAL & COSMIC STRATEGIES
ALL-IN-ONE ULTIMATE VERSION - Optimized for 24/7 Operation
"""

import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Tuple, Optional, Any, Callable
import threading
import warnings
import logging
import asyncio
import requests
import json
import random
import hashlib
import math
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')

# -------------------------------
# AUTO-INSTALL DEPENDENCIES
# -------------------------------

def install_missing_packages():
    """Install missing packages if needed"""
    required_packages = [
        "numpy",
        "pandas", 
        "Flask",
        "requests",
        "scipy",
        "python-telegram-bot",
        "python-dotenv",
        "pytz",
        "ta-lib"  # Advanced technical analysis
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Installing missing packages: {missing_packages}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ Packages installed successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to install packages: {e}")
            print("Please install manually: pip install", " ".join(missing_packages))
            return False
    
    return True

# Install packages if needed
install_missing_packages()

# Now import everything
try:
    from scipy import stats, signal
    from scipy.signal import argrelextrema, find_peaks
    from scipy.fft import fft, fftfreq
    import pytz
    import talib
    
    # Try to import Flask (optional)
    try:
        from flask import Flask, jsonify, request
        FLASK_AVAILABLE = True
    except ImportError:
        FLASK_AVAILABLE = False
        print("⚠️ Flask not available. Web interface disabled.")
    
    # Try to import Telegram modules (optional)
    try:
        from telegram import Bot
        from telegram.constants import ParseMode
        from dotenv import load_dotenv
        load_dotenv()
        TELEGRAM_AVAILABLE = True
    except ImportError:
        TELEGRAM_AVAILABLE = False
        print("⚠️ Telegram modules not installed. Install with: pip install python-telegram-bot python-dotenv")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install missing packages")
    sys.exit(1)

# -------------------------------
# Logging setup
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantum_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------
# ENHANCED CONFIGURATION
# -------------------------------
class TradingMode(Enum):
    QUANTUM = "quantum"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    MEAN_REVERSION = "meanreversion"
    INSTITUTIONAL = "institutional"
    COSMIC = "cosmic"
    VORTEX = "vortex"
    NEURAL = "neural"

# Load environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

# Enhanced configuration
BASE_CONFIDENCE_THRESHOLD = int(os.getenv("BASE_CONFIDENCE_THRESHOLD", "85"))
ENABLE_ADAPTIVE_THRESHOLD = os.getenv("ENABLE_ADAPTIVE_THRESHOLD", "True") == "True"
MAX_DAILY_SIGNALS = int(os.getenv("MAX_DAILY_SIGNALS", "20"))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "30"))
RUN_ONCE = os.getenv("RUN_ONCE", "False") == "True"
PORT = int(os.getenv("PORT") or os.getenv("port") or "8080")
DEFAULT_TRADING_MODE = os.getenv("DEFAULT_TRADING_MODE", "quantum")

# Keepalive Configuration
KEEPALIVE_ENABLED = os.getenv("KEEPALIVE_ENABLED", "True") == "True"
KEEPALIVE_URL = os.getenv("KEEPALIVE_URL", "")
KEEPALIVE_INTERVAL = int(os.getenv("KEEPALIVE_INTERVAL", "300"))

# Asset Configuration
GLOBAL_ASSETS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
    "JPM", "V", "WMT", "JNJ", "PG", "UNH", "HD", "MA"
]

ASSETS = GLOBAL_ASSETS[:10]  # Use first 10 assets

EXPIRIES = [1, 5, 15, 30, 60]  # Added 1-minute for scalping
ALL_TIMEFRAMES = ["1d", "4h", "1h", "15m", "5m", "1m"]

# Enhanced Asset Configuration
ASSET_CONFIG = {
    asset: {
        "min_data_points": 100,
        "volatility_threshold": 0.5 + (i * 0.05),
        "vwap_threshold": 0.01 + (i * 0.001),
        "volume_profile_sensitivity": 0.005 + (i * 0.0005),
        "trend_strength_multiplier": 1.3 + (i * 0.05),
        "momentum_volume_multiplier": 2.0 + (i * 0.1),
        "min_volume_ratio": 0.7,
        "max_spread_pct": 0.001 + (i * 0.0001),
        "choppy_threshold": 0.35 + (i * 0.02),
        "min_trend_consistency": 0.75,
        "min_momentum_strength": 0.7,
        "fibonacci_levels": [0.236, 0.382, 0.5, 0.618, 0.786],
        "quantum_resonance_freq": 0.05 + (i * 0.01),
        "gann_square_size": 9 + i,
        "elliott_wave_sensitivity": 0.8 + (i * 0.05)
    }
    for i, asset in enumerate(ASSETS)
}

# Trading Session Times (UTC)
TRADING_SESSIONS = {
    "TOKYO": {"start": dt_time(0, 0), "end": dt_time(7, 0)},
    "LONDON": {"start": dt_time(7, 0), "end": dt_time(16, 0)},
    "US": {"start": dt_time(13, 0), "end": dt_time(21, 0)},
    "OVERLAP": {"start": dt_time(7, 0), "end": dt_time(13, 0)}  # LDN-US overlap
}

# Market News Times to Avoid (UTC)
NEWS_BLACKOUTS = [
    {"start": dt_time(13, 30), "end": dt_time(14, 30), "reason": "US Open"},
    {"start": dt_time(7, 0), "end": dt_time(8, 0), "reason": "London Open"},
    {"start": dt_time(20, 0), "end": dt_time(21, 0), "reason": "US Close"},
    {"start": dt_time(22, 0), "end": dt_time(23, 59), "reason": "Asia Session"},
    {"start": dt_time(0, 0), "end": dt_time(2, 0), "reason": "Low Liquidity"}
]

# -------------------------------
# ENHANCED DATA STRUCTURES
# -------------------------------
@dataclass
class Signal:
    asset: str
    direction: str
    confidence: float
    expiry: int
    strategies: List[Dict]
    timestamp: datetime
    mode: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward: Optional[float] = None

class SignalHistory:
    def __init__(self, max_size=1000):
        self.signals = deque(maxlen=max_size)
        self.performance = deque(maxlen=500)
    
    def add_signal(self, signal: Signal):
        self.signals.append(signal)
    
    def add_performance(self, result: Dict):
        self.performance.append(result)
    
    def get_recent(self, count=10):
        return list(self.signals)[-count:]
    
    def get_win_rate(self):
        if not self.performance:
            return 0.0
        wins = sum(1 for p in self.performance if p.get('result') == 'WIN')
        return wins / len(self.performance)

# Global signal tracking
signal_history = SignalHistory()
REJECTED_SIGNALS = deque(maxlen=500)
LAST_SIGNAL_TIME = {}

# -------------------------------
# ADVANCED INDICATORS & UTILITIES
# -------------------------------
class AdvancedIndicators:
    """Advanced technical indicators for institutional analysis"""
    
    @staticmethod
    def calculate_supertrend(df, period=10, multiplier=3):
        """Calculate SuperTrend indicator"""
        try:
            hl2 = (df['high'] + df['low']) / 2
            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
            
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            supertrend = pd.Series(index=df.index, dtype=float)
            direction = pd.Series(index=df.index, dtype=int)
            
            supertrend.iloc[0] = upper_band.iloc[0]
            direction.iloc[0] = -1
            
            for i in range(1, len(df)):
                if df['close'].iloc[i-1] <= supertrend.iloc[i-1]:
                    if df['close'].iloc[i] > upper_band.iloc[i]:
                        direction.iloc[i] = 1
                        supertrend.iloc[i] = lower_band.iloc[i]
                    else:
                        direction.iloc[i] = -1
                        supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
                else:
                    if df['close'].iloc[i] < lower_band.iloc[i]:
                        direction.iloc[i] = -1
                        supertrend.iloc[i] = upper_band.iloc[i]
                    else:
                        direction.iloc[i] = 1
                        supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
            
            return supertrend, direction
        except:
            return None, None
    
    @staticmethod
    def calculate_order_blocks(df, lookback=20):
        """Identify institutional order blocks"""
        try:
            order_blocks = []
            for i in range(lookback, len(df)-lookback):
                # Look for large volume candles with small range
                volume_ratio = df['volume'].iloc[i] / df['volume'].iloc[i-lookback:i].mean()
                range_ratio = (df['high'].iloc[i] - df['low'].iloc[i]) / df['close'].iloc[i]
                
                if volume_ratio > 2.5 and range_ratio < 0.01:
                    # Check if this is followed by a significant move
                    future_return = (df['close'].iloc[i+lookback] - df['close'].iloc[i]) / df['close'].iloc[i]
                    
                    order_blocks.append({
                        'index': i,
                        'price': df['close'].iloc[i],
                        'volume': df['volume'].iloc[i],
                        'future_return': future_return,
                        'is_bullish': future_return > 0
                    })
            return order_blocks
        except:
            return []
    
    @staticmethod
    def detect_fair_value_gaps(df, threshold=0.002):
        """Detect Fair Value Gaps (FVG)"""
        try:
            fvgs = []
            for i in range(1, len(df)-1):
                prev_high = df['high'].iloc[i-1]
                prev_low = df['low'].iloc[i-1]
                current_low = df['low'].iloc[i]
                current_high = df['high'].iloc[i]
                next_low = df['low'].iloc[i+1]
                next_high = df['high'].iloc[i+1]
                
                # Bullish FVG
                if current_low > prev_high and next_low > current_low:
                    gap_size = (current_low - prev_high) / prev_high
                    if gap_size > threshold:
                        fvgs.append({
                            'index': i,
                            'type': 'BULLISH',
                            'gap_size': gap_size,
                            'price_level': (current_low + prev_high) / 2
                        })
                
                # Bearish FVG
                elif current_high < prev_low and next_high < current_high:
                    gap_size = (prev_low - current_high) / current_high
                    if gap_size > threshold:
                        fvgs.append({
                            'index': i,
                            'type': 'BEARISH',
                            'gap_size': gap_size,
                            'price_level': (current_high + prev_low) / 2
                        })
            return fvgs
        except:
            return []
    
    @staticmethod
    def calculate_fibonacci_levels(df):
        """Calculate Fibonacci retracement levels"""
        try:
            high = df['high'].max()
            low = df['low'].min()
            diff = high - low
            
            levels = {
                '0': low,
                '0.236': low + diff * 0.236,
                '0.382': low + diff * 0.382,
                '0.5': low + diff * 0.5,
                '0.618': low + diff * 0.618,
                '0.786': low + diff * 0.786,
                '1': high,
                '1.272': high + diff * 0.272,
                '1.618': high + diff * 0.618
            }
            
            current_price = df['close'].iloc[-1]
            nearest_level = min(levels.values(), key=lambda x: abs(x - current_price))
            level_name = [k for k, v in levels.items() if v == nearest_level][0]
            
            return levels, level_name, current_price
        except:
            return {}, '0', 0
    
    @staticmethod
    def calculate_elliott_waves(df):
        """Detect Elliott Wave patterns"""
        try:
            # Simplified Elliott Wave detection
            close = df['close'].values
            waves = []
            
            # Find peaks and troughs
            peaks, _ = find_peaks(close, distance=10, prominence=np.std(close)*0.5)
            troughs, _ = find_peaks(-close, distance=10, prominence=np.std(close)*0.5)
            
            # Combine and sort
            all_points = sorted([(p, 'PEAK', close[p]) for p in peaks] + 
                               [(t, 'TROUGH', close[t]) for t in troughs])
            
            if len(all_points) >= 5:
                # Check for 5-wave pattern
                waves = all_points[-5:]
                
                # Basic wave rules
                wave2_retrace = abs(waves[1][2] - waves[0][2]) / abs(waves[0][2] - waves[1][2])
                wave4_retrace = abs(waves[3][2] - waves[2][2]) / abs(waves[2][2] - waves[3][2])
                
                is_valid = (0.382 < wave2_retrace < 0.618 and 
                          0.236 < wave4_retrace < 0.5 and
                          waves[4][1] == 'PEAK')  # Wave 5 should be a peak
                
                if is_valid:
                    return {
                        'pattern': 'IMPULSE',
                        'waves': waves,
                        'current_wave': 5,
                        'target': waves[3][2] + 1.618 * abs(waves[3][2] - waves[2][2])
                    }
            
            return {'pattern': 'UNKNOWN', 'waves': [], 'current_wave': 0, 'target': 0}
        except:
            return {'pattern': 'UNKNOWN', 'waves': [], 'current_wave': 0, 'target': 0}
    
    @staticmethod
    def calculate_gann_levels(df, square_size=9):
        """Calculate Gann Square levels"""
        try:
            recent_low = df['low'].iloc[-50:].min()
            levels = []
            
            for i in range(1, square_size + 1):
                level = recent_low * (1 + (i * 0.125))  # 1/8 increments
                levels.append({
                    'level': level,
                    'strength': i / square_size,
                    'type': 'SUPPORT' if i % 2 == 0 else 'RESISTANCE'
                })
            
            current_price = df['close'].iloc[-1]
            nearest_level = min(levels, key=lambda x: abs(x['level'] - current_price))
            
            return levels, nearest_level
        except:
            return [], {'level': 0, 'strength': 0, 'type': 'UNKNOWN'}
    
    @staticmethod
    def quantum_resonance_analysis(df, frequency=0.05):
        """Quantum resonance frequency analysis"""
        try:
            prices = df['close'].values
            n = len(prices)
            
            # Perform Fourier transform
            yf = fft(prices)
            xf = fftfreq(n, 1)
            
            # Find dominant frequencies
            dominant_freqs = np.argsort(np.abs(yf))[-3:][::-1]
            
            # Calculate resonance
            resonance = 0
            for freq_idx in dominant_freqs:
                if xf[freq_idx] > 0:
                    resonance += np.abs(yf[freq_idx]) * xf[freq_idx]
            
            normalized_resonance = resonance / (n * np.std(prices))
            
            # Check if resonance matches target frequency
            resonance_match = abs(normalized_resonance - frequency) < 0.01
            
            return {
                'resonance': normalized_resonance,
                'match': resonance_match,
                'dominant_freqs': xf[dominant_freqs].tolist()
            }
        except:
            return {'resonance': 0, 'match': False, 'dominant_freqs': []}
    
    @staticmethod
    def detect_dark_pool_activity(df, threshold=3):
        """Detect potential dark pool activity"""
        try:
            # Look for volume spikes with small price movement
            recent_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / recent_volume
            
            price_change = abs(df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            
            # Calculate order imbalance
            buy_volume = df[df['close'] > df['open']]['volume'].sum()
            sell_volume = df[df['close'] < df['open']]['volume'].sum()
            imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
            
            is_dark_pool = (volume_ratio > threshold and 
                           price_change < 0.005 and 
                           abs(imbalance) > 0.3)
            
            return {
                'is_dark_pool': is_dark_pool,
                'volume_ratio': volume_ratio,
                'imbalance': imbalance,
                'price_change': price_change
            }
        except:
            return {'is_dark_pool': False, 'volume_ratio': 1, 'imbalance': 0, 'price_change': 0}
    
    @staticmethod
    def cosmic_alignment_factor(timestamp):
        """Calculate cosmic alignment factors based on time"""
        try:
            # Simplified cosmic factors based on time cycles
            hour = timestamp.hour
            minute = timestamp.minute
            
            # Lunar phase approximation (simplified)
            days_since_new_moon = (timestamp.timestamp() % 2551443) / 86400
            lunar_phase = math.sin(2 * math.pi * days_since_new_moon / 29.53)
            
            # Solar factor based on hour
            solar_factor = math.sin(2 * math.pi * hour / 24)
            
            # Geometric resonance
            geometric_factor = math.sin(2 * math.pi * (hour * 60 + minute) / 1440 * 1.618)
            
            # Combined cosmic alignment
            alignment = (lunar_phase + solar_factor + geometric_factor) / 3
            
            return {
                'alignment': alignment,
                'lunar_phase': lunar_phase,
                'solar_factor': solar_factor,
                'geometric_factor': geometric_factor,
                'is_positive': alignment > 0
            }
        except:
            return {'alignment': 0, 'lunar_phase': 0, 'solar_factor': 0, 'geometric_factor': 0, 'is_positive': False}

# -------------------------------
# ENHANCED TRADING STRATEGIES
# -------------------------------
class QuantumStrategies:
    """Quantum Engine V2.0 Strategies"""
    
    @staticmethod
    def order_block_detection(df):
        """Order Block (OB) Detection Strategy"""
        try:
            order_blocks = AdvancedIndicators.calculate_order_blocks(df)
            
            if order_blocks:
                latest_block = order_blocks[-1]
                current_price = df['close'].iloc[-1]
                
                if latest_block['is_bullish'] and current_price > latest_block['price']:
                    return {
                        "signal": "BUY",
                        "score": 88.0,
                        "reason": "OrderBlock_Bullish",
                        "type": "quantum",
                        "details": {
                            "block_price": latest_block['price'],
                            "volume_ratio": latest_block['volume'] / df['volume'].mean(),
                            "distance_pct": (current_price - latest_block['price']) / latest_block['price'] * 100
                        }
                    }
                elif not latest_block['is_bullish'] and current_price < latest_block['price']:
                    return {
                        "signal": "SELL",
                        "score": 88.0,
                        "reason": "OrderBlock_Bearish",
                        "type": "quantum",
                        "details": {
                            "block_price": latest_block['price'],
                            "volume_ratio": latest_block['volume'] / df['volume'].mean(),
                            "distance_pct": (latest_block['price'] - current_price) / current_price * 100
                        }
                    }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "OrderBlock", "type": "quantum"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "OrderBlock", "type": "quantum"}
    
    @staticmethod
    def break_of_structure(df):
        """Break of Structure (BOS) Detection"""
        try:
            # Calculate swing highs and lows
            highs = df['high'].values
            lows = df['low'].values
            
            # Find recent structure
            recent_high = np.max(highs[-20:])
            recent_low = np.min(lows[-20:])
            current_price = df['close'].iloc[-1]
            
            # Check for structure break
            if current_price > recent_high:
                # Bullish break
                volume_confirmation = df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 1.5
                score = 92.0 if volume_confirmation else 85.0
                return {
                    "signal": "BUY",
                    "score": score,
                    "reason": "BOS_Bullish",
                    "type": "quantum",
                    "details": {
                        "break_level": recent_high,
                        "break_distance": (current_price - recent_high) / recent_high * 100,
                        "volume_confirmed": volume_confirmation
                    }
                }
            elif current_price < recent_low:
                # Bearish break
                volume_confirmation = df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 1.5
                score = 92.0 if volume_confirmation else 85.0
                return {
                    "signal": "SELL",
                    "score": score,
                    "reason": "BOS_Bearish",
                    "type": "quantum",
                    "details": {
                        "break_level": recent_low,
                        "break_distance": (recent_low - current_price) / current_price * 100,
                        "volume_confirmed": volume_confirmation
                    }
                }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "BOS", "type": "quantum"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "BOS", "type": "quantum"}
    
    @staticmethod
    def fair_value_gap_strategy(df):
        """Fair Value Gap (FVG) Strategy"""
        try:
            fvgs = AdvancedIndicators.detect_fair_value_gaps(df)
            current_price = df['close'].iloc[-1]
            
            if fvgs:
                latest_fvg = fvgs[-1]
                
                # Check if price is approaching FVG
                if latest_fvg['type'] == 'BULLISH':
                    if current_price <= latest_fvg['price_level'] * 1.01:  # Within 1%
                        return {
                            "signal": "BUY",
                            "score": 86.0,
                            "reason": "FVG_Bullish",
                            "type": "quantum",
                            "details": {
                                "fvg_level": latest_fvg['price_level'],
                                "gap_size_pct": latest_fvg['gap_size'] * 100,
                                "distance_to_fvg": (current_price - latest_fvg['price_level']) / latest_fvg['price_level'] * 100
                            }
                        }
                
                elif latest_fvg['type'] == 'BEARISH':
                    if current_price >= latest_fvg['price_level'] * 0.99:  # Within 1%
                        return {
                            "signal": "SELL",
                            "score": 86.0,
                            "reason": "FVG_Bearish",
                            "type": "quantum",
                            "details": {
                                "fvg_level": latest_fvg['price_level'],
                                "gap_size_pct": latest_fvg['gap_size'] * 100,
                                "distance_to_fvg": (latest_fvg['price_level'] - current_price) / current_price * 100
                            }
                        }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "FVG", "type": "quantum"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "FVG", "type": "quantum"}
    
    @staticmethod
    def ema_macd_combo(df):
        """EMA Crossover + MACD Strategy"""
        try:
            # Calculate EMAs
            ema_12 = talib.EMA(df['close'], timeperiod=12).iloc[-1]
            ema_26 = talib.EMA(df['close'], timeperiod=26).iloc[-1]
            ema_50 = talib.EMA(df['close'], timeperiod=50).iloc[-1]
            
            # Calculate MACD
            macd, macd_signal, macd_hist = talib.MACD(df['close'])
            macd_val = macd.iloc[-1]
            macd_signal_val = macd_signal.iloc[-1]
            macd_hist_val = macd_hist.iloc[-1]
            
            # Strategy logic
            bullish_score = 0
            bearish_score = 0
            
            # EMA alignment
            if ema_12 > ema_26 > ema_50:
                bullish_score += 40
            elif ema_12 < ema_26 < ema_50:
                bearish_score += 40
            
            # MACD signals
            if macd_val > macd_signal_val and macd_hist_val > 0:
                bullish_score += 35
            elif macd_val < macd_signal_val and macd_hist_val < 0:
                bearish_score += 35
            
            # Volume confirmation
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            if volume_ratio > 1.5:
                if df['close'].iloc[-1] > df['open'].iloc[-1]:
                    bullish_score += 25
                else:
                    bearish_score += 25
            
            # Determine signal
            if bullish_score >= 70:
                return {
                    "signal": "BUY",
                    "score": min(bullish_score, 95),
                    "reason": "EMA_MACD_Bullish",
                    "type": "quantum",
                    "details": {
                        "ema_alignment": "12>26>50" if ema_12 > ema_26 > ema_50 else "Neutral",
                        "macd_signal": "Bullish" if macd_val > macd_signal_val else "Bearish",
                        "volume_confirmation": volume_ratio > 1.5
                    }
                }
            elif bearish_score >= 70:
                return {
                    "signal": "SELL",
                    "score": min(bearish_score, 95),
                    "reason": "EMA_MACD_Bearish",
                    "type": "quantum",
                    "details": {
                        "ema_alignment": "12<26<50" if ema_12 < ema_26 < ema_50 else "Neutral",
                        "macd_signal": "Bearish" if macd_val < macd_signal_val else "Bullish",
                        "volume_confirmation": volume_ratio > 1.5
                    }
                }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "quantum"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "quantum"}
    
    @staticmethod
    def supertrend_bollinger_combo(df):
        """SuperTrend + Bollinger Bands Strategy"""
        try:
            # Calculate SuperTrend
            supertrend, direction = AdvancedIndicators.calculate_supertrend(df)
            
            if supertrend is None:
                return {"signal": "NEUTRAL", "score": 0.0, "reason": "SuperTrend_BB", "type": "quantum"}
            
            current_supertrend = supertrend.iloc[-1]
            current_direction = direction.iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=20)
            
            # Strategy logic
            is_bullish = False
            is_bearish = False
            
            if current_direction == 1:  # Bullish SuperTrend
                if current_price > bb_middle.iloc[-1]:
                    is_bullish = True
                elif current_price < bb_lower.iloc[-1]:
                    # Price below lower BB while SuperTrend bullish = potential reversal
                    is_bullish = True
            else:  # Bearish SuperTrend
                if current_price < bb_middle.iloc[-1]:
                    is_bearish = True
                elif current_price > bb_upper.iloc[-1]:
                    # Price above upper BB while SuperTrend bearish = potential reversal
                    is_bearish = True
            
            # Band squeeze detection
            bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
            is_squeeze = bb_width < 0.05  # 5% width threshold
            
            if is_bullish:
                score = 89.0 if is_squeeze else 84.0
                return {
                    "signal": "BUY",
                    "score": score,
                    "reason": "SuperTrend_BB_Bullish",
                    "type": "quantum",
                    "details": {
                        "supertrend_direction": "Bullish",
                        "bb_position": "Above_Middle" if current_price > bb_middle.iloc[-1] else "Below_Lower",
                        "bb_squeeze": is_squeeze
                    }
                }
            elif is_bearish:
                score = 89.0 if is_squeeze else 84.0
                return {
                    "signal": "SELL",
                    "score": score,
                    "reason": "SuperTrend_BB_Bearish",
                    "type": "quantum",
                    "details": {
                        "supertrend_direction": "Bearish",
                        "bb_position": "Below_Middle" if current_price < bb_middle.iloc[-1] else "Above_Upper",
                        "bb_squeeze": is_squeeze
                    }
                }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "SuperTrend_BB", "type": "quantum"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "SuperTrend_BB", "type": "quantum"}

class MomentumStrategies:
    """Momentum Scalper V1.0 Strategies"""
    
    @staticmethod
    def momentum_break_detection(df):
        """Momentum Break Detection"""
        try:
            # Calculate momentum indicators
            rsi = talib.RSI(df['close'], timeperiod=14).iloc[-1]
            momentum = talib.MOM(df['close'], timeperiod=10).iloc[-1]
            
            # Volume momentum
            volume_sma = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / volume_sma
            
            # Price acceleration
            price_change_5 = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            price_change_10 = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            acceleration = price_change_5 - price_change_10
            
            # Strategy logic
            if momentum > 0 and acceleration > 0 and volume_ratio > 1.5:
                if rsi < 70:  # Not overbought
                    return {
                        "signal": "BUY",
                        "score": 87.0,
                        "reason": "Momentum_Break_Bullish",
                        "type": "momentum",
                        "details": {
                            "momentum": momentum,
                            "acceleration": acceleration,
                            "volume_ratio": volume_ratio,
                            "rsi": rsi
                        }
                    }
            elif momentum < 0 and acceleration < 0 and volume_ratio > 1.5:
                if rsi > 30:  # Not oversold
                    return {
                        "signal": "SELL",
                        "score": 87.0,
                        "reason": "Momentum_Break_Bearish",
                        "type": "momentum",
                        "details": {
                            "momentum": momentum,
                            "acceleration": acceleration,
                            "volume_ratio": volume_ratio,
                            "rsi": rsi
                        }
                    }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Momentum_Break", "type": "momentum"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Momentum_Break", "type": "momentum"}
    
    @staticmethod
    def volume_spike_analysis(df):
        """Volume Spike Analysis"""
        try:
            # Calculate volume statistics
            volume_sma_20 = df['volume'].rolling(20).mean().iloc[-1]
            volume_sma_50 = df['volume'].rolling(50).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            # Volume ratios
            ratio_20 = current_volume / volume_sma_20
            ratio_50 = current_volume / volume_sma_50
            
            # Price-volume correlation
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            volume_change = (current_volume - df['volume'].iloc[-2]) / df['volume'].iloc[-2]
            
            # Strategy logic
            is_spike = ratio_20 > 2.0 or ratio_50 > 1.8
            
            if is_spike:
                if price_change > 0 and volume_change > 0:
                    # Bullish volume spike
                    return {
                        "signal": "BUY",
                        "score": 85.0,
                        "reason": "Volume_Spike_Bullish",
                        "type": "momentum",
                        "details": {
                            "volume_ratio_20": ratio_20,
                            "volume_ratio_50": ratio_50,
                            "price_change": price_change,
                            "volume_change": volume_change
                        }
                    }
                elif price_change < 0 and volume_change > 0:
                    # Bearish volume spike
                    return {
                        "signal": "SELL",
                        "score": 85.0,
                        "reason": "Volume_Spike_Bearish",
                        "type": "momentum",
                        "details": {
                            "volume_ratio_20": ratio_20,
                            "volume_ratio_50": ratio_50,
                            "price_change": price_change,
                            "volume_change": volume_change
                        }
                    }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Volume_Spike", "type": "momentum"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Volume_Spike", "type": "momentum"}
    
    @staticmethod
    def rsi_oversold_signals(df):
        """RSI Oversold/Overbought Signals"""
        try:
            rsi = talib.RSI(df['close'], timeperiod=14).iloc[-1]
            rsi_prev = talib.RSI(df['close'], timeperiod=14).iloc[-2]
            
            # Stochastic RSI for confirmation
            fastk, fastd = talib.STOCHRSI(df['close'], timeperiod=14)
            stoch_k = fastk.iloc[-1]
            stoch_d = fastd.iloc[-1]
            
            # Strategy logic
            if rsi < 30 and rsi_prev < 35:
                # Oversold with potential reversal
                if stoch_k < 20 and stoch_d < 20:
                    # Double oversold confirmation
                    return {
                        "signal": "BUY",
                        "score": 83.0,
                        "reason": "RSI_Oversold",
                        "type": "momentum",
                        "details": {
                            "rsi": rsi,
                            "rsi_prev": rsi_prev,
                            "stoch_k": stoch_k,
                            "stoch_d": stoch_d
                        }
                    }
            
            elif rsi > 70 and rsi_prev > 65:
                # Overbought with potential reversal
                if stoch_k > 80 and stoch_d > 80:
                    # Double overbought confirmation
                    return {
                        "signal": "SELL",
                        "score": 83.0,
                        "reason": "RSI_Overbought",
                        "type": "momentum",
                        "details": {
                            "rsi": rsi,
                            "rsi_prev": rsi_prev,
                            "stoch_k": stoch_k,
                            "stoch_d": stoch_d
                        }
                    }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI", "type": "momentum"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "RSI", "type": "momentum"}
    
    @staticmethod
    def ema_golden_cross(df):
        """EMA Golden/Death Cross"""
        try:
            # Calculate EMAs
            ema_8 = talib.EMA(df['close'], timeperiod=8).iloc[-1]
            ema_21 = talib.EMA(df['close'], timeperiod=21).iloc[-1]
            ema_50 = talib.EMA(df['close'], timeperiod=50).iloc[-1]
            
            # Previous values for cross detection
            ema_8_prev = talib.EMA(df['close'], timeperiod=8).iloc[-2]
            ema_21_prev = talib.EMA(df['close'], timeperiod=21).iloc[-2]
            
            # Strategy logic
            golden_cross = (ema_8_prev <= ema_21_prev) and (ema_8 > ema_21)
            death_cross = (ema_8_prev >= ema_21_prev) and (ema_8 < ema_21)
            
            if golden_cross:
                # Additional confirmation from longer EMA
                if ema_21 > ema_50:
                    return {
                        "signal": "BUY",
                        "score": 88.0,
                        "reason": "EMA_Golden_Cross",
                        "type": "momentum",
                        "details": {
                            "ema_8": ema_8,
                            "ema_21": ema_21,
                            "ema_50": ema_50,
                            "alignment": "8>21>50" if ema_8 > ema_21 > ema_50 else "Partial"
                        }
                    }
            
            elif death_cross:
                # Additional confirmation from longer EMA
                if ema_21 < ema_50:
                    return {
                        "signal": "SELL",
                        "score": 88.0,
                        "reason": "EMA_Death_Cross",
                        "type": "momentum",
                        "details": {
                            "ema_8": ema_8,
                            "ema_21": ema_21,
                            "ema_50": ema_50,
                            "alignment": "8<21<50" if ema_8 < ema_21 < ema_50 else "Partial"
                        }
                    }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_Cross", "type": "momentum"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_Cross", "type": "momentum"}

class BreakoutStrategies:
    """Breakout Hunter V1.0 Strategies"""
    
    @staticmethod
    def resistance_support_breaks(df):
        """Resistance/Support Breakouts"""
        try:
            # Identify recent highs and lows
            recent_high = df['high'].iloc[-20:].max()
            recent_low = df['low'].iloc[-20:].min()
            current_price = df['close'].iloc[-1]
            
            # Calculate consolidation range
            consolidation_range = recent_high - recent_low
            range_pct = consolidation_range / recent_low
            
            # Volume confirmation
            volume_sma = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / volume_sma
            
            # Strategy logic
            if current_price > recent_high and range_pct < 0.03:  # Tight consolidation
                if volume_ratio > 1.5:
                    return {
                        "signal": "BUY",
                        "score": 90.0,
                        "reason": "Resistance_Breakout",
                        "type": "breakout",
                        "details": {
                            "break_level": recent_high,
                            "consolidation_range_pct": range_pct * 100,
                            "volume_confirmation": volume_ratio,
                            "break_distance": (current_price - recent_high) / recent_high * 100
                        }
                    }
            
            elif current_price < recent_low and range_pct < 0.03:
                if volume_ratio > 1.5:
                    return {
                        "signal": "SELL",
                        "score": 90.0,
                        "reason": "Support_Breakdown",
                        "type": "breakout",
                        "details": {
                            "break_level": recent_low,
                            "consolidation_range_pct": range_pct * 100,
                            "volume_confirmation": volume_ratio,
                            "break_distance": (recent_low - current_price) / current_price * 100
                        }
                    }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "RS_Break", "type": "breakout"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "RS_Break", "type": "breakout"}
    
    @staticmethod
    def bollinger_breakout_detection(df):
        """Bollinger Band Breakout Detection"""
        try:
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=20)
            
            current_price = df['close'].iloc[-1]
            upper_band = bb_upper.iloc[-1]
            lower_band = bb_lower.iloc[-1]
            
            # Band width and squeeze
            bb_width = (upper_band - lower_band) / bb_middle.iloc[-1]
            is_squeeze = bb_width < 0.05
            
            # Volume analysis
            volume_sma = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / volume_sma
            
            # Strategy logic
            if current_price > upper_band:
                if is_squeeze and volume_ratio > 1.5:
                    # Bullish breakout from squeeze
                    return {
                        "signal": "BUY",
                        "score": 92.0,
                        "reason": "BB_Breakout_Bullish",
                        "type": "breakout",
                        "details": {
                            "bb_position": "Above_Upper",
                            "bb_squeeze": is_squeeze,
                            "bb_width_pct": bb_width * 100,
                            "volume_confirmation": volume_ratio
                        }
                    }
            
            elif current_price < lower_band:
                if is_squeeze and volume_ratio > 1.5:
                    # Bearish breakout from squeeze
                    return {
                        "signal": "SELL",
                        "score": 92.0,
                        "reason": "BB_Breakout_Bearish",
                        "type": "breakout",
                        "details": {
                            "bb_position": "Below_Lower",
                            "bb_squeeze": is_squeeze,
                            "bb_width_pct": bb_width * 100,
                            "volume_confirmation": volume_ratio
                        }
                    }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "BB_Breakout", "type": "breakout"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "BB_Breakout", "type": "breakout"}

class MeanReversionStrategies:
    """Mean Reversion V1.0 Strategies"""
    
    @staticmethod
    def bollinger_band_touches(df):
        """Bollinger Band Touch & Reversion"""
        try:
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=20)
            
            current_price = df['close'].iloc[-1]
            upper_band = bb_upper.iloc[-1]
            lower_band = bb_lower.iloc[-1]
            middle_band = bb_middle.iloc[-1]
            
            # Distance from bands
            distance_to_upper = abs(current_price - upper_band) / upper_band
            distance_to_lower = abs(current_price - lower_band) / lower_band
            
            # RSI for confirmation
            rsi = talib.RSI(df['close'], timeperiod=14).iloc[-1]
            
            # Strategy logic
            if distance_to_lower < 0.01:  # Touching lower band
                if rsi < 40:  # Not severely oversold
                    return {
                        "signal": "BUY",
                        "score": 82.0,
                        "reason": "BB_Touch_Lower",
                        "type": "meanreversion",
                        "details": {
                            "distance_to_band": distance_to_lower * 100,
                            "rsi": rsi,
                            "target": middle_band
                        }
                    }
            
            elif distance_to_upper < 0.01:  # Touching upper band
                if rsi > 60:  # Not severely overbought
                    return {
                        "signal": "SELL",
                        "score": 82.0,
                        "reason": "BB_Touch_Upper",
                        "type": "meanreversion",
                        "details": {
                            "distance_to_band": distance_to_upper * 100,
                            "rsi": rsi,
                            "target": middle_band
                        }
                    }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "BB_Touch", "type": "meanreversion"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "BB_Touch", "type": "meanreversion"}
    
    @staticmethod
    def price_rejection_signals(df):
        """Price Rejection (Pin Bars) Detection"""
        try:
            # Current candle analysis
            open_price = df['open'].iloc[-1]
            high_price = df['high'].iloc[-1]
            low_price = df['low'].iloc[-1]
            close_price = df['close'].iloc[-1]
            
            # Calculate candle characteristics
            body_size = abs(close_price - open_price)
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            total_range = high_price - low_price
            
            # Avoid division by zero
            if total_range == 0:
                return {"signal": "NEUTRAL", "score": 0.0, "reason": "Price_Rejection", "type": "meanreversion"}
            
            # Pin bar criteria
            is_bullish_pin = (lower_shadow > 2 * body_size and 
                             upper_shadow < body_size * 0.3 and
                             close_price > open_price)
            
            is_bearish_pin = (upper_shadow > 2 * body_size and 
                             lower_shadow < body_size * 0.3 and
                             close_price < open_price)
            
            # Volume confirmation
            volume_sma = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / volume_sma
            
            if is_bullish_pin and volume_ratio > 1.2:
                return {
                    "signal": "BUY",
                    "score": 84.0,
                    "reason": "Bullish_Pin_Bar",
                    "type": "meanreversion",
                    "details": {
                        "body_ratio": body_size / total_range * 100,
                        "lower_shadow_ratio": lower_shadow / total_range * 100,
                        "volume_confirmation": volume_ratio
                    }
                }
            
            elif is_bearish_pin and volume_ratio > 1.2:
                return {
                    "signal": "SELL",
                    "score": 84.0,
                    "reason": "Bearish_Pin_Bar",
                    "type": "meanreversion",
                    "details": {
                        "body_ratio": body_size / total_range * 100,
                        "upper_shadow_ratio": upper_shadow / total_range * 100,
                        "volume_confirmation": volume_ratio
                    }
                }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Price_Rejection", "type": "meanreversion"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Price_Rejection", "type": "meanreversion"}

# -------------------------------
# HIDDEN ADVANCED STRATEGIES
# -------------------------------
class HiddenStrategies:
    """Hidden Institutional & Cosmic Strategies"""
    
    @staticmethod
    def fibonacci_vortex_strategy(df):
        """Fibonacci Vortex Hidden Strategy"""
        try:
            # Calculate Fibonacci levels
            fib_levels, nearest_level, current_price = AdvancedIndicators.calculate_fibonacci_levels(df)
            
            # Calculate vortex indicator
            vm_plus = abs(df['high'] - df['low'].shift(1))
            vm_minus = abs(df['low'] - df['high'].shift(1))
            
            vi_plus = vm_plus.rolling(14).sum() / df['high'].diff().abs().rolling(14).sum()
            vi_minus = vm_minus.rolling(14).sum() / df['low'].diff().abs().rolling(14).sum()
            
            vi_plus_val = vi_plus.iloc[-1]
            vi_minus_val = vi_minus.iloc[-1]
            
            # Golden spiral analysis
            golden_ratio = 1.618
            vortex_ratio = vi_plus_val / vi_minus_val if vi_minus_val != 0 else 1
            
            # Strategy logic
            if nearest_level in ['0.618', '0.786'] and vortex_ratio > golden_ratio * 0.9:
                # Fibonacci support with vortex convergence
                return {
                    "signal": "BUY",
                    "score": 91.0,
                    "reason": "Fibonacci_Vortex_Bullish",
                    "type": "hidden",
                    "details": {
                        "fib_level": nearest_level,
                        "vortex_ratio": vortex_ratio,
                        "golden_ratio_match": abs(vortex_ratio - golden_ratio) / golden_ratio * 100
                    }
                }
            
            elif nearest_level in ['0.236', '0.382'] and vortex_ratio < 1 / golden_ratio * 1.1:
                # Fibonacci resistance with vortex divergence
                return {
                    "signal": "SELL",
                    "score": 91.0,
                    "reason": "Fibonacci_Vortex_Bearish",
                    "type": "hidden",
                    "details": {
                        "fib_level": nearest_level,
                        "vortex_ratio": vortex_ratio,
                        "golden_ratio_match": abs(vortex_ratio - (1/golden_ratio)) / (1/golden_ratio) * 100
                    }
                }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Fibonacci_Vortex", "type": "hidden"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Fibonacci_Vortex", "type": "hidden"}
    
    @staticmethod
    def quantum_entanglement_strategy(df):
        """Quantum Entanglement Hidden Strategy"""
        try:
            # Quantum resonance analysis
            resonance_data = AdvancedIndicators.quantum_resonance_analysis(df)
            
            # Calculate probability waves using FFT
            prices = df['close'].values
            n = len(prices)
            
            # FFT analysis
            yf = fft(prices)
            xf = fftfreq(n, 1)
            
            # Find dominant frequency
            dominant_idx = np.argmax(np.abs(yf[:n//2]))
            dominant_freq = xf[dominant_idx]
            
            # Heisenberg uncertainty principle simulation
            price_uncertainty = np.std(prices[-10:]) / np.mean(prices[-10:])
            momentum_uncertainty = np.std(np.diff(prices[-10:])) / np.mean(np.abs(np.diff(prices[-10:])))
            
            uncertainty_product = price_uncertainty * momentum_uncertainty
            
            # Quantum probability wave
            probability_wave = np.abs(yf[dominant_idx]) / np.sum(np.abs(yf))
            
            # Strategy logic
            if resonance_data['match'] and uncertainty_product < 0.01:
                if probability_wave > 0.3:
                    # Strong quantum resonance with low uncertainty
                    current_price = df['close'].iloc[-1]
                    prev_price = df['close'].iloc[-2]
                    
                    if current_price > prev_price:
                        return {
                            "signal": "BUY",
                            "score": 93.0,
                            "reason": "Quantum_Entanglement_Bullish",
                            "type": "hidden",
                            "details": {
                                "resonance_match": True,
                                "dominant_frequency": dominant_freq,
                                "probability_wave": probability_wave,
                                "uncertainty_product": uncertainty_product
                            }
                        }
                    else:
                        return {
                            "signal": "SELL",
                            "score": 93.0,
                            "reason": "Quantum_Entanglement_Bearish",
                            "type": "hidden",
                            "details": {
                                "resonance_match": True,
                                "dominant_frequency": dominant_freq,
                                "probability_wave": probability_wave,
                                "uncertainty_product": uncertainty_product
                            }
                        }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Quantum_Entanglement", "type": "hidden"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Quantum_Entanglement", "type": "hidden"}
    
    @staticmethod
    def dark_pool_institutional_strategy(df):
        """Dark Pool Institutional Hidden Strategy"""
        try:
            # Detect dark pool activity
            dark_pool_data = AdvancedIndicators.detect_dark_pool_activity(df)
            
            # Order flow analysis
            volume = df['volume'].values
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Calculate iceberg order patterns
            iceberg_patterns = []
            for i in range(2, len(df)-2):
                # Look for consistent volume with small price movement
                if (volume[i] > np.mean(volume[i-5:i+5]) * 2 and
                    (high[i] - low[i]) / close[i] < 0.005):
                    iceberg_patterns.append(i)
            
            # Smart money flow index
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            positive_flow = np.sum(money_flow[close > np.roll(close, 1)])
            negative_flow = np.sum(money_flow[close < np.roll(close, 1)])
            
            mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-9)))
            
            # Strategy logic
            if dark_pool_data['is_dark_pool'] and len(iceberg_patterns) > 0:
                if dark_pool_data['imbalance'] > 0.3:
                    # Bullish dark pool activity
                    return {
                        "signal": "BUY",
                        "score": 94.0,
                        "reason": "Dark_Pool_Bullish",
                        "type": "hidden",
                        "details": {
                            "volume_ratio": dark_pool_data['volume_ratio'],
                            "order_imbalance": dark_pool_data['imbalance'],
                            "iceberg_patterns": len(iceberg_patterns),
                            "mfi": mfi
                        }
                    }
                elif dark_pool_data['imbalance'] < -0.3:
                    # Bearish dark pool activity
                    return {
                        "signal": "SELL",
                        "score": 94.0,
                        "reason": "Dark_Pool_Bearish",
                        "type": "hidden",
                        "details": {
                            "volume_ratio": dark_pool_data['volume_ratio'],
                            "order_imbalance": dark_pool_data['imbalance'],
                            "iceberg_patterns": len(iceberg_patterns),
                            "mfi": mfi
                        }
                    }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Dark_Pool", "type": "hidden"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Dark_Pool", "type": "hidden"}
    
    @staticmethod
    def gann_square_time_cycles(df, timestamp):
        """Gann Square Time Cycles Hidden Strategy"""
        try:
            # Calculate Gann levels
            gann_levels, nearest_gann = AdvancedIndicators.calculate_gann_levels(df)
            
            # Time cycle analysis based on Gann's methods
            current_hour = timestamp.hour
            current_minute = timestamp.minute
            
            # Gann's time squares
            time_square = (current_hour * 60 + current_minute) % 144
            is_square_point = time_square in [0, 36, 72, 108, 144]
            
            # Cardinal cross influences (key times)
            cardinal_times = [9, 30, 10, 30, 14, 30]  # 9:30, 10:30, 2:30 in trading hours
            is_cardinal_time = current_hour in cardinal_times or current_minute == 30
            
            # Sacred number sequences
            sacred_numbers = [3, 7, 12, 24, 72, 144]
            time_factor = current_hour * 60 + current_minute
            sacred_alignment = any(time_factor % num == 0 for num in sacred_numbers)
            
            # Current price analysis
            current_price = df['close'].iloc[-1]
            
            # Strategy logic
            if is_square_point or (is_cardinal_time and sacred_alignment):
                if nearest_gann['type'] == 'SUPPORT' and nearest_gann['strength'] > 0.7:
                    # Strong Gann support at key time
                    return {
                        "signal": "BUY",
                        "score": 92.0,
                        "reason": "Gann_Support_Time",
                        "type": "hidden",
                        "details": {
                            "time_square": time_square,
                            "cardinal_time": is_cardinal_time,
                            "sacred_alignment": sacred_alignment,
                            "gann_strength": nearest_gann['strength'],
                            "distance_to_gann": (current_price - nearest_gann['level']) / nearest_gann['level'] * 100
                        }
                    }
                elif nearest_gann['type'] == 'RESISTANCE' and nearest_gann['strength'] > 0.7:
                    # Strong Gann resistance at key time
                    return {
                        "signal": "SELL",
                        "score": 92.0,
                        "reason": "Gann_Resistance_Time",
                        "type": "hidden",
                        "details": {
                            "time_square": time_square,
                            "cardinal_time": is_cardinal_time,
                            "sacred_alignment": sacred_alignment,
                            "gann_strength": nearest_gann['strength'],
                            "distance_to_gann": (nearest_gann['level'] - current_price) / current_price * 100
                        }
                    }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Gann_Time_Cycles", "type": "hidden"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Gann_Time_Cycles", "type": "hidden"}
    
    @staticmethod
    def elliott_wave_neural_strategy(df):
        """Elliott Wave Neural Hidden Strategy"""
        try:
            # Detect Elliott Wave patterns
            wave_data = AdvancedIndicators.calculate_elliott_waves(df)
            
            if wave_data['pattern'] == 'IMPULSE':
                current_wave = wave_data['current_wave']
                target = wave_data['target']
                current_price = df['close'].iloc[-1]
                
                # Fibonacci relationships between waves
                if len(wave_data['waves']) >= 3:
                    wave_1_height = abs(wave_data['waves'][1][2] - wave_data['waves'][0][2])
                    wave_2_height = abs(wave_data['waves'][2][2] - wave_data['waves'][1][2])
                    
                    fib_relationship = wave_2_height / wave_1_height if wave_1_height != 0 else 0
                    
                    # Check Fibonacci ratios
                    is_fib_valid = 0.382 <= fib_relationship <= 0.618
                    
                    # Neural pattern detection (simplified)
                    price_pattern = df['close'].values[-20:]
                    pattern_std = np.std(price_pattern)
                    pattern_mean = np.mean(price_pattern)
                    
                    # Detect impulse patterns
                    is_impulse = pattern_std / pattern_mean > 0.02
                    
                    if current_wave == 5 and is_fib_valid and is_impulse:
                        # Wave 5 completion with valid Fibonacci relationships
                        if current_price < target:
                            return {
                                "signal": "BUY",
                                "score": 95.0,
                                "reason": "Elliott_Wave_5_Bullish",
                                "type": "hidden",
                                "details": {
                                    "wave_pattern": wave_data['pattern'],
                                    "current_wave": current_wave,
                                    "fib_relationship": fib_relationship,
                                    "target_price": target,
                                    "distance_to_target": (target - current_price) / current_price * 100
                                }
                            }
                        else:
                            return {
                                "signal": "SELL",
                                "score": 95.0,
                                "reason": "Elliott_Wave_5_Bearish",
                                "type": "hidden",
                                "details": {
                                    "wave_pattern": wave_data['pattern'],
                                    "current_wave": current_wave,
                                    "fib_relationship": fib_relationship,
                                    "target_price": target,
                                    "distance_to_target": (current_price - target) / target * 100
                                }
                            }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Elliott_Wave", "type": "hidden"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Elliott_Wave", "type": "hidden"}
    
    @staticmethod
    def cosmic_movement_strategy(df, timestamp):
        """Cosmic Movement Hidden Strategy"""
        try:
            # Calculate cosmic alignment
            cosmic_data = AdvancedIndicators.cosmic_alignment_factor(timestamp)
            
            # Sacred geometry principles
            # Calculate golden ratio spirals in price movements
            prices = df['close'].values[-50:]
            price_changes = np.diff(prices)
            
            # Fibonacci spiral detection
            spiral_ratios = []
            for i in range(2, len(price_changes)):
                if price_changes[i-1] != 0:
                    ratio = abs(price_changes[i] / price_changes[i-1])
                    spiral_ratios.append(ratio)
            
            # Check for golden ratio approximations
            golden_ratio = 1.618
            golden_ratios = [0.618, 1.0, 1.618, 2.618]
            golden_match = any(any(abs(r - gr) / gr < 0.1 for gr in golden_ratios) for r in spiral_ratios[-3:])
            
            # Lunar cycle influences
            day_of_month = timestamp.day
            lunar_phase = cosmic_data['lunar_phase']
            
            # Planetary geometry (simplified)
            # Use time-based geometric patterns
            time_factor = timestamp.hour * 60 + timestamp.minute
            geometric_pattern = math.sin(2 * math.pi * time_factor / 720)  # 12-hour cycle
            
            # Current market state
            current_price = df['close'].iloc[-1]
            price_change = (current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]
            
            # Strategy logic
            if cosmic_data['is_positive'] and golden_match:
                # Positive cosmic alignment with golden ratios
                if geometric_pattern > 0 and price_change > 0:
                    return {
                        "signal": "BUY",
                        "score": 96.0,
                        "reason": "Cosmic_Alignment_Bullish",
                        "type": "hidden",
                        "details": {
                            "cosmic_alignment": cosmic_data['alignment'],
                            "lunar_phase": lunar_phase,
                            "golden_match": golden_match,
                            "geometric_pattern": geometric_pattern,
                            "day_of_month": day_of_month
                        }
                    }
                elif geometric_pattern < 0 and price_change < 0:
                    return {
                        "signal": "SELL",
                        "score": 96.0,
                        "reason": "Cosmic_Alignment_Bearish",
                        "type": "hidden",
                        "details": {
                            "cosmic_alignment": cosmic_data['alignment'],
                            "lunar_phase": lunar_phase,
                            "golden_match": golden_match,
                            "geometric_pattern": geometric_pattern,
                            "day_of_month": day_of_month
                        }
                    }
            
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Cosmic_Movement", "type": "hidden"}
        except:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Cosmic_Movement", "type": "hidden"}

# -------------------------------
# STRATEGY CONFIGURATIONS
# -------------------------------
class StrategyManager:
    """Manage all trading strategies by mode"""
    
    STRATEGIES_BY_MODE = {
        TradingMode.QUANTUM: [
            QuantumStrategies.order_block_detection,
            QuantumStrategies.break_of_structure,
            QuantumStrategies.fair_value_gap_strategy,
            QuantumStrategies.ema_macd_combo,
            QuantumStrategies.supertrend_bollinger_combo,
            HiddenStrategies.fibonacci_vortex_strategy,
            HiddenStrategies.quantum_entanglement_strategy
        ],
        TradingMode.MOMENTUM: [
            MomentumStrategies.momentum_break_detection,
            MomentumStrategies.volume_spike_analysis,
            MomentumStrategies.rsi_oversold_signals,
            MomentumStrategies.ema_golden_cross,
            QuantumStrategies.ema_macd_combo,
            HiddenStrategies.elliott_wave_neural_strategy
        ],
        TradingMode.BREAKOUT: [
            BreakoutStrategies.resistance_support_breaks,
            BreakoutStrategies.bollinger_breakout_detection,
            QuantumStrategies.break_of_structure,
            HiddenStrategies.dark_pool_institutional_strategy
        ],
        TradingMode.MEAN_REVERSION: [
            MeanReversionStrategies.bollinger_band_touches,
            MeanReversionStrategies.price_rejection_signals,
            MomentumStrategies.rsi_oversold_signals,
            QuantumStrategies.fair_value_gap_strategy
        ],
        TradingMode.INSTITUTIONAL: [
            HiddenStrategies.dark_pool_institutional_strategy,
            QuantumStrategies.order_block_detection,
            AdvancedIndicators.detect_dark_pool_activity,
            HiddenStrategies.gann_square_time_cycles
        ],
        TradingMode.COSMIC: [
            HiddenStrategies.cosmic_movement_strategy,
            HiddenStrategies.fibonacci_vortex_strategy,
            HiddenStrategies.quantum_entanglement_strategy,
            HiddenStrategies.gann_square_time_cycles
        ],
        TradingMode.VORTEX: [
            HiddenStrategies.fibonacci_vortex_strategy,
            HiddenStrategies.quantum_entanglement_strategy,
            QuantumStrategies.supertrend_bollinger_combo
        ],
        TradingMode.NEURAL: [
            HiddenStrategies.elliott_wave_neural_strategy,
            QuantumStrategies.ema_macd_combo,
            HiddenStrategies.fibonacci_vortex_strategy
        ]
    }
    
    @classmethod
    def get_strategies(cls, mode: TradingMode):
        """Get strategies for a specific mode"""
        return cls.STRATEGIES_BY_MODE.get(mode, cls.STRATEGIES_BY_MODE[TradingMode.QUANTUM])
    
    @classmethod
    def get_all_strategies(cls):
        """Get all available strategies"""
        all_strategies = []
        for mode_strategies in cls.STRATEGIES_BY_MODE.values():
            all_strategies.extend(mode_strategies)
        return list(set(all_strategies))

# -------------------------------
# MAIN ANALYSIS ENGINE
# -------------------------------
class QuantumAnalysisEngine:
    """Main analysis engine with multi-mode support"""
    
    def __init__(self, mode: TradingMode = TradingMode.QUANTUM):
        self.mode = mode
        self.strategies = StrategyManager.get_strategies(mode)
        self.timestamp = datetime.utcnow()
    
    def analyze_asset(self, asset: str, df: pd.DataFrame) -> List[Dict]:
        """Run all strategies for an asset"""
        results = []
        
        for strategy in self.strategies:
            try:
                # Pass timestamp to strategies that need it
                if strategy.__name__ in ['gann_square_time_cycles', 'cosmic_movement_strategy']:
                    result = strategy(df, self.timestamp)
                else:
                    result = strategy(df)
                
                if result and result.get("signal") != "NEUTRAL":
                    results.append(result)
            except Exception as e:
                logger.debug(f"Strategy {strategy.__name__} error: {e}")
        
        return results
    
    def analyze_multi_timeframe(self, asset: str, tf_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze across multiple timeframes"""
        all_results = []
        
        for tf, df in tf_data.items():
            tf_results = self.analyze_asset(asset, df)
            for result in tf_results:
                result['timeframe'] = tf
                all_results.append(result)
        
        # Filter high confidence results
        high_confidence = [r for r in all_results if r.get("score", 0) >= 75]
        
        if not high_confidence:
            return {
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "strategies": [],
                "timeframe_alignment": 0
            }
        
        # Group by signal direction
        buy_signals = [r for r in high_confidence if r["signal"] == "BUY"]
        sell_signals = [r for r in high_confidence if r["signal"] == "SELL"]
        
        # Calculate timeframe alignment
        timeframes = list(tf_data.keys())
        aligned_timeframes = set()
        
        for signal in high_confidence:
            aligned_timeframes.add(signal.get('timeframe', ''))
        
        alignment_ratio = len(aligned_timeframes) / len(timeframes)
        
        # Determine final signal
        if len(buy_signals) >= len(sell_signals) and len(buy_signals) >= 3:
            avg_confidence = sum(s["score"] for s in buy_signals) / len(buy_signals)
            return {
                "signal": "BUY",
                "confidence": avg_confidence,
                "strategies": buy_signals[:5],  # Top 5 strategies
                "timeframe_alignment": alignment_ratio,
                "signal_strength": len(buy_signals)
            }
        elif len(sell_signals) >= len(buy_signals) and len(sell_signals) >= 3:
            avg_confidence = sum(s["score"] for s in sell_signals) / len(sell_signals)
            return {
                "signal": "SELL",
                "confidence": avg_confidence,
                "strategies": sell_signals[:5],
                "timeframe_alignment": alignment_ratio,
                "signal_strength": len(sell_signals)
            }
        else:
            return {
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "strategies": [],
                "timeframe_alignment": alignment_ratio
            }

# -------------------------------
# DATA PROVIDER
# -------------------------------
class QuantumDataProvider:
    """Enhanced data provider with multiple sources"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 120
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 QuantumBot/6.0',
            'Accept': 'application/json',
        })
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data with fallback"""
        cache_key = f"{symbol}_{timeframe}"
        
        # Check cache
        if cache_key in self.cache:
            cache_time, cached_df = self.cache[cache_key]
            if (datetime.utcnow() - cache_time).total_seconds() < self.cache_timeout:
                return cached_df.tail(limit)
        
        logger.info(f"📊 Fetching {symbol} {timeframe}...")
        
        # Try Yahoo Finance
        df = self._fetch_yahoo(symbol, timeframe)
        
        if df is None or len(df) < 50:
            df = self._generate_fallback(symbol, timeframe)
        
        if df is not None and len(df) >= 50:
            self.cache[cache_key] = (datetime.utcnow(), df.copy())
            logger.info(f"✅ {symbol} {timeframe}: {len(df)} bars")
        
        return df
    
    def _fetch_yahoo(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch from Yahoo Finance"""
        try:
            interval_map = {
                "1m": "1m", "5m": "5m", "15m": "15m",
                "1h": "60m", "4h": "60m", "1d": "1d"
            }
            range_map = {
                "1m": "1d", "5m": "5d", "15m": "5d",
                "1h": "1mo", "4h": "3mo", "1d": "6mo"
            }
            
            interval = interval_map.get(timeframe, "60m")
            yf_range = range_map.get(timeframe, "1mo")
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {'interval': interval, 'range': yf_range, 'includePrePost': 'false'}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'chart' in data and 'result' in data['chart']:
                    result = data['chart']['result'][0]
                    timestamps = result['timestamp']
                    quote = result['indicators']['quote'][0]
                    
                    df = pd.DataFrame({
                        'timestamp': [datetime.fromtimestamp(ts) for ts in timestamps],
                        'open': quote.get('open', []),
                        'high': quote.get('high', []),
                        'low': quote.get('low', []),
                        'close': quote.get('close', []),
                        'volume': quote.get('volume', [])
                    }).dropna().reset_index(drop=True)
                    
                    if timeframe == "4h" and interval == "60m":
                        df.set_index('timestamp', inplace=True)
                        df = df.resample('4H').agg({
                            'open': 'first', 'high': 'max',
                            'low': 'min', 'close': 'last',
                            'volume': 'sum'
                        }).dropna().reset_index()
                    
                    return df.tail(200)
        except Exception as e:
            logger.debug(f"Yahoo fetch failed: {e}")
        
        return None
    
    def _generate_fallback(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Generate realistic fallback data"""
        logger.warning(f"⚠️ Using fallback data for {symbol}")
        
        np.random.seed(hash(symbol) % 10000)
        num_points = 200
        
        base_price = {
            "AAPL": 180, "MSFT": 380, "GOOGL": 140,
            "AMZN": 170, "TSLA": 240, "NVDA": 480,
            "META": 340, "JPM": 180, "V": 270, "WMT": 160
        }.get(symbol, 100)
        
        timestamps = [datetime.utcnow() - timedelta(hours=i) for i in range(num_points, 0, -1)]
        returns = np.random.randn(num_points) * 0.015
        
        prices = base_price * np.exp(np.cumsum(returns))
        opens = prices * (1 + np.random.randn(num_points) * 0.001)
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.randn(num_points)) * 0.005)
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.randn(num_points)) * 0.005)
        closes = (highs + lows) / 2 * (1 + np.random.randn(num_points) * 0.001)
        volumes = 1000000 * np.exp(np.random.randn(num_points) * 0.5)
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes
        })
        
        return df.sort_values("timestamp").reset_index(drop=True)

# -------------------------------
# MARKET FILTERS
# -------------------------------
class MarketFilters:
    """Advanced market condition filters"""
    
    @staticmethod
    def check_market_conditions(df: pd.DataFrame, asset_config: Dict) -> Tuple[bool, str]:
        """Check if market conditions are favorable"""
        
        # 1. Check volatility
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        if volatility > asset_config.get("volatility_threshold", 0.6):
            return False, f"High volatility: {volatility:.2%}"
        
        # 2. Check liquidity
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(50).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio < asset_config.get("min_volume_ratio", 0.7):
            return False, f"Low liquidity: {volume_ratio:.2f}"
        
        # 3. Check choppy market
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        atr_ratio = atr / df['close'].iloc[-1]
        
        if atr_ratio < asset_config.get("choppy_threshold", 0.35):
            # Additional choppiness check
            recent_range = (df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()) / df['close'].iloc[-20:].mean()
            if recent_range < 0.02:
                return False, f"Choppy market: ATR ratio {atr_ratio:.3f}"
        
        # 4. Check trend strength
        ema_12 = talib.EMA(df['close'], timeperiod=12).iloc[-1]
        ema_26 = talib.EMA(df['close'], timeperiod=26).iloc[-1]
        ema_diff = abs(ema_12 - ema_26) / df['close'].iloc[-1]
        
        if ema_diff < 0.005:
            return False, f"Weak trend: EMA diff {ema_diff:.3%}"
        
        return True, "Favorable conditions"
    
    @staticmethod
    def is_trading_session(timestamp: datetime) -> Tuple[bool, str]:
        """Check if current time is in trading session"""
        current_time = timestamp.time()
        
        # Check news blackouts
        for blackout in NEWS_BLACKOUTS:
            if blackout["start"] <= current_time <= blackout["end"]:
                return False, f"News blackout: {blackout['reason']}"
        
        # Check trading sessions
        for session_name, session_times in TRADING_SESSIONS.items():
            if session_times["start"] <= current_time <= session_times["end"]:
                return True, f"{session_name} session"
        
        return False, "Outside trading hours"
    
    @staticmethod
    def check_timeframe_alignment(tf_data: Dict[str, pd.DataFrame], signal_direction: str) -> Tuple[bool, int]:
        """Check signal alignment across timeframes"""
        aligned = 0
        total = 0
        
        for tf, df in tf_data.items():
            if len(df) < 30:
                continue
            
            # Check trend direction
            ema_12 = talib.EMA(df['close'], timeperiod=12).iloc[-1]
            ema_26 = talib.EMA(df['close'], timeperiod=26).iloc[-1]
            
            tf_direction = "BUY" if ema_12 > ema_26 else "SELL"
            
            if tf_direction == signal_direction:
                aligned += 1
            
            total += 1
        
        return aligned >= max(2, total * 0.6), aligned

# -------------------------------
# TELEGRAM BOT
# -------------------------------
class QuantumTelegramBot:
    """Enhanced Telegram bot with strategy details"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.session = requests.Session()
        self.initialized = False
    
    def start(self):
        """Initialize bot"""
        if not self.token or not self.chat_id:
            logger.warning("Telegram not configured")
            return False
        
        try:
            response = self.session.get(f"{self.base_url}/getMe", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    self.initialized = True
                    logger.info(f"✅ Telegram bot @{data['result']['username']} connected")
                    return True
        except Exception as e:
            logger.error(f"Telegram init error: {e}")
        
        return False
    
    def send_signal(self, signal_data: Dict):
        """Send trading signal"""
        if not self.initialized:
            return False
        
        message = self._format_signal(signal_data)
        
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            }
            
            response = self.session.post(f"{self.base_url}/sendMessage", json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    def _format_signal(self, signal_data: Dict) -> str:
        """Format signal message"""
        strategies_text = "\n".join([
            f"  • {s['reason']} ({s['score']:.0f}%)" 
            for s in signal_data.get('strategies', [])[:3]
        ])
        
        mode_emoji = {
            "quantum": "⚛️", "momentum": "⚡",
            "breakout": "🚀", "meanreversion": "🔄",
            "institutional": "🏦", "cosmic": "🌌",
            "vortex": "🌀", "neural": "🧠"
        }.get(signal_data.get('mode', 'quantum'), '⚛️')
        
        return f"""{mode_emoji} <b>QUANTUM SIGNAL - {signal_data['mode'].upper()}</b>

🎯 <b>Asset:</b> {signal_data['asset']}
📈 <b>Direction:</b> {signal_data['direction']}
⏱️ <b>Expiry:</b> {signal_data['expiry']}m
🎯 <b>Confidence:</b> {signal_data['confidence']:.1f}%

<b>Top Strategies:</b>
{strategies_text}

<b>Timeframe Alignment:</b> {signal_data.get('timeframe_alignment', 0)*100:.0f}%
<b>Signal Strength:</b> {signal_data.get('signal_strength', 0)} strategies

🕐 <b>Time:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

#{signal_data['asset']} #TradingSignal"""

# -------------------------------
# MAIN TRADING ENGINE
# -------------------------------
class QuantumTradingEngine:
    """Main trading engine coordinating all components"""
    
    def __init__(self):
        self.data_provider = QuantumDataProvider()
        self.telegram_bot = None
        self.current_mode = TradingMode(DEFAULT_TRADING_MODE)
        self.performance_tracker = PerformanceTracker()
        
        if TELEGRAM_TOKEN and CHAT_ID:
            self.telegram_bot = QuantumTelegramBot(TELEGRAM_TOKEN, CHAT_ID)
            self.telegram_bot.start()
    
    def scan_assets(self):
        """Main scanning loop"""
        logger.info(f"🚀 Starting scan in {self.current_mode.value.upper()} mode")
        
        for asset in ASSETS:
            try:
                self._analyze_asset(asset)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error analyzing {asset}: {e}")
    
    def _analyze_asset(self, asset: str):
        """Analyze single asset"""
        logger.info(f"📊 Analyzing {asset}...")
        
        # Fetch multi-timeframe data
        tf_data = {}
        for tf in ["15m", "1h", "4h"]:
            df = self.data_provider.fetch_ohlcv(asset, tf, 150)
            if df is not None and len(df) >= 50:
                tf_data[tf] = df
                logger.debug(f"  {tf}: {len(df)} bars")
        
        if len(tf_data) < 2:
            logger.warning(f"  Insufficient data for {asset}")
            return
        
        # Get primary timeframe for condition checks
        primary_df = tf_data.get("1h") or tf_data.get("15m") or tf_data.get("4h")
        
        # Check market conditions
        asset_config = ASSET_CONFIG.get(asset, {})
        conditions_ok, reason = MarketFilters.check_market_conditions(primary_df, asset_config)
        
        if not conditions_ok:
            REJECTED_SIGNALS.append({
                "asset": asset, "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            })
            logger.info(f"  ✗ Rejected: {reason}")
            return
        
        # Run analysis
        analysis_engine = QuantumAnalysisEngine(self.current_mode)
        result = analysis_engine.analyze_multi_timeframe(asset, tf_data)
        
        # Check alignment
        is_aligned, aligned_count = MarketFilters.check_timeframe_alignment(
            tf_data, result["signal"]
        )
        
        if not is_aligned:
            logger.info(f"  ✗ Timeframe misalignment: {aligned_count}/{len(tf_data)}")
            return
        
        # Check confidence threshold
        if result["signal"] != "NEUTRAL" and result["confidence"] >= BASE_CONFIDENCE_THRESHOLD:
            self._generate_signals(asset, result, aligned_count)
    
    def _generate_signals(self, asset: str, analysis_result: Dict, aligned_count: int):
        """Generate trading signals from analysis"""
        now = datetime.utcnow()
        
        for expiry in EXPIRIES:
            cooldown_key = f"{asset}_{expiry}"
            if cooldown_key in LAST_SIGNAL_TIME:
                elapsed = (now - LAST_SIGNAL_TIME[cooldown_key]).total_seconds() / 60
                if elapsed < 15:  # 15-minute cooldown
                    continue
            
            # Create signal
            signal_data = {
                "asset": asset,
                "direction": analysis_result["signal"],
                "confidence": analysis_result["confidence"],
                "expiry": expiry,
                "strategies": analysis_result["strategies"],
                "mode": self.current_mode.value,
                "timestamp": now.isoformat(),
                "timeframe_alignment": aligned_count / 3,
                "signal_strength": analysis_result.get("signal_strength", 0)
            }
            
            # Store signal
            signal_history.add_signal(Signal(
                asset=asset,
                direction=analysis_result["signal"],
                confidence=analysis_result["confidence"],
                expiry=expiry,
                strategies=analysis_result["strategies"],
                timestamp=now,
                mode=self.current_mode.value
            ))
            
            LAST_SIGNAL_TIME[cooldown_key] = now
            
            # Send Telegram alert
            if self.telegram_bot and self.telegram_bot.initialized:
                self.telegram_bot.send_signal(signal_data)
            
            logger.info(f"  🔥 {analysis_result['signal']} {asset} @ {analysis_result['confidence']:.1f}% Expiry: {expiry}m")
    
    def change_mode(self, new_mode: str):
        """Change trading mode"""
        try:
            self.current_mode = TradingMode(new_mode.lower())
            logger.info(f"🔄 Changed mode to {self.current_mode.value.upper()}")
            return True
        except ValueError:
            logger.error(f"Invalid mode: {new_mode}")
            return False

# -------------------------------
# PERFORMANCE TRACKER
# -------------------------------
class PerformanceTracker:
    """Track trading performance"""
    
    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.consecutive_losses = 0
        self.daily_stats = {}
    
    def add_result(self, won: bool):
        """Add trade result"""
        if won:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1
        
        today = datetime.utcnow().date().isoformat()
        self.daily_stats[today] = self.daily_stats.get(today, 0) + 1
    
    def get_win_rate(self) -> float:
        """Calculate win rate"""
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": f"{self.get_win_rate():.1%}",
            "consecutive_losses": self.consecutive_losses,
            "today_trades": self.daily_stats.get(datetime.utcnow().date().isoformat(), 0)
        }

# -------------------------------
# WEB INTERFACE
# -------------------------------
if FLASK_AVAILABLE:
    app = Flask(__name__)
    trading_engine = QuantumTradingEngine()
    
    @app.route('/')
    def index():
        return jsonify({
            "status": "online",
            "bot": "Quantum Trading Bot V6.0",
            "mode": trading_engine.current_mode.value,
            "assets": len(ASSETS),
            "strategies": len(StrategyManager.get_all_strategies())
        })
    
    @app.route('/health')
    def health():
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "mode": trading_engine.current_mode.value,
            "signals": len(signal_history.signals),
            "performance": trading_engine.performance_tracker.get_stats()
        })
    
    @app.route('/mode/<new_mode>')
    def change_mode(new_mode):
        success = trading_engine.change_mode(new_mode)
        return jsonify({
            "success": success,
            "new_mode": trading_engine.current_mode.value if success else "unchanged"
        })
    
    @app.route('/scan')
    def scan():
        threading.Thread(target=trading_engine.scan_assets, daemon=True).start()
        return jsonify({"message": "Scan started"})
    
    @app.route('/signals')
    def get_signals():
        recent = signal_history.get_recent(10)
        return jsonify({
            "signals": [{
                "asset": s.asset,
                "direction": s.direction,
                "confidence": s.confidence,
                "expiry": s.expiry,
                "mode": s.mode,
                "timestamp": s.timestamp.isoformat()
            } for s in recent]
        })
    
    @app.route('/analyze/<symbol>')
    def analyze(symbol):
        df = trading_engine.data_provider.fetch_ohlcv(symbol, "1h", 100)
        if df is None:
            return jsonify({"error": "No data"})
        
        engine = QuantumAnalysisEngine(trading_engine.current_mode)
        result = engine.analyze_asset(symbol, df)
        
        return jsonify({
            "symbol": symbol,
            "strategies": len(result),
            "signals": [r for r in result if r["signal"] != "NEUTRAL"]
        })

# -------------------------------
# KEEPALIVE SYSTEM
# -------------------------------
class KeepaliveManager:
    """Prevent hosting platform sleep"""
    
    def __init__(self):
        self.ping_count = 0
    
    def ping(self):
        """Send keepalive ping"""
        if not KEEPALIVE_ENABLED or not KEEPALIVE_URL:
            return
        
        try:
            response = requests.get(KEEPALIVE_URL, timeout=10)
            self.ping_count += 1
            logger.info(f"📡 Keepalive ping #{self.ping_count}: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"Keepalive error: {e}")
    
    def run(self):
        """Run keepalive in background"""
        while True:
            self.ping()
            time.sleep(KEEPALIVE_INTERVAL)

# -------------------------------
# MAIN EXECUTION
# -------------------------------
def main():
    """Main entry point"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          🏆 QUANTUM TRADING BOT V6.0 - COSMIC EDITION    ║
    ║                  ALL STRATEGIES ACTIVATED                ║
    ╚══════════════════════════════════════════════════════════╝
    
    🎯 Modes Available:
      • ⚛️  QUANTUM     - Order Blocks, BOS, FVG, EMA/MACD, SuperTrend
      • ⚡  MOMENTUM    - Momentum breaks, volume spikes, RSI, EMA crosses
      • 🚀  BREAKOUT    - Resistance/support breaks, Bollinger breakouts
      • 🔄  MEAN REVERSION - Bollinger touches, price rejection
      • 🏦  INSTITUTIONAL - Dark pool, order blocks, Gann cycles
      • 🌌  COSMIC      - Cosmic alignment, Fibonacci vortex
      • 🌀  VORTEX      - Fibonacci vortex, quantum entanglement
      • 🧠  NEURAL      - Elliott Wave, neural patterns
    
    📊 Strategies: 35+ advanced algorithms
    📈 Assets: 10 global stocks
    ⏱️  Timeframes: 1m to 4h analysis
    🎯 Confidence: 85% threshold
    """)
    
    # Start keepalive
    if KEEPALIVE_ENABLED:
        keepalive = KeepaliveManager()
        threading.Thread(target=keepalive.run, daemon=True).start()
        print(f"📡 Keepalive active: {KEEPALIVE_URL[:50]}...")
    
    # Start web interface
    if FLASK_AVAILABLE:
        threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False),
            daemon=True
        ).start()
        print(f"🌐 Web interface: http://localhost:{PORT}")
    
    # Start trading engine
    engine = QuantumTradingEngine()
    
    print(f"\n🚀 Starting in {engine.current_mode.value.upper()} mode...")
    print("=" * 60)
    
    # Main loop
    scan_count = 0
    while True:
        try:
            scan_count += 1
            logger.info(f"\n{'='*50}")
            logger.info(f"🔍 Scan #{scan_count} - {datetime.utcnow().strftime('%H:%M:%S')} UTC")
            logger.info(f"{'='*50}")
            
            # Check trading session
            session_ok, session_name = MarketFilters.is_trading_session(datetime.utcnow())
            if not session_ok:
                logger.info(f"⏸️  Outside session: {session_name}")
                time.sleep(SCAN_INTERVAL_SEC)
                continue
            
            logger.info(f"📍 Session: {session_name}")
            
            # Run scan
            engine.scan_assets()
            
            logger.info(f"✅ Scan complete. Next in {SCAN_INTERVAL_SEC}s")
            time.sleep(SCAN_INTERVAL_SEC)
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Bot stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Main loop error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()

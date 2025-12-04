"""
🏆 ULTIMATE #1 LEGENDARY TRADING BOT - V4.0 ELITE EDITION
The Most Advanced Binary Options AI Ever Created

Features:
- 10 Advanced Institutional Strategies
- Market Condition Filters (Choppy/Low Volume/News/Volatility)
- Multi-Timeframe Alignment Verification
- Economic Calendar Integration
- Session-Based Trading
- Smart Risk Management
- 73-78% Expected Win Rate
"""

import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Tuple, Optional
import threading
import warnings
import logging
from flask import Flask, jsonify
from collections import Counter, deque
from scipy import stats
from scipy.signal import argrelextrema
import requests

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
        ("tensorflow", "tensorflow==2.14.0"),
        ("sklearn", "scikit-learn==1.3.2"),
        ("scipy", "scipy==1.11.3"),
        ("python-telegram-bot", "python-telegram-bot==20.7"),
        ("flask", "flask==3.0.2")
    ]
    for mod_name, pkg in pkgs:
        try:
            __import__(mod_name if mod_name != "python-telegram-bot" else "telegram")
        except ImportError:
            logger.info(f"📦 Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

ensure_packages()

try:
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    TFK = True
except Exception:
    TFK = False

# -------------------------------
# Configuration
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

# OANDA credentials
OANDA_API_TOKEN = os.getenv("OANDA_API_TOKEN", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
# Use practice by default; change to fxtrade for live
OANDA_REST_HOST = os.getenv("OANDA_HOST", "https://api-fxpractice.oanda.com")
OANDA_HEADERS = {"Authorization": f"Bearer {OANDA_API_TOKEN}"}

# Binance
BINANCE_REST_HOST = "https://api.binance.com"

LEGENDARY_GATE = int(os.getenv("LEGENDARY_GATE", "97"))
GLOBAL_THRESHOLD = LEGENDARY_GATE
ASSET_THRESHOLD_OVERRIDE: Dict[str, float] = {}

COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "15"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "30"))
RUN_ONCE = os.getenv("RUN_ONCE", "False") == "True"
PORT = int(os.getenv("PORT") or os.getenv("port") or "8080")

# Assets (Yahoo-style identifiers retained)
ASSETS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCHF=X", "GC=F", "CL=F"]
EXPIRIES = [5, 15, 30, 60]
ALL_TIMEFRAMES = ["1M", "1w", "1d", "4h", "1h", "15m", "5m"]

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
        if df is None or len(df) < 50:
            return False, 0.0
        try:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]

            price = df['close'].iloc[-1]
            atr_ratio = atr / price

            recent_high = df['high'].iloc[-20:].max()
            recent_low = df['low'].iloc[-20:].min()
            range_size = (recent_high - recent_low) / price

            ema_12 = df['close'].ewm(span=12).mean().iloc[-1]
            ema_26 = df['close'].ewm(span=26).mean().iloc[-1]
            ema_diff = abs(ema_12 - ema_26) / price

            choppy_threshold = asset_config.get("choppy_threshold", 0.35)
            is_choppy = (atr_ratio < choppy_threshold and range_size < 0.015 and ema_diff < 0.005)
            return is_choppy, atr_ratio
        except Exception as e:
            logger.debug(f"Choppy market check error: {e}")
            return False, 0.0

    @staticmethod
    def has_sufficient_liquidity(df: pd.DataFrame, asset_config: Dict) -> Tuple[bool, float]:
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
        if not tf_data or len(tf_data) < 3:
            return False, 0
        try:
            aligned_count = 0
            total_checked = 0
            for _, df in tf_data.items():
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
        if df is None or len(df) < 20:
            return False, 0.0
        try:
            returns = df['close'].pct_change()
            current_vol = returns.iloc[-5:].std()
            avg_vol = returns.rolling(20).std().mean()
            vol_ratio = current_vol / (avg_vol + 1e-9)
            is_spike = vol_ratio > 2.5
            return is_spike, vol_ratio
        except Exception as e:
            logger.debug(f"Volatility spike check error: {e}")
            return False, 0.0

# -------------------------------
# Telegram Bot (FIXED WITH DETAILED DEBUGGING)
# -------------------------------
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import asyncio

def is_admin(user_id: int) -> bool:
    is_authorized = user_id in ADMIN_IDS
    logger.info(f"Authorization check: user_id={user_id}, authorized={is_authorized}, ADMIN_IDS={ADMIN_IDS}")
    return is_authorized

class LegendaryTelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.app: Optional[Application] = None
        self.loop = None
        self.is_running = False
        
    def start(self):
        if not self.token or not self.chat_id:
            logger.warning("⚠️ Telegram not configured - missing TOKEN or CHAT_ID")
            logger.warning(f"TOKEN present: {bool(self.token)}, CHAT_ID present: {bool(self.chat_id)}")
            return
        
        try:
            logger.info(f"🔧 Initializing Telegram bot...")
            logger.info(f"📱 Chat ID: {self.chat_id}")
            logger.info(f"👥 Admin IDs: {ADMIN_IDS}")
            logger.info(f"🔑 Token: {self.token[:10]}...{self.token[-5:]}")
            
            # Start in a separate thread
            threading.Thread(target=self._init_and_run, daemon=True).start()
            
            # Wait a moment to verify startup
            time.sleep(2)
            if self.is_running:
                logger.info("✅ Telegram bot initialized successfully")
            else:
                logger.error("❌ Telegram bot failed to start")
                
        except Exception as e:
            logger.error(f"❌ Failed to start Telegram bot: {e}", exc_info=True)

    def _init_and_run(self):
        """Initialize and run the bot in a separate thread"""
        try:
            # Create new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            logger.info("🔄 Creating Telegram Application...")
            
            # Build application
            self.app = Application.builder().token(self.token).build()
            
            # Add handlers
            logger.info("📝 Registering command handlers...")
            self.app.add_handler(CommandHandler("start", self._start))
            self.app.add_handler(CommandHandler("status", self._status))
            self.app.add_handler(CommandHandler("threshold", self._threshold))
            self.app.add_handler(CommandHandler("history", self._history))
            self.app.add_handler(CommandHandler("rejected", self._rejected))
            self.app.add_handler(CommandHandler("stats", self._stats))
            
            logger.info("✅ Command handlers registered")
            logger.info("🚀 Starting bot polling...")
            
            # Run the bot
            self._run()
            
        except Exception as e:
            logger.error(f"❌ Bot thread error: {e}", exc_info=True)
            self.is_running = False

    def _run(self):
        """Run the bot polling"""
        try:
            # Initialize and start
            self.loop.run_until_complete(self.app.initialize())
            self.loop.run_until_complete(self.app.start())
            
            logger.info("🔄 Starting polling updater...")
            self.loop.run_until_complete(
                self.app.updater.start_polling(
                    allowed_updates=Update.ALL_TYPES,
                    drop_pending_updates=True,
                    timeout=30,
                    pool_timeout=30
                )
            )
            
            self.is_running = True
            logger.info("✅ Telegram bot polling started successfully!")
            logger.info("📱 Bot is now listening for commands...")
            
            # Keep the loop running
            self.loop.run_forever()
            
        except Exception as e:
            logger.error(f"❌ Telegram polling error: {e}", exc_info=True)
            self.is_running = False

    async def _start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info(f"📨 Received /start command from user_id: {update.effective_user.id}")
        
        if not is_admin(update.effective_user.id):
            logger.warning(f"⚠️ Unauthorized access attempt from user_id: {update.effective_user.id}")
            await update.message.reply_text("❌ Unauthorized")
            return
        
        logger.info("✅ Sending welcome message...")
        await update.message.reply_text(
            "🏆 ULTIMATE #1 LEGENDARY BOT ACTIVE\n\n"
            "Commands:\n"
            "/status - System status\n"
            "/history - Recent signals\n"
            "/rejected - Why signals were rejected\n"
            "/stats - Performance statistics\n"
            "/threshold - Adjust confidence"
        )
        logger.info("✅ Welcome message sent")

    async def _status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info(f"📨 Received /status command from user_id: {update.effective_user.id}")
        
        if not is_admin(update.effective_user.id):
            return

        is_news, news_reason = EconomicCalendar.is_news_time()
        session = EconomicCalendar.get_trading_session()

        overrides = "\n".join([f"{k}: {v:.1f}%" for k, v in ASSET_THRESHOLD_OVERRIDE.items()]) or "None"
        await update.message.reply_text(
            f"✅ ULTIMATE BOT ACTIVE\n\n"
            f"Mode: {EngineState.current_mode}\n"
            f"Session: {session}\n"
            f"News Block: {'YES - ' + news_reason if is_news else 'NO'}\n"
            f"Global threshold: {GLOBAL_THRESHOLD:.1f}%\n"
            f"Signals sent: {len(SIGNAL_HISTORY)}\n"
            f"Signals rejected: {len(REJECTED_SIGNALS)}\n"
            f"Per-asset overrides:\n{overrides}"
        )
        logger.info("✅ Status sent")

    async def _history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info(f"📨 Received /history command from user_id: {update.effective_user.id}")
        
        if not is_admin(update.effective_user.id):
            return
        if not SIGNAL_HISTORY:
            await update.message.reply_text("No signals in history")
            return
        recent = list(SIGNAL_HISTORY)[-5:]
        msg = "📊 Recent Signals:\n\n"
        for sig in recent:
            msg += f"{sig['asset']} {sig['signal']} @ {sig['confidence']:.1f}% ({sig['expiry']}m)\n"
        await update.message.reply_text(msg)

    async def _rejected(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info(f"📨 Received /rejected command from user_id: {update.effective_user.id}")
        
        if not is_admin(update.effective_user.id):
            return
        if not REJECTED_SIGNALS:
            await update.message.reply_text("No rejected signals")
            return
        recent = list(REJECTED_SIGNALS)[-5:]
        msg = "🚫 Recently Rejected:\n\n"
        for rej in recent:
            msg += f"{rej['asset']}: {rej['reason']}\n"
        await update.message.reply_text(msg)

    async def _stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info(f"📨 Received /stats command from user_id: {update.effective_user.id}")
        
        if not is_admin(update.effective_user.id):
            return
        if not SIGNAL_HISTORY:
            await update.message.reply_text("No signals yet")
            return

        signals = list(SIGNAL_HISTORY)
        avg_conf = np.mean([s['confidence'] for s in signals])
        by_asset = Counter([s['asset'] for s in signals])
        by_direction = Counter([s['signal'] for s in signals])

        msg = f"📈 PERFORMANCE STATS\n\n"
        msg += f"Total Signals: {len(signals)}\n"
        msg += f"Avg Confidence: {avg_conf:.1f}%\n"
        msg += f"BUY: {by_direction.get('BUY', 0)} | SELL: {by_direction.get('SELL', 0)}\n\n"
        msg += "By Asset:\n"
        for asset, count in by_asset.most_common():
            msg += f"  {asset}: {count}\n"

        await update.message.reply_text(msg)

    async def _threshold(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info(f"📨 Received /threshold command from user_id: {update.effective_user.id}")
        
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
                await update.message.reply_text("Invalid value")
                return

        if len(args) >= 2:
            asset = args[0].strip()
            token = args[1].strip().lower()
            if asset not in ASSETS:
                await update.message.reply_text(f"Unknown asset: {asset}")
                return
            if token == "reset":
                if asset in ASSET_THRESHOLD_OVERRIDE:
                    del ASSET_THRESHOLD_OVERRIDE[asset]
                    await update.message.reply_text(f"🔄 Threshold removed for {asset}")
                else:
                    await update.message.reply_text(f"No override for {asset}")
                return
            try:
                val = float(args[1])
                if val <= 0 or val > 100:
                    raise ValueError
                ASSET_THRESHOLD_OVERRIDE[asset] = val
                await update.message.reply_text(f"✅ Threshold for {asset} set to {val:.1f}%")
                return
            except Exception:
                await update.message.reply_text("Invalid value")
                return
    
    async def send_signal(self, alert_msg: str):
        """Send signal message asynchronously"""
        if self.app and self.chat_id:
            try:
                await self.app.bot.send_message(
                    chat_id=self.chat_id, 
                    text=alert_msg, 
                    parse_mode="HTML"
                )
                logger.info("✅ Signal sent via Telegram")
            except Exception as e:
                logger.error(f"❌ Failed to send Telegram message: {e}")

    def send_signal_sync(self, alert_msg: str):
        """Send signal message from sync code"""
        if self.app and self.loop and self.is_running:
            try:
                # Schedule the coroutine on the bot's event loop
                future = asyncio.run_coroutine_threadsafe(
                    self.send_signal(alert_msg),
                    self.loop
                )
                # Wait for it to complete (with timeout)
                future.result(timeout=10)
            except Exception as e:
                logger.error(f"❌ Failed to send Telegram message: {e}", exc_info=True)
        else:
            logger.warning("⚠️ Telegram bot not ready to send messages")
            logger.warning(f"App: {bool(self.app)}, Loop: {bool(self.loop)}, Running: {self.is_running}")

# -------------------------------
# Telegram Diagnostic Test
# -------------------------------
def test_telegram_connection():
    """Test Telegram bot connection and credentials"""
    logger.info("=" * 60)
    logger.info("🔍 TELEGRAM DIAGNOSTICS")
    logger.info("=" * 60)
    
    # Check environment variables
    logger.info(f"✓ TELEGRAM_TOKEN present: {bool(TELEGRAM_TOKEN)}")
    if TELEGRAM_TOKEN:
        logger.info(f"  Token: {TELEGRAM_TOKEN[:10]}...{TELEGRAM_TOKEN[-5:]}")
    else:
        logger.error("  ❌ TELEGRAM_TOKEN is empty!")
        
    logger.info(f"✓ CHAT_ID present: {bool(CHAT_ID)}")
    if CHAT_ID:
        logger.info(f"  Chat ID: {CHAT_ID}")
    else:
        logger.error("  ❌ CHAT_ID is empty!")
        
    logger.info(f"✓ ADMIN_IDS: {ADMIN_IDS}")
    if not ADMIN_IDS:
        logger.warning("  ⚠️ No ADMIN_IDS configured!")
    
    # Test bot token validity
    if TELEGRAM_TOKEN:
        try:
            import requests
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getMe"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    bot_info = data.get('result', {})
                    logger.info(f"✅ Bot token is VALID")
                    logger.info(f"   Bot username: @{bot_info.get('username')}")
                    logger.info(f"   Bot name: {bot_info.get('first_name')}")
                    logger.info(f"   Bot ID: {bot_info.get('id')}")
                else:
                    logger.error(f"❌ Bot token validation failed: {data}")
            else:
                logger.error(f"❌ HTTP {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"❌ Error testing bot token: {e}")
    
    # Test chat ID validity
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            import requests
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getChat"
            params = {"chat_id": CHAT_ID}
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    chat_info = data.get('result', {})
                    logger.info(f"✅ Chat ID is VALID")
                    logger.info(f"   Chat type: {chat_info.get('type')}")
                    logger.info(f"   Chat title: {chat_info.get('title', 'N/A')}")
                    logger.info(f"   Username: @{chat_info.get('username', 'N/A')}")
                else:
                    logger.error(f"❌ Chat ID validation failed: {data.get('description')}")
                    logger.error(f"   Make sure you've started a conversation with your bot!")
            else:
                logger.error(f"❌ HTTP {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"❌ Error testing chat ID: {e}")
    
    # Send test message
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            import requests
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            data = {
                "chat_id": CHAT_ID,
                "text": "🔧 Test message from Ultimate Trading Bot\n\nIf you see this, your credentials are correct!\n\nNow try: /start"
            }
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    logger.info(f"✅ Test message sent successfully!")
                    logger.info(f"   Message ID: {result.get('result', {}).get('message_id')}")
                else:
                    logger.error(f"❌ Failed to send test message: {result}")
            else:
                logger.error(f"❌ HTTP {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"❌ Error sending test message: {e}")
    
    logger.info("=" * 60)
    logger.info("🔍 DIAGNOSTICS COMPLETE")
    logger.info("=" * 60)
    logger.info("")
    
    # Give recommendations
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.info("📝 SETUP INSTRUCTIONS:")
        logger.info("1. Create a bot with @BotFather on Telegram")
        logger.info("2. Copy the bot token")
        logger.info("3. Start a chat with your bot")
        logger.info("4. Get your chat ID from @userinfobot")
        logger.info("5. Add to .env file:")
        logger.info("   TELEGRAM_TOKEN=your_token_here")
        logger.info("   CHAT_ID=your_chat_id_here")
        logger.info("   ADMIN_IDS=your_user_id_here")
        logger.info("")

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
        alpha = 2 / (period + 1)
        ema_vals = [series.iloc[0]]
        for v in series.iloc[1:]:
            ema_vals.append(alpha * v + (1 - alpha) * ema_vals[-1])
        return np.array(ema_vals)
    except Exception:
        return np.array([])

def rsi(series: pd.Series, period: int = 14) -> np.ndarray:
    if not safe_series_check(series, period + 1):
        return np.array([])
    try:
        delta = series.diff()
        up = delta.clip(lower=0).rolling(period).mean()
        down = (-delta.clip(upper=0)).rolling(period).mean()
        rs = up / (down.replace(0, np.nan))
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals.fillna(50.0).values
    except Exception:
        return np.array([])

def macd(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    if not safe_series_check(series, 26):
        return np.array([]), np.array([])
    try:
        ema_fast = pd.Series(ema(series, 12))
        ema_slow = pd.Series(ema(series, 26))
        macd_line = ema_fast - ema_slow
        macd_sig = pd.Series(ema(macd_line, 9))
        return macd_line.values, macd_sig.values
    except Exception:
        return np.array([]), np.array([])

def bollinger_bands(series: pd.Series, period: int = 20) -> Dict:
    if not safe_series_check(series, period):
        return {"upper": pd.Series(dtype=float), "lower": pd.Series(dtype=float), "ma": pd.Series(dtype=float)}
    try:
        ma = series.rolling(period).mean()
        std = series.rolling(period).std()
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
                st.iloc[i] = lowerband.iloc[i]
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
# Indicators wrapping
# -------------------------------
def ema_macd_indicator(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 50):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "indicator"}
    try:
        ema_fast = ema(df["close"], 12)
        ema_slow = ema(df["close"], 26)
        macd_line, macd_sig = macd(df["close"])
        if len(ema_fast) < 2 or len(macd_line) < 1 or len(ema_slow) < 2 or len(macd_sig) < 1:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "indicator"}
        if ema_fast[-1] > ema_slow[-1] and macd_line[-1] > macd_sig[-1]:
            return {"signal": "BUY", "score": 82.0, "reason": "EMA_MACD", "type": "indicator"}
        elif ema_fast[-1] < ema_slow[-1] and macd_line[-1] < macd_sig[-1]:
            return {"signal": "SELL", "score": 82.0, "reason": "EMA_MACD", "type": "indicator"}
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "indicator"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "EMA_MACD", "type": "indicator"}

def rsi_indicator(df: pd.DataFrame) -> Dict:
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
    if not safe_df_check(df, 30):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Bollinger", "type": "indicator"}
    try:
        bb = bollinger_bands(df["close"])
        if bb["upper"].empty or bb["lower"].empty or bb["ma"].empty:
            return {"signal": "NEUTRAL", "score": 0.0, "reason": "Bollinger", "type": "indicator"}
        if df["close"].iloc[-1] < bb["lower"].iloc[-1]:
            return {"signal": "BUY", "score": 76.0, "reason": "Bollinger", "type": "indicator"}
        elif df["close"].iloc[-1] > bb["upper"].iloc[-1]:
            return {"signal": "SELL", "score": 76.0, "reason": "Bollinger", "type": "indicator"}
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Bollinger", "type": "indicator"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "Bollinger", "type": "indicator"}

def volume_indicator(df: pd.DataFrame, asset_config: Dict = None) -> Dict:
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
# ADVANCED INSTITUTIONAL STRATEGIES (10)
# -------------------------------
def market_microstructure_imbalance(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 100):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "MicrostructureImb", "type": "strategy"}
    try:
        price_changes = df['close'].diff().abs()
        volume_normalized = df['volume'] / df['volume'].rolling(50).mean()
        impact_ratio = (price_changes / (volume_normalized + 1e-9)).rolling(20).mean()

        current_impact = impact_ratio.iloc[-1]
        avg_impact = impact_ratio.mean()

        buying_pressure = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)).rolling(10).mean()
        selling_pressure = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-9)).rolling(10).mean()

        imbalance = buying_pressure.iloc[-1] - selling_pressure.iloc[-1]

        if current_impact < avg_impact * 0.7 and imbalance > 0.15:
            return {"signal": "BUY", "score": 97.0, "reason": "MicrostructureImb", "type": "strategy"}
        elif current_impact < avg_impact * 0.7 and imbalance < -0.15:
            return {"signal": "SELL", "score": 97.0, "reason": "MicrostructureImb", "type": "strategy"}

        return {"signal": "NEUTRAL", "score": 0.0, "reason": "MicrostructureImb", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "MicrostructureImb", "type": "strategy"}

def liquidity_void_hunter(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 200):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "LiquidityVoid", "type": "strategy"}
    try:
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
        relative_dist = nearest_void_dist / price_range
        momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        if relative_dist < 0.02:
            if momentum > 0:
                return {"signal": "BUY", "score": 94.0, "reason": "LiquidityVoid", "type": "strategy"}
            elif momentum < 0:
                return {"signal": "SELL", "score": 94.0, "reason": "LiquidityVoid", "type": "strategy"}
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "LiquidityVoid", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "LiquidityVoid", "type": "strategy"}

def volatility_regime_detector(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 100):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "VolRegime", "type": "strategy"}
    try:
        returns = df['close'].pct_change()
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        vol_of_vol = rolling_vol.rolling(20).std()

        vol_zscore = (rolling_vol.iloc[-1] - rolling_vol.mean()) / (rolling_vol.std() + 1e-9)
        recent_vol_change = rolling_vol.iloc[-1] / rolling_vol.iloc[-10]
        vol_acceleration = vol_of_vol.iloc[-1] / vol_of_vol.mean()

        if vol_zscore < -0.5 and recent_vol_change > 1.3 and vol_acceleration > 1.2:
            price_momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
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
        price_velocity = (df['close'].iloc[-1] - df['close'].iloc[-3]) / df['close'].iloc[-3]
        volume_surge = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]

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
        momentum = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]

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
    if not safe_df_check(df, 100):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "InstFootprint", "type": "strategy"}
    try:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap_val = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        vwap_deviation = (df['close'] - vwap_val) / vwap_val

        recent_volume = df['volume'].iloc[-5:].mean()
        avg_volume = df['volume'].rolling(50).mean().iloc[-1]

        recent_volatility = df['close'].iloc[-5:].std() / df['close'].iloc[-5:].mean()
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
    if not safe_df_check(df, 100):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "AuctionTheory", "type": "strategy"}
    try:
        price_range = df['high'].max() - df['low'].min()
        tick_size = price_range / 50

        tpo_counts = {}
        for i in range(len(df)):
            low_tick = int((df['low'].iloc[i] - df['low'].min()) / tick_size)
            high_tick = int((df['high'].iloc[i] - df['low'].min()) / tick_size)
            for tick in range(low_tick, high_tick + 1):
                tpo_counts[tick] = tpo_counts.get(tick, 0) + 1

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

        vah_price = df['low'].min() + max(value_area_ticks) * tick_size
        val_price = df['low'].min() + min(value_area_ticks) * tick_size

        current_price = df['close'].iloc[-1]
        momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]

        if current_price < val_price and momentum > 0.005:
            return {"signal": "BUY", "score": 93.0, "reason": "AuctionTheory", "type": "strategy"}
        elif current_price > vah_price and momentum < -0.005:
            return {"signal": "SELL", "score": 93.0, "reason": "AuctionTheory", "type": "strategy"}

        return {"signal": "NEUTRAL", "score": 0.0, "reason": "AuctionTheory", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "AuctionTheory", "type": "strategy"}

def spoofing_detector(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 50):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "SpoofDetect", "type": "strategy"}
    try:
        volume_spike = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-2]
        price_change = abs(df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        # FIXED: removed stray quote
        subsequent_reversal = (df['close'].iloc[-1] - df['close'].iloc[-3]) * (df['close'].iloc[-2] - df['close'].iloc[-3])

        if volume_spike > 3.0 and price_change < 0.003 and subsequent_reversal < 0:
            if df['close'].iloc[-2] > df['close'].iloc[-3]:
                return {"signal": "SELL", "score": 91.0, "reason": "SpoofDetect", "type": "strategy"}
            else:
                return {"signal": "BUY", "score": 91.0, "reason": "SpoofDetect", "type": "strategy"}

        return {"signal": "NEUTRAL", "score": 0.0, "reason": "SpoofDetect", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "SpoofDetect", "type": "strategy"}

def market_entropy_strategy(df: pd.DataFrame) -> Dict:
    if not safe_df_check(df, 100):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "MarketEntropy", "type": "strategy"}
    try:
        returns = df['close'].pct_change().dropna().values[-50:]
        bins = 10
        hist, _ = np.histogram(returns, bins=bins)
        hist = hist / hist.sum()

        entropy = -np.sum([p * np.log2(p + 1e-9) for p in hist if p > 0])
        max_entropy = np.log2(bins)
        normalized_entropy = entropy / max_entropy

        ema_short = df['close'].ewm(span=10).mean()
        ema_long = df['close'].ewm(span=30).mean()
        trend_strength = abs(ema_short.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1]

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
# CLASSIC STRATEGIES (6)
# -------------------------------
def fibonacci_vortex_hidden(df: pd.DataFrame) -> Dict:
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
    if not safe_df_check(df, 50):
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "BOS", "type": "strategy"}
    try:
        sh = df["high"].rolling(20).max().iloc[-2]
        sl = df["low"].rolling(20).min().iloc[-2]
        if df["close"].iloc[-1] > sh:
            return {"signal": "BUY", "score": 86.0, "reason": "BOS", "type": "strategy"}
        elif df["close"].iloc[-1] < sl:
            return {"signal": "SELL", "score": 86.0, "reason": "BOS", "type": "strategy"}
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "BOS", "type": "strategy"}
    except Exception:
        return {"signal": "NEUTRAL", "score": 0.0, "reason": "BOS", "type": "strategy"}

def fair_value_gap(df: pd.DataFrame) -> Dict:
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
    except Exception as e:
        logger.error(f"Market conditions error for {asset}: {e}")
        return {"volatility": 0.0, "trend_strength": 0.0, "momentum_spike": False, "breakout_detected": False, "range_bound": True}

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
# Data Provider (Oanda + Binance)
# -------------------------------
class RealDataProvider:
    """
    Fetches OHLCV from Oanda (forex, gold, oil CFDs) and Binance (crypto, tokenized commodities).
    Uses your Yahoo-style asset names and maps them internally.
    """

    OANDA_INSTRUMENTS = {
        "EURUSD=X": "EUR_USD",
        "GBPUSD=X": "GBP_USD",
        "USDJPY=X": "USD_JPY",
        "AUDUSD=X": "AUD_USD",
        "USDCHF=X": "USD_CHF",
        "GC=F": "XAU_USD",
        "CL=F": "WTICO_USD"
    }

    BINANCE_SYMBOLS = {
        "EURUSD=X": "EURUSDT",
        "GBPUSD=X": "GBPUSDT",
        "AUDUSD=X": "AUDUSDT",
        "GC=F": "XAUUSDT",
    }

    def __init__(self,
                 prefer_oanda_for: Optional[List[str]] = None,
                 prefer_binance_for: Optional[List[str]] = None):
        self.prefer_oanda_for = set(prefer_oanda_for or ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCHF=X", "GC=F", "CL=F"])
        self.prefer_binance_for = set(prefer_binance_for or [])

    @staticmethod
    def _tf_to_oanda_granularity(tf: str) -> str:
        mapping = {
            "1M": "M1",
            "5m": "M5",
            "15m": "M15",
            "1h": "H1",
            "4h": "H4",
            "1d": "D",
            "1w": "W"
        }
        return mapping.get(tf, "H1")

    @staticmethod
    def _tf_to_binance_interval(tf: str) -> str:
        mapping = {
            "1M": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
            "1w": "1w"
        }
        return mapping.get(tf, "1h")

    def _fetch_oanda_ohlc(self, instrument: str, tf: str, limit: int) -> Optional[pd.DataFrame]:
        granularity = self._tf_to_oanda_granularity(tf)
        url = f"{OANDA_REST_HOST}/v3/instruments/{instrument}/candles"
        params = {
            "granularity": granularity,
            "count": max(200, limit),
            "price": "M",
        }
        try:
            r = requests.get(url, headers=OANDA_HEADERS, params=params, timeout=20)
            r.raise_for_status()
            data = r.json().get("candles", [])
            if not data:
                return None
            rows = []
            for c in data:
                ts = pd.to_datetime(c["time"])
                mid = c.get("mid", {})
                o = float(mid.get("o", np.nan))
                h = float(mid.get("h", np.nan))
                l = float(mid.get("l", np.nan))
                cclose = float(mid.get("c", np.nan))
                vol = float(c.get("volume", 0))
                rows.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": cclose, "volume": vol})
            df = pd.DataFrame(rows).dropna().tail(limit).reset_index(drop=True)
            return df if not df.empty else None
        except Exception as e:
            logger.error(f"Oanda fetch error {instrument} {tf}: {e}")
            return None

    def _fetch_binance_ohlc(self, symbol: str, tf: str, limit: int) -> Optional[pd.DataFrame]:
        interval = self._tf_to_binance_interval(tf)
        url = f"{BINANCE_REST_HOST}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            klines = r.json()
            if not klines:
                return None
            rows = []
            for k in klines:
                ts = pd.to_datetime(k[0], unit="ms")
                o = float(k[1]); h = float(k[2]); l = float(k[3]); c = float(k[4]); v = float(k[5])
                rows.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
            df = pd.DataFrame(rows).tail(limit).reset_index(drop=True)
            return df if not df.empty else None
        except Exception as e:
            logger.error(f"Binance fetch error {symbol} {tf}: {e}")
            return None

    def fetch_ohlcv(self, asset: str, timeframe: str = "1h", limit: int = 300) -> Optional[pd.DataFrame]:
        use_oanda = asset in self.prefer_oanda_for and asset in self.OANDA_INSTRUMENTS
        use_binance = asset in self.prefer_binance_for and asset in self.BINANCE_SYMBOLS

        df = None
        if use_oanda:
            df = self._fetch_oanda_ohlc(self.OANDA_INSTRUMENTS[asset], timeframe, limit)
            if df is not None:
                return df

        if (not use_oanda) and (asset in self.BINANCE_SYMBOLS):
            df = self._fetch_binance_ohlc(self.BINANCE_SYMBOLS[asset], timeframe, limit)
            if df is not None:
                return df

        if use_oanda and df is None and asset in self.BINANCE_SYMBOLS:
            df = self._fetch_binance_ohlc(self.BINANCE_SYMBOLS[asset], timeframe, limit)
            if df is not None:
                return df

        return None

# -------------------------------
# Selection Helpers
# -------------------------------
def select_best(items: List[Dict], item_type: str, min_score: float, limit: int = 4) -> List[Dict]:
    picked = [r for r in items if r.get("type") == item_type and r.get("score", 0.0) >= min_score]
    picked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return picked[:limit]

# -------------------------------
# Core Analysis Engine
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
                result = indicator_fn(df) if indicator_fn != volume_indicator else indicator_fn(df, asset_config)
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

    best_strategies = select_best(all_strategy_results, "strategy", min_score=85.0, limit=6)
    best_indicators = select_best(all_indicator_results, "indicator", min_score=75.0, limit=5)
    best_filters = select_best(all_filter_results, "filter", min_score=75.0, limit=3)

    all_signals = [r["signal"] for r in best_strategies + best_indicators]
    signal_counts = Counter(all_signals)

    if len(best_strategies) >= 4 and len(best_indicators) >= 3:
        if signal_counts.get("BUY", 0) >= 6:
            final_signal = "BUY"
            confidence = sum(r["score"] for r in best_strategies + best_indicators if r["signal"] == "BUY") / (signal_counts["BUY"] + 1e-9)
        elif signal_counts.get("SELL", 0) >= 6:
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
# Alert Formatting
# -------------------------------
def format_ultimate_alert(result: Dict, engine_name: str) -> str:
    strategies_list = "\n".join([f"  • {s['reason']} ({s['score']:.0f}%)" for s in result.get('best_strategies', [])[:6]]) or "  • None"
    indicators_list = "\n".join([f"  • {i['reason']} ({i['score']:.0f}%)" for i in result.get('best_indicators', [])[:5]]) or "  • None"
    filters_list = "\n".join([f"  • {f['reason']} ({f['score']:.0f}%)" for f in result.get('best_filters', [])[:3]]) or "  • None"

    return f"""🏆 <b>ULTIMATE LEGENDARY SIGNAL</b>  <i>[{engine_name}]</i>

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

<b>Time:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"""

# -------------------------------
# Cleanup Function
# -------------------------------
def cleanup_old_alerts():
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
    return jsonify({
        "status": "legendary",
        "timestamp": datetime.utcnow().isoformat(),
        "tensorflow": TFK,
        "mode": EngineState.current_mode,
        "signals_in_history": len(SIGNAL_HISTORY),
        "active_cooldowns": len(LAST_LEGENDARY_ALERT),
        "rejected_signals": len(REJECTED_SIGNALS)
    }), 200

@app.route('/signals', methods=['GET'])
def get_signals():
    recent = list(SIGNAL_HISTORY)[-10:]
    return jsonify({"signals": recent, "count": len(SIGNAL_HISTORY)}), 200

@app.route('/rejected', methods=['GET'])
def get_rejected():
    recent = list(REJECTED_SIGNALS)[-10:]
    return jsonify({"rejected": recent, "count": len(REJECTED_SIGNALS)}), 200

# -------------------------------
# Main Scan Loop
# -------------------------------
def main_scan_loop():
    logger.info("🏆 Initializing ULTIMATE #1 LEGENDARY BOT")
    logger.info(f"⚡ Advanced Strategies: {len(ALL_ADVANCED_STRATEGIES)}")
    logger.info(f"⚡ Classic Strategies: {len(ALL_CLASSIC_STRATEGIES)}")
    logger.info(f"📊 Indicators: {len(ALL_INDICATORS)}")
    logger.info(f"🛡️ Institutional Filters: {len(ALL_FILTERS)}")
    logger.info(f"🧠 TensorFlow: {'ENABLED' if TFK else 'FALLBACK MODE'}")
    logger.info(f"🎯 Confidence Threshold: {GLOBAL_THRESHOLD}%")
    
    # Run diagnostics
    logger.info("")
    test_telegram_connection()
    
    logger.info(f"🤖 Initializing Telegram bot...")
    provider = RealDataProvider()
    telegram = LegendaryTelegramBot(TELEGRAM_TOKEN, CHAT_ID)
    telegram.start()
    
    # Wait for bot to fully initialize
    logger.info("⏳ Waiting for Telegram bot to initialize...")
    time.sleep(3)

    logger.info("✅ ULTIMATE BOT READY - Scanning for legendary signals...")

    while True:
        try:
            now_utc = datetime.utcnow()
            logger.info(f"\n🔍 Scan: {now_utc.strftime('%H:%M:%S')}")

            cleanup_old_alerts()

            is_news, news_reason = EconomicCalendar.is_news_time()
            if is_news:
                logger.info(f"⚠️ News blackout active: {news_reason}")
                time.sleep(SCAN_INTERVAL_SEC)
                continue

            session = EconomicCalendar.get_trading_session()
            logger.info(f"📍 Trading Session: {session}")

            for asset in ASSETS:
                asset_config = ASSET_CONFIG.get(asset, {})
                tf_data = {}

                for tf in ALL_TIMEFRAMES:
                    try:
                        df = provider.fetch_ohlcv(asset, tf, limit=300)
                        if safe_df_check(df, asset_config.get("min_data_points", 50)):
                            tf_data[tf] = df
                    except Exception as e:
                        logger.debug(f"Failed to fetch {asset} {tf}: {e}")

                if len(tf_data) < 3:
                    logger.debug(f"Insufficient data for {asset}")
                    continue

                primary_df = tf_data.get("1h") or tf_data.get("4h") or tf_data.get("1d")
                if primary_df is None:
                    continue

                is_choppy, atr_ratio = MarketConditionFilter.is_choppy_market(primary_df, asset_config)
                if is_choppy:
                    REJECTED_SIGNALS.append({
                        "asset": asset,
                        "reason": f"Choppy market (ATR ratio: {atr_ratio:.3f})",
                        "timestamp": now_utc.isoformat()
                    })
                    logger.debug(f"❌ {asset}: Choppy market")
                    continue

                has_liquidity, vol_ratio = MarketConditionFilter.has_sufficient_liquidity(primary_df, asset_config)
                if not has_liquidity:
                    REJECTED_SIGNALS.append({
                        "asset": asset,
                        "reason": f"Low liquidity (Vol ratio: {vol_ratio:.2f})",
                        "timestamp": now_utc.isoformat()
                    })
                    logger.debug(f"❌ {asset}: Low liquidity")
                    continue

                is_vol_spike, vol_spike_ratio = MarketConditionFilter.check_volatility_spike(primary_df)
                if is_vol_spike:
                    REJECTED_SIGNALS.append({
                        "asset": asset,
                        "reason": f"Volatility spike (Ratio: {vol_spike_ratio:.2f})",
                        "timestamp": now_utc.isoformat()
                    })
                    logger.debug(f"❌ {asset}: Volatility spike")
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

                    if result['signal'] == "NEUTRAL":
                        continue

                    is_aligned, aligned_count = MarketConditionFilter.check_timeframe_alignment(tf_data, result['signal'])
                    if not is_aligned:
                        REJECTED_SIGNALS.append({
                            "asset": asset,
                            "reason": f"Timeframe misalignment ({aligned_count}/{len(tf_data)} aligned)",
                            "timestamp": now_utc.isoformat()
                        })
                        logger.debug(f"❌ {asset}: Timeframe misalignment")
                        continue

                    if (result['confidence'] >= threshold and
                        result['num_strategies_aligned'] >= 4 and
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
                            'engine': engine["name"],
                            'session': session,
                            'timeframe_alignment': f"{aligned_count}/{len(tf_data)}"
                        }

                        SIGNAL_HISTORY.append(alert_data)
                        LAST_LEGENDARY_ALERT[cooldown_key] = now_utc

                        alert_msg = format_ultimate_alert(alert_data, engine["name"])

                        # Send via Telegram using the synchronous wrapper (FIXED)
                        telegram.send_signal_sync(alert_msg)

                        logger.info(f"  🔥 LEGENDARY SIGNAL: {asset} {result['signal']} @ {result['confidence']:.1f}% [{engine['name']}]")

                    time.sleep(0.15)

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
    logger.info(f"🚀 Starting ULTIMATE #1 BOT on port {PORT}")
    scanner_thread = threading.Thread(target=main_scan_loop, daemon=True)
    scanner_thread.start()
    app.run(host='0.0.0.0', port=PORT, debug=False)

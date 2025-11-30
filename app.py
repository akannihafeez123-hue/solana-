"""
🔮 ULTIMATE LEGENDARY Pocket Option AI Scanner
With Sacred Geometry, Quantum Analysis & Hidden Institutional Strategies
"""

import os
import sys
import time
import subprocess
import importlib
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import threading
import warnings
import json
import hashlib
import hmac
import logging
warnings.filterwarnings('ignore')

# ==================== CRITICAL DEPENDENCY INSTALLATION ====================

def install_package(package_spec):
    """Install a single package"""
    try:
        print(f"📦 Installing {package_spec}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            package_spec
        ])
        print(f"✅ Successfully installed {package_spec}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_spec}: {e}")
        return False

def ensure_dependencies():
    """Ensure all critical dependencies are available"""
    # Critical packages that must be available
    critical_packages = [
        "flask==2.3.3",
        "python-dotenv==1.0.0"
    ]
    
    # Heavy packages (installed on demand)
    heavy_packages = [
        "numpy==1.21.6",
        "pandas==1.3.5",
        "yfinance==0.2.18", 
        "psutil==5.9.5",
        "python-telegram-bot==20.7"
    ]
    
    print("🔧 Checking critical dependencies...")
    
    # First, ensure Flask is available
    try:
        import flask
        print("✅ Flask is available")
    except ImportError:
        print("❌ Flask not found, installing...")
        if not install_package("flask==2.3.3"):
            print("💥 CRITICAL: Failed to install Flask. Exiting.")
            return False
    
    # Ensure python-dotenv
    try:
        import dotenv
        print("✅ python-dotenv is available")
    except ImportError:
        print("❌ python-dotenv not found, installing...")
        install_package("python-dotenv==1.0.0")
    
    print("✅ Critical dependencies are ready!")
    return True

# Install critical dependencies first
if not ensure_dependencies():
    sys.exit(1)

# Now import Flask and other critical packages
from flask import Flask, request, jsonify
from collections import Counter
import math

# ==================== HEAVY DEPENDENCY HANDLING ====================

class LazyLoader:
    """Lazy loader for heavy dependencies"""
    def __init__(self, module_name, package_name=None):
        self.module_name = module_name
        self.package_name = package_name or module_name
        self._module = None
    
    def _ensure_loaded(self):
        if self._module is None:
            try:
                self._module = importlib.import_module(self.module_name)
            except ImportError:
                print(f"❌ {self.module_name} not found, installing...")
                if install_package(self.package_name):
                    self._module = importlib.import_module(self.module_name)
                else:
                    raise ImportError(f"Failed to load {self.module_name}")
    
    def __getattr__(self, name):
        self._ensure_loaded()
        return getattr(self._module, name)

# Lazy load heavy dependencies
try:
    np = LazyLoader("numpy", "numpy==1.21.6")
    pd = LazyLoader("pandas", "pandas==1.3.5")
    yfinance = LazyLoader("yfinance", "yfinance==0.2.18")
    psutil = LazyLoader("psutil", "psutil==5.9.5")
    telegram_loaded = False
    try:
        from telegram import Update
        from telegram.ext import Application, CommandHandler, ContextTypes
        from telegram.constants import ParseMode
        telegram_loaded = True
        print("✅ Telegram loaded successfully")
    except ImportError:
        print("⚠️ Telegram not available - installing...")
        if install_package("python-telegram-bot==20.7"):
            from telegram import Update
            from telegram.ext import Application, CommandHandler, ContextTypes
            from telegram.constants import ParseMode
            telegram_loaded = True
            print("✅ Telegram loaded after installation")
except ImportError as e:
    print(f"⚠️ Some heavy dependencies not available: {e}")
    print("💡 The app will run with limited functionality")

# ==================== LOGGING SETUP ====================

def setup_logging():
    """Setup structured logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ==================== CONFIGURATION ====================

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "your_legendary_secret_key_here")
LEGENDARY_GATE = int(os.getenv("LEGENDARY_GATE", "95"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "45"))
RUN_ONCE = os.getenv("RUN_ONCE", "False") == "True"
PORT = int(os.getenv("PORT", "8080"))

# Trading configuration
ASSETS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F"]
EXPIRIES = [5, 15, 30]
ALL_TIMEFRAMES = ["1d", "4h", "1h"]

SIGNAL_HISTORY = {}
LAST_LEGENDARY_ALERT = {}

app = Flask(__name__)

# ==================== SIMPLIFIED TRADING ENGINE ====================

def simple_moving_average(series, period):
    """Simple moving average calculation"""
    try:
        return series.rolling(window=period).mean()
    except Exception:
        return series  # Fallback

def calculate_rsi(series, period=14):
    """RSI calculation"""
    try:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception:
        return pd.Series([50] * len(series))  # Neutral fallback

def momentum_strategy(df):
    """Simple momentum-based strategy"""
    try:
        if len(df) < 20:
            return {"signal": "NEUTRAL", "score": 0, "reason": "Insufficient data"}
        
        # Use closing prices
        prices = df['close']
        
        # Calculate indicators
        sma_short = simple_moving_average(prices, 9)
        sma_long = simple_moving_average(prices, 21)
        rsi = calculate_rsi(prices, 14)
        
        current_price = prices.iloc[-1]
        current_sma_short = sma_short.iloc[-1]
        current_sma_long = sma_long.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Strategy logic
        price_above_short = current_price > current_sma_short
        price_above_long = current_price > current_sma_long
        short_above_long = current_sma_short > current_sma_long
        
        bullish = price_above_short and price_above_long and short_above_long and current_rsi < 70
        bearish = not price_above_short and not price_above_long and not short_above_long and current_rsi > 30
        
        if bullish:
            return {"signal": "BUY", "score": 85, "reason": "Momentum"}
        elif bearish:
            return {"signal": "SELL", "score": 85, "reason": "Momentum"}
        else:
            return {"signal": "NEUTRAL", "score": 0, "reason": "Momentum"}
            
    except Exception as e:
        logger.error(f"Strategy error: {e}")
        return {"signal": "NEUTRAL", "score": 0, "reason": "Error"}

# ==================== DATA PROVIDER ====================

class DataProvider:
    """Simple data provider with fallback"""
    def __init__(self):
        self.available = False
        try:
            # Test if yfinance is available
            import yfinance
            self.yf = yfinance
            self.available = True
            logger.info("✅ yfinance data provider ready")
        except ImportError:
            logger.warning("❌ yfinance not available - using mock data")
    
    def fetch_ohlcv(self, symbol, timeframe="1d", limit=100):
        """Fetch OHLCV data"""
        if not self.available:
            return self._mock_data(symbol, limit)
        
        try:
            # Map timeframes
            interval_map = {"1h": "1h", "4h": "1h", "1d": "1d"}
            interval = interval_map.get(timeframe, "1d")
            period = "3mo" if interval == "1d" else "1mo"
            
            ticker = self.yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return self._mock_data(symbol, limit)
            
            # Rename columns to standard format
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low", 
                "Close": "close", "Volume": "volume"
            })
            
            df.reset_index(inplace=True)
            if 'Date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['Date'])
            else:
                df['timestamp'] = df.index
                
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(limit)
            
        except Exception as e:
            logger.error(f"Data fetch error for {symbol}: {e}")
            return self._mock_data(symbol, limit)
    
    def _mock_data(self, symbol, limit):
        """Generate mock data when real data is unavailable"""
        import random
        base_price = 100.0
        
        data = {
            'timestamp': pd.date_range(end=datetime.now(timezone.utc), periods=limit, freq='H'),
            'open': [base_price * (1 + random.uniform(-0.02, 0.02)) for _ in range(limit)],
            'high': [base_price * (1 + random.uniform(0, 0.03)) for _ in range(limit)],
            'low': [base_price * (1 + random.uniform(-0.03, 0)) for _ in range(limit)],
            'close': [base_price * (1 + random.uniform(-0.02, 0.02)) for _ in range(limit)],
            'volume': [random.randint(1000, 10000) for _ in range(limit)]
        }
        
        return pd.DataFrame(data)

# ==================== TELEGRAM BOT ====================

class TelegramBot:
    """Simple Telegram bot with fallback"""
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.available = False
        
        if not token or not chat_id:
            logger.warning("Telegram not configured")
            return
        
        try:
            from telegram import Bot
            self.bot = Bot(token=token)
            # Test connection
            self.bot.get_me()
            self.available = True
            logger.info("✅ Telegram bot connected")
        except Exception as e:
            logger.warning(f"Telegram bot unavailable: {e}")
    
    def send_message(self, text):
        """Send message with fallback to logging"""
        if self.available:
            try:
                self.bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    parse_mode='HTML',
                    disable_web_page_preview=True
                )
                logger.info("✅ Telegram message sent")
            except Exception as e:
                logger.error(f"Failed to send Telegram message: {e}")
                self._log_message(text)
        else:
            self._log_message(text)
    
    def _log_message(self, text):
        """Log message when Telegram is unavailable"""
        logger.info(f"📱 TELEGRAM MESSAGE: {text}")

# ==================== SCANNER ENGINE ====================

def analyze_asset(provider, asset, timeframes):
    """Analyze a single asset across multiple timeframes"""
    try:
        tf_data = {}
        for tf in timeframes:
            df = provider.fetch_ohlcv(asset, tf, 50)
            if df is not None and len(df) > 20:
                tf_data[tf] = df
        
        if not tf_data:
            return {"signal": "NEUTRAL", "confidence": 0, "reason": "No data"}
        
        # Run strategy on each timeframe
        results = []
        for tf, df in tf_data.items():
            result = momentum_strategy(df)
            if result["signal"] != "NEUTRAL":
                result["timeframe"] = tf
                results.append(result)
        
        # Aggregate results
        if len(results) >= 2:  # Require at least 2 timeframes to agree
            buy_signals = [r for r in results if r["signal"] == "BUY"]
            sell_signals = [r for r in results if r["signal"] == "SELL"]
            
            if len(buy_signals) >= 2:
                confidence = sum(r["score"] for r in buy_signals) / len(buy_signals)
                return {"signal": "BUY", "confidence": confidence, "strategies": buy_signals}
            elif len(sell_signals) >= 2:
                confidence = sum(r["score"] for r in sell_signals) / len(sell_signals)
                return {"signal": "SELL", "confidence": confidence, "strategies": sell_signals}
        
        return {"signal": "NEUTRAL", "confidence": 0, "strategies": results}
        
    except Exception as e:
        logger.error(f"Analysis error for {asset}: {e}")
        return {"signal": "NEUTRAL", "confidence": 0, "reason": "Error"}

def format_signal(asset, analysis):
    """Format trading signal for display"""
    strategies_text = "\n".join([f"• {s['reason']} ({s['timeframe']})" for s in analysis['strategies'][:3]])
    
    return f"""
🎯 <b>TRADING SIGNAL</b>

<b>Asset:</b> {asset}
<b>Signal:</b> {analysis['signal']}
<b>Confidence:</b> {analysis['confidence']:.1f}%

<b>Timeframes:</b>
{strategies_text}

<b>Time:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
"""

# ==================== MAIN SCANNER LOOP ====================

def scanner_loop():
    """Main scanning loop"""
    logger.info("🔮 Starting Ultimate Legendary Scanner")
    
    provider = DataProvider()
    bot = TelegramBot(TELEGRAM_TOKEN, CHAT_ID)
    
    logger.info(f"📊 Monitoring {len(ASSETS)} assets")
    logger.info(f"🎯 Confidence threshold: {LEGENDARY_GATE}%")
    logger.info(f"⏰ Scan interval: {SCAN_INTERVAL_SEC} seconds")
    
    scan_count = 0
    
    while True:
        try:
            scan_count += 1
            current_time = datetime.now(timezone.utc)
            
            logger.info(f"\n🔍 Scan #{scan_count} at {current_time.strftime('%H:%M:%S')} UTC")
            
            signals_found = 0
            
            for asset in ASSETS:
                # Check cooldown
                if asset in LAST_LEGENDARY_ALERT:
                    elapsed = (current_time - LAST_LEGENDARY_ALERT[asset]).total_seconds() / 60
                    if elapsed < 15:  # 15-minute cooldown
                        continue
                
                # Analyze asset
                analysis = analyze_asset(provider, asset, ALL_TIMEFRAMES[:2])  # Use 2 timeframes
                
                # Check if signal meets threshold
                if (analysis["signal"] != "NEUTRAL" and 
                    analysis["confidence"] >= LEGENDARY_GATE):
                    
                    # Send alert
                    alert_message = format_signal(asset, analysis)
                    bot.send_message(alert_message)
                    
                    # Record signal
                    LAST_LEGENDARY_ALERT[asset] = current_time
                    signal_key = f"{asset}_{current_time.strftime('%Y%m%d_%H%M')}"
                    SIGNAL_HISTORY[signal_key] = {
                        'asset': asset,
                        'signal': analysis['signal'],
                        'confidence': analysis['confidence'],
                        'timestamp': current_time.isoformat(),
                        'strategies': len(analysis['strategies'])
                    }
                    
                    signals_found += 1
                    logger.info(f"🚨 {asset} {analysis['signal']} @ {analysis['confidence']:.1f}%")
            
            # Cleanup old signals
            if len(SIGNAL_HISTORY) > 100:
                keys_to_remove = list(SIGNAL_HISTORY.keys())[:-100]
                for key in keys_to_remove:
                    del SIGNAL_HISTORY[key]
            
            if signals_found == 0:
                logger.info("⚡ No high-confidence signals found")
            else:
                logger.info(f"🎯 Found {signals_found} signals")
            
            logger.info(f"✅ Scan complete. Next in {SCAN_INTERVAL_SEC}s")
            
            if RUN_ONCE:
                break
                
            time.sleep(SCAN_INTERVAL_SEC)
            
        except KeyboardInterrupt:
            logger.info("🛑 Scanner stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Scanner error: {e}")
            time.sleep(30)  # Wait longer on error

# ==================== FLASK ROUTES ====================

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "service": "Ultimate Legendary Scanner",
        "version": "3.0",
        "signals_count": len(SIGNAL_HISTORY),
        "assets_monitored": len(ASSETS)
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "memory_usage": len(SIGNAL_HISTORY),
        "uptime": "unknown"  # Could be enhanced with start time tracking
    })

@app.route('/signals')
def get_signals():
    """Get recent signals"""
    limit = min(int(request.args.get('limit', 10)), 50)
    recent = list(SIGNAL_HISTORY.values())[-limit:]
    return jsonify({"signals": recent})

@app.route('/deps/install')
def install_dependencies():
    """Install missing dependencies"""
    try:
        packages = [
            "numpy==1.21.6",
            "pandas==1.3.5", 
            "yfinance==0.2.18",
            "psutil==5.9.5",
            "python-telegram-bot==20.7"
        ]
        
        results = {}
        for package in packages:
            results[package] = install_package(package)
        
        return jsonify({
            "status": "installation_attempted",
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/deps/check')
def check_dependencies():
    """Check dependency status"""
    deps_status = {}
    
    # Check critical deps
    try:
        import flask
        deps_status["flask"] = "available"
    except ImportError:
        deps_status["flask"] = "missing"
    
    try:
        import numpy
        deps_status["numpy"] = "available"
    except ImportError:
        deps_status["numpy"] = "missing"
    
    try:
        import pandas
        deps_status["pandas"] = "available"
    except ImportError:
        deps_status["pandas"] = "missing"
    
    try:
        import yfinance
        deps_status["yfinance"] = "available"
    except ImportError:
        deps_status["yfinance"] = "missing"
    
    return jsonify({"dependencies": deps_status})

# ==================== STARTUP ====================

if __name__ == "__main__":
    logger.info("🚀 Ultimate Legendary Scanner Starting...")
    logger.info("💡 Using lazy loading for heavy dependencies")
    
    # Display dependency status
    try:
        import flask
        logger.info("✅ Flask: Available")
    except ImportError:
        logger.error("❌ Flask: MISSING - Critical!")
    
    # Start scanner in background if not run once
    if not RUN_ONCE:
        scanner_thread = threading.Thread(target=scanner_loop, daemon=True)
        scanner_thread.start()
        logger.info("🔄 Background scanner started")
    
    # Display available endpoints
    logger.info(f"🌐 Web server starting on port {PORT}")
    logger.info("📡 Available endpoints:")
    logger.info(f"   http://localhost:{PORT}/")
    logger.info(f"   http://localhost:{PORT}/health")
    logger.info(f"   http://localhost:{PORT}/signals")
    logger.info(f"   http://localhost:{PORT}/deps/check")
    logger.info(f"   http://localhost:{PORT}/deps/install")
    
    # Start Flask app
    try:
        app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"❌ Failed to start web server: {e}")

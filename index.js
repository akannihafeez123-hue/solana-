// src/index.js
// 🏆 ULTIMATE QUANTUM TRADING BOT - V6.0 COSMIC EDITION (Node.js Cloudflare Worker)
// Confirmation-only Telegram, high-confidence signals, multi-timeframe analysis

// Environment bindings come from wrangler.toml and secrets
// Bindings: TELEGRAM_TOKEN, CHAT_ID, BASE_CONFIDENCE_THRESHOLD, TELEGRAM_MIN_CONFIDENCE,
// TELEGRAM_COOLDOWN_MINUTES, SCAN_INTERVAL_SEC, DEFAULT_TRADING_MODE, KEEPALIVE_URL

const GLOBAL_ASSETS = [
  "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "WMT", "JNJ", "PG", "UNH", "HD", "MA",
];
const ASSETS = GLOBAL_ASSETS.slice(0, 10);
const EXPIRIES = [1, 5, 15, 30, 60];

const TRADING_SESSIONS = [
  { name: "TOKYO", start: [0, 0], end: [7, 0] },
  { name: "LONDON", start: [7, 0], end: [16, 0] },
  { name: "US", start: [13, 0], end: [21, 0] },
  { name: "OVERLAP", start: [7, 0], end: [13, 0] },
];
const NEWS_BLACKOUTS = [
  { start: [13, 30], end: [14, 30], reason: "US Open" },
  { start: [7, 0], end: [8, 0], reason: "London Open" },
  { start: [20, 0], end: [21, 0], reason: "US Close" },
  { start: [22, 0], end: [23, 59], reason: "Asia Session" },
  { start: [0, 0], end: [2, 0], reason: "Low Liquidity" },
];

const DEFAULT_ASSET_CONFIG = (() => {
  const cfg = {};
  for (let i = 0; i < ASSETS.length; i++) {
    const asset = ASSETS[i];
    cfg[asset] = {
      min_data_points: 100,
      volatility_threshold: 0.5 + i * 0.05,
      vwap_threshold: 0.01 + i * 0.001,
      volume_profile_sensitivity: 0.005 + i * 0.0005,
      trend_strength_multiplier: 1.3 + i * 0.05,
      momentum_volume_multiplier: 2.0 + i * 0.1,
      min_volume_ratio: 0.7,
      max_spread_pct: 0.001 + i * 0.0001,
      choppy_threshold: 0.35 + i * 0.02,
      min_trend_consistency: 0.75,
      min_momentum_strength: 0.7,
      fibonacci_levels: [0.236, 0.382, 0.5, 0.618, 0.786],
    };
  }
  return cfg;
})();

const TradingMode = {
  QUANTUM: "quantum",
  MOMENTUM: "momentum",
  BREAKOUT: "breakout",
  MEAN_REVERSION: "meanreversion",
};

// ————————————————————————————————
// In-memory state (ephemeral per Worker instance)
// ————————————————————————————————
const signalHistory = [];
const rejectedSignals = [];
const lastSignalTime = new Map();
let lastTelegramMessageAt = null;

// ————————————————————————————————
// Utilities
// ————————————————————————————————
function pctChange(series) {
  const out = [];
  for (let i = 1; i < series.length; i++) {
    const prev = series[i - 1];
    const curr = series[i];
    if (prev && curr) out.push((curr - prev) / prev);
  }
  return out;
}
function rollingMean(arr, window) {
  const out = new Array(arr.length).fill(null);
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    sum += arr[i] ?? 0;
    if (i >= window) sum -= arr[i - window] ?? 0;
    if (i >= window - 1) out[i] = sum / window;
  }
  return out;
}
function rollingStd(arr, window) {
  const out = new Array(arr.length).fill(null);
  const mean = rollingMean(arr, window);
  for (let i = window - 1; i < arr.length; i++) {
    let s2 = 0;
    for (let j = i - window + 1; j <= i; j++) {
      const d = (arr[j] ?? 0) - (mean[i] ?? 0);
      s2 += d * d;
    }
    out[i] = Math.sqrt(s2 / window);
  }
  return out;
}
function ema(arr, span) {
  const out = new Array(arr.length).fill(null);
  const alpha = 2 / (span + 1);
  let prev = arr[0] ?? 0;
  out[0] = prev;
  for (let i = 1; i < arr.length; i++) {
    const val = arr[i] ?? prev;
    const emaVal = alpha * val + (1 - alpha) * prev;
    out[i] = emaVal;
    prev = emaVal;
  }
  return out;
}
function nowUTC() {
  const d = new Date();
  return new Date(d.toISOString()); // ensure UTC
}
function withinTimeRange(date, startHHMM, endHHMM) {
  const hh = date.getUTCHours();
  const mm = date.getUTCMinutes();
  const start = startHHMM[0] * 60 + startHHMM[1];
  const end = endHHMM[0] * 60 + endHHMM[1];
  const cur = hh * 60 + mm;
  return cur >= start && cur <= end;
}

// ————————————————————————————————
// Data Provider (Yahoo + fallback)
// ————————————————————————————————
async function fetchYahoo(symbol, timeframe, limit = 150) {
  const intervalMap = { "1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "4h": "60m", "1d": "1d" };
  const rangeMap = { "1m": "1d", "5m": "5d", "15m": "5d", "1h": "1mo", "4h": "3mo", "1d": "6mo" };
  const interval = intervalMap[timeframe] ?? "60m";
  const yfRange = rangeMap[timeframe] ?? "1mo";

  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=${interval}&range=${yfRange}&includePrePost=false`;
  const res = await fetch(url, { headers: { "User-Agent": "QuantumBot/6.0" } });
  if (!res.ok) return null;
  const data = await res.json();
  const result = data?.chart?.result?.[0];
  if (!result?.timestamp || !result?.indicators?.quote?.[0]) return null;

  const quote = result.indicators.quote[0];
  const ts = result.timestamp;
  const rows = [];
  for (let i = 0; i < ts.length; i++) {
    const row = {
      timestamp: new Date(ts[i] * 1000),
      open: quote.open?.[i] ?? null,
      high: quote.high?.[i] ?? null,
      low: quote.low?.[i] ?? null,
      close: quote.close?.[i] ?? null,
      volume: quote.volume?.[i] ?? null,
    };
    if (Object.values(row).every((v) => v !== null)) rows.push(row);
  }

  // Resample 4h if needed
  if (timeframe === "4h" && interval === "60m") {
    const buckets = new Map();
    rows.forEach((r) => {
      const bucketKey = Math.floor(r.timestamp.getTime() / (4 * 3600 * 1000));
      const b = buckets.get(bucketKey) || {
        timestamp: new Date(bucketKey * 4 * 3600 * 1000),
        open: r.open,
        high: r.high,
        low: r.low,
        close: r.close,
        volume: 0,
      };
      b.high = Math.max(b.high, r.high);
      b.low = Math.min(b.low, r.low);
      b.close = r.close;
      b.volume += r.volume;
      buckets.set(bucketKey, b);
    });
    const resampled = Array.from(buckets.values()).sort((a, b) => a.timestamp - b.timestamp);
    return resampled.slice(-limit);
  }

  return rows.slice(-limit);
}

function generateFallback(symbol, points = 200) {
  const baseMap = { AAPL: 180, MSFT: 380, GOOGL: 140, AMZN: 170, TSLA: 240, NVDA: 480, META: 340, JPM: 180, V: 270, WMT: 160 };
  const base = baseMap[symbol] ?? 100;
  const timestamps = [];
  const returns = [];
  const prices = [];
  const opens = [];
  const highs = [];
  const lows = [];
  const closes = [];
  const volumes = [];
  let price = base;

  for (let i = points; i >= 1; i--) {
    const ts = new Date(Date.now() - i * 3600 * 1000);
    timestamps.push(ts);
  }
  for (let i = 0; i < points; i++) {
    const r = randn() * 0.015;
    returns.push(r);
    price = (i === 0 ? base : prices[i - 1]) * Math.exp(r);
    prices.push(price);
    const o = price * (1 + randn() * 0.001);
    opens.push(o);
    const h = Math.max(o, price) * (1 + Math.abs(randn()) * 0.005);
    highs.push(h);
    const l = Math.min(o, price) * (1 - Math.abs(randn()) * 0.005);
    lows.push(l);
    const c = ((h + l) / 2) * (1 + randn() * 0.001);
    closes.push(c);
    const v = 1_000_000 * Math.exp(randn() * 0.5);
    volumes.push(v);
  }
  return timestamps.map((t, i) => ({
    timestamp: t,
    open: opens[i],
    high: highs[i],
    low: lows[i],
    close: closes[i],
    volume: volumes[i],
  }));
}
function randn() {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

async function fetchOHLCV(symbol, timeframe, limit = 150) {
  let df = await fetchYahoo(symbol, timeframe, limit);
  if (!df || df.length < 50) df = generateFallback(symbol, 200);
  return df;
}

// ————————————————————————————————
// Indicators & strategies (Workers-safe)
// ————————————————————————————————
function calculateSupertrend(df, period = 10, multiplier = 3) {
  if (df.length < period) return { supertrend: null, direction: null };
  const hl2 = df.map((r) => (r.high + r.low) / 2);
  const tr = df.map((r, i) => {
    if (i === 0) return r.high - r.low;
    const prevClose = df[i - 1].close;
    return Math.max(
      r.high - r.low,
      Math.abs(r.high - prevClose),
      Math.abs(r.low - prevClose)
    );
  });
  const atr = rollingMean(tr, period);
  const upper = hl2.map((v, i) => v + (atr[i] ?? atr[atr.length - 1]) * multiplier);
  const lower = hl2.map((v, i) => v - (atr[i] ?? atr[atr.length - 1]) * multiplier);

  const supertrend = new Array(df.length);
  const direction = new Array(df.length);
  supertrend[0] = upper[0];
  direction[0] = -1;

  for (let i = 1; i < df.length; i++) {
    const prevClose = df[i - 1].close;
    const close = df[i].close;
    if (prevClose <= supertrend[i - 1]) {
      if (close > upper[i]) {
        direction[i] = 1;
        supertrend[i] = lower[i];
      } else {
        direction[i] = -1;
        supertrend[i] = Math.min(upper[i], supertrend[i - 1]);
      }
    } else {
      if (close < lower[i]) {
        direction[i] = -1;
        supertrend[i] = upper[i];
      } else {
        direction[i] = 1;
        supertrend[i] = Math.max(lower[i], supertrend[i - 1]);
      }
    }
  }
  return { supertrend, direction };
}

function bollingerBands(series, period = 20, stdMult = 2) {
  const mid = rollingMean(series, period);
  const s = rollingStd(series, period);
  const upper = mid.map((m, i) => (m ?? mid[mid.length - 1]) + (s[i] ?? s[s.length - 1]) * stdMult);
  const lower = mid.map((m, i) => (m ?? mid[mid.length - 1]) - (s[i] ?? s[s.length - 1]) * stdMult);
  return { mid, upper, lower };
}

function detectFairValueGaps(df, threshold = 0.002) {
  if (df.length < 3) return [];
  const out = [];
  for (let i = 1; i < df.length - 1; i++) {
    const prev = df[i - 1];
    const cur = df[i];
    const next = df[i + 1];
    // Bullish FVG
    if (cur.low > prev.high && next.low > cur.low) {
      const gap = (cur.low - prev.high) / prev.high;
      if (gap > threshold) {
        out.push({ index: i, type: "BULLISH", gap_size: gap, price_level: (cur.low + prev.high) / 2 });
      }
    } else if (cur.high < prev.low && next.high < cur.high) {
      const gap = (prev.low - cur.high) / cur.high;
      if (gap > threshold) {
        out.push({ index: i, type: "BEARISH", gap_size: gap, price_level: (cur.high + prev.low) / 2 });
      }
    }
  }
  return out;
}

function emaMacdCombo(df) {
  if (df.length < 50) return neutral("EMA_MACD", "quantum");
  const closes = df.map((r) => r.close);
  const ema12 = ema(closes, 12);
  const ema26 = ema(closes, 26);
  const ema50 = ema(closes, 50);
  const macdLine = closes.map((_, i) => (ema12[i] ?? 0) - (ema26[i] ?? 0));
  const macdSignal = ema(macdLine, 9);
  const macdHist = macdLine.map((v, i) => v - (macdSignal[i] ?? 0));

  let bull = 0, bear = 0;
  const last = df.length - 1;
  if (ema12[last] > ema26[last] && ema26[last] > ema50[last]) bull += 40;
  else if (ema12[last] < ema26[last] && ema26[last] < ema50[last]) bear += 40;

  if (macdLine[last] > macdSignal[last] && macdHist[last] > 0) bull += 35;
  else if (macdLine[last] < macdSignal[last] && macdHist[last] < 0) bear += 35;

  const vol20 = rollingMean(df.map((r) => r.volume), 20);
  const volRatio = df[last].volume / (vol20[last] || df[last].volume);
  if (volRatio > 1.5) {
    if (df[last].close > df[last].open) bull += 25;
    else bear += 25;
  }

  if (bull >= 70) {
    return signal("BUY", Math.min(bull, 95), "EMA_MACD_Bullish", "quantum", {
      ema_alignment: ema12[last] > ema26[last] && ema26[last] > ema50[last] ? "12>26>50" : "Neutral",
      macd_signal: macdLine[last] > macdSignal[last] ? "Bullish" : "Bearish",
      volume_confirmation: volRatio > 1.5,
    });
  }
  if (bear >= 70) {
    return signal("SELL", Math.min(bear, 95), "EMA_MACD_Bearish", "quantum", {
      ema_alignment: ema12[last] < ema26[last] && ema26[last] < ema50[last] ? "12<26<50" : "Neutral",
      macd_signal: macdLine[last] < macdSignal[last] ? "Bearish" : "Bullish",
      volume_confirmation: volRatio > 1.5,
    });
  }
  return neutral("EMA_MACD", "quantum");
}

function fairValueGapStrategy(df) {
  const fvgs = detectFairValueGaps(df);
  const last = df[df.length - 1];
  if (fvgs.length) {
    const fvg = fvgs[fvgs.length - 1];
    if (fvg.type === "BULLISH") {
      if (last.close <= fvg.price_level * 1.01) {
        return signal("BUY", 86, "FVG_Bullish", "quantum", {
          fvg_level: fvg.price_level,
          gap_size_pct: fvg.gap_size * 100,
          distance_to_fvg: ((last.close - fvg.price_level) / fvg.price_level) * 100,
        });
      }
    } else {
      if (last.close >= fvg.price_level * 0.99) {
        return signal("SELL", 86, "FVG_Bearish", "quantum", {
          fvg_level: fvg.price_level,
          gap_size_pct: fvg.gap_size * 100,
          distance_to_fvg: ((fvg.price_level - last.close) / last.close) * 100,
        });
      }
    }
  }
  return neutral("FVG", "quantum");
}

function supertrendBollingerCombo(df) {
  if (df.length < 30) return neutral("SuperTrend_BB", "quantum");
  const { supertrend, direction } = calculateSupertrend(df);
  if (!supertrend || !direction) return neutral("SuperTrend_BB", "quantum");
  const closes = df.map((r) => r.close);
  const { mid, upper, lower } = bollingerBands(closes, 20, 2);
  const last = df.length - 1;
  const stDir = direction[last];
  const price = df[last].close;
  const midVal = mid[last] ?? closes[last];
  const upperVal = upper[last] ?? midVal * 1.02;
  const lowerVal = lower[last] ?? midVal * 0.98;

  let bull = false, bear = false;
  if (stDir === 1) {
    if (price > midVal || price < lowerVal) bull = true;
  } else {
    if (price < midVal || price > upperVal) bear = true;
  }
  const bbWidth = midVal ? (upperVal - lowerVal) / midVal : 0;
  const squeeze = bbWidth < 0.05;

  if (bull) {
    return signal("BUY", squeeze ? 89 : 84, "SuperTrend_BB_Bullish", "quantum", {
      supertrend_direction: "Bullish",
      bb_position: price > midVal ? "Above_Middle" : "Below_Lower",
      bb_squeeze: squeeze,
    });
  }
  if (bear) {
    return signal("SELL", squeeze ? 89 : 84, "SuperTrend_BB_Bearish", "quantum", {
      supertrend_direction: "Bearish",
      bb_position: price < midVal ? "Below_Middle" : "Above_Upper",
      bb_squeeze: squeeze,
    });
  }
  return neutral("SuperTrend_BB", "quantum");
}

function breakOfStructure(df) {
  if (df.length < 20) return neutral("BOS", "quantum");
  const last = df.length - 1;
  const highs = df.slice(-20).map((r) => r.high);
  const lows = df.slice(-20).map((r) => r.low);
  const recentHigh = Math.max(...highs);
  const recentLow = Math.min(...lows);
  const curr = df[last];
  const vol20 = rollingMean(df.map((r) => r.volume), 20);
  const volConfirmed = curr.volume > (vol20[last] || curr.volume) * 1.5;

  if (curr.close > recentHigh) {
    return signal("BUY", volConfirmed ? 92 : 85, "BOS_Bullish", "quantum", {
      break_level: recentHigh,
      break_distance: ((curr.close - recentHigh) / recentHigh) * 100,
      volume_confirmed: !!volConfirmed,
    });
  } else if (curr.close < recentLow) {
    return signal("SELL", volConfirmed ? 92 : 85, "BOS_Bearish", "quantum", {
      break_level: recentLow,
      break_distance: ((recentLow - curr.close) / curr.close) * 100,
      volume_confirmed: !!volConfirmed,
    });
  }
  return neutral("BOS", "quantum");
}

// Momentum strategies (subset to keep Worker lean)
function momentumBreakDetection(df) {
  if (df.length < 15) return neutral("Momentum_Break", "momentum");
  const last = df.length - 1;
  const c = df[last].close;
  const c5 = df[Math.max(0, last - 5)].close;
  const c10 = df[Math.max(0, last - 10)].close;
  const momentum = (c - c5) / c5;
  const priceChange5 = (c - c5) / c5;
  const priceChange10 = (c - c10) / c10;
  const acceleration = priceChange5 - priceChange10;
  const vol20 = rollingMean(df.map((r) => r.volume), 20);
  const volRatio = df[last].volume / (vol20[last] || df[last].volume);

  // RSI
  const deltas = [];
  const closes = df.map((r) => r.close);
  for (let i = 1; i < closes.length; i++) deltas.push(closes[i] - closes[i - 1]);
  const gains = deltas.map((d) => (d > 0 ? d : 0));
  const losses = deltas.map((d) => (d < 0 ? -d : 0));
  const gain14 = rollingMean(gains, 14);
  const loss14 = rollingMean(losses, 14);
  const lastRS = (gain14[gain14.length - 1] || 0.0001) / (loss14[loss14.length - 1] || 0.0001);
  const rsi = 100 - 100 / (1 + lastRS);

  if (momentum > 0 && acceleration > 0 && volRatio > 1.5 && rsi < 70) {
    return signal("BUY", 87, "Momentum_Break_Bullish", "momentum", { momentum, acceleration, volRatio, rsi });
  }
  if (momentum < 0 && acceleration < 0 && volRatio > 1.5 && rsi > 30) {
    return signal("SELL", 87, "Momentum_Break_Bearish", "momentum", { momentum, acceleration, volRatio, rsi });
  }
  return neutral("Momentum_Break", "momentum");
}

function resistanceSupportBreaks(df) {
  if (df.length < 20) return neutral("RS_Break", "breakout");
  const last = df.length - 1;
  const window = df.slice(-20);
  const recentHigh = Math.max(...window.map((r) => r.high));
  const recentLow = Math.min(...window.map((r) => r.low));
  const current = df[last].close;
  const consolidationRange = recentHigh - recentLow;
  const rangePct = consolidationRange / (recentLow || 1);
  const vol20 = rollingMean(df.map((r) => r.volume), 20);
  const volRatio = df[last].volume / (vol20[last] || df[last].volume);

  if (current > recentHigh && rangePct < 0.03 && volRatio > 1.5) {
    return signal("BUY", 90, "Resistance_Breakout", "breakout", {
      break_level: recentHigh,
      consolidation_range_pct: rangePct * 100,
      volume_confirmation: volRatio,
      break_distance: ((current - recentHigh) / recentHigh) * 100,
    });
  }
  if (current < recentLow && rangePct < 0.03 && volRatio > 1.5) {
    return signal("SELL", 90, "Support_Breakdown", "breakout", {
      break_level: recentLow,
      consolidation_range_pct: rangePct * 100,
      volume_confirmation: volRatio,
      break_distance: ((recentLow - current) / current) * 100,
    });
  }
  return neutral("RS_Break", "breakout");
}

// Helpers to build strategy results
function signal(dir, score, reason, type, details = {}) {
  return { signal: dir, score, reason, type, details };
}
function neutral(reason, type) {
  return { signal: "NEUTRAL", score: 0, reason, type };
}

// ————————————————————————————————
// Strategy manager and analysis engine
// ————————————————————————————————
function strategiesByMode(mode) {
  const m = mode || TradingMode.QUANTUM;
  const map = {
    [TradingMode.QUANTUM]: [breakOfStructure, fairValueGapStrategy, emaMacdCombo, supertrendBollingerCombo],
    [TradingMode.MOMENTUM]: [momentumBreakDetection, emaMacdCombo],
    [TradingMode.BREAKOUT]: [resistanceSupportBreaks, breakOfStructure],
    [TradingMode.MEAN_REVERSION]: [fairValueGapStrategy, momentumBreakDetection],
  };
  return map[m] || map[TradingMode.QUANTUM];
}

function analyzeAsset(df, mode) {
  const results = [];
  const strategies = strategiesByMode(mode);
  for (const strat of strategies) {
    try {
      const r = strat(df);
      if (r && r.signal !== "NEUTRAL") results.push(r);
    } catch (e) {
      // swallow in Worker
    }
  }
  return results;
}

function analyzeMultiTimeframe(asset, tfData, mode) {
  const all = [];
  for (const [tf, df] of Object.entries(tfData)) {
    const res = analyzeAsset(df, mode);
    res.forEach((r) => all.push({ ...r, timeframe: tf }));
  }
  const high = all.filter((r) => (r.score || 0) >= 75);
  if (!high.length) return { signal: "NEUTRAL", confidence: 0, strategies: [], timeframe_alignment: 0 };
  const buys = high.filter((r) => r.signal === "BUY");
  const sells = high.filter((r) => r.signal === "SELL");

  const timeframes = Object.keys(tfData);
  const aligned = new Set(high.map((r) => r.timeframe));
  const alignmentRatio = timeframes.length ? aligned.size / timeframes.length : 0;

  if (buys.length >= sells.length && buys.length >= 3) {
    const avg = buys.reduce((a, b) => a + (b.score || 0), 0) / buys.length;
    return { signal: "BUY", confidence: avg, strategies: buys.slice(0, 5), timeframe_alignment: alignmentRatio, signal_strength: buys.length };
  }
  if (sells.length >= buys.length && sells.length >= 3) {
    const avg = sells.reduce((a, b) => a + (b.score || 0), 0) / sells.length;
    return { signal: "SELL", confidence: avg, strategies: sells.slice(0, 5), timeframe_alignment: alignmentRatio, signal_strength: sells.length };
  }
  return { signal: "NEUTRAL", confidence: 0, strategies: [], timeframe_alignment: alignmentRatio };
}

// ————————————————————————————————
// Market filters
// ————————————————————————————————
function checkMarketConditions(df, assetCfg) {
  if (!df || df.length < 50) return [false, "Insufficient data"];
  const closes = df.map((r) => r.close);
  const rets = pctChange(closes);
  if (!rets.length) return [false, "No return data"];
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const sd = Math.sqrt(rets.reduce((a, b) => a + (b - mean) ** 2, 0) / rets.length);
  const volatility = sd * Math.sqrt(252);
  if (volatility > (assetCfg?.volatility_threshold ?? 0.6)) return [false, `High volatility: ${(volatility * 100).toFixed(2)}%`];

  const vol = df.map((r) => r.volume);
  const avgVol = rollingMean(vol, 50);
  const last = df.length - 1;
  const volRatio = vol[last] / (avgVol[last] || vol[last]);
  if (volRatio < (assetCfg?.min_volume_ratio ?? 0.7)) return [false, `Low liquidity: ${volRatio.toFixed(2)}`];

  const tr = df.map((r, i) => {
    if (i === 0) return r.high - r.low;
    const prevClose = df[i - 1].close;
    return Math.max(r.high - r.low, Math.abs(r.high - prevClose), Math.abs(r.low - prevClose));
  });
  const atr14 = rollingMean(tr, 14);
  const atrRatio = (atr14[last] || atr14[atr14.length - 1] || 0) / (df[last].close || 1);
  if (atrRatio < (assetCfg?.choppy_threshold ?? 0.35)) {
    const recent = df.slice(-20);
    const rh = Math.max(...recent.map((r) => r.high));
    const rl = Math.min(...recent.map((r) => r.low));
    const rm = recent.reduce((a, r) => a + r.close, 0) / recent.length;
    const range = (rh - rl) / (rm || 1);
    if (range < 0.02) return [false, `Choppy market: ATR ratio ${atrRatio.toFixed(3)}`];
  }

  return [true, "Favorable conditions"];
}

function isTradingSession(date) {
  const current = [date.getUTCHours(), date.getUTCMinutes()];
  for (const b of NEWS_BLACKOUTS) {
    if (withinTimeRange(date, b.start, b.end)) return [false, `News blackout: ${b.reason}`];
  }
  for (const s of TRADING_SESSIONS) {
    if (withinTimeRange(date, s.start, s.end)) return [true, `${s.name} session`];
  }
  return [false, "Outside trading hours"];
}

function checkTimeframeAlignment(tfData, direction) {
  let aligned = 0, total = 0;
  for (const df of Object.values(tfData)) {
    if (df.length < 30) continue;
    const closes = df.map((r) => r.close);
    const e12 = ema(closes, 12);
    const e26 = ema(closes, 26);
    const last = df.length - 1;
    const tfDir = e12[last] > e26[last] ? "BUY" : "SELL";
    if (tfDir === direction) aligned++;
    total++;
  }
  const ok = aligned >= Math.max(2, Math.floor(total * 0.6));
  return [ok, aligned];
}

// ————————————————————————————————
// Telegram (HTTP API via fetch)
// ————————————————————————————————
function shouldSendTelegram(lastAt, cooldownMin) {
  if (!lastAt) return true;
  const minutes = (nowUTC().getTime() - lastAt.getTime()) / 60000;
  return minutes >= cooldownMin;
}

async function sendTelegramMessage(env, text) {
  const url = `https://api.telegram.org/bot${env.TELEGRAM_TOKEN}/sendMessage`;
  const payload = { chat_id: env.CHAT_ID, text, parse_mode: "HTML", disable_web_page_preview: true };
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  return res.ok;
}

function formatSignalMessage(env, signal) {
  const modeEmoji = { quantum: "⚛️", momentum: "⚡", breakout: "🚀", meanreversion: "🔄" }[signal.mode] || "⚛️";
  return `${modeEmoji} <b>QUANTUM SIGNAL CONFIRMED</b>

🎯 <b>${signal.asset}</b> - ${signal.direction}
📈 <b>Confidence:</b> ${signal.confidence.toFixed(1)}%
⏱️ <b>Expiry:</b> ${signal.expiry}m

🕐 ${new Date().toISOString().slice(11, 19)} UTC
#${signal.asset} #${signal.direction}`;
}

// ————————————————————————————————
// Scanning and signal generation
// ————————————————————————————————
async function scanAssets(env, mode) {
  const baseThreshold = Number(env.BASE_CONFIDENCE_THRESHOLD || 85);
  const telegramMin = Number(env.TELEGRAM_MIN_CONFIDENCE || 90);

  for (const asset of ASSETS) {
    const tfData = {};
    for (const tf of ["15m", "1h", "4h"]) {
      const df = await fetchOHLCV(asset, tf, 150);
      if (df && df.length >= 50) tfData[tf] = df;
    }
    if (Object.keys(tfData).length < 2) continue;

    const primary = tfData["1h"] || tfData["15m"] || tfData["4h"];
    const [ok, reason] = checkMarketConditions(primary, DEFAULT_ASSET_CONFIG[asset]);
    if (!ok) {
      rejectedSignals.push({ asset, reason, timestamp: nowUTC().toISOString() });
      continue;
    }

    const analysis = analyzeMultiTimeframe(asset, tfData, mode);
    const [alignedOk, alignedCount] = checkTimeframeAlignment(tfData, analysis.signal);
    if (!alignedOk) continue;

    if (analysis.signal !== "NEUTRAL" && analysis.confidence >= baseThreshold) {
      const now = nowUTC();
      for (const expiry of EXPIRIES) {
        const key = `${asset}_${expiry}`;
        const lastAt = lastSignalTime.get(key);
        const elapsed = lastAt ? (now.getTime() - lastAt.getTime()) / 60000 : Infinity;
        if (elapsed < 15) continue;

        const sig = {
          asset,
          direction: analysis.signal,
          confidence: analysis.confidence,
          expiry,
          strategies: analysis.strategies,
          mode,
          timestamp: now.toISOString(),
          timeframe_alignment: alignedCount / Object.keys(tfData).length,
          signal_strength: analysis.signal_strength || 0,
        };
        signalHistory.push(sig);
        lastSignalTime.set(key, now);

        // Telegram only for high confidence
        if (analysis.confidence >= telegramMin) {
          if (shouldSendTelegram(lastTelegramMessageAt, Number(env.TELEGRAM_COOLDOWN_MINUTES || 5))) {
            const msg = formatSignalMessage(env, sig);
            const okSend = await sendTelegramMessage(env, msg);
            if (okSend) lastTelegramMessageAt = nowUTC();
          }
        }
      }
    }
  }
}

// ————————————————————————————————
// Worker entry points
// ————————————————————————————————
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const mode = (env.DEFAULT_TRADING_MODE || TradingMode.QUANTUM).toLowerCase();

    if (url.pathname === "/") {
      const [sessionOk, sessionName] = isTradingSession(nowUTC());
      return new Response(JSON.stringify({
        status: "online",
        bot: "Quantum Trading Bot V6.0 (Worker)",
        mode,
        assets: ASSETS.length,
        session: sessionOk ? sessionName : `Paused: ${sessionName}`,
        telegram_configured: !!(env.TELEGRAM_TOKEN && env.CHAT_ID),
      }), { headers: { "Content-Type": "application/json" } });
    }

    if (url.pathname === "/scan") {
      const [sessionOk] = isTradingSession(nowUTC());
      if (!sessionOk) return new Response(JSON.stringify({ message: "Outside trading session" }), { headers: { "Content-Type": "application/json" } });
      await scanAssets(env, mode);
      return new Response(JSON.stringify({ message: "Scan completed", signals: signalHistory.slice(-10) }), { headers: { "Content-Type": "application/json" } });
    }

    if (url.pathname === "/signals") {
      return new Response(JSON.stringify({ signals: signalHistory.slice(-10) }), { headers: { "Content-Type": "application/json" } });
    }

    if (url.pathname.startsWith("/mode/")) {
      const newMode = url.pathname.split("/").pop();
      const valid = Object.values(TradingMode).includes(newMode);
      if (!valid) {
        return new Response(JSON.stringify({ success: false, error: "Invalid mode" }), { headers: { "Content-Type": "application/json" }, status: 400 });
      }
      return new Response(JSON.stringify({ success: true, new_mode: newMode }), { headers: { "Content-Type": "application/json" } });
    }

    return new Response("Not found", { status: 404 });
  },

  // Optional scheduled scans (cron)
  async scheduled(event, env) {
    const [sessionOk] = isTradingSession(nowUTC());
    if (!sessionOk) return;
    await scanAssets(env, env.DEFAULT_TRADING_MODE || TradingMode.QUANTUM);
  },
};

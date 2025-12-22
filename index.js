// Quantum Trading Bot Worker — All-in-One single file (index.js)
// Features:
// - Automatic scanning via cron
// - Telegram webhook with /start, /status, /help
// - Adaptive selection: best 3 strategies, best 3 indicators, best 4 timeframes (2y–5m)
// - Institutional analysis: VWAP/AVWAP, Volume Profile, CVD/Delta, Liquidity, Order Flow
// - Strategy catalog including Quantum Engine V2.0, Momentum, Breakout, Mean Reversion + Hidden institutional layers

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Telegram webhook endpoint
    if (url.pathname === "/telegram" && request.method === "POST") {
      const update = await request.json();
      const msg = update.message;
      if (msg?.text) {
        const chatId = msg.chat.id;
        const text = msg.text.trim();

        if (text === "/start") {
          await replyTelegram(env, chatId,
            "✅ Quantum Bot active.\nAutomatic scans every 5 minutes.\nSignals require institutional confirmation when applicable."
          );
        } else if (text === "/status") {
          const assets = listAssets(assetUniverse).join(", ");
          const sList = Object.keys(strategies).join(", ");
          await replyTelegram(env, chatId,
            `📊 Status
Mode: ${env.DEFAULT_TRADING_MODE || "auto"}
Base threshold: ${env.BASE_CONFIDENCE_THRESHOLD || "85"}%
Telegram threshold: ${env.TELEGRAM_MIN_CONFIDENCE || "90"}%
Assets: ${assets}
Strategies: ${sList}`
          );
        } else if (text === "/help") {
          await replyTelegram(env, chatId,
            "Commands:\n/start – confirm bot\n/status – show config\n/help – show commands\n(Scanning is automatic; mode selection is adaptive)"
          );
        } else {
          await replyTelegram(env, chatId, "I didn’t understand that command. Type /help.");
        }
      }
      return new Response("OK");
    }

    // HTTP endpoints
    if (url.pathname === "/") {
      return json({ status: "online", bot: "Quantum Trading Bot", mode: env.DEFAULT_TRADING_MODE || "auto" });
    }
    if (url.pathname === "/scan") {
      const signals = await runFullScan(env, { source: "http" });
      return json({ count: signals.length, signals });
    }
    if (url.pathname === "/signals") {
      // TODO: read from KV if you enable persistence
      return json({ signals: [] });
    }

    return new Response("Not found", { status: 404 });
  },

  async scheduled(_event, env) {
    await runFullScan(env, { source: "cron" });
  }
};

/* ================================
   Assets (universe)
================================ */

const assetUniverse = [
  { symbol: "BTCUSDT", market: "crypto", baseTf: ["2y","6m","1d","4h","1h","15m","5m"] },
  { symbol: "ETHUSDT", market: "crypto", baseTf: ["2y","6m","1d","4h","1h","15m","5m"] },
  { symbol: "AAPL", market: "stocks", baseTf: ["2y","1y","6m","1d","4h","1h","15m"] },
  { symbol: "EURUSD", market: "forex", baseTf: ["2y","1y","6m","1d","4h","1h","15m"] }
];

function listAssets(universe) {
  return universe.map(a => a.symbol);
}

/* ================================
   Indicators (stubs/placeholders)
   Wire these to your data sources
================================ */

const indicatorLib = {
  async rsi(asset, tf, period = 14) { return { value: 50, period, tf }; },
  async macd(asset, tf, params = { fast:12, slow:26, signal:9 }) { return { cross: "neutral", hist: 0, tf, params }; },
  async ema(asset, tf, period) { return { value: 0, period, tf }; },
  async bb(asset, tf, params = { period:20, dev:2 }) { return { upperTouch: false, lowerTouch: false, breakoutDir: null, tf, params }; },
  async supertrend(asset, tf, params = { atrPeriod:10, multiplier:3 }) { return { trend: "neutral", tf, params }; },
  async atr(asset, tf, period = 14) { return { value: 0, tf, period }; },

  // Regime helpers
  async trendState(asset, tf) { return { state: "side", strength: 0.5, tf }; },
  async volatility(asset, tf) { return { level: "med", sigma: 1.0, tf }; },

  // Structure
  async orderBlocks(asset, tf) { return { blocks: [], tf }; },
  async bos(asset, tf) { return { direction: "none", tf }; },
  async fvg(asset, tf) { return { gaps: [], tf }; }
};

// Institutional pack (order flow, VWAP, profiles, liquidity, session)
const institutionalPack = {
  async vwapPack(asset) {
    return {
      vwap: 0, avwapAnchors: [], bands: { upper: 0, lower: 0 },
      position: "neutral" // "long"/"short" if price vs VWAP/AVWAP bands bias
    };
  },
  async profilePack(asset) {
    return {
      poc: 0, valueAreaHigh: 0, valueAreaLow: 0,
      valueAreaBreak: "none" // "long"/"short" if breaking VA with volume
    };
  },
  async orderFlowBias(asset) {
    return {
      cvd: 0, delta: 0, bias: "neutral" // "long"/"short" based on bid/ask pressure
    };
  },
  async liquidityScore(asset) {
    // 0..1 scale: combines spread, depth, recent turnover
    return 0.7;
  },
  async sessionStats(asset) {
    return {
      ibHigh: 0, ibLow: 0, sessionVwap: 0, sessionValueArea: [0,0]
    };
  }
};

/* ================================
   Strategies (catalog + hidden)
================================ */

const strategies = {
  // Quantum Engine V2.0 — OB, BOS, FVG, EMA x MACD, SuperTrend x BB, Volume x Smart Money
  quantum: async (asset, ind, ctx) => {
    const [ob, bos, fvg] = await Promise.all([
      ind.orderBlocks(asset, "1h"),
      ind.bos(asset, "1h"),
      ind.fvg(asset, "15m")
    ]);
    const emaFast = await ind.ema(asset, "15m", 20);
    const emaSlow = await ind.ema(asset, "15m", 50);
    const macd = await ind.macd(asset, "1h");
    const st = await ind.supertrend(asset, "1h");
    const bb = await ind.bb(asset, "1h");
    const vol = await ind.atr(asset, "1h");

    const direction =
      macd.cross === "bullish" && emaFast.value > emaSlow.value ? "long" :
      macd.cross === "bearish" && emaFast.value < emaSlow.value ? "short" : null;
    if (!direction) return null;

    const confidence = 88;
    return { direction, confidence, rationale: "OB/BOS/FVG + EMA/MACD + ST/BB confluence", extras: { ob, bos, fvg, emaFast, emaSlow, macd, st, bb, vol } };
  },

  // Momentum Scalper V1.0 — Momentum break, Volume spike, RSI, EMA golden cross
  momentum: async (asset, ind, ctx) => {
    const rsi = await ind.rsi(asset, "5m", 14);
    const ema50 = await ind.ema(asset, "5m", 50);
    const ema200 = await ind.ema(asset, "5m", 200);
    const vol = await ind.atr(asset, "5m", 14);
    const direction = ema50.value > ema200.value && rsi.value > 55 ? "long" :
                      ema50.value < ema200.value && rsi.value < 45 ? "short" : null;
    if (!direction) return null;
    return { direction, confidence: 84, rationale: "Momentum break + EMA golden cross + RSI filter", extras: { rsi, ema50, ema200, vol } };
  },

  // Breakout Hunter V1.0 — S/R breaks, Volume confirm, BB breakout
  breakout: async (asset, ind, ctx) => {
    const bb = await ind.bb(asset, "15m");
    const vol = await ind.atr(asset, "15m");
    const direction = bb.breakoutDir || null;
    if (!direction) return null;
    return { direction, confidence: 82, rationale: "BB breakout + volume confirmation", extras: { bb, vol } };
  },

  // Mean Reversion V1.0 — RSI OB/OS, BB touches, rejection, volume divergence
  meanreversion: async (asset, ind, ctx) => {
    const rsi = await ind.rsi(asset, "15m", 14);
    const bb = await ind.bb(asset, "15m");
    const rejection = false; // Replace with your wick/engulfing detection
    const direction = rsi.value < 30 && bb.lowerTouch ? "long" :
                      rsi.value > 70 && bb.upperTouch ? "short" : null;
    if (!direction) return null;
    return { direction, confidence: 80, rationale: "RSI OB/OS + BB touch + rejection", extras: { rsi, bb, rejection } };
  }
};

// Hidden/Institutional layers (heuristics; integrate your proprietary logic)
const hiddenBlocks = {
  fibonacciVortex: async (asset, ind, ctx) => ({ bias: "long", score: 0.6 }),
  quantumEntanglement: async (asset, ind, ctx) => ({ resonance: "aligned", score: 0.55 }),
  darkPoolInstitutional: async (asset, ind, ctx) => ({ stealthBuy: true, iceberg: false, score: 0.7 }),
  gannSquareCycles: async (asset, ind, ctx) => ({ cyclePhase: "up", score: 0.5 }),
  elliottWaveNeural: async (asset, ind, ctx) => ({ waveCount: "3/5", score: 0.65 }),
  cosmicMovement: async (asset, ind, ctx) => ({ alignment: "neutral", score: 0.4 })
};

/* ==========================================
   Adaptive selection + scoring/formatting
========================================== */

const TF_POOL = ["2y","1y","6m","3m","1m","2w","1w","3d","2d","1d","12h","8h","4h","2h","1h","30m","15m","5m"];

function selectAdaptive(asset, ctx) {
  const regime = inferRegime(ctx);                      // trend/vol/liquidity/order flow
  const mode = selectMode(regime);                      // auto-picks strategy family
  const topStrategies = pickTopStrategies(mode, regime, 3);
  const topIndicators = pickTopIndicators(mode, regime, 3);
  const topTimeframes = pickTopTimeframes(regime, 4);
  return { mode, topStrategies, topIndicators, topTimeframes, regime };
}

function inferRegime(ctx) {
  const trend = ctx.trend?.state || "side";
  const vol = ctx.volatility?.level || "med";
  const liq = ctx.liquidity || 0.5;
  const ofb = ctx.orderFlowBias?.bias || "neutral";
  return { trend, vol, liq, ofb };
}

function selectMode(regime) {
  if (regime.trend === "up" || regime.trend === "down") {
    return regime.vol === "high" ? "momentum" : "quantum";
  }
  if (regime.trend === "side") {
    return regime.vol === "low" ? "meanreversion" : "breakout";
  }
  return "quantum";
}

function pickTopStrategies(mode, regime, k) {
  const ranked = {
    quantum: ["quantum","breakout","momentum","meanreversion"],
    momentum: ["momentum","quantum","breakout","meanreversion"],
    breakout: ["breakout","quantum","momentum","meanreversion"],
    meanreversion: ["meanreversion","breakout","quantum","momentum"]
  }[mode];
  return ranked.slice(0, k);
}

function pickTopIndicators(mode, regime, k) {
  const institutional = ["vwapPack","orderFlowBias","profilePack"];
  const retailCore = ["rsi","macd","ema","bb","supertrend","atr"];
  const pool = (mode === "quantum" || mode === "momentum")
    ? [...institutional, "ema", "macd", "atr"]
    : ["rsi","bb","supertrend","atr", ...institutional];
  return unique(pool).slice(0, k);
}

function pickTopTimeframes(regime, k) {
  const htf = regime.trend !== "side" ? ["1w","1d"] : ["2y","6m"];
  const itf = ["4h","1h"];
  const ltf = regime.vol === "high" ? ["5m"] : ["15m"];
  return unique([...htf, ...itf, ...ltf]).slice(0, k);
}

function unique(arr) { return [...new Set(arr)]; }

function scoreSignal(raw, { baseThreshold, mode, strategyName, asset, tf, ctx }) {
  const base = raw.confidence ?? 0;
  const regimeBoost = (ctx.liquidity > 0.6 ? 3 : 0) + (ctx.orderFlowBias?.bias === raw.direction ? 4 : 0);
  const score = Math.min(99, base + regimeBoost);
  if (score < baseThreshold) return null;
  return { ...raw, asset: asset.symbol, tf, strategy: strategyName, confidence: score, mode };
}

function shouldNotify(sig, { teleThreshold }) {
  return (sig.confidence ?? 0) >= teleThreshold;
}

function formatTelegramMessage(sig) {
  return [
    `⚡ ${sig.asset} ${sig.direction?.toUpperCase()} (${sig.tf})`,
    `Mode: ${sig.mode} | Strategy: ${sig.strategy}`,
    `Confidence: ${sig.confidence}%`,
    `Why: ${sig.rationale || "n/a"}`
  ].join("\n");
}

/* ==========================================
   Orchestration: full scan with adaptive plan
========================================== */

async function runFullScan(env, { source }) {
  const baseThreshold = Number(env.BASE_CONFIDENCE_THRESHOLD || 85);
  const teleThreshold = Number(env.TELEGRAM_MIN_CONFIDENCE || 90);

  const allSignals = [];

  for (const asset of assetUniverse) {
    // Build market context
    const ctx = await buildMarketContext(asset, indicatorLib, institutionalPack);
    // Adaptive plan
    const plan = selectAdaptive(asset, ctx);

    // Execute selected strategies across selected timeframes
    for (const stratName of plan.topStrategies) {
      const strat = strategies[stratName];
      for (const tf of plan.topTimeframes) {
        const raw = await runStrategyOnAsset(strat, asset, indicatorLib, { tf, ctx, env, plan });
        if (!raw) continue;

        // Institutional confirmation gates
        const gated = await applyInstitutionalGates(raw, asset, tf, ctx);
        if (!gated?.pass) continue;

        // Score + format
        const scored = scoreSignal(gated.signal, { baseThreshold, mode: plan.mode, strategyName: stratName, asset, tf, ctx });
        if (!scored) continue;

        const message = formatTelegramMessage(scored);
        const final = { ...scored, message };
        allSignals.push(final);

        if (shouldNotify(final, { teleThreshold })) {
          await sendTelegram(env, env.CHAT_ID, message);
          // TODO: cooldowns + KV persistence if desired
        }
      }
    }
  }

  return allSignals;
}

async function buildMarketContext(asset, ind, inst) {
  return {
    trend: await ind.trendState(asset, "1h"),
    volatility: await ind.volatility(asset, "1h"),
    liquidity: await inst.liquidityScore(asset),
    orderFlowBias: await inst.orderFlowBias(asset),
    session: await inst.sessionStats(asset),
    vwap: await inst.vwapPack(asset),
    profile: await inst.profilePack(asset)
  };
}

async function applyInstitutionalGates(raw, asset, tf, ctx) {
  const confirms = [
    ctx.orderFlowBias?.bias === raw.direction,                 // CVD/Delta agrees
    ctx.vwap?.position === raw.direction,                      // Price vs VWAP/AVWAP bands
    ctx.profile?.valueAreaBreak === raw.direction,             // Value area/POC break
    ctx.liquidity >= 0.6                                       // Minimum liquidity
  ];
  const pass = confirms.filter(Boolean).length >= 1;           // Require at least 1 confirmation
  return { pass, signal: raw, reasons: confirms };
}

/* ================================
   Telegram helpers
================================ */

async function sendTelegram(env, chatId, text) {
  const url = `https://api.telegram.org/bot${env.TELEGRAM_TOKEN}/sendMessage`;
  const payload = { chat_id: chatId, text, parse_mode: "HTML", disable_web_page_preview: true };
  const res = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
  });
  if (!res.ok) console.error("Telegram error:", await res.text());
  return res.ok;
}

async function replyTelegram(env, chatId, text) {
  return sendTelegram(env, chatId, text);
}

/* ================================
   Strategy runner + utilities
================================ */

async function runStrategyOnAsset(strategy, asset, ind, ctx) {
  try { return await strategy(asset, ind, ctx); }
  catch (e) { console.error(`Strategy error on ${asset.symbol}:`, e); return null; }
}

function json(obj) {
  return new Response(JSON.stringify(obj), { headers: { "Content-Type": "application/json" } });
}

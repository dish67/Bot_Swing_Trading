"""Crypto spot swing trading bot with backtest and live (paper) modes.

This module is based on the user's original script but includes a number of
quality-of-life fixes and small correctness improvements:
    * explicit handling of the cooldown guard in the backtest loop
    * safeguards against NaN indicator values during the warmup window
    * Wilder-style smoothing for RSI/ATR/ADX computations for better
      compatibility with trading platforms
    * light refactors to factor out repeated risk-management code and to
      refresh the higher timeframe bias periodically

The behaviour of the public API (``run_backtest`` / ``run_live_paper``) is kept
compatible with the original script so that existing automation still works.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo
from typing import Dict, Optional, Tuple

import ccxt
import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
load_dotenv()

API_KEY = os.getenv("MEXC_API_KEY", "")
API_SECRET = os.getenv("MEXC_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

MODE = os.getenv("BOT_MODE", "backtest").strip().lower()
BACKTEST = MODE == "backtest"
PAPER_TRADING = MODE == "paper"
LIVE_TRADING = MODE == "live"
BT_START = "2025-10-01"
BT_END = "2025-11-01"

SYMBOLS = ["ETH/USDT", "SOL/USDT", "DOGE/USDT"]
TIMEFRAME = "15m"
CAPITAL_PER_TRADE = 5.0

LOG_FILE = "trade_log.csv"
PAUSE_FILE = "pause.flag"
LAST_MSG_FILE = "last_msg.txt"

FEES_ROUNDTRIP = 0.003

PRESET_NAME = os.getenv("BOT_PRESET", "optimise")

TAKE_PROFIT_R = 2.2
STOP_ATR_MULT = 1.3
MIN_SL_PCT = 0.007
MAX_SL_PCT = 0.025
BE_ARM_R = 0.7
TRAIL_ATR_MULT = 1.1

ADX_MIN = 22
RSI_LONG_MIN = 52
RSI_SHORT_MAX = 48
VOL_MULT = 1.25
BODY_ATR_MIN = 0.35
BREAKOUT_LOOKBACK = 20

ALLOWED_HOURS_UTC = (12, 22)
COOLDOWN_MIN = 90
MAX_TRADES_PER_DAY = 3

DAILY_STOP_R = -3.0
SERIE_STOP_LOSS = 3

MARKET_FILTER_SYMBOL = "BTC/USDT"
MARKET_FILTER_TF = "1h"
MARKET_RSI_LIMIT = 45
MARKET_RSI_BEAR_MAX = 55
USE_MARKET_FILTER = True

INDICATOR_WARMUP = 50
HTF_REFRESH_MIN = 60
MARKET_FILTER_REFRESH_MIN = 60

PARAM_PRESETS = {
    "actif": {
        "TAKE_PROFIT_R": 2.2,
        "STOP_ATR_MULT": 1.3,
        "MIN_SL_PCT": 0.007,
        "MAX_SL_PCT": 0.025,
        "BE_ARM_R": 0.7,
        "TRAIL_ATR_MULT": 1.1,
        "ADX_MIN": 22,
        "RSI_LONG_MIN": 52,
        "RSI_SHORT_MAX": 48,
        "VOL_MULT": 1.25,
        "BODY_ATR_MIN": 0.35,
        "BREAKOUT_LOOKBACK": 20,
        "COOLDOWN_MIN": 90,
        "MAX_TRADES_PER_DAY": 3,
        "DAILY_STOP_R": -3.0,
        "SERIE_STOP_LOSS": 3,
    },
    "optimise": {
        "TAKE_PROFIT_R": 1.45,
        "STOP_ATR_MULT": 0.85,
        "MIN_SL_PCT": 0.0045,
        "MAX_SL_PCT": 0.018,
        "BE_ARM_R": 0.55,
        "TRAIL_ATR_MULT": 1.0,
        "ADX_MIN": 18,
        "RSI_LONG_MIN": 55,
        "RSI_SHORT_MAX": 45,
        "VOL_MULT": 1.1,
        "BODY_ATR_MIN": 0.25,
        "BREAKOUT_LOOKBACK": 14,
        "COOLDOWN_MIN": 120,
        "MAX_TRADES_PER_DAY": 2,
        "DAILY_STOP_R": -2.0,
        "SERIE_STOP_LOSS": 2,
    },
}


def _apply_preset(name: str) -> str:
    key = name.lower()
    params = PARAM_PRESETS.get(key)
    if params is None:
        key = "actif"
        params = PARAM_PRESETS[key]
    for preset_key, value in params.items():
        globals()[preset_key] = value
    return key


PRESET_NAME = _apply_preset(PRESET_NAME)

_raw_paper_runtime = os.getenv("BOT_PAPER_RUNTIME_HOURS")
if _raw_paper_runtime is None:
    PAPER_RUNTIME_HOURS = 48.0 if PAPER_TRADING else 0.0
else:
    try:
        PAPER_RUNTIME_HOURS = float(_raw_paper_runtime)
    except ValueError:
        PAPER_RUNTIME_HOURS = 48.0 if PAPER_TRADING else 0.0
if PAPER_RUNTIME_HOURS < 0:
    PAPER_RUNTIME_HOURS = 0.0

ALLOW_LIVE_SHORTS = os.getenv("BOT_ALLOW_SHORTS", "false").strip().lower() == "true"

LOCAL_TZ = ZoneInfo("America/Toronto")


def _now_str() -> str:
    return datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _log_candle_scan(symbol: str, candle_ts: pd.Timestamp) -> None:
    if pd.notna(candle_ts):
        if candle_ts.tzinfo is None:
            ts_local = candle_ts.tz_localize(timezone.utc).astimezone(LOCAL_TZ)
        else:
            ts_local = candle_ts.tz_convert(LOCAL_TZ)
        ts_str = ts_local.strftime("%Y-%m-%d %H:%M")
    else:
        ts_str = "unknown"
    print(f"[{_now_str()}] Analyse de la bougie {TIMEFRAME} ({ts_str} heure locale) sur {symbol}")

# ---------------------------------------------------------------------------
# EXCHANGE INITIALISATION
# ---------------------------------------------------------------------------
exchange = ccxt.mexc(
    {
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    }
)
exchange.load_markets()


# ---------------------------------------------------------------------------
# TELEGRAM UTILITIES
# ---------------------------------------------------------------------------
def tg_send(text: str) -> None:
    if BACKTEST:
        return
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
    except Exception:
        # Silence network errors – the trading loop should not crash on Telegram
        # interruptions.  We intentionally swallow the exception to keep behaviour
        # identical to the legacy script.
        pass


def tg_check_pause() -> bool:
    if BACKTEST:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
        response = requests.get(url, timeout=10)
        data = response.json().get("result", [])
        last_id = int(open(LAST_MSG_FILE).read()) if os.path.exists(LAST_MSG_FILE) else 0
        paused_now = os.path.exists(PAUSE_FILE)
        for message in data:
            uid = message.get("update_id", 0)
            if uid <= last_id:
                continue
            text = (message.get("message", {}).get("text") or "").lower()
            if "pause" in text and not paused_now:
                open(PAUSE_FILE, "w").close()
                tg_send("⏸ Bot mis en pause via Telegram")
                paused_now = True
            elif "resume" in text and paused_now:
                try:
                    os.remove(PAUSE_FILE)
                except Exception:
                    pass
                tg_send("▶️ Bot relancé via Telegram")
                paused_now = False
            with open(LAST_MSG_FILE, "w") as handle:
                handle.write(str(uid))
        return paused_now
    except Exception:
        return os.path.exists(PAUSE_FILE)


# ---------------------------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------------------------
def fetch_ohlcv(symbol: str, tf: str, since: Optional[int] = None, limit: int = 500) -> pd.DataFrame:
    data = exchange.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def fetch_ohlcv_range(symbol: str, tf: str, start_iso: str, end_iso: str, limit: int = 1000) -> pd.DataFrame:
    start_ms = exchange.parse8601(start_iso)
    end_ms = exchange.parse8601(end_iso)
    frames = []
    cursor = start_ms

    if tf.endswith("m"):
        step = int(tf[:-1]) * 60 * 1000 * (limit - 10)
    elif tf.endswith("h"):
        step = int(tf[:-1]) * 60 * 60 * 1000 * (limit - 10)
    else:
        step = 60 * 1000 * (limit - 10)

    while cursor < end_ms:
        df = fetch_ohlcv(symbol, tf, since=cursor, limit=limit)
        if df.empty:
            break
        frames.append(df)
        last_ts = int(df["timestamp"].iloc[-1].value // 10**6)
        cursor = last_ts + 1
        if len(df) < limit // 4:
            cursor += step

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
    )
    start = pd.to_datetime(start_iso, utc=True)
    end = pd.to_datetime(end_iso, utc=True)
    out = out[(out["timestamp"] >= start) & (out["timestamp"] < end)].copy()
    out.reset_index(drop=True, inplace=True)
    return out


def build_market_filter(start_iso: str, end_iso: str) -> Optional[pd.DataFrame]:
    if not USE_MARKET_FILTER:
        return None
    try:
        market_df = fetch_ohlcv_range(
            MARKET_FILTER_SYMBOL,
            MARKET_FILTER_TF,
            start_iso,
            end_iso,
            limit=1000,
        )
        if market_df.empty:
            return None
        market_df = apply_indicators(market_df)
        market_df["bull_ok"] = (
            (market_df["ema_fast"] > market_df["ema_slow"])
            & (market_df["rsi"] >= MARKET_RSI_LIMIT)
        )
        market_df["bear_ok"] = (
            (market_df["ema_fast"] < market_df["ema_slow"])
            & (market_df["rsi"] <= MARKET_RSI_BEAR_MAX)
        )
        return market_df[["timestamp", "bull_ok", "bear_ok", "rsi", "ema_fast", "ema_slow"]].copy()
    except Exception:
        return None


def market_bias_allows(market_df: Optional[pd.DataFrame], ts: pd.Timestamp, bullish: bool) -> bool:
    if not USE_MARKET_FILTER or market_df is None or market_df.empty:
        return True
    try:
        position = market_df["timestamp"].searchsorted(ts)
        idx = int(position) - 1
        if idx < 0:
            return False
        column = "bull_ok" if bullish else "bear_ok"
        return bool(market_df[column].iloc[idx])
    except Exception:
        return True


# ---------------------------------------------------------------------------
# INDICATORS (WILDER SMOOTHING)
# ---------------------------------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _wilders(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _wilders(gain, period)
    avg_loss = _wilders(loss, period)
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.ffill()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return _wilders(tr, period)


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = _wilders(tr, period)

    plus_di = 100 * (_wilders(plus_dm, period) / atr.replace(0, pd.NA))
    minus_di = 100 * (_wilders(minus_dm, period) / atr.replace(0, pd.NA))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)) * 100
    return _wilders(dx, period).ffill()


def apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["ema_fast"] = ema(out["close"], 9)
    out["ema_slow"] = ema(out["close"], 21)
    out["rsi"] = compute_rsi(out["close"], 14)
    out["atr"] = compute_atr(out, 14)
    out["adx"] = compute_adx(out, 14)
    out["vol_ma"] = out["volume"].rolling(30).mean()
    out["body"] = (out["close"] - out["open"]).abs()
    return out


# ---------------------------------------------------------------------------
# HIGHER TIMEFRAME BIAS
# ---------------------------------------------------------------------------
def fetch_htf_bias(symbol: str) -> Dict[str, Optional[float]]:
    try:
        df_higher = fetch_ohlcv(symbol, "1h", limit=200)
        if df_higher.empty:
            return {"bull": False, "bear": False, "rsi": None}
        df_higher = apply_indicators(df_higher)
        candle = df_higher.iloc[-1]
        bull = bool(candle["ema_fast"] > candle["ema_slow"] and candle["rsi"] >= 50)
        bear = bool(candle["ema_fast"] < candle["ema_slow"] and candle["rsi"] <= 50)
        rsi = float(candle["rsi"]) if pd.notna(candle["rsi"]) else None
        return {"bull": bull, "bear": bear, "rsi": rsi}
    except Exception:
        return {"bull": False, "bear": False, "rsi": None}


# ---------------------------------------------------------------------------
# ENTRY CONDITIONS
# ---------------------------------------------------------------------------
def _vol_threshold(df: pd.DataFrame, vol_mult: float) -> float:
    vol_ma = df["vol_ma"].iloc[-1]
    if pd.isna(vol_ma):
        return 0.0
    return vol_ma * max(1.0, vol_mult)


def _atr_threshold(df: pd.DataFrame) -> Optional[float]:
    atr_value = df["atr"].iloc[-1]
    if pd.isna(atr_value):
        return None
    return atr_value


def _has_breakout(df: pd.DataFrame, bullish: bool) -> bool:
    if len(df) < BREAKOUT_LOOKBACK + 1:
        return False
    recent_slice = df.iloc[-BREAKOUT_LOOKBACK - 1 : -1]
    current = df.iloc[-1]
    if bullish:
        return current["close"] > recent_slice["high"].max()
    return current["close"] < recent_slice["low"].min()


def should_long(df: pd.DataFrame, vol_mult: float) -> Tuple[bool, Dict[str, bool]]:
    if len(df) < INDICATOR_WARMUP:
        return False, {"warmup": True}
    current = df.iloc[-1]
    atr_value = _atr_threshold(df)
    conds = {
        "trend": bool(current["ema_fast"] > current["ema_slow"]),
        "rsi": bool(pd.notna(current["rsi"]) and current["rsi"] >= RSI_LONG_MIN),
        "adx": bool(pd.notna(current["adx"]) and current["adx"] >= ADX_MIN),
        "volume": bool(current["volume"] >= _vol_threshold(df, vol_mult)),
        "body": bool(atr_value is not None and current["body"] >= atr_value * BODY_ATR_MIN),
        "breakout": _has_breakout(df, bullish=True),
    }
    ok = all(conds.values())
    return ok, conds


def should_short(df: pd.DataFrame, vol_mult: float) -> Tuple[bool, Dict[str, bool]]:
    if len(df) < INDICATOR_WARMUP:
        return False, {"warmup": True}
    current = df.iloc[-1]
    atr_value = _atr_threshold(df)
    conds = {
        "trend": bool(current["ema_fast"] < current["ema_slow"]),
        "rsi": bool(pd.notna(current["rsi"]) and current["rsi"] <= RSI_SHORT_MAX),
        "adx": bool(pd.notna(current["adx"]) and current["adx"] >= ADX_MIN),
        "volume": bool(current["volume"] >= _vol_threshold(df, vol_mult)),
        "body": bool(atr_value is not None and current["body"] >= atr_value * BODY_ATR_MIN),
        "breakout": _has_breakout(df, bullish=False),
    }
    ok = all(conds.values())
    return ok, conds


# ---------------------------------------------------------------------------
# LOGGING UTILITIES
# ---------------------------------------------------------------------------
def ensure_log_header() -> None:
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, "w", encoding="utf-8") as handle:
            handle.write("timestamp,Paire,Direction,Entrée,Sortie,PnL (%),PnL ($),Résultat\n")


def log_trade(
    symbol: str,
    side: str,
    entry: float,
    exit_price: float,
    pnl_pct: float,
    reason: Optional[str] = None,
) -> None:
    ensure_log_header()
    result = "Gagné" if pnl_pct > 0 else "Perdu"
    row = [
        datetime.now(timezone.utc).isoformat(),
        symbol,
        side,
        f"{entry:.5f}",
        f"{exit_price:.5f}",
        f"{pnl_pct * 100:.2f}%",
        f"{(pnl_pct * CAPITAL_PER_TRADE):.2f}$",
        result,
    ]
    with open(LOG_FILE, "a", encoding="utf-8") as handle:
        handle.write(",".join(map(str, row)) + "\n")

    outcome_pct = pnl_pct * 100
    outcome_cash = pnl_pct * CAPITAL_PER_TRADE
    reason_hint = f" ({reason})" if reason else ""
    emoji = "✅" if pnl_pct > 0 else "❌"
    print(
        f"   {emoji} {symbol} {side} fermé{reason_hint} | sortie={exit_price:.5f} | "
        f"PnL={outcome_pct:.2f}% ({outcome_cash:.2f}$)"
    )


def _notional_to_amount(symbol: str, price: float) -> Optional[float]:
    if price <= 0:
        return None
    amount = CAPITAL_PER_TRADE / price
    try:
        return float(exchange.amount_to_precision(symbol, amount))
    except Exception:
        return float(round(amount, 8))


def _extract_fill_price(order: dict, fallback: float) -> float:
    for key in ("average", "avgPrice", "price", "info"):
        value = order.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
        if isinstance(value, dict):
            nested_price = value.get("price")
            if nested_price:
                try:
                    return float(nested_price)
                except (TypeError, ValueError):
                    continue
    return float(fallback)


# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------
def is_allowed_hour(ts_utc: pd.Timestamp) -> bool:
    start_hour, end_hour = ALLOWED_HOURS_UTC
    return start_hour <= int(ts_utc.hour) < end_hour


@dataclass
class Position:
    side: str
    entry: float
    sl_price: float
    tp_price: float
    sl_pct: float
    amount: Optional[float] = None

    def unrealised_r_multiple(self, price: float) -> float:
        gross_pct = (
            (price - self.entry) / self.entry
            if self.side == "LONG"
            else (self.entry - price) / self.entry
        )
        return gross_pct / max(abs(self.sl_pct), 1e-6)


def _update_trailing_stop(position: Position, close_price: float, atr_value: Optional[float]) -> None:
    if atr_value is None or math.isnan(atr_value) or atr_value <= 0:
        return
    if position.side == "LONG":
        trail = close_price - TRAIL_ATR_MULT * atr_value
        position.sl_price = max(position.sl_price, trail)
    else:
        trail = close_price + TRAIL_ATR_MULT * atr_value
        position.sl_price = min(position.sl_price, trail)


def _check_exit(position: Position, candle: pd.Series) -> Tuple[Optional[float], Optional[str]]:
    if position.side == "LONG":
        hit_tp = candle["high"] >= position.tp_price
        hit_sl = candle["low"] <= position.sl_price
    else:
        hit_tp = candle["low"] <= position.tp_price
        hit_sl = candle["high"] >= position.sl_price

    if hit_tp:
        return position.tp_price, "TP"
    if hit_sl:
        return position.sl_price, "SL"
    return None, None


def _refresh_bias_if_needed(
    symbol: str,
    last_refresh: Optional[pd.Timestamp],
    current_ts: pd.Timestamp,
    cached_bias: Dict[str, Optional[float]],
) -> Tuple[pd.Timestamp, Dict[str, Optional[float]]]:
    if last_refresh is None or (current_ts - last_refresh) >= pd.Timedelta(minutes=HTF_REFRESH_MIN):
        return current_ts, fetch_htf_bias(symbol)
    return last_refresh, cached_bias


def run_backtest() -> None:
    start_iso = f"{BT_START}T00:00:00Z"
    end_iso = f"{BT_END}T00:00:00Z"
    print(f"\n📊 Backtest {BT_START} → {BT_END} | tf={TIMEFRAME}")

    with open(LOG_FILE, "w", encoding="utf-8") as handle:
        handle.write("timestamp,Paire,Direction,Entrée,Sortie,PnL (%),PnL ($),Résultat\n")

    market_filter_df = build_market_filter(start_iso, end_iso)
    if market_filter_df is not None and not market_filter_df.empty:
        last_row = market_filter_df.iloc[-1]
        bull_label = "Oui" if last_row["bull_ok"] else "Non"
        bear_label = "Oui" if last_row["bear_ok"] else "Non"
        print(
            "[Filtre marché] "
            f"RSI={last_row['rsi']:.1f} | EMA9 {int(last_row['ema_fast'])} > "
            f"EMA21 {int(last_row['ema_slow'])}? {bull_label} | Bear? {bear_label}"
        )
    else:
        print("[Filtre marché] Indisponible (pas de données)")

    total_trades = 0
    wins = 0
    pnl_net_dollars = 0.0
    pnl_gains_dollars = 0.0
    pnl_losses_dollars = 0.0
    pct_gains_total = 0.0
    pct_losses_total = 0.0
    pair_pnl: Dict[str, float] = {}

    for symbol in SYMBOLS:
        print(f"🚀 {symbol}…")
        df = fetch_ohlcv_range(symbol, TIMEFRAME, start_iso, end_iso, limit=1000)
        if df.empty or len(df) < INDICATOR_WARMUP:
            print("  (pas assez de données)")
            continue
        df = apply_indicators(df)

        last_bias_refresh: Optional[pd.Timestamp] = None
        bias_cache = {"bull": False, "bear": False, "rsi": None}

        last_trade_ts = pd.Timestamp(0, tz=timezone.utc)
        daily_r: Dict[Tuple[str, date], float] = {}
        streak_losses = 0
        trades_today: Dict[Tuple[str, date], int] = {}

        position: Optional[Position] = None

        for i in range(INDICATOR_WARMUP, len(df)):
            candle = df.iloc[i]
            candle_ts = df["timestamp"].iloc[i]

            if (candle_ts - last_trade_ts) < pd.Timedelta(minutes=COOLDOWN_MIN):
                continue

            last_bias_refresh, bias_cache = _refresh_bias_if_needed(
                symbol, last_bias_refresh, candle_ts, bias_cache
            )

            if not is_allowed_hour(candle_ts):
                if position is not None:
                    atr_value = df["atr"].iloc[i]
                    atr = float(atr_value) if pd.notna(atr_value) else None
                    _update_trailing_stop(position, candle["close"], atr)
                continue

            day_key = (symbol, candle_ts.date())
            if daily_r.get(day_key, 0.0) <= DAILY_STOP_R:
                continue
            if streak_losses >= SERIE_STOP_LOSS:
                continue
            if trades_today.get(day_key, 0) >= MAX_TRADES_PER_DAY:
                continue

            atr_value = df["atr"].iloc[i]
            atr = float(atr_value) if pd.notna(atr_value) else None

            if position is not None:
                _update_trailing_stop(position, candle["close"], atr)
                exit_price, reason = _check_exit(position, candle)
                if exit_price is None:
                    continue

                gross_pct = (
                    (exit_price - position.entry) / position.entry
                    if position.side == "LONG"
                    else (position.entry - exit_price) / position.entry
                )
                pnl_pct = gross_pct - FEES_ROUNDTRIP
                log_trade(symbol, position.side, position.entry, exit_price, pnl_pct, reason)

                sl_pct_abs = max(abs(position.sl_pct), 1e-6)
                r_multiple = pnl_pct / sl_pct_abs
                if pnl_pct <= 0:
                    streak_losses += 1
                    pnl_losses_dollars += abs(pnl_pct * CAPITAL_PER_TRADE)
                    pct_losses_total += abs(pnl_pct * 100)
                else:
                    streak_losses = 0
                    wins += 1
                    pnl_gains_dollars += pnl_pct * CAPITAL_PER_TRADE
                    pct_gains_total += pnl_pct * 100

                pnl_net_dollars += pnl_pct * CAPITAL_PER_TRADE
                pair_pnl[symbol] = pair_pnl.get(symbol, 0.0) + (pnl_pct * CAPITAL_PER_TRADE)

                daily_r[day_key] = daily_r.get(day_key, 0.0) + r_multiple
                trades_today[day_key] = trades_today.get(day_key, 0) + 1
                last_trade_ts = candle_ts
                total_trades += 1
                position = None
                continue

            ok_long, _ = should_long(df.iloc[: i + 1], VOL_MULT)
            ok_short, _ = should_short(df.iloc[: i + 1], VOL_MULT)
            bull_market = market_bias_allows(market_filter_df, candle_ts, bullish=True)
            bear_market = market_bias_allows(market_filter_df, candle_ts, bullish=False)

            side: Optional[str]
            if ok_long and not ok_short and bias_cache.get("bull") and bull_market:
                side = "LONG"
            elif ok_short and not ok_long and bias_cache.get("bear") and bear_market:
                side = "SHORT"
            else:
                side = None

            if side is None or atr is None or math.isnan(atr) or atr <= 0:
                continue

            sl_pct = STOP_ATR_MULT * (atr / float(candle["close"]))
            sl_pct = min(max(sl_pct, MIN_SL_PCT), MAX_SL_PCT)
            tp_pct = TAKE_PROFIT_R * sl_pct

            entry_price = float(candle["close"])
            if side == "LONG":
                sl_price = entry_price * (1 - sl_pct)
                tp_price = entry_price * (1 + tp_pct)
            else:
                sl_price = entry_price * (1 + sl_pct)
                tp_price = entry_price * (1 - tp_pct)

            position = Position(
                side=side,
                entry=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                sl_pct=sl_pct,
            )
            entry_emoji = "🟢" if side == "LONG" else "🔴"
            print(
                f"   {entry_emoji} {symbol} {side} ouvert @ {entry_price:.5f} | "
                f"SL={sl_price:.5f} | TP={tp_price:.5f}"
            )
            last_trade_ts = candle_ts

    if total_trades == 0:
        print("\nℹ️ Aucun trade loggé.")
        return

    losses = total_trades - wins
    winrate = (wins / total_trades * 100) if total_trades else 0.0
    best_pair = max(pair_pnl.items(), key=lambda item: item[1])[0] if pair_pnl else "N/A"

    print(
        "\n"
        f"📈 Résumé Backtest {BT_START} → {BT_END} (preset={PRESET_NAME}) "
        f"Trades : {total_trades} | Gagnés : {wins} | Perdus : {losses} | Winrate : {winrate:.2f}% "
        f"PnL net : {pnl_net_dollars:.2f}$ | Gains : {pnl_gains_dollars:.2f}$ ({pct_gains_total:.2f}%) | "
        f"Pertes : {pnl_losses_dollars:.2f}$ ({pct_losses_total:.2f}%) | Meilleure paire : {best_pair}"
    )


# ---------------------------------------------------------------------------
# LIVE (PAPER)
# ---------------------------------------------------------------------------
def run_live_paper() -> None:
    tg_send("🟢 Bot lancé (paper) 15m + biais 1h")
    open_pos: Dict[str, Position] = {}
    last_trade_ts = {symbol: pd.Timestamp(0, tz=timezone.utc) for symbol in SYMBOLS}
    daily_r: Dict[Tuple[str, date], float] = {}
    streak_losses = {symbol: 0 for symbol in SYMBOLS}
    trades_today: Dict[Tuple[str, date], int] = {}
    bias_cache: Dict[str, Dict[str, Optional[float]]] = {symbol: {"bull": False, "bear": False, "rsi": None} for symbol in SYMBOLS}
    bias_refresh: Dict[str, Optional[pd.Timestamp]] = {symbol: None for symbol in SYMBOLS}
    market_bias_live: Dict[str, Optional[float]] = {"bull": True, "bear": True, "rsi": None}
    last_market_refresh: Optional[pd.Timestamp] = None
    last_logged_candle: Dict[str, Optional[pd.Timestamp]] = {symbol: None for symbol in SYMBOLS}
    runtime_deadline: Optional[pd.Timestamp] = None
    if PAPER_RUNTIME_HOURS > 0:
        runtime_deadline = pd.Timestamp(datetime.now(timezone.utc)) + pd.Timedelta(hours=PAPER_RUNTIME_HOURS)

    def _announce_runtime_stop() -> None:
        hours_value = int(PAPER_RUNTIME_HOURS) if PAPER_RUNTIME_HOURS.is_integer() else PAPER_RUNTIME_HOURS
        message = f"⏹ Durée paper de {hours_value}h atteinte, arrêt du bot."
        print(message)
        tg_send(message)

    while True:
        now_utc = datetime.now(timezone.utc)
        if runtime_deadline is not None and now_utc >= runtime_deadline:
            _announce_runtime_stop()
            return
        if tg_check_pause():
            print("⏸ Pause active…")
            time.sleep(15)
            continue
        if USE_MARKET_FILTER and (
            last_market_refresh is None
            or (now_utc - last_market_refresh) >= pd.Timedelta(minutes=MARKET_FILTER_REFRESH_MIN)
        ):
            market_bias_live = fetch_htf_bias(MARKET_FILTER_SYMBOL)
            last_market_refresh = now_utc
        for symbol in SYMBOLS:
            try:
                df = fetch_ohlcv(symbol, TIMEFRAME, limit=300)
                if df.empty or len(df) < INDICATOR_WARMUP:
                    continue
                df = apply_indicators(df)
                candle = df.iloc[-1]
                candle_ts = df["timestamp"].iloc[-1]

                if (
                    last_logged_candle[symbol] is None
                    or candle_ts > last_logged_candle[symbol]
                ):
                    _log_candle_scan(symbol, candle_ts)
                    last_logged_candle[symbol] = candle_ts

                if (candle_ts - last_trade_ts[symbol]) < pd.Timedelta(minutes=COOLDOWN_MIN):
                    continue

                bias_refresh[symbol], bias_cache[symbol] = _refresh_bias_if_needed(
                    symbol, bias_refresh[symbol], candle_ts, bias_cache[symbol]
                )

                if not is_allowed_hour(candle_ts):
                    if symbol in open_pos:
                        atr_value = df["atr"].iloc[-1]
                        atr = float(atr_value) if pd.notna(atr_value) else None
                        _update_trailing_stop(open_pos[symbol], candle["close"], atr)
                    continue

                day_key = (symbol, candle_ts.date())
                if daily_r.get(day_key, 0.0) <= DAILY_STOP_R:
                    continue
                if streak_losses[symbol] >= SERIE_STOP_LOSS:
                    continue
                if trades_today.get(day_key, 0) >= MAX_TRADES_PER_DAY:
                    continue

                atr_value = df["atr"].iloc[-1]
                atr = float(atr_value) if pd.notna(atr_value) else None

                if symbol in open_pos:
                    position = open_pos[symbol]
                    _update_trailing_stop(position, candle["close"], atr)
                    exit_price, reason = _check_exit(position, candle)
                    if exit_price is None:
                        continue
                    gross_pct = (
                        (exit_price - position.entry) / position.entry
                        if position.side == "LONG"
                        else (position.entry - exit_price) / position.entry
                    )
                    pnl_pct = gross_pct - FEES_ROUNDTRIP
                    log_trade(symbol, position.side, position.entry, exit_price, pnl_pct, reason)
                    sl_pct_abs = max(abs(position.sl_pct), 1e-6)
                    r_multiple = pnl_pct / sl_pct_abs
                    if pnl_pct <= 0:
                        streak_losses[symbol] += 1
                    else:
                        streak_losses[symbol] = 0
                    daily_r[day_key] = daily_r.get(day_key, 0.0) + r_multiple
                    trades_today[day_key] = trades_today.get(day_key, 0) + 1
                    last_trade_ts[symbol] = candle_ts
                    reason_suffix = f" ({reason})" if reason else ""
                    tg_send(
                        f"✅ {symbol} {position.side} fermé{reason_suffix} | pnl={pnl_pct * 100:.2f}%"
                    )
                    del open_pos[symbol]
                    continue

                ok_long, _ = should_long(df, VOL_MULT)
                ok_short, _ = should_short(df, VOL_MULT)

                if ok_long and not ok_short and bias_cache[symbol].get("bull") and market_bias_live.get("bull"):
                    side = "LONG"
                elif ok_short and not ok_long and bias_cache[symbol].get("bear") and market_bias_live.get("bear"):
                    side = "SHORT"
                else:
                    continue

                if atr is None or math.isnan(atr) or atr <= 0:
                    continue

                sl_pct = STOP_ATR_MULT * (atr / float(candle["close"]))
                sl_pct = min(max(sl_pct, MIN_SL_PCT), MAX_SL_PCT)
                tp_pct = TAKE_PROFIT_R * sl_pct

                entry_price = float(candle["close"])
                if side == "LONG":
                    sl_price = entry_price * (1 - sl_pct)
                    tp_price = entry_price * (1 + tp_pct)
                else:
                    sl_price = entry_price * (1 + sl_pct)
                    tp_price = entry_price * (1 - tp_pct)

                open_pos[symbol] = Position(
                    side=side,
                    entry=entry_price,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    sl_pct=sl_pct,
                )
                last_trade_ts[symbol] = candle_ts
                entry_emoji = "🟢" if side == "LONG" else "🔴"
                print(
                    f"   {entry_emoji} {symbol} {side} ouvert @ {entry_price:.5f} | "
                    f"SL={sl_price:.5f} | TP={tp_price:.5f}"
                )
                tg_send(f"📥 {symbol} {side} (paper) | entry={entry_price}")
            except Exception as exc:
                print(f"[{symbol}] erreur: {exc}")
        if runtime_deadline is not None and datetime.now(timezone.utc) >= runtime_deadline:
            _announce_runtime_stop()
            return
        time.sleep(20)


def run_live_trading() -> None:
    tg_send("🟢 Bot lancé (live) 15m + biais 1h")
    open_pos: Dict[str, Position] = {}
    last_trade_ts = {symbol: pd.Timestamp(0, tz=timezone.utc) for symbol in SYMBOLS}
    daily_r: Dict[Tuple[str, date], float] = {}
    streak_losses = {symbol: 0 for symbol in SYMBOLS}
    trades_today: Dict[Tuple[str, date], int] = {}
    bias_cache: Dict[str, Dict[str, Optional[float]]] = {symbol: {"bull": False, "bear": False, "rsi": None} for symbol in SYMBOLS}
    bias_refresh: Dict[str, Optional[pd.Timestamp]] = {symbol: None for symbol in SYMBOLS}
    market_bias_live: Dict[str, Optional[float]] = {"bull": True, "bear": True, "rsi": None}
    last_market_refresh: Optional[pd.Timestamp] = None
    last_logged_candle: Dict[str, Optional[pd.Timestamp]] = {symbol: None for symbol in SYMBOLS}

    while True:
        now_utc = datetime.now(timezone.utc)
        if tg_check_pause():
            print("⏸ Pause active…")
            time.sleep(15)
            continue
        if USE_MARKET_FILTER and (
            last_market_refresh is None
            or (now_utc - last_market_refresh) >= pd.Timedelta(minutes=MARKET_FILTER_REFRESH_MIN)
        ):
            market_bias_live = fetch_htf_bias(MARKET_FILTER_SYMBOL)
            last_market_refresh = now_utc
        for symbol in SYMBOLS:
            try:
                df = fetch_ohlcv(symbol, TIMEFRAME, limit=300)
                if df.empty or len(df) < INDICATOR_WARMUP:
                    continue
                df = apply_indicators(df)
                candle = df.iloc[-1]
                candle_ts = df["timestamp"].iloc[-1]

                if (
                    last_logged_candle[symbol] is None
                    or candle_ts > last_logged_candle[symbol]
                ):
                    _log_candle_scan(symbol, candle_ts)
                    last_logged_candle[symbol] = candle_ts

                if (candle_ts - last_trade_ts[symbol]) < pd.Timedelta(minutes=COOLDOWN_MIN):
                    continue

                bias_refresh[symbol], bias_cache[symbol] = _refresh_bias_if_needed(
                    symbol, bias_refresh[symbol], candle_ts, bias_cache[symbol]
                )

                if not is_allowed_hour(candle_ts):
                    if symbol in open_pos:
                        atr_value = df["atr"].iloc[-1]
                        atr = float(atr_value) if pd.notna(atr_value) else None
                        _update_trailing_stop(open_pos[symbol], candle["close"], atr)
                    continue

                day_key = (symbol, candle_ts.date())
                if daily_r.get(day_key, 0.0) <= DAILY_STOP_R:
                    continue
                if streak_losses[symbol] >= SERIE_STOP_LOSS:
                    continue
                if trades_today.get(day_key, 0) >= MAX_TRADES_PER_DAY:
                    continue

                atr_value = df["atr"].iloc[-1]
                atr = float(atr_value) if pd.notna(atr_value) else None

                if symbol in open_pos:
                    position = open_pos[symbol]
                    _update_trailing_stop(position, candle["close"], atr)
                    exit_price, reason = _check_exit(position, candle)
                    if exit_price is None:
                        continue

                    try:
                        exit_amount = position.amount or _notional_to_amount(symbol, position.entry)
                        if exit_amount is None or exit_amount <= 0:
                            raise ValueError("invalid exit amount")
                        exit_side = "sell" if position.side == "LONG" else "buy"
                        exit_order = exchange.create_order(symbol, "market", exit_side, exit_amount)
                        exit_fill = _extract_fill_price(exit_order, exit_price)
                    except Exception as exc:
                        print(f"[{symbol}] fermeture live impossible: {exc}")
                        continue

                    gross_pct = (
                        (exit_fill - position.entry) / position.entry
                        if position.side == "LONG"
                        else (position.entry - exit_fill) / position.entry
                    )
                    pnl_pct = gross_pct - FEES_ROUNDTRIP
                    log_trade(symbol, position.side, position.entry, exit_fill, pnl_pct, reason)

                    sl_pct_abs = max(abs(position.sl_pct), 1e-6)
                    r_multiple = pnl_pct / sl_pct_abs
                    if pnl_pct <= 0:
                        streak_losses[symbol] += 1
                    else:
                        streak_losses[symbol] = 0
                    daily_r[day_key] = daily_r.get(day_key, 0.0) + r_multiple
                    trades_today[day_key] = trades_today.get(day_key, 0) + 1
                    last_trade_ts[symbol] = candle_ts
                    reason_suffix = f" ({reason})" if reason else ""
                    tg_send(f"✅ {symbol} {position.side} fermé{reason_suffix} | pnl={pnl_pct * 100:.2f}%")
                    del open_pos[symbol]
                    continue

                ok_long, _ = should_long(df, VOL_MULT)
                ok_short, _ = should_short(df, VOL_MULT)

                if ok_long and not ok_short and bias_cache[symbol].get("bull") and market_bias_live.get("bull"):
                    side = "LONG"
                elif ok_short and not ok_long and bias_cache[symbol].get("bear") and market_bias_live.get("bear"):
                    side = "SHORT"
                else:
                    continue

                if side == "SHORT" and not ALLOW_LIVE_SHORTS:
                    print(f"[{symbol}] Signal SHORT ignoré (BOT_ALLOW_SHORTS désactivé en live).")
                    continue

                if atr is None or math.isnan(atr) or atr <= 0:
                    continue

                sl_pct = STOP_ATR_MULT * (atr / float(candle["close"]))
                sl_pct = min(max(sl_pct, MIN_SL_PCT), MAX_SL_PCT)
                tp_pct = TAKE_PROFIT_R * sl_pct

                entry_price = float(candle["close"])
                order_amount = _notional_to_amount(symbol, entry_price)
                if order_amount is None or order_amount <= 0:
                    continue

                order_side = "buy" if side == "LONG" else "sell"

                try:
                    order = exchange.create_order(symbol, "market", order_side, order_amount)
                    fill_price = _extract_fill_price(order, entry_price)
                except Exception as exc:
                    print(f"[{symbol}] ouverture live impossible: {exc}")
                    continue

                if side == "LONG":
                    sl_price = fill_price * (1 - sl_pct)
                    tp_price = fill_price * (1 + tp_pct)
                else:
                    sl_price = fill_price * (1 + sl_pct)
                    tp_price = fill_price * (1 - tp_pct)

                open_pos[symbol] = Position(
                    side=side,
                    entry=fill_price,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    sl_pct=sl_pct,
                    amount=order_amount,
                )
                last_trade_ts[symbol] = candle_ts
                entry_emoji = "🟢" if side == "LONG" else "🔴"
                print(
                    f"   {entry_emoji} {symbol} {side} (live) ouvert @ {fill_price:.5f} | "
                    f"SL={sl_price:.5f} | TP={tp_price:.5f} | taille={order_amount}"
                )
                tg_send(f"📥 {symbol} {side} (live) | entry={fill_price}")
            except Exception as exc:
                print(f"[{symbol}] erreur live: {exc}")
        time.sleep(20)


# ---------------------------------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if BACKTEST:
        run_backtest()
    elif PAPER_TRADING:
        run_live_paper()
    elif LIVE_TRADING:
        run_live_trading()
    else:
        raise SystemExit(
            "BOT_MODE inconnu. Utilisez 'backtest', 'paper' ou 'live'."
        )

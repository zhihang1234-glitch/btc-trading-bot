import time
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "http://localhost:8080/webhook")
SYMBOL = "BTC-USD"
GRANULARITY = 3600  # 1 hour candles
CANDLE_LIMIT = 100  # enough for ATR(14), EMA(21)
COOLDOWN_SECONDS = 4 * 3600  # 4 hours

last_signal_time = {"LONG": 0, "SHORT": 0}

# ===== FETCH CANDLES =====
def get_candles():
    url = f"https://api.exchange.coinbase.com/products/{SYMBOL}/candles"
    params = {"granularity": GRANULARITY}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # Coinbase returns [time, low, high, open, close, volume]
    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df.astype(float)
    return df

# ===== INDICATORS =====
def calculate_indicators(df):
    # EMA
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    # VWAP (rolling daily approximation over all candles)
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (df["tp"] * df["volume"]).cumsum() / df["volume"].cumsum()

    # ATR(14)
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = df[["high", "low", "prev_close"]].apply(
        lambda row: max(
            row["high"] - row["low"],
            abs(row["high"] - row["prev_close"]),
            abs(row["low"] - row["prev_close"])
        ), axis=1
    )
    df["atr"] = df["tr"].ewm(span=14, adjust=False).mean()

    return df

# ===== SIGNAL CHECK =====
def check_signals(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close = latest["close"]
    ema9 = latest["ema9"]
    ema21 = latest["ema21"]
    vwap = latest["vwap"]
    volume = latest["volume"]
    atr = latest["atr"]

    long_signal = (
        close > vwap and
        ema9 > ema21 and
        close <= ema21 * 1.003  # within 0.3% of ema21 (pullback)
    )

    short_signal = (
        close < vwap and
        ema9 < ema21 and
        close >= ema21 * 0.997  # within 0.3% of ema21 (pullback)
    )

    return long_signal, short_signal, {
        "price": round(close, 2),
        "ema9": round(ema9, 2),
        "ema21": round(ema21, 2),
        "vwap": round(vwap, 2),
        "volume": round(volume, 8),
        "atr": round(atr, 2)
    }

# ===== COOLDOWN CHECK =====
def is_on_cooldown(signal):
    elapsed = time.time() - last_signal_time[signal]
    return elapsed < COOLDOWN_SECONDS

# ===== SEND SIGNAL =====
def send_signal(signal, values):
    payload = {"signal": signal, **values}
    try:
        resp = requests.post(WEBHOOK_URL, json=payload, timeout=5)
        print(f"✅ Signal sent: {signal} @ {values['price']} → {resp.status_code}")
        last_signal_time[signal] = time.time()
    except Exception as e:
        print(f"❌ Failed to send signal: {e}")

# ===== MAIN LOOP =====
def main():
    print(f"🔍 Scanner started — checking {SYMBOL} every 60s")
    print(f"📡 Sending signals to: {WEBHOOK_URL}")

    while True:
        try:
            df = get_candles()
            df = calculate_indicators(df)
            long_signal, short_signal, values = check_signals(df)

            now = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{now}] price={values['price']} ema9={values['ema9']} ema21={values['ema21']} vwap={values['vwap']} atr={values['atr']}")

            if long_signal:
                if is_on_cooldown("LONG"):
                    print("⏳ LONG signal skipped — cooldown active")
                else:
                    print("🟢 LONG signal detected!")
                    send_signal("LONG", values)

            elif short_signal:
                if is_on_cooldown("SHORT"):
                    print("⏳ SHORT signal skipped — cooldown active")
                else:
                    print("🔴 SHORT signal detected!")
                    send_signal("SHORT", values)

            else:
                print("— No signal")

        except Exception as e:
            print(f"❌ Scanner error: {e}")

        time.sleep(60)

if __name__ == "__main__":
    main()

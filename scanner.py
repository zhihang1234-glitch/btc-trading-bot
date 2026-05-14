# Fixed `scanner.py`

```python
import time
import requests
import pandas as pd
import os
import psycopg2
from datetime import datetime, timezone

WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "http://localhost:8080/webhook")
DATABASE_URL = os.environ.get("DATABASE_URL")
SYMBOL = "BTC-USD"
GRANULARITY = 3600  # 1 hour candles

# RSI filter range
RSI_MIN = 40
RSI_MAX = 60

# ===== MEMORY =====
last_signal_candle = {
    "LONG": None,
    "SHORT": None
}

# ===== DB =====
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def init_db_tables():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cooldowns (
                    signal TEXT PRIMARY KEY,
                    last_fired TIMESTAMP
                )
            """)
        conn.commit()
    print("✅ DB tables ready")


# ===== OPEN TRADE CHECK =====
def has_open_trade(signal):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM trades WHERE signal = %s AND status = 'OPEN'",
                    (signal,)
                )
                count = cur.fetchone()[0]
                return count > 0
    except Exception as e:
        print(f"❌ Open trade check failed: {e}")
        return False


# ===== FETCH CANDLES =====
def get_candles():
    url = f"https://api.exchange.coinbase.com/products/{SYMBOL}/candles"
    params = {"granularity": GRANULARITY}

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()

    data = resp.json()

    df = pd.DataFrame(
        data,
        columns=["time", "low", "high", "open", "close", "volume"]
    )

    df = df.sort_values("time").reset_index(drop=True)
    df = df.astype(float)

    return df


# ===== INDICATORS =====
def calculate_indicators(df):
    # EMA
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    # VWAP
    df["tp_price"] = (
        df["high"] + df["low"] + df["close"]
    ) / 3

    df["vwap"] = (
        (df["tp_price"] * df["volume"]).cumsum()
        / df["volume"].cumsum()
    )

    # ATR
    df["prev_close"] = df["close"].shift(1)

    df["tr"] = df[["high", "low", "prev_close"]].apply(
        lambda row: max(
            row["high"] - row["low"],
            abs(row["high"] - row["prev_close"]),
            abs(row["low"] - row["prev_close"])
        ),
        axis=1
    )

    df["atr"] = df["tr"].ewm(span=14, adjust=False).mean()

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)

    df["rsi"] = 100 - (100 / (1 + rs))

    return df


# ===== SIGNAL CHECK =====
def check_signals(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    candle_time = latest["time"]

    close = latest["close"]
    ema9 = latest["ema9"]
    ema21 = latest["ema21"]
    vwap = latest["vwap"]
    volume = latest["volume"]
    atr = latest["atr"]
    rsi = latest["rsi"]

    # ATR validation
    if pd.isna(atr) or atr < 10:
        print(f"⚠️ Invalid ATR ({atr:.2f}), skipping")
        return False, False, {}, candle_time

    # RSI validation
    if pd.isna(rsi):
        print("⚠️ RSI not ready, skipping")
        return False, False, {}, candle_time

    # RSI filter
    rsi_ok = RSI_MIN <= rsi <= RSI_MAX

    # ===== CROSS DETECTION =====
    bullish_cross = (
        prev["ema9"] <= prev["ema21"]
        and latest["ema9"] > latest["ema21"]
    )

    bearish_cross = (
        prev["ema9"] >= prev["ema21"]
        and latest["ema9"] < latest["ema21"]
    )

    # ===== LONG =====
    long_signal = (
        bullish_cross
        and close > vwap
        and close <= ema21 * 1.003
        and rsi_ok
    )

    # ===== SHORT =====
    short_signal = (
        bearish_cross
        and close < vwap
        and close >= ema21 * 0.997
        and rsi_ok
    )

    values = {
        "price": round(close, 2),
        "ema9": round(ema9, 2),
        "ema21": round(ema21, 2),
        "vwap": round(vwap, 2),
        "volume": round(volume, 8),
        "atr": round(atr, 2),
        "rsi": round(rsi, 2)
    }

    return long_signal, short_signal, values, candle_time


# ===== SEND SIGNAL =====
def send_signal(signal, values):
    payload = {
        "signal": signal,
        **values
    }

    try:
        resp = requests.post(
            WEBHOOK_URL,
            json=payload,
            timeout=5
        )

        print(
            f"✅ Signal sent: {signal} @ {values['price']} → {resp.status_code}"
        )

    except Exception as e:
        print(f"❌ Failed to send signal: {e}")


# ===== MAIN LOOP =====
def main():
    global last_signal_candle

    print(f"🔍 Scanner started — checking {SYMBOL}")
    print(f"📡 Sending signals to: {WEBHOOK_URL}")
    print(f"📊 RSI filter: {RSI_MIN}-{RSI_MAX}")

    init_db_tables()

    while True:
        try:
            df = get_candles()
            df = calculate_indicators(df)

            long_signal, short_signal, values, candle_time = check_signals(df)

            if not values:
                time.sleep(60)
                continue

            now = datetime.now(timezone.utc).strftime("%H:%M:%S")

            print(
                f"[{now}] "
                f"price={values['price']} "
                f"ema9={values['ema9']} "
                f"ema21={values['ema21']} "
                f"vwap={values['vwap']} "
                f"atr={values['atr']} "
                f"rsi={values['rsi']}"
            )

            # ===== LONG =====
            if long_signal:
                if candle_time == last_signal_candle["LONG"]:
                    print("⚠️ Duplicate LONG signal blocked")

                elif has_open_trade("LONG"):
                    print("⚠️ Existing LONG trade already open")

                else:
                    print("🟢 LONG signal detected!")
                    send_signal("LONG", values)
                    last_signal_candle["LONG"] = candle_time

            # ===== SHORT =====
            elif short_signal:
                if candle_time == last_signal_candle["SHORT"]:
                    print("⚠️ Duplicate SHORT signal blocked")

                elif has_open_trade("SHORT"):
                    print("⚠️ Existing SHORT trade already open")

                else:
                    print("🔴 SHORT signal detected!")
                    send_signal("SHORT", values)
                    last_signal_candle["SHORT"] = candle_time

            else:
                rsi = values['rsi']

                if rsi > RSI_MAX:
                    print(f"— No signal (RSI {rsi} too high)")

                elif rsi < RSI_MIN:
                    print(f"— No signal (RSI {rsi} too low)")

                else:
                    print("— No signal")

        except Exception as e:
            print(f"❌ Scanner error: {e}")

        time.sleep(60)


if __name__ == "__main__":
    main()
```

# What Changed

## Fixed duplicate spam alerts

* Prevents repeated alerts on the same candle.

## Added EMA crossover detection

* Signals only fire on fresh trend transitions.

## Added open trade lock

* Prevents stacking identical open positions.

## Improved signal quality

* No more constant 100/100 spam every minute.

## Kept your architecture intact

* Compatible with your existing `bot.py`
* Works with your PostgreSQL setup
* Keeps Discord integration unchanged

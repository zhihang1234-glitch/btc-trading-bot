import time
import requests
import pandas as pd
import os
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone

WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "http://localhost:8080/webhook")
DATABASE_URL = os.environ.get("DATABASE_URL")
SYMBOL = "BTC-USD"
GRANULARITY = 3600  # 1 hour candles
COOLDOWN_SECONDS = 4 * 3600  # 4 hours

# ===== DB COOLDOWN =====
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def init_cooldown_table():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cooldowns (
                    signal TEXT PRIMARY KEY,
                    last_fired TIMESTAMP
                )
            """)
            # ensure rows exist
            cur.execute("""
                INSERT INTO cooldowns (signal, last_fired)
                VALUES ('LONG', '2000-01-01'), ('SHORT', '2000-01-01')
                ON CONFLICT (signal) DO NOTHING
            """)
        conn.commit()
    print("✅ Cooldown table ready")

def is_on_cooldown(signal):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT last_fired FROM cooldowns WHERE signal = %s", (signal,))
            row = cur.fetchone()
            if not row:
                return False
            last_fired = row[0].replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - last_fired).total_seconds()
            remaining = COOLDOWN_SECONDS - elapsed
            if remaining > 0:
                hours = int(remaining // 3600)
                mins = int((remaining % 3600) // 60)
                print(f"⏳ {signal} cooldown: {hours}h {mins}m remaining")
                return True
            return False

def set_cooldown(signal):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO cooldowns (signal, last_fired)
                VALUES (%s, NOW())
                ON CONFLICT (signal) DO UPDATE SET last_fired = NOW()
            """, (signal,))
        conn.commit()

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
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    df["tp_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (df["tp_price"] * df["volume"]).cumsum() / df["volume"].cumsum()

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

    close = latest["close"]
    ema9 = latest["ema9"]
    ema21 = latest["ema21"]
    vwap = latest["vwap"]
    volume = latest["volume"]
    atr = latest["atr"]

    # Guard against NaN
    if any(pd.isna([ema9, ema21, vwap, atr])):
        return False, False, {}

    long_signal = (
        close > vwap and
        ema9 > ema21 and
        close <= ema21 * 1.003
    )

    short_signal = (
        close < vwap and
        ema9 < ema21 and
        close >= ema21 * 0.997
    )

    return long_signal, short_signal, {
        "price": round(close, 2),
        "ema9": round(ema9, 2),
        "ema21": round(ema21, 2),
        "vwap": round(vwap, 2),
        "volume": round(volume, 8),
        "atr": round(atr, 2)
    }

# ===== SEND SIGNAL =====
def send_signal(signal, values):
    payload = {"signal": signal, **values}
    try:
        resp = requests.post(WEBHOOK_URL, json=payload, timeout=5)
        print(f"✅ Signal sent: {signal} @ {values['price']} → {resp.status_code}")
        set_cooldown(signal)
    except Exception as e:
        print(f"❌ Failed to send signal: {e}")

# ===== MAIN LOOP =====
def main():
    print(f"🔍 Scanner started — checking {SYMBOL} every 60s")
    print(f"📡 Sending signals to: {WEBHOOK_URL}")

    init_cooldown_table()

    while True:
        try:
            df = get_candles()
            df = calculate_indicators(df)
            long_signal, short_signal, values = check_signals(df)

            if not values:
                print("⚠️ Skipping — NaN in indicators")
                time.sleep(60)
                continue

            now = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{now}] price={values['price']} ema9={values['ema9']} ema21={values['ema21']} vwap={values['vwap']} atr={values['atr']}")

            if long_signal:
                if is_on_cooldown("LONG"):
                    pass  # message already printed in is_on_cooldown
                else:
                    print("🟢 LONG signal detected!")
                    send_signal("LONG", values)

            elif short_signal:
                if is_on_cooldown("SHORT"):
                    pass
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

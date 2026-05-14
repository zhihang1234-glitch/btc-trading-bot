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

# RSI filter range — only take signals when RSI is in this band
RSI_MIN = 40
RSI_MAX = 60

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

# ===== FETCH CANDLES =====
def get_candles():
    url = f"https://api.exchange.coinbase.com/products/{SYMBOL}/candles"
    params = {"granularity": GRANULARITY}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df.astype(float)
    return df

# ===== INDICATORS =====
def calculate_indicators(df):
    # EMA
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    # VWAP
    df["tp_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (df["tp_price"] * df["volume"]).cumsum() / df["volume"].cumsum()

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

    # RSI(14)
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

    close = latest["close"]
    ema9 = latest["ema9"]
    ema21 = latest["ema21"]
    vwap = latest["vwap"]
    volume = latest["volume"]
    atr = latest["atr"]
    rsi = latest["rsi"]

    # Guard against NaN or invalid ATR
    if pd.isna(atr) or atr < 10:
        print(f"⚠️ Invalid ATR ({atr:.2f}), skipping")
        return False, False, {}

    if pd.isna(rsi):
        print("⚠️ RSI not ready, skipping")
        return False, False, {}

    # RSI filter — only trade when RSI is in neutral zone
    rsi_ok = RSI_MIN <= rsi <= RSI_MAX

    long_signal = (
        close > vwap and
        ema9 > ema21 and
        close <= ema21 * 1.003 and
        rsi_ok
    )

    short_signal = (
        close < vwap and
        ema9 < ema21 and
        close >= ema21 * 0.997 and
        rsi_ok
    )

    return long_signal, short_signal, {
        "price": round(close, 2),
        "ema9": round(ema9, 2),
        "ema21": round(ema21, 2),
        "vwap": round(vwap, 2),
        "volume": round(volume, 8),
        "atr": round(atr, 2),
        "rsi": round(rsi, 2)
    }

# ===== SEND SIGNAL =====
def send_signal(signal, values):
    payload = {"signal": signal, **values}
    try:
        resp = requests.post(WEBHOOK_URL, json=payload, timeout=5)
        print(f"✅ Signal sent: {signal} @ {values['price']} → {resp.status_code}")
    except Exception as e:
        print(f"❌ Failed to send signal: {e}")

# ===== MAIN LOOP =====
def main():
    print(f"🔍 Scanner started — checking {SYMBOL} every 60s")
    print(f"📡 Sending signals to: {WEBHOOK_URL}")
    print(f"📊 RSI filter: {RSI_MIN}-{RSI_MAX}")

    init_db_tables()

    while True:
        try:
            df = get_candles()
            df = calculate_indicators(df)
            long_signal, short_signal, values = check_signals(df)

            if not values:
                time.sleep(60)
                continue

            now = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{now}] price={values['price']} ema9={values['ema9']} ema21={values['ema21']} vwap={values['vwap']} atr={values['atr']} rsi={values['rsi']}")

            if long_signal:
                print("🟢 LONG signal detected!")
                send_signal("LONG", values)

            elif short_signal:
                print("🔴 SHORT signal detected!")
                send_signal("SHORT", values)

            else:
                rsi = values['rsi']
                if rsi > RSI_MAX:
                    print(f"— No signal (RSI {rsi} too high for LONG)")
                elif rsi < RSI_MIN:
                    print(f"— No signal (RSI {rsi} too low for SHORT)")
                else:
                    print("— No signal")

        except Exception as e:
            print(f"❌ Scanner error: {e}")

        time.sleep(60)

if __name__ == "__main__":
    main()

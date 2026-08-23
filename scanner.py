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

# RSI filter range. Backtest (300d, 1h, 210+ trades) showed RSI has no
# measurable effect: 30-70 and 45-55 land within noise of each other in every
# sweep row. Set as wide as tested — a narrower band discards trades for no
# demonstrated gain, and a smaller sample makes future evidence worse.
RSI_MIN = 30
RSI_MAX = 70

# ===== DB =====
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def init_db_tables():
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Records the last candle we acted on. Stored in Postgres, not in
            # memory, so a Railway restart cannot replay a candle already traded.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS scanner_state (
                    key TEXT PRIMARY KEY,
                    value BIGINT
                )
            """)
            cur.execute("""
                INSERT INTO scanner_state (key, value) VALUES ('last_candle', 0)
                ON CONFLICT (key) DO NOTHING
            """)
        conn.commit()
    print("✅ DB tables ready")


def get_last_candle():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM scanner_state WHERE key = 'last_candle'")
            row = cur.fetchone()
            return row[0] if row else 0


def set_last_candle(candle_time):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO scanner_state (key, value) VALUES ('last_candle', %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """, (candle_time,))
        conn.commit()


def has_open_trade():
    """The backtest holds one position at a time. Live must do the same, or the
    two are not the same strategy and the backtested expectancy does not apply."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM trades WHERE status = 'OPEN'")
            return cur.fetchone()[0] > 0

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
    # VWAP resets each UTC day. A running cumsum over the whole response would
    # make VWAP depend on how many candles the API happened to return.
    df["tp_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["date"] = pd.to_datetime(df["time"], unit="s").dt.date
    cum_tpv = (df["tp_price"] * df["volume"]).groupby(df["date"]).cumsum()
    cum_vol = df["volume"].groupby(df["date"]).cumsum()
    df["vwap"] = cum_tpv / cum_vol

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
    # iloc[-1] is the CURRENTLY FORMING candle — its close, volume and RSI all
    # keep changing until the hour ends. The backtest evaluates closed candles
    # only, so using [-1] live means testing one strategy and trading another.
    # iloc[-2] is the last fully closed candle.
    if len(df) < 2:
        return False, False, {}

    latest = df.iloc[-2]
    candle_time = int(latest["time"])

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
        "candle_time": candle_time,
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
    payload = {"signal": signal,
               **{k: v for k, v in values.items() if k != "candle_time"}}
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
            candle_time = values["candle_time"]
            candle_str = datetime.fromtimestamp(candle_time, timezone.utc).strftime("%m-%d %H:%M")
            print(f"[{now}] candle={candle_str} price={values['price']} "
                  f"ema9={values['ema9']} ema21={values['ema21']} "
                  f"vwap={values['vwap']} atr={values['atr']} rsi={values['rsi']}")

            if not (long_signal or short_signal):
                print("— No signal")
                time.sleep(60)
                continue

            direction = "LONG" if long_signal else "SHORT"

            # GATE 1: one entry per closed candle. Without this the same candle
            # re-fires every 60s for the whole hour — that is what logged ~150
            # duplicate trades in three hours.
            if candle_time <= get_last_candle():
                print(f"⏸️  {direction} already handled for candle {candle_str}")
                time.sleep(60)
                continue

            # GATE 2: one position at a time, matching the backtest.
            if has_open_trade():
                print(f"⏸️  {direction} skipped — a position is already open")
                set_last_candle(candle_time)
                time.sleep(60)
                continue

            emoji = "🟢" if long_signal else "🔴"
            print(f"{emoji} {direction} signal detected on closed candle {candle_str}")
            send_signal(direction, values)
            set_last_candle(candle_time)

        except Exception as e:
            print(f"❌ Scanner error: {e}")

        time.sleep(60)

if __name__ == "__main__":
    main()

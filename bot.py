print("STARTING BOT...")
import discord
import asyncio
import json
import threading
import time
import requests
import joblib
import os
import psycopg2
import psycopg2.extras
import psycopg2.pool
from flask import Flask, request
from datetime import datetime, timezone, timedelta

# ===== CONFIG =====
TOKEN = os.environ.get("DISCORD_TOKEN")
CHANNEL_ID = int(os.environ.get("CHANNEL_ID", "1497320466715775080"))
DATABASE_URL = os.environ.get("DATABASE_URL")
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
PORT = int(os.environ.get("PORT", 5000))

app = Flask(__name__)
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# ===== CONNECTION POOL =====
# FIX #3: Use a threaded connection pool instead of a new connection per call
_pool = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=DATABASE_URL,
            sslmode="require"
        )
    return _pool

def get_conn():
    return get_pool().getconn()

def release_conn(conn):
    get_pool().putconn(conn)

# Context manager for cleaner usage
class ManagedConn:
    def __enter__(self):
        self.conn = get_conn()
        return self.conn
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        release_conn(self.conn)
        return False

# ===== DATABASE =====
def init_db():
    with ManagedConn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    time TIMESTAMP DEFAULT NOW(),
                    signal TEXT,
                    entry FLOAT,
                    sl FLOAT,
                    tp FLOAT,
                    atr FLOAT,
                    status TEXT DEFAULT 'OPEN',
                    score INT,
                    features JSONB
                )
            """)
            # FIX: only add column if truly missing; avoid running ALTER on every boot
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name='trades' AND column_name='atr'
            """)
            if not cur.fetchone():
                cur.execute("ALTER TABLE trades ADD COLUMN atr FLOAT")
    print("✅ DB initialized")

def log_trade(data, score, features, sl, tp):
    entry = float(data["price"])
    signal = data["signal"]
    atr = float(data.get("atr", 0))

    with ManagedConn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades (signal, entry, sl, tp, atr, score, features)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (signal, entry, sl, tp, atr, score, json.dumps(features)))

def get_open_trades():
    with ManagedConn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM trades WHERE status = 'OPEN'")
            return cur.fetchall()

def update_trade_status(trade_id, status, exit_price):
    # FIX #7: store actual exit price so P&L is accurate
    with ManagedConn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE trades SET status = %s, exit_price = %s WHERE id = %s",
                (status, exit_price, trade_id)
            )

def ensure_exit_price_column():
    """Ensure exit_price column exists for accurate P&L tracking."""
    with ManagedConn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name='trades' AND column_name='exit_price'
            """)
            if not cur.fetchone():
                cur.execute("ALTER TABLE trades ADD COLUMN exit_price FLOAT")

# ===== MODEL =====
model = None

def load_model():
    global model
    try:
        # FIX: use absolute path so working directory doesn't matter
        model = joblib.load(MODEL_FILE)
        print("✅ Model loaded")
    except Exception as e:
        print(f"⚠️ No model found at {MODEL_FILE}: {e}")
        model = None

# ===== PRICE =====
def get_price():
    try:
        url = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
        return float(requests.get(url, timeout=5).json()["price"])
    except Exception as e:
        print(f"❌ Price fetch failed: {e}")
        return None

# ===== SAFE FLOAT =====
def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

# ===== ATR-BASED SL/TP =====
def calculate_sl_tp(signal, entry, atr):
    if atr and atr > 0:
        sl_distance = atr * 1.5
        tp_distance = atr * 2.5
    else:
        sl_distance = entry * 0.005
        tp_distance = entry * 0.01

    if signal == "LONG":
        sl = entry - sl_distance
        tp = entry + tp_distance
    else:
        sl = entry + sl_distance
        tp = entry - tp_distance

    return round(sl, 2), round(tp, 2)

# ===== RULE SCORE =====
def evaluate_trade(data):
    price = safe_float(data.get("price"))
    ema9 = safe_float(data.get("ema9"))
    ema21 = safe_float(data.get("ema21"))
    vwap = safe_float(data.get("vwap"))
    volume = safe_float(data.get("volume"))
    signal = data.get("signal", "")

    if price is None:
        return 50

    score = 50

    if ema9 is not None and ema21 is not None:
        if (signal == "LONG" and ema9 > ema21) or (signal == "SHORT" and ema9 < ema21):
            score += 20

    if vwap is not None:
        if abs(price - vwap) / vwap < 0.004:
            score += 15
        if (signal == "LONG" and price > vwap) or (signal == "SHORT" and price < vwap):
            score += 20

    if ema21 is not None:
        if abs(price - ema21) / ema21 < 0.003:
            score += 15

    if volume is not None and volume > 0:
        score += 10

    return min(100, score)

# ===== FEATURES =====
def build_features(data):
    price = safe_float(data.get("price"))
    ema9 = safe_float(data.get("ema9"))
    ema21 = safe_float(data.get("ema21"))
    vwap = safe_float(data.get("vwap"))
    volume = safe_float(data.get("volume"))
    signal = data.get("signal", "")

    return {
        "trend": (ema9 > ema21) if (ema9 is not None and ema21 is not None) else None,
        "vwap_ok": (abs(price - vwap) / vwap < 0.004) if (price and vwap) else None,
        "pullback": (abs(price - ema21) / ema21 < 0.003) if (price and ema21) else None,
        "volume_nonzero": (volume > 0) if volume is not None else None,
        "direction": (
            (signal == "LONG" and price > vwap) or (signal == "SHORT" and price < vwap)
        ) if (price and vwap) else None
    }

# ===== ML =====
def ml_predict(features, score):
    if model is None:
        return None
    if any(v is None for v in features.values()):
        return None
    X = [[
        int(features["trend"]),
        int(features["vwap_ok"]),
        int(features["pullback"]),
        int(features["volume_nonzero"]),
        int(features["direction"]),
        score
    ]]
    return model.predict_proba(X)[0][1]

# ===== DISCORD SEND HELPER =====
# FIX #2 & #4: safe wrapper that checks client.is_ready() before using the loop
def send_discord(msg):
    if not client.is_ready():
        print("⚠️ Discord not ready, dropping message")
        return
    ch = client.get_channel(CHANNEL_ID)
    if ch is None:
        print("❌ CHANNEL NOT FOUND")
        return
    future = asyncio.run_coroutine_threadsafe(ch.send(msg), client.loop)
    try:
        # FIX #1: wait for the coroutine to complete so failures surface
        future.result(timeout=10)
        print("✅ MESSAGE SENT")
    except Exception as e:
        print(f"❌ Discord send failed: {e}")

# ===== WEBHOOK =====
@app.route("/webhook", methods=["POST"])
def webhook():
    raw = request.get_data(as_text=True)
    print("RAW BODY:", raw)

    try:
        data = json.loads(raw)
    except Exception as e:
        print(f"❌ JSON parse error: {e}")
        return "bad request", 400

    if not data:
        return "empty", 400

    # FIX #8: validate signal field before doing anything
    signal = data.get("signal")
    if signal not in ("LONG", "SHORT"):
        print(f"❌ Invalid signal: {signal!r}")
        return "invalid signal", 400

    entry = safe_float(data.get("price"))
    # FIX #8: validate price
    if entry is None:
        print("❌ Missing or invalid price")
        return "invalid price", 400

    atr = safe_float(data.get("atr"))

    if not atr:
        print("⚠️ ATR missing — signal skipped")
        return "ok"

    print("WEBHOOK RECEIVED:", data)

    sl, tp = calculate_sl_tp(signal, entry, atr)
    score = evaluate_trade(data)
    features = build_features(data)
    prob = ml_predict(features, score)

    if prob is None:
        decision = "📋 RULE ONLY"
    elif prob > 0.7:
        decision = "🔥 STRONG"
    elif prob > 0.55:
        decision = "✅ TAKE"
    else:
        decision = "❌ SKIP"

    rsi = safe_float(data.get("rsi"))
    rsi_str = f"`{rsi}`" if rsi else "`N/A`"

    log_trade(data, score, features, sl, tp)

    msg = (
        f"📊 **TRADE SIGNAL**\n"
        f"**{signal}** @ `{entry}`\n\n"
        f"Score: `{score}/100`\n"
        f"ML Prob: `{round(prob * 100, 2) if prob else 'N/A'}%`\n"
        f"Decision: {decision}\n"
        f"RSI: {rsi_str} | ATR: `{atr}`\n"
        f"SL: `{sl}` | TP: `{tp}`"
    )

    print("SENDING TO DISCORD...")
    send_discord(msg)

    return "ok"

@app.route("/health", methods=["GET"])
def health():
    return "ok"

# ===== DISCORD COMMANDS =====
@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # !status
    if message.content == "!status":
        try:
            with ManagedConn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM trades WHERE status='OPEN'")
                    open_count = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM trades WHERE status='WIN'")
                    wins = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM trades WHERE status='LOSS'")
                    losses = cur.fetchone()[0]

            total = wins + losses
            wr = round((wins / total) * 100, 1) if total > 0 else 0
            await message.channel.send(
                f"📈 **Trade Status**\n"
                f"Open: `{open_count}`\n"
                f"Wins: `{wins}` | Losses: `{losses}`\n"
                f"Win Rate: `{wr}%` ({total} closed)"
            )
        except Exception as e:
            await message.channel.send(f"❌ DB error: {e}")

    # !trades
    elif message.content == "!trades":
        try:
            trades = get_open_trades()
            if not trades:
                await message.channel.send("📭 No open trades right now.")
                return
            price = get_price()
            lines = ["📂 **Open Trades**"]
            for t in trades:
                if price:
                    if t["signal"] == "LONG":
                        pnl = round(price - t["entry"], 2)
                    else:
                        pnl = round(t["entry"] - price, 2)
                    pnl_str = f"`{'+'if pnl>=0 else ''}{pnl}`"
                else:
                    pnl_str = "`N/A`"
                lines.append(
                    f"**{t['signal']}** @ `{t['entry']}` | SL `{t['sl']}` | TP `{t['tp']}` | P&L {pnl_str}"
                )
            await message.channel.send("\n".join(lines))
        except Exception as e:
            await message.channel.send(f"❌ DB error: {e}")

    # !pnl — FIX #7: use actual exit_price instead of tp/sl
    elif message.content == "!pnl":
        try:
            with ManagedConn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT * FROM trades WHERE status IN ('WIN', 'LOSS')")
                    closed = cur.fetchall()

            if not closed:
                await message.channel.send("📭 No closed trades yet.")
                return

            total_pnl = 0
            for t in closed:
                exit_price = t.get("exit_price")
                if exit_price is None:
                    # fallback for trades closed before this fix
                    exit_price = t["tp"] if t["status"] == "WIN" else t["sl"]

                if t["signal"] == "LONG":
                    total_pnl += exit_price - t["entry"]
                else:
                    total_pnl += t["entry"] - exit_price

            emoji = "🟢" if total_pnl >= 0 else "🔴"
            await message.channel.send(
                f"{emoji} **Total P&L (points)**\n"
                f"`{'+'if total_pnl>=0 else ''}{round(total_pnl, 2)}` pts across {len(closed)} closed trades"
            )
        except Exception as e:
            await message.channel.send(f"❌ DB error: {e}")

    # !history
    elif message.content == "!history":
        try:
            with ManagedConn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT * FROM trades WHERE status IN ('WIN','LOSS') ORDER BY time DESC LIMIT 10")
                    trades = cur.fetchall()
            if not trades:
                await message.channel.send("📭 No closed trades yet.")
                return
            lines = ["📜 **Last 10 Trades**"]
            for t in trades:
                emoji = "🟢" if t["status"] == "WIN" else "🔴"
                lines.append(f"{emoji} **{t['signal']}** @ `{t['entry']}` | {t['status']}")
            await message.channel.send("\n".join(lines))
        except Exception as e:
            await message.channel.send(f"❌ DB error: {e}")

# ===== MONITOR =====
def monitor():
    while True:
        try:
            price = get_price()
            if price is None:
                time.sleep(10)
                continue

            trades = get_open_trades()

            for t in trades:
                new_status = None

                if t["signal"] == "LONG":
                    if price >= t["tp"]:
                        new_status = "WIN"
                    elif price <= t["sl"]:
                        new_status = "LOSS"

                elif t["signal"] == "SHORT":
                    if price <= t["tp"]:
                        new_status = "WIN"
                    elif price >= t["sl"]:
                        new_status = "LOSS"

                if new_status:
                    # FIX #7: pass actual current price as exit price
                    update_trade_status(t["id"], new_status, exit_price=price)
                    emoji = "🟢" if new_status == "WIN" else "🔴"
                    msg = (
                        f"{emoji} **Trade Closed — {new_status}**\n"
                        f"{t['signal']} | Entry: `{t['entry']}` | Exit: `{price}`\n"
                        f"SL was `{t['sl']}` | TP was `{t['tp']}`"
                    )
                    send_discord(msg)

        except Exception as e:
            print(f"Monitor error: {e}")

        time.sleep(10)

# ===== DAILY SUMMARY =====
# FIX #5: guard flag so on_ready reconnects don't spawn duplicate tasks
_summary_started = False

async def daily_summary():
    await client.wait_until_ready()
    while True:
        now = datetime.now(timezone.utc)

        # FIX #6: compute next midnight correctly regardless of start time
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        seconds_until_midnight = (next_midnight - now).total_seconds()
        await asyncio.sleep(seconds_until_midnight)

        try:
            with ManagedConn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM trades
                        WHERE status IN ('WIN', 'LOSS')
                        AND time >= NOW() - INTERVAL '24 hours'
                    """)
                    trades = cur.fetchall()

            if not trades:
                continue

            wins = sum(1 for t in trades if t["status"] == "WIN")
            losses = sum(1 for t in trades if t["status"] == "LOSS")
            total = len(trades)
            wr = round((wins / total) * 100, 1) if total > 0 else 0

            ch = client.get_channel(CHANNEL_ID)
            if ch:
                await ch.send(
                    f"📅 **Daily Summary**\n"
                    f"Signals today: `{total}`\n"
                    f"Wins: `{wins}` | Losses: `{losses}`\n"
                    f"Win Rate: `{wr}%`"
                )
        except Exception as e:
            print(f"Daily summary error: {e}")

# ===== STARTUP =====
@client.event
async def on_ready():
    global _summary_started
    print(f"✅ Bot ready: {client.user}")
    # FIX #5: only start once even if on_ready fires multiple times
    if not _summary_started:
        _summary_started = True
        client.loop.create_task(daily_summary())

def run_flask():
    app.run(host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    load_model()
    init_db()
    ensure_exit_price_column()

    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=monitor, daemon=True).start()

    client.run(TOKEN)

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
from flask import Flask, request
from datetime import datetime

# ===== CONFIG =====
TOKEN = os.environ.get("DISCORD_TOKEN")
CHANNEL_ID = int(os.environ.get("CHANNEL_ID", "1497320466715775080"))
DATABASE_URL = os.environ.get("DATABASE_URL")
MODEL_FILE = "model.pkl"
PORT = int(os.environ.get("PORT", 5000))

app = Flask(__name__)
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# ===== DATABASE =====
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    time TIMESTAMP DEFAULT NOW(),
                    signal TEXT,
                    entry FLOAT,
                    sl FLOAT,
                    tp FLOAT,
                    status TEXT DEFAULT 'OPEN',
                    score INT,
                    features JSONB
                )
            """)
        conn.commit()
    print("✅ DB initialized")

def log_trade(data, score, features):
    entry = float(data["price"])
    signal = data["signal"]

    if signal == "LONG":
        sl = entry * 0.995
        tp = entry * 1.01
    else:
        sl = entry * 1.005
        tp = entry * 0.99

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades (signal, entry, sl, tp, score, features)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (signal, entry, sl, tp, score, json.dumps(features)))
        conn.commit()

def get_open_trades():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM trades WHERE status = 'OPEN'")
            return cur.fetchall()

def update_trade_status(trade_id, status):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE trades SET status = %s WHERE id = %s", (status, trade_id))
        conn.commit()

# ===== MODEL =====
model = None

def load_model():
    global model
    try:
        model = joblib.load(MODEL_FILE)
        print("✅ Model loaded")
    except Exception as e:
        print(f"⚠️ No model found: {e}")
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

    print("WEBHOOK RECEIVED:", data)

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

    indicator_note = ""
    if safe_float(data.get("ema9")) is None:
        indicator_note = "\n⚠️ _Indicators not resolved — score based on price only_"

    log_trade(data, score, features)

    msg = (
        f"📊 **TRADE SIGNAL**\n"
        f"**{data.get('signal')}** @ `{data.get('price')}`\n\n"
        f"Score: `{score}/100`\n"
        f"ML Prob: `{round(prob * 100, 2) if prob else 'N/A'}%`\n"
        f"Decision: {decision}"
        f"{indicator_note}"
    )

    print("SENDING TO DISCORD...")
    ch = client.get_channel(CHANNEL_ID)
    if ch is None:
        print("❌ CHANNEL NOT FOUND")
    else:
        asyncio.run_coroutine_threadsafe(ch.send(msg), client.loop)
        print("✅ MESSAGE SENT")

    return "ok"

@app.route("/health", methods=["GET"])
def health():
    return "ok"

# ===== DISCORD COMMANDS =====
@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content == "!status":
        try:
            with get_conn() as conn:
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
                    update_trade_status(t["id"], new_status)
                    emoji = "🟢" if new_status == "WIN" else "🔴"
                    msg = (
                        f"{emoji} **Trade Closed — {new_status}**\n"
                        f"{t['signal']} | Entry: `{t['entry']}` | Exit: `{price}`"
                    )
                    ch = client.get_channel(CHANNEL_ID)
                    if ch:
                        asyncio.run_coroutine_threadsafe(ch.send(msg), client.loop)

        except Exception as e:
            print(f"Monitor error: {e}")

        time.sleep(10)

# ===== STARTUP =====
@client.event
async def on_ready():
    print(f"✅ Bot ready: {client.user}")

def run_flask():
    app.run(host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    load_model()
    init_db()

    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=monitor, daemon=True).start()

    client.run(TOKEN)

print("STARTING BOT...")
import discord, asyncio, json, threading, time, requests, joblib
from flask import Flask, request
from datetime import datetime

import os
TOKEN = os.environ.get("DISCORD_TOKEN")
CHANNEL_ID = 1496879851037261847  # replace with your channel ID

LOG_FILE = "trades_log.json"
MODEL_FILE = "model.pkl"

app = Flask(__name__)
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# ===== LOAD MODEL =====
def load_model():
    try:
        return joblib.load(MODEL_FILE)
    except:
        return None

# ===== PRICE =====
def get_price():
    url = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
    return float(requests.get(url).json()["price"])

# ===== RULE SCORE =====
def evaluate_trade(data):
    price = float(data["price"])
    ema9 = float(data["ema9"])
    ema21 = float(data["ema21"])
    vwap = float(data["vwap"])
    volume = float(data["volume"])
    signal = data["signal"]

    score = 50

    if (signal=="LONG" and ema9>ema21) or (signal=="SHORT" and ema9<ema21):
        score += 20

    if abs(price-vwap)/vwap < 0.004:
        score += 15

    if abs(price-ema21)/ema21 < 0.003:
        score += 15

    if volume > 0:
        score += 10

    if (signal=="LONG" and price>vwap) or (signal=="SHORT" and price<vwap):
        score += 20

    return min(100, score)

# ===== FEATURES =====
def build_features(data):
    price = float(data["price"])
    ema9 = float(data["ema9"])
    ema21 = float(data["ema21"])
    vwap = float(data["vwap"])
    signal = data["signal"]

    return {
        "trend": ema9 > ema21,
        "vwap_ok": abs(price-vwap)/vwap < 0.004,
        "pullback": abs(price-ema21)/ema21 < 0.003,
        "volume": True,
        "direction": (signal=="LONG" and price>vwap) or (signal=="SHORT" and price<vwap)
    }

# ===== ML =====
def ml_predict(model, features, score):
    if model is None:
        return None
    X = [[
        int(features["trend"]),
        int(features["vwap_ok"]),
        int(features["pullback"]),
        int(features["volume"]),
        int(features["direction"]),
        score
    ]]
    return model.predict_proba(X)[0][1]

# ===== LOG TRADE =====
def log_trade(data, score, features):
    entry = float(data["price"])
    sl = entry * 0.995
    tp = entry * 1.01

    trade = {
        "time": str(datetime.now()),
        "signal": data["signal"],
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "status": "OPEN",
        "score": score,
        "features": features
    }

    try:
        logs = json.load(open(LOG_FILE))
    except:
        logs = []

    logs.append(trade)
    json.dump(logs, open(LOG_FILE, "w"), indent=2)

# ===== WEBHOOK =====
@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json

    score = evaluate_trade(data)
    features = build_features(data)

    model = load_model()
    prob = ml_predict(model, features, score)

    if prob:
        if prob > 0.7:
            decision = "🔥 STRONG"
        elif prob > 0.55:
            decision = "✅ TAKE"
        else:
            decision = "❌ SKIP"
    else:
        decision = "RULE ONLY"

    log_trade(data, score, features)

    msg = f"""
📊 TRADE SIGNAL
{data['signal']} @ {data['price']}

Score: {score}
ML Prob: {round(prob*100,2) if prob else "N/A"}%
Decision: {decision}
"""

    ch = client.get_channel(CHANNEL_ID)
    asyncio.run_coroutine_threadsafe(ch.send(msg), client.loop)

    return "ok"

# ===== MONITOR =====
def monitor():
    while True:
        try:
            trades = json.load(open(LOG_FILE))
        except:
            trades = []

        price = get_price()

        for t in trades:
            if t["status"] != "OPEN":
                continue

            if t["signal"] == "LONG":
                if price >= t["tp"]:
                    t["status"] = "WIN"
                elif price <= t["sl"]:
                    t["status"] = "LOSS"

            if t["signal"] == "SHORT":
                if price <= t["tp"]:
                    t["status"] = "WIN"
                elif price >= t["sl"]:
                    t["status"] = "LOSS"

        json.dump(trades, open(LOG_FILE, "w"), indent=2)
        time.sleep(10)

import os

def run_flask():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

@client.event
async def on_ready():
    print("Bot is running")

threading.Thread(target=run_flask).start()
threading.Thread(target=monitor).start()

client.run(TOKEN)
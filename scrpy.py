#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests
import feedparser
from newspaper import Article
from datetime import datetime
import pytz
import yfinance as yf
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from colorama import Fore, Style, init as colorama_init
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ---------------------------
# CONFIG
# ---------------------------
FINNHUB_API_KEY = "d1ilckhr01qhbuvqnvk0d1ilckhr01qhbuvqnvkg"
NIFTY_THRESHOLD = 20000
STOCKS_FILE = "stocks.txt"

colorama_init(autoreset=True)

tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

vader = SentimentIntensityAnalyzer()

# ---------------------------
# Symbol search with multiple options
# ---------------------------
def search_symbol(company_name):
    url = f"https://finnhub.io/api/v1/search?q={company_name}&token={FINNHUB_API_KEY}"
    response = requests.get(url).json()
    results = response.get("result", [])

    if not results:
        return None, None

    print("\nPossible matches found:")
    for i, res in enumerate(results[:5], 1):
        symbol = res.get("symbol")
        desc = res.get("description")
        print(f"{i}) {symbol} — {desc}")

    choice = input(f"Choose 1-{len(results[:5])} (press Enter for first): ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(results[:5]):
            selected = results[idx]
            return selected["symbol"], selected["description"]

    selected = results[0]
    return selected["symbol"], selected["description"]

# ---------------------------
# Fetch sentiment from headlines
# ---------------------------
def fetch_sentiment(company):
    feed_url = f"https://news.google.com/rss/search?q={company.replace(' ', '+')}"
    feed = feedparser.parse(feed_url)
    sentiments = []

    for entry in feed.entries[:5]:
        article = Article(entry.link)
        try:
            article.download()
            article.parse()
            text = f"{article.title}. {article.text[:200]}"
            finbert_result = finbert(text)[0]["label"].lower()
            vs = vader.polarity_scores(text)
            score = vs["compound"]

            if finbert_result == "positive":
                sentiments.append(0.5 + score / 2)
            elif finbert_result == "negative":
                sentiments.append(-0.5 + score / 2)
            else:
                sentiments.append(score)
        except:
            continue

    avg_sent = np.mean(sentiments) if sentiments else 0
    return round(avg_sent, 4)

# ---------------------------
# Simple XGBoost next close prediction
# ---------------------------
def predict_next_close(symbol):
    df = yf.download(symbol, period="1y", auto_adjust=True)
    df = df.dropna()

    df["MA5"] = df["Close"].rolling(5).mean()
    df["Target"] = df["Close"].shift(-1)
    df = df.dropna()

    X = df[["Close", "Volume", "MA5"]]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = XGBRegressor(objective="reg:squarederror")
    model.fit(X_train, y_train)

    last_row = X.tail(1)
    prediction = model.predict(last_row)[0]

    return round(prediction, 2)

# ---------------------------
# NIFTY Info
# ---------------------------
def get_nifty_info():
    nifty = yf.Ticker("^NSEI")
    info = nifty.info
    return {
        "currentPrice": info.get("regularMarketPrice"),
    }

def check_nifty_alert(current_price):
    if current_price is None:
        return f"{Fore.RED}⚠️ Could not fetch NIFTY data."

    status = f"{Fore.RED if current_price > NIFTY_THRESHOLD else Fore.GREEN}"
    alert = f"{status}NIFTY: {current_price} ({'Above' if current_price > NIFTY_THRESHOLD else 'Below'} {NIFTY_THRESHOLD})"
    return alert

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    tz = pytz.timezone("Asia/Kolkata")
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    print("\n📌 Choose:\n1. Run for ONE company name\n2. Run for MULTIPLE from stocks.txt")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        companies = [input("Enter Company Name: ").strip()]
    else:
        with open(STOCKS_FILE, "r", encoding="utf-8") as f:
            companies = [line.strip() for line in f if line.strip()]

    nifty_price = get_nifty_info()["currentPrice"]
    nifty_summary = check_nifty_alert(nifty_price)

    for company in companies:
        print(f"\n🔵 Running for: {company}")
        symbol, description = search_symbol(company)

        if not symbol:
            print(f"{Fore.RED}❌ Could not find ticker for: {company}")
            continue

        sentiment_score = fetch_sentiment(company)
        stock = yf.Ticker(symbol)
        current = stock.history(period="1d")["Close"].iloc[-1]

        predicted_next = predict_next_close(symbol)
        adjusted = round(predicted_next * (1 + 0.01 * sentiment_score), 2)

        print(f"{Fore.CYAN}📄 {company} ({symbol}) — {description}")
        print(f"{Fore.YELLOW}🧠 Sentiment Score: {sentiment_score}")
        print(f"{Fore.LIGHTYELLOW_EX}💰 Current Close: {round(current,2)}")
        arrow = "🔼" if adjusted > current else "🔽"
        print(f"{Fore.GREEN if adjusted>current else Fore.RED}📈 Predicted Next Close: {adjusted} {arrow}")
        print(nifty_summary)
        
    # ---------------------------
    # SAVE REPORT TO FILE
    # ---------------------------
    report_lines = [
        f"Company: {company} ({symbol}) — {description}",
        f"Sentiment Score: {sentiment_score}",
        f"Current Close: {round(current,2)}",
        f"Predicted Next Close: {adjusted}",
        nifty_summary,
        f"Generated: {now}"
    ]

    os.makedirs("reports", exist_ok=True)
    filename = f"{symbol}_{now[:10]}.txt"
    filepath = os.path.join("reports", filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"{Fore.CYAN}💾 Report saved to: {filepath}")


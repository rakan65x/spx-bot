import subprocess
import sys
import time
from datetime import datetime, timedelta
import json
import os
import warnings
import asyncio
warnings.filterwarnings('ignore')

print("Installing packages...")
packages = ['yfinance', 'pandas', 'numpy', 'scikit-learn', 'schedule', 'python-telegram-bot==20.8']
for pkg in packages:
    try:
        __import__(pkg.replace('-', '_').split('==')[0])
    except:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from telegram.ext import ApplicationBuilder
import schedule

TELEGRAM_TOKEN = "8400621815:AAF8ebrgJXjjmUsz6Y7Y3Z624-Vwetlc5C4"
CHAT_ID = "803731534"
CAPITAL = 100000000
TICKER = "SPY"
DAILY_TRADES = 5
PREDICTIONS_FILE = "predictions_history.json"

class SmartBot:
    def __init__(self):
        self.capital = CAPITAL
        self.data = None
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10)
        self.scaler = StandardScaler()

    def save_prediction(self, date, signals, current_price):
        predictions = self.load_predictions()
        predictions[date] = {'signals': signals, 'price': current_price, 'verified': False}
        with open(PREDICTIONS_FILE, 'w') as f:
            json.dump(predictions, f, indent=2)

    def load_predictions(self):
        if os.path.exists(PREDICTIONS_FILE):
            with open(PREDICTIONS_FILE, 'r') as f:
                return json.load(f)
        return {}

    def verify_yesterday(self):
        predictions = self.load_predictions()
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        if yesterday not in predictions or predictions[yesterday].get('verified'):
            return None
        pred_data = predictions[yesterday]
        yesterday_price = pred_data['price']
        try:
            ticker = yf.Ticker(TICKER)
            today_data = ticker.history(period="2d")
            if len(today_data) >= 2:
                today_price = today_data['Close'].iloc[-1]
                actual_change = ((today_price - yesterday_price) / yesterday_price) * 100
                correct = 0
                total = len(pred_data['signals'])
                for sig in pred_data['signals']:
                    if sig['type'] == 'CALL' and actual_change > 0:
                        correct += 1
                    elif sig['type'] == 'PUT' and actual_change < 0:
                        correct += 1
                accuracy = (correct / total * 100) if total > 0 else 0
                predictions[yesterday]['verified'] = True
                predictions[yesterday]['actual_change'] = actual_change
                predictions[yesterday]['accuracy'] = accuracy
                with open(PREDICTIONS_FILE, 'w') as f:
                    json.dump(predictions, f, indent=2)
                return {'date': yesterday, 'predicted': pred_data['signals'][0]['type'], 'actual_change': actual_change, 'correct': correct, 'total': total, 'accuracy': accuracy}
        except:
            pass
        return None

    def download_data(self):
        print("Downloading data...")
        try:
            ticker = yf.Ticker(TICKER)
            self.data = ticker.history(start="2020-01-01", end=datetime.now())
            self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
            self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
            delta = self.data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            self.data['RSI'] = 100 - (100 / (1 + rs))
            exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
            self.data['MACD'] = exp1 - exp2
            self.data['Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
            self.data['Next_Return'] = (self.data['Close'].shift(-1) - self.data['Close']) / self.data['Close']
            self.data['Target'] = (self.data['Next_Return'] > 0).astype(int)
            self.data = self.data.dropna()
            print(f"Downloaded {len(self.data)} days")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def train_model(self):
        print("Training AI...")
        features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal']
        X = self.data[features].iloc[:-1]
        y = self.data['Target'].iloc[:-1]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        accuracy = self.model.score(X_scaled, y)
        print(f"AI Accuracy: {accuracy*100:.1f}%")
        return accuracy

    def generate_signals(self):
        print("Generating signals...")
        latest = self.data.tail(1)
        signals = []
        for i in range(DAILY_TRADES):
            features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal']
            X = latest[features].values
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)[0]
            prob = self.model.predict_proba(X_scaled)[0]
            confidence = max(prob) * 100
            current_price = latest['Close'].values[0]
            if pred == 1:
                signal_type = "CALL"
                strike = current_price * 1.015
            else:
                signal_type = "PUT"
                strike = current_price * 0.985
            premium = abs(current_price - strike) * 0.05
            signals.append({'number': i+1, 'type': signal_type, 'strike': strike, 'premium': premium, 'confidence': confidence, 'tp': premium*1.5, 'sl': premium*0.7})
        return signals, current_price

    def send_telegram(self, signals, yesterday_result):
        message = f"SPX BOT REPORT\n{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        if yesterday_result:
            emoji = "GOOD" if yesterday_result['accuracy'] > 60 else "BAD"
            direction = "UP" if yesterday_result['actual_change'] > 0 else "DOWN"
            message += f"YESTERDAY {emoji}\nDate: {yesterday_result['date']}\nPredicted: {yesterday_result['predicted']}\nActual: {direction} {yesterday_result['actual_change']:.2f}%\nAccuracy: {yesterday_result['accuracy']:.0f}%\nCorrect: {yesterday_result['correct']}/{yesterday_result['total']}\n\n"
        message += "TODAY SIGNALS:\n"
        for sig in signals:
            message += f"\n#{sig['number']} {sig['type']}\nStrike: ${sig['strike']:.2f}\nPremium: ${sig['premium']:.2f}\nConfidence: {sig['confidence']:.0f}%\nTP: ${sig['tp']:.2f} SL: ${sig['sl']:.2f}\n"
        message += f"\nCapital: ${self.capital:,}\nPaper Trading"
        try:
            async def send():
                app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
                await app.bot.send_message(chat_id=CHAT_ID, text=message)
            asyncio.run(send())
            print("Telegram sent!")
        except Exception as e:
            print(f"Telegram error: {e}")

    def daily_routine(self):
        print("\nDAILY ROUTINE START")
        yesterday_result = self.verify_yesterday()
        if not self.download_data():
            return
        self.train_model()
        signals, current_price = self.generate_signals()
        today = datetime.now().strftime('%Y-%m-%d')
        self.save_prediction(today, signals, current_price)
        for sig in signals:
            print(f"#{sig['number']} {sig['type']} ${sig['strike']:.2f}")
        self.send_telegram(signals, yesterday_result)
        print("DONE\n")

def run_bot():
    bot = SmartBot()
    schedule.every().day.at("13:00").do(bot.daily_routine)
    print("SPX BOT RUNNING")
    print("Daily at 13:00 UTC")
    bot.daily_routine()
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    try:
        run_bot()
    except KeyboardInterrupt:
        print("Stopped")
    except Exception as e:
        print(f"Error: {e}")

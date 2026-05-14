import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from datetime import datetime, timedelta

# Daftar beberapa saham unggulan (LQ45 / Blue Chips)
TOP_STOCKS = [
    # Bank & Keuangan
    'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BBNI.JK', 'BRIS.JK', 'ARTO.JK', 'BBTN.JK',
    # Telekomunikasi & Tech
    'TLKM.JK', 'GOTO.JK', 'ISAT.JK', 'EXCL.JK', 'MTEL.JK', 'BUKA.JK',
    # Energi & Pertambangan
    'ADRO.JK', 'PTBA.JK', 'UNTR.JK', 'ITMG.JK', 'PGAS.JK', 'MEDC.JK', 'AMMN.JK', 'MDKA.JK', 'PGEO.JK', 'BUMI.JK', 'BRPT.JK', 'CUAN.JK',
    # Konsumer & Ritel
    'UNVR.JK', 'ICBP.JK', 'INDF.JK', 'AMRT.JK', 'MYOR.JK', 'MAPI.JK', 'ACES.JK',
    # Infrastruktur & Properti
    'ASII.JK', 'KLBF.JK', 'AKRA.JK', 'SMGR.JK', 'INTP.JK', 'CTRA.JK', 'BSDE.JK', 'JSMR.JK'
]

def fetch_data(ticker, period="2y"):
    """Fetch historical data from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def add_features(df):
    """Add technical indicators to the dataframe."""
    df = df.copy()
    
    # RSI (14 days)
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()
    
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # SMA (Moving Averages)
    sma20 = SMAIndicator(close=df['Close'], window=20)
    sma50 = SMAIndicator(close=df['Close'], window=50)
    df['SMA_20'] = sma20.sma_indicator()
    df['SMA_50'] = sma50.sma_indicator()
    
    # Daily Returns & Volatility
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=10).std()
    
    # Target: 1 if tomorrow's close is higher than today's close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop NaNs (from rolling windows and shift)
    # We don't drop the last row yet, because we need it for prediction
    return df

def train_and_predict(df):
    """Trains a Random Forest and predicts the next day's direction."""
    # Prepare data
    features = ['RSI', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_50', 'Daily_Return', 'Volatility']
    
    # Drop rows with NaN in features
    df_clean = df.dropna(subset=features)
    
    if len(df_clean) < 50:
        return None, None
        
    # The last row doesn't have a valid Target (it's NaN before dropping, or 0 if filled)
    # Actually, df['Target'] calculation makes the last row target deterministic but wrong 
    # because shift(-1) creates NaN, and astype(int) might make it 0.
    # Let's cleanly separate training data and prediction data
    
    # For training, drop the very last row because we don't know tomorrow's close yet
    train_df = df_clean.iloc[:-1]
    
    X_train = train_df[features]
    y_train = train_df['Target']
    
    # Train Model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict for the last row (today)
    X_pred = df_clean.iloc[-1:][features]
    
    # Get probability of class 1 (Up)
    prob_up = model.predict_proba(X_pred)[0][1]
    
    # Current RSI
    current_rsi = df_clean.iloc[-1]['RSI']
    current_close = df_clean.iloc[-1]['Close']
    
    return prob_up, current_rsi, current_close

def analyze_stocks(tickers=TOP_STOCKS):
    """Analyzes a list of stocks and returns a summary DataFrame."""
    results = []
    
    for ticker in tickers:
        df = fetch_data(ticker)
        if df is not None:
            df_feat = add_features(df)
            prob, rsi, close = train_and_predict(df_feat)
            
            if prob is not None:
                # Determine RSI Signal
                rsi_signal = "Neutral"
                if rsi < 30:
                    rsi_signal = "Oversold (Buy Signal)"
                elif rsi > 70:
                    rsi_signal = "Overbought (Sell Signal)"
                    
                # Determine ML Signal
                ml_signal = "Neutral"
                if prob > 0.55:
                    ml_signal = "Bullish (Up)"
                elif prob < 0.45:
                    ml_signal = "Bearish (Down)"
                    
                results.append({
                    'Ticker': ticker.replace('.JK', ''),
                    'Last Close (Rp)': int(close),
                    'RSI': round(rsi, 2),
                    'RSI Signal': rsi_signal,
                    'Probabilitas Naik (%)': round(prob * 100, 1),
                    'ML Signal': ml_signal
                })
                
    if results:
        res_df = pd.DataFrame(results)
        # Sort by Probability of going up
        res_df = res_df.sort_values(by='Probabilitas Naik (%)', ascending=False).reset_index(drop=True)
        return res_df
    return pd.DataFrame()

def evaluate_historical(tickers, days_back=3):
    """Evaluates the model by predicting T-1 to T-3."""
    results = []
    
    for ticker in tickers:
        df = fetch_data(ticker, period="2y")
        if df is None or df.empty:
            continue
            
        df_feat = add_features(df)
        features = ['RSI', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_50', 'Daily_Return', 'Volatility']
        df_clean = df_feat.dropna(subset=features)
        
        if len(df_clean) < 50:
            continue
            
        for day in range(1, days_back + 1):
            # day=1 means predicting H-1.
            # To predict H-1, we use features from H-2.
            # In python negative indexing: -1 is H (Today), -2 is H-1, -3 is H-2.
            idx = -(day + 2)
            
            # Training data must not contain idx or anything after it, 
            # because the 'Target' of idx contains the answer for idx+1.
            train_df = df_clean.iloc[:idx]
            
            # If not enough training data
            if len(train_df) < 50:
                continue
                
            X_train = train_df[features]
            y_train = train_df['Target']
            
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict using features from H-2 (idx)
            X_pred = df_clean.iloc[idx:idx+1][features]
            prob_up = model.predict_proba(X_pred)[0][1]
            
            prediction_label = "Bullish" if prob_up > 0.55 else ("Bearish" if prob_up < 0.45 else "Neutral")
            
            # Actual prices: base is H-2, target is H-1
            close_base = df_clean.iloc[idx]['Close']
            close_target = df_clean.iloc[idx+1]['Close']
            date_base = df_clean.index[idx].strftime("%Y-%m-%d")
            date_target = df_clean.index[idx+1].strftime("%Y-%m-%d")
            
            actual_label = "Naik" if close_target > close_base else ("Turun" if close_target < close_base else "Tetap")
            
            is_correct = "BENAR" if (prob_up > 0.55 and close_target > close_base) or (prob_up < 0.45 and close_target <= close_base) else ("SALAH" if prob_up > 0.55 or prob_up < 0.45 else "NETRAL")
            
            results.append({
                'Ticker': ticker.replace('.JK', ''),
                'Hari Uji': f"H-{day}",
                'Tgl Latih': date_base,
                'Prediksi Arah': prediction_label,
                'Tgl Target': date_target,
                'Realita': actual_label,
                'Hasil': is_correct
            })
        
    return pd.DataFrame(results)

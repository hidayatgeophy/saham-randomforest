import yfinance as yf
import pandas as pd
from ml_engine import fetch_data, add_features
from sklearn.ensemble import RandomForestClassifier

stocks_to_test = ['BBCA.JK', 'BMRI.JK', 'TLKM.JK', 'ASII.JK', 'GOTO.JK', 'ADRO.JK']

print("=== PENGUJIAN AKURASI MODEL (H-1) ===")

for ticker in stocks_to_test:
    df = fetch_data(ticker, period="2y")
    if df is None or df.empty:
        continue
        
    df_feat = add_features(df)
    features = ['RSI', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_50', 'Daily_Return', 'Volatility']
    df_clean = df_feat.dropna(subset=features)
    
    if len(df_clean) < 10:
        continue
        
    # We want to predict the LAST day in the dataset (T-0).
    # To do this fairly, we train on data up to T-2.
    # The input to predict T-0 is the features from T-1.
    
    # Target in df is: df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    # This means df.iloc[X]['Target'] represents whether Close[X+1] > Close[X].
    
    # If df_clean has N rows (index 0 to N-1).
    # N-1 is the latest day (Today/Yesterday's close). Its Target is NaN/0 because there is no N.
    # N-2 is the day before the latest day. Its Target is whether Close[N-1] > Close[N-2].
    
    # To predict what happened on N-1, we must use features from N-2.
    # The model must NOT see N-1 during training.
    # So we train on rows up to N-3.
    
    train_df = df_clean.iloc[:-2] # Up to N-3
    X_train = train_df[features]
    y_train = train_df['Target']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # The features from N-2
    X_pred = df_clean.iloc[-2:-1][features]
    
    # Predict the target for N-2 (which means predicting if N-1 is higher than N-2)
    prob_up = model.predict_proba(X_pred)[0][1]
    
    prediction_label = "Bullish (Naik)" if prob_up > 0.55 else ("Bearish (Turun)" if prob_up < 0.45 else "Neutral")
    
    # Actual outcome
    close_n2 = df_clean.iloc[-2]['Close']
    close_n1 = df_clean.iloc[-1]['Close']
    date_n2 = df_clean.index[-2].strftime("%Y-%m-%d")
    date_n1 = df_clean.index[-1].strftime("%Y-%m-%d")
    
    actual_label = "Naik" if close_n1 > close_n2 else ("Turun" if close_n1 < close_n2 else "Tetap")
    
    # Correctness
    is_correct = "BENAR [V]" if (prob_up > 0.55 and close_n1 > close_n2) or (prob_up < 0.45 and close_n1 <= close_n2) else ("SALAH [X]" if prob_up > 0.55 or prob_up < 0.45 else "NETRAL [-]")
    
    print(f"\nSaham: {ticker.replace('.JK', '')}")
    print(f"Prediksi Model dari data {date_n2}: {prob_up*100:.1f}% -> Sinyal: {prediction_label}")
    print(f"Kenyataan di tanggal {date_n1}: Harga dari Rp {close_n2:,.0f} menjadi Rp {close_n1:,.0f} ({actual_label})")
    print(f"Hasil Tebakan: {is_correct}")


import yfinance as yf
import pandas as pd
from ml_engine import fetch_data, add_features
from sklearn.ensemble import RandomForestClassifier

stocks_to_test = ['BBCA.JK', 'BMRI.JK', 'TLKM.JK', 'ASII.JK', 'GOTO.JK', 'ADRO.JK']

print("=== MENCARI BUKTI SINYAL BULLISH (30 HARI TERAKHIR) ===")

for ticker in stocks_to_test:
    df = fetch_data(ticker, period="2y")
    if df is None or df.empty:
        continue
        
    df_feat = add_features(df)
    features = ['RSI', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_50', 'Daily_Return', 'Volatility']
    df_clean = df_feat.dropna(subset=features)
    
    if len(df_clean) < 100:
        continue
        
    # Latih model dengan data hingga 30 hari yang lalu
    train_df = df_clean.iloc[:-30]
    test_df = df_clean.iloc[-30:] # 30 hari terakhir untuk diuji
    
    X_train = train_df[features]
    y_train = train_df['Target']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Lakukan prediksi untuk 30 hari terakhir
    X_test = test_df[features]
    probs = model.predict_proba(X_test)[:, 1]
    
    for i in range(len(test_df) - 1):
        prob_up = probs[i]
        if prob_up > 0.55: # Hanya cari yang diprediksi kuat NAIK
            # Hari H (saat sinyal muncul)
            date_signal = test_df.index[i].strftime("%Y-%m-%d")
            close_signal = test_df.iloc[i]['Close']
            
            # Hari H+1 (kenyataan)
            date_actual = test_df.index[i+1].strftime("%Y-%m-%d")
            close_actual = test_df.iloc[i+1]['Close']
            
            if close_actual > close_signal:
                print(f"\n[BUKTI VALID] Saham: {ticker.replace('.JK', '')}")
                print(f"Tanggal Sinyal: {date_signal}")
                print(f"Prediksi: BULLISH (Probabilitas {prob_up*100:.1f}%)")
                print(f"Kenyataan Besoknya ({date_actual}): Naik dari Rp {close_signal:,.0f} menjadi Rp {close_actual:,.0f} [BENAR]")

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ml_engine import analyze_stocks, TOP_STOCKS, fetch_data, add_features, train_and_predict, evaluate_historical

st.set_page_config(page_title="IDX Stock Predictor", layout="wide", page_icon="📈")

st.title("📈 IDX Stock Predictor & Screener")
st.markdown("""
Aplikasi ini memprediksi pergerakan saham harian menggunakan **Random Forest Classifier** dan mendeteksi momentum menggunakan indikator **RSI**.
Fokus utama adalah pada saham-saham Blue Chip/LQ45 Indonesia.
""")

def style_signal(val):
    if "Buy" in str(val) or "Up" in str(val) or "Bullish" in str(val):
        color = 'green'
    elif "Sell" in str(val) or "Down" in str(val) or "Bearish" in str(val):
        color = 'red'
    else:
        color = 'gray'
    return f'color: {color}; font-weight: bold;'

st.header("1. Automatic Screener (Top LQ45)")
st.write(f"Tekan tombol di bawah untuk memindai {len(TOP_STOCKS)} saham unggulan (LQ45 & Second Liners) dan mendapatkan rekomendasi otomatis.")

if st.button("Run Stock Screener"):
    with st.spinner(f"Mengunduh data dan menganalisis {len(TOP_STOCKS)} saham. Proses ini memakan waktu sekitar 1-2 menit, harap tunggu..."):
        results_df = analyze_stocks()
        if not results_df.empty:
            st.success("Screener selesai!")
            st.dataframe(
                results_df.style.map(style_signal, subset=['RSI Signal', 'ML Signal'])
                                .format({'Last Close (Rp)': "{:,.0f}", 'RSI': "{:.2f}", 'Probabilitas Naik (%)': "{:.1f}%"}),
                use_container_width=True,
                hide_index=True
            )
            
            st.info("""
            **Cara Membaca Tabel:**
            - **RSI Signal**: Jika 'Oversold (Buy Signal)', artinya harga sudah turun dalam dan berpotensi naik. Jika 'Overbought (Sell Signal)', harga rawan koreksi.
            - **Probabilitas Naik (%)**: Prediksi dari model Machine Learning (Random Forest) tentang peluang harga ditutup lebih tinggi esok hari.
            - **ML Signal**: Kesimpulan dari Probabilitas (>55% Bullish, <45% Bearish).
            
            **Rekomendasi Aksi:** Cari saham dengan kondisi **Oversold** DAN **Probabilitas Naik tinggi** sebagai kandidat kuat untuk dibeli.
            """)
        else:
            st.error("Gagal mengambil data. Pastikan koneksi internet aktif.")

st.divider()

st.header("2. Analisis Saham Individual")
st.write("Masukkan kode saham (tanpa .JK) untuk melihat grafik dan indikator secara detail.")

ticker_input = st.text_input("Kode Saham (contoh: BBCA, TLKM, GOTO)", "BBCA").upper()

if ticker_input:
    full_ticker = f"{ticker_input}.JK"
    with st.spinner(f"Menganalisis {ticker_input}..."):
        df = fetch_data(full_ticker, period="2y")
        
        if df is not None and not df.empty:
            df_feat = add_features(df)
            
            # Plotting Candlestick (hanya tampilkan 6 bulan terakhir untuk grafik)
            df_plot = df.iloc[-125:] # 125 hari bursa = 6 bulan
            
            fig = go.Figure(data=[go.Candlestick(x=df_plot.index,
                            open=df_plot['Open'],
                            high=df_plot['High'],
                            low=df_plot['Low'],
                            close=df_plot['Close'],
                            name='Price')])
            
            fig.update_layout(
                title=f"Pergerakan Harga {ticker_input} (6 Bulan Terakhir)",
                yaxis_title="Harga (IDR)",
                xaxis_rangeslider_visible=False,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show Metrics
            last_close = df_feat.iloc[-1]['Close']
            last_rsi = df_feat.iloc[-1]['RSI']
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Harga Penutupan Terakhir", f"Rp {int(last_close):,}")
            
            rsi_delta_color = "normal"
            if last_rsi < 30: rsi_desc = "Oversold 🟢"
            elif last_rsi > 70: rsi_desc = "Overbought 🔴"
            else: rsi_desc = "Neutral ⚪"
            
            col2.metric("RSI (14 Hari)", f"{last_rsi:.2f}", rsi_desc, delta_color="off")
            
            # ML Prediction
            prob, _, _ = train_and_predict(df_feat)
            if prob is not None:
                ml_signal_text = "Neutral ⚪"
                if prob > 0.55: ml_signal_text = "Bullish (Up) 🟢"
                elif prob < 0.45: ml_signal_text = "Bearish (Down) 🔴"
                
                col3.metric("Prediksi ML Besok (Prob Naik)", f"{prob * 100:.1f}%", ml_signal_text, delta_color="off")
            else:
                col3.metric("Prediksi ML Besok", "N/A", "Data kurang")
            
        else:
            st.warning(f"Data untuk {ticker_input} tidak ditemukan atau gagal diakses.")

st.divider()

st.header("3. Pengujian Akurasi (Backtest H-1 sd H-3)")
st.write("Uji keakuratan algoritma Random Forest untuk memprediksi arah pergerakan harga 3 hari terakhir secara mundur.")

if st.button("Jalankan Uji Backtest H-1 sd H-3"):
    with st.spinner(f"Melakukan backtest pada seluruh {len(TOP_STOCKS)} saham. Proses ini memakan waktu sekitar 1-2 menit, harap tunggu..."):
        eval_df = evaluate_historical(TOP_STOCKS)
        if not eval_df.empty:
            def style_hasil(val):
                if val == "BENAR": return 'color: green; font-weight: bold;'
                elif val == "SALAH": return 'color: red; font-weight: bold;'
                return 'color: gray;'
                
            st.dataframe(
                eval_df.style.map(style_hasil, subset=['Hasil']),
                use_container_width=True,
                hide_index=True
            )
            
            benar_count = len(eval_df[eval_df['Hasil'] == 'BENAR'])
            salah_count = len(eval_df[eval_df['Hasil'] == 'SALAH'])
            total_valid = benar_count + salah_count
            if total_valid > 0:
                akurasi = (benar_count / total_valid) * 100
                st.info(f"**Tingkat Akurasi (Arah): {akurasi:.1f}%** (Dari {total_valid} tebakan non-netral)")
        else:
            st.error("Gagal melakukan backtest.")

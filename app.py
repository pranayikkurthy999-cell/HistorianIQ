import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.fft import rfft, rfftfreq
from src.core.data_loader import HistorianDataLoader
from src.core.frequency_analyzer import FrequencyAnalyzer
from src.core.quality_analyzer import QualityScorer
from src.core.missing_data import MissingDataImputer
from src.core.filter_engine import SignalConditioner
from src.core.report_generator import PDFReportGenerator

st.set_page_config(page_title="HistorianIQ", layout="wide")

# --- PIPELINE UI ---
st.markdown("<h2 style='text-align: center;'>🏭 HistorianIQ Data Pipeline</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><b>CSV Import ➔ Diagnostics ➔ Imputation ➔ Filtering ➔ ML Export</b></p>", unsafe_allow_html=True)
st.divider()

st.sidebar.header("1. Data Import")
uploaded_file = st.sidebar.file_uploader("Upload Historian CSV", type=["csv"])

if uploaded_file:
    loader = HistorianDataLoader()
    df, time_col, numeric_cols = loader.load_csv(uploaded_file)
    freq_analyzer = FrequencyAnalyzer()
    scorer = QualityScorer()
    
    stats = freq_analyzer.analyze(df, time_col)
    target_col = st.sidebar.selectbox("Target Process Variable", numeric_cols)
    raw_signal = df[target_col]
    quality_metrics = scorer.assess_signal(raw_signal)
    fs = stats['estimated_hz'] if stats['estimated_hz'] > 0 else 1.0

    # --- TAB LAYOUT ---
    tab1, tab2, tab3 = st.tabs(["🔍 Diagnostics & Recommendations", "🎛️ Engineering Studio (FFT & Filters)", "📑 ML Readiness & Export"])
    
    # === TAB 1: DIAGNOSTICS ===
    with tab1:
        colA, colB = st.columns([2, 1])
        with colA:
            st.subheader("Data Loss Transparency")
            c1, c2, c3 = st.columns(3)
            c1.metric("Original Samples", f"{quality_metrics['total_samples']:,}")
            c2.metric("Missing (NaNs)", f"{quality_metrics['missing_samples']:,}")
            c3.metric("Usable Rate", f"{100 - quality_metrics['missing_pct']:.1f}%")
            
            fig_raw = go.Figure()
            fig_raw.add_trace(go.Scattergl(x=df[time_col], y=raw_signal, mode='lines', line=dict(color='gray')))
            fig_raw.update_layout(title="Raw Historian Trend", height=300)
            st.plotly_chart(fig_raw, use_container_width=True)

        with colB:
            st.subheader("Engineering Recommendation")
            st.info("**Detected Characteristics:**")
            if quality_metrics['missing_pct'] > 0: st.markdown("- ⚠ Contains Missing Data")
            if stats['inconsistent_sampling']: st.markdown("- ⚠ Compression Deadbands Detected")
            
            st.success("**Recommended Workflow:**\n\n1. Cubic Spline Interpolation\n2. Butterworth Filter (Lowpass)\n3. Resample to strict Uniform Grid")

    # === TAB 2: CONDITIONING STUDIO ===
    with tab2:
        imputer = MissingDataImputer()
        conditioner = SignalConditioner()
        
        st.sidebar.header("2. Conditioning Controls")
        impute_type = st.sidebar.selectbox("Imputation Strategy", ["Linear Interpolation", "Cubic Spline", "Forward Fill"])
        filter_type = st.sidebar.selectbox("Filter Algorithm", ["Butterworth (Lowpass)", "Savitzky-Golay", "Moving Average", "None"])
        
        imputed_signal = imputer.apply_imputation(raw_signal, impute_type)
        clean_signal = imputed_signal
        
        if filter_type == "Butterworth (Lowpass)":
            cutoff = st.sidebar.slider("Cutoff (Hz)", 0.001, float(fs/2)*0.9, 0.05, 0.001)
            order = st.sidebar.slider("Order", 1, 6, 2)
            clean_signal = conditioner.apply_butterworth(imputed_signal, cutoff, fs, order)
        elif filter_type == "Moving Average":
            window = st.sidebar.slider("Window", 2, 100, 10)
            clean_signal = conditioner.apply_moving_average(imputed_signal, window)
            
        # --- FIXED SNR CALCULATION ---
        snr_after = scorer.calculate_snr(raw_signal, clean_signal)
        
        col1, col2 = st.columns(2)
        if filter_type != "None":
            col1.metric("Filter Signal-to-Noise Ratio (SNR)", f"{snr_after:.1f} dB", "Clean vs Raw")
        else:
            col1.metric("Filter Signal-to-Noise Ratio (SNR)", "N/A", "Apply a filter to calculate")

        # --- FIXED FFT CALCULATION (Removing DC Offset) ---
        imputed_centered = imputed_signal.dropna() - imputed_signal.dropna().mean()
        clean_centered = clean_signal.dropna() - clean_signal.dropna().mean()

        N = len(clean_centered)
        xf = rfftfreq(N, 1/fs)
        yf_raw = np.abs(rfft(imputed_centered.values))
        yf_clean = np.abs(rfft(clean_centered.values))

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scattergl(x=df[time_col], y=raw_signal, mode='lines', name='Raw', opacity=0.3))
        fig_compare.add_trace(go.Scattergl(x=df[time_col], y=clean_signal, mode='lines', name='Clean', line=dict(color='red')))
        fig_compare.update_layout(title="Time Domain", height=350)
        
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scattergl(x=xf, y=yf_raw, mode='lines', name='Raw FFT', opacity=0.4))
        fig_fft.add_trace(go.Scattergl(x=xf, y=yf_clean, mode='lines', name='Clean FFT', line=dict(color='red')))
        fig_fft.update_layout(title="Frequency Spectrum (FFT)", xaxis_title="Frequency (Hz)", height=350)

        st.plotly_chart(fig_compare, use_container_width=True)
        st.plotly_chart(fig_fft, use_container_width=True)

    # === TAB 3: EXPORT ===
    with tab3:
        # --- FIXED ML SCORE LOGIC ---
        clean_quality_metrics = scorer.assess_signal(clean_signal)
        
        is_inconsistent = stats['inconsistent_sampling']
        if filter_type != "None":
            is_inconsistent = False 
            
        ml_score, recommendations = scorer.get_ml_readiness(
            clean_quality_metrics['quality_score'], 
            clean_quality_metrics['missing_pct'], 
            is_inconsistent
        )
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = ml_score,
            title = {'text': "ML Readiness Score"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "green" if ml_score > 90 else "orange"}}
        ))
        
        c1, c2 = st.columns([1, 2])
        with c1: st.plotly_chart(fig_gauge, use_container_width=True)
        with c2:
            st.markdown("### Deployment Recommendations")
            for rec in recommendations: st.markdown(rec)
            
            st.divider()
            
            export_df = pd.DataFrame({time_col: df[time_col], f"{target_col}_Cleaned": clean_signal})
            csv_data = export_df.to_csv(index=False).encode('utf-8')
            
            pdf_gen = PDFReportGenerator()
            pdf_data = pdf_gen.generate_report(target_col, quality_metrics, filter_type, 0, snr_after, ml_score, recommendations)
            
            st.download_button("📥 Download Cleaned CSV", data=csv_data, file_name=f"{target_col}_clean.csv", mime="text/csv")
            st.download_button("📄 Download Engineering QA Report (PDF)", data=pdf_data, file_name=f"{target_col}_QA_Report.pdf", mime="application/pdf")
import pandas as pd
import numpy as np

class QualityScorer:
    def assess_signal(self, series):
        total_rows = len(series)
        if total_rows == 0: return None
            
        missing_pct = series.isnull().sum() / total_rows * 100
        diffs = series.dropna().diff()
        frozen_pct = (diffs == 0).sum() / total_rows * 100
        
        score = 100 - (missing_pct * 2.0) - (frozen_pct * 0.5)
        score = max(0, min(100, score))
        if series.var() == 0: score = 0
            
        return {
            "total_samples": total_rows,
            "missing_samples": series.isnull().sum(),
            "missing_pct": missing_pct,
            "frozen_pct": frozen_pct,
            "quality_score": score
        }

    def calculate_snr(self, raw_series, clean_series):
        """Calculates Signal-to-Noise Ratio in dB."""
        # Align series to drop any remaining NaNs for the calculation
        df = pd.DataFrame({'raw': raw_series, 'clean': clean_series}).dropna()
        if len(df) == 0: return 0
        
        signal_power = np.var(df['clean'])
        noise_power = np.var(df['raw'] - df['clean'])
        
        if noise_power == 0: return float('inf') # Perfect signal
        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db

    def get_ml_readiness(self, score, missing_pct, inconsistent_sampling):
        """Determines ML suitability based on industrial standards."""
        if inconsistent_sampling:
            return 50, ["⚠ Resampling required before FFT/Modeling", "⚠ Time-series forecasting not suitable"]
        if missing_pct > 1:
            return 75, ["✔ Suitable for basic regression", "⚠ Missing data imputation limits APC use"]
        if score > 90:
            return 95, ["✔ Ready for Advanced Process Control (APC)", "✔ Ready for Soft Sensor Development", "✔ Ready for High-Frequency FFT"]
        return 80, ["✔ Suitable for Regression", "⚠ Check noise levels before Digital Twin use"]
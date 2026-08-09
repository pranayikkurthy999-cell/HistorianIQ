import pandas as pd

class FrequencyAnalyzer:
    def analyze(self, df, time_col):
        """Determines sampling frequency and checks for inconsistencies."""
        # Calculate the time difference between each row in seconds
        time_deltas = df[time_col].diff().dt.total_seconds().dropna()
        
        # Calculate median and mean
        median_dt = time_deltas.median()
        mean_dt = time_deltas.mean()
        
        # Calculate sampling frequency (Hz)
        fs = 1.0 / median_dt if median_dt > 0 else 0
        
        # Flag if the mean deviates significantly from the median 
        # (This indicates missing chunks of data or heavy deadband compression)
        is_inconsistent = abs(median_dt - mean_dt) > (0.1 * median_dt)
        
        return {
            "median_interval_s": median_dt,
            "mean_interval_s": mean_dt,
            "estimated_hz": fs,
            "inconsistent_sampling": is_inconsistent
        }
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt, savgol_filter

class SignalConditioner:
    def _fill_gaps(self, series):
        """Filters crash on NaNs. Temporarily interpolate for filtering purposes."""
        return series.interpolate(method='linear').bfill().ffill()

    def apply_moving_average(self, series, window=5):
        return series.rolling(window=window, center=True, min_periods=1).mean()
        
    def apply_butterworth(self, series, cutoff_hz, fs, order=2):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff_hz / nyquist
        if normal_cutoff >= 1.0 or normal_cutoff <= 0:
            return series # Fallback if cutoff is invalid
            
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filled_series = self._fill_gaps(series)
        return pd.Series(filtfilt(b, a, filled_series), index=series.index)
        
    def apply_median(self, series, kernel_size=5):
        if kernel_size % 2 == 0: 
            kernel_size += 1 # Kernel must be odd
        filled_series = self._fill_gaps(series)
        return pd.Series(medfilt(filled_series, kernel_size=kernel_size), index=series.index)
        
    def apply_savgol(self, series, window=11, polyorder=2):
        if window % 2 == 0: 
            window += 1
        if polyorder >= window:
            polyorder = window - 1
        filled_series = self._fill_gaps(series)
        return pd.Series(savgol_filter(filled_series, window_length=window, polyorder=polyorder), index=series.index)
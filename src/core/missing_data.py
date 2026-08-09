import pandas as pd

class MissingDataImputer:
    def apply_imputation(self, series, strategy="Linear Interpolation"):
        """Fills NaN values using industrial standard techniques."""
        if series.isnull().sum() == 0:
            return series  # No missing data, return as-is
            
        if strategy == "Linear Interpolation":
            # Connects point A to point B with a straight line
            return series.interpolate(method='linear').bfill().ffill()
            
        elif strategy == "Forward Fill (Zero-Order Hold)":
            # Holds the last known transmitter value (Good for setpoints/valves)
            return series.ffill().bfill()
            
        elif strategy == "Cubic Spline":
            # Fits a smooth curve (Good for slow-moving tank levels)
            try:
                return series.interpolate(method='cubic').bfill().ffill()
            except:
                return series.interpolate(method='linear').bfill().ffill()
                
        return series
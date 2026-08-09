import pandas as pd

class HistorianDataLoader:
    def load_csv(self, file_object):
        """Loads data and automatically detects time and numeric columns."""
        df = pd.read_csv(file_object)
        
        # Auto-detect time column (assuming first column for MVP)
        time_col = df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Drop rows where time couldn't be parsed and sort chronologically
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        
        # Isolate numeric columns for processing
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        return df, time_col, numeric_cols
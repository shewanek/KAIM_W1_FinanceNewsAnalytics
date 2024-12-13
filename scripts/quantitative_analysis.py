import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import pynance as pn
import glob
import os

class StockEDA:
    """Class for exploratory data analysis of stock price data"""

    def __init__(self, folder_path):
        """Initialize with path to folder containing stock CSV files

        Args:
            folder_path (str): Path to folder containing stock price CSV files
        """
        self.folder_path = folder_path
        self.data = None

    def load_data(self):
        """Load and combine stock price data from CSV files

        Returns:
            pd.DataFrame: Combined dataframe with stock price data
        """
        # Get list of all CSV files in folder
        csv_files = glob.glob(os.path.join(self.folder_path, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.folder_path}")

        # Load each file into a list of dataframes
        df_list = []
        for file_path in csv_files:
            try:
                # Extract stock symbol from filename
                stock_symbol = os.path.basename(file_path).split('_')[0]

                # Read CSV and add stock symbol column
                df = pd.read_csv(file_path)
                df['stock_symbol'] = stock_symbol
                df_list.append(df)

            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue

        if not df_list:
            raise ValueError("No valid data could be loaded from CSV files")

        # Combine all dataframes and convert date
        self.data = pd.concat(df_list, ignore_index=True)
        self.data['Date'] = pd.to_datetime(self.data['Date'])

        return self.data

    def data_descriptive(self):
        """Generate descriptive statistics for stock price data

        Returns:
            dict: Dictionary containing key descriptive statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        descriptive_stats = {}

        # Basic price and volume stats
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        descriptive_stats['basic_stats'] = self.data[numeric_cols].describe()

        # Calculate daily returns
        self.data['Daily_Return'] = self.data.groupby('stock_symbol')['Close'].pct_change()

        # Key return metrics
        descriptive_stats['return_stats'] = {
            'mean_return': self.data['Daily_Return'].mean(),
            'return_std': self.data['Daily_Return'].std(),
            'max_return': self.data['Daily_Return'].max(),
            'min_return': self.data['Daily_Return'].min()
        }

        # Volume metrics
        descriptive_stats['volume_stats'] = {
            'avg_daily_volume': self.data['Volume'].mean(),
            'max_volume': self.data['Volume'].max(),
            'min_volume': self.data['Volume'].min()
        }

        # Price metrics
        descriptive_stats['price_stats'] = {
            'avg_daily_range': (self.data['High'] - self.data['Low']).mean(),
            'max_price': self.data['High'].max(),
            'min_price': self.data['Low'].min()
        }

        # Date range
        descriptive_stats['time_stats'] = {
            'start_date': self.data['Date'].min(),
            'end_date': self.data['Date'].max(),
            'trading_days': self.data['Date'].nunique()
        }

        # Per-stock summary
        descriptive_stats['stock_summary'] = self.data.groupby('stock_symbol').agg({
            'Close': ['mean', 'std'],
            'Volume': 'mean',
            'Daily_Return': 'mean'
        })

        return descriptive_stats

   
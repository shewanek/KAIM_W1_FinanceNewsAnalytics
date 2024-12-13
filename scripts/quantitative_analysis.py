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

    def calculate_technical_indicators(self):
        """Calculate various technical indicators using TA-Lib

        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Make sure data is sorted by date
        self.data = self.data.sort_values('Date')

        # Calculate Simple Moving Averages
        self.data['SMA_20'] = talib.SMA(self.data['Close'], timeperiod=20)
        self.data['SMA_50'] = talib.SMA(self.data['Close'], timeperiod=50)
        self.data['SMA_200'] = talib.SMA(self.data['Close'], timeperiod=200)

        # Calculate Exponential Moving Averages
        self.data['EMA_12'] = talib.EMA(self.data['Close'], timeperiod=12)
        self.data['EMA_26'] = talib.EMA(self.data['Close'], timeperiod=26)

        # Calculate MACD
        macd, macd_signal, macd_hist = talib.MACD(self.data['Close'])
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = macd_signal
        self.data['MACD_Hist'] = macd_hist

        # Calculate RSI
        self.data['RSI'] = talib.RSI(self.data['Close'], timeperiod=14)

        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(self.data['Close'])
        self.data['BB_Upper'] = upper
        self.data['BB_Middle'] = middle
        self.data['BB_Lower'] = lower

        return self.data

    def calculate_financial_metrics(self):
        """Calculate key financial metrics using PyNance

        Returns:
            dict: Dictionary containing calculated financial metrics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        metrics = {}

        # Calculate daily returns
        self.data['Daily_Return'] = self.data['Close'].pct_change()

        # Calculate volatility (standard deviation of returns)
        metrics['volatility'] = self.data['Daily_Return'].std() * np.sqrt(252)  # Annualized

        # Calculate Sharpe Ratio (assuming risk-free rate of 0.01)
        risk_free_rate = 0.01
        excess_returns = self.data['Daily_Return'] - risk_free_rate/252
        metrics['sharpe_ratio'] = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

        # Calculate maximum drawdown
        cum_returns = (1 + self.data['Daily_Return']).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns/rolling_max - 1
        metrics['max_drawdown'] = drawdowns.min()

        return metrics

    def plot_price_and_indicators(self, stock_symbol=None):
        """Create visualizations of price data and technical indicators

        Args:
            stock_symbol (str, optional): Filter data for specific stock symbol
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if stock_symbol:
            plot_data = self.data[self.data['stock_symbol'] == stock_symbol].copy()
        else:
            plot_data = self.data.copy()

        # Create figure with secondary y-axis
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), height_ratios=[3, 1, 1])

        # Plot price and moving averages
        ax1.plot(plot_data['Date'], plot_data['Close'], label='Close Price')
        ax1.plot(plot_data['Date'], plot_data['SMA_20'], label='20-day SMA')
        ax1.plot(plot_data['Date'], plot_data['SMA_50'], label='50-day SMA')
        ax1.plot(plot_data['Date'], plot_data['BB_Upper'], 'g--', label='BB Upper')
        ax1.plot(plot_data['Date'], plot_data['BB_Lower'], 'r--', label='BB Lower')
        ax1.set_title('Stock Price with Technical Indicators')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        # ax1.legend()
        ax1.grid(True)

        # Plot MACD
        ax2.plot(plot_data['Date'], plot_data['MACD'], label='MACD')
        ax2.plot(plot_data['Date'], plot_data['MACD_Signal'], label='Signal Line')
        ax2.bar(plot_data['Date'], plot_data['MACD_Hist'], label='MACD Histogram')
        ax2.set_title('MACD')
        # ax2.legend()
        ax2.grid(True)

        # Plot RSI
        ax3.plot(plot_data['Date'], plot_data['RSI'], label='RSI')
        ax3.axhline(y=70, color='r', linestyle='--')
        ax3.axhline(y=30, color='g', linestyle='--')
        ax3.set_title('RSI')
        ax3.set_ylabel('RSI')
        # ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

    

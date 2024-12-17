import pandas as pd
import numpy as np
from textblob import TextBlob
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os

class NewsStockCorrelation:
    """Class for analyzing correlation between news sentiment and stock movements"""

    def __init__(self, news_data_path, stock_data_path):
        """Initialize with paths to news and stock data files

        Args:
            news_data_path (str): Path to processed news data CSV
            stock_data_path (str): Path to stock price data CSV
        """
        self.news_data_path = news_data_path
        self.stock_data_path = stock_data_path
        self.news_df = None
        self.stock_df = None
        self.merged_df = None

    def load_data(self):
        """Load news and stock data from CSV files and align dates"""
        try:
            self.news_df = pd.read_csv(self.news_data_path)
            self.stock_df = pd.read_csv(self.stock_data_path)

            # Convert date columns to datetime with UTC timezone
            self.stock_df['Date'] = pd.to_datetime(self.stock_df['Date'], utc=True).dt.date
            self.news_df['date'] = pd.to_datetime(self.news_df['date'], utc=True).dt.date

            # Standardize date and stock column name to 'Date' and 'stock_symbol'
            self.news_df = self.news_df.rename(columns={'date': 'Date'})
            self.news_df = self.news_df.rename(columns={'stock': 'stock_symbol'})

            # # Normalize dates to start of day for consistent alignment
            # self.stock_df['Date'] = self.stock_df['Date'].dt.normalize()
            # self.news_df['Date'] = self.news_df['Date'].dt.normalize()

            # Sort both dataframes by date
            self.stock_df = self.stock_df.sort_values('Date')
            self.news_df = self.news_df.sort_values('Date')

            # Ensure dates overlap between datasets
            min_date = max(self.stock_df['Date'].min(), self.news_df['Date'].min())
            max_date = min(self.stock_df['Date'].max(), self.news_df['Date'].max())

            # Filter news data to match stock trading days
            self.news_df = self.news_df[self.news_df['Date'].isin(self.stock_df['Date'])]

            # Filter stock data to match news dates
            self.stock_df = self.stock_df[self.stock_df['Date'].isin(self.news_df['Date'])]

            print("Data loaded and aligned successfully")
            return self.news_df, self.stock_df

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False


    def align_data(self):
        """Align news and stock data by date and stock symbol"""
        if self.news_df is None or self.stock_df is None:
            raise ValueError("News or stock data not loaded. Call load_data() first.")

        # Merge on date and stock symbol
        self.merged_df = pd.merge(self.news_df, self.stock_df,
                                left_on=['Date', 'stock_symbol'], right_on=['Date', 'stock_symbol'],
                                how='inner')
        return self.merged_df

    
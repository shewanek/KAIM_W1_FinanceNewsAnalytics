# Financial Analysis Scripts

This directory contains Python scripts for analyzing financial news data and stock market data.

## financial_analysis.py

### Overview
The `financial_analysis.py` script provides a comprehensive set of analysis tools for processing and visualizing financial news data. It contains a main `FinancialAnalysis` class with methods for various types of analysis.

### Dependencies
- pandas
- matplotlib
- seaborn
- textblob
- scikit-learn
- numpy

### Key Features
- Data loading and cleaning
- Exploratory data analysis
- Sentiment analysis
- Topic modeling
- Publication pattern analysis
- Publisher analytics

### Class Methods

#### load_data()
- Loads data from CSV file into pandas DataFrame
- Default filepath: '../data/raw_analyst_ratings/raw_analyst_ratings.csv'

#### explore_data()
- Performs initial exploratory analysis
- Shows dataset overview, missing values, duplicates
- Provides basic statistics and categorical summaries

#### clean_data()
- Handles missing values and duplicates
- Standardizes date formats
- Cleans text fields
- Adds derived features like year, month, day of week

## quantitative_analysis.py

### Overview
The `quantitative_analysis.py` script provides tools for analyzing stock price data using technical and quantitative methods. It contains a main `StockEDA` class for exploratory data analysis.

### Dependencies
- pandas
- numpy
- matplotlib
- talib
- pynance

### Key Features
- Stock price data loading and processing
- Technical indicator calculations
- Financial metric analysis
- Data visualization

### Class Methods

#### load_data()
- Loads stock price data from CSV files
- Combines data from multiple stocks
- Adds stock symbol identifiers

#### data_descriptive()
- Generates descriptive statistics
- Calculates price and volume metrics
- Provides per-stock summaries

#### calculate_technical_indicators()
- Calculates moving averages (SMA, EMA)
- Generates MACD indicators
- Computes RSI values
- Calculates Bollinger Bands

#### calculate_financial_metrics()
- Computes volatility metrics
- Calculates Sharpe ratio
- Determines maximum drawdown

#### plot_price_and_indicators()
- Creates price charts with technical indicators
- Shows MACD and RSI plots
- Visualizes Bollinger Bands

#### analyze_stock()
- Performs comprehensive stock analysis
- Generates technical indicators
- Calculates financial metrics
- Creates visualizations


## correlation_analysis.py

### Overview
The `correlation_analysis.py` script provides tools for analyzing the correlation between news sentiment and stock movements. It contains a main `NewsStockCorrelation` class for correlation analysis.

### Dependencies
- pandas
- numpy
- textblob
- scipy
- matplotlib
- seaborn
### Class Methods

#### load_data()
- Loads news and stock data from CSV files
- Aligns data by date and stock symbol
- Calculates daily stock returns

#### align_data()
- Merges news and stock data on date and stock symbol
- Ensures data is ready for correlation analysis

#### calculate_sentiment_scores()
- Computes sentiment scores for news headlines
- Aggregates sentiment scores by date

#### calculate_stock_returns()
- Calculates daily stock returns for each stock
- Aggregates returns by date

#### calculate_correlation()
- Computes Pearson correlation coefficient between sentiment and stock returns
- Returns correlation results for each stock

#### plot_correlation()
- Visualizes the correlation results using bar plots
- Displays correlation coefficients for each stock symbol

#### scatter_plot()
- Creates scatter plots of sentiment scores vs. daily stock returns
- Visualizes relationships between sentiment and stock performance


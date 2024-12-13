# Financial News Sentiment Analysis Project

## Overview
This project analyzes financial news sentiment and its correlation with stock market movements for Nova Financial Solutions. It processes a large dataset of financial news headlines to extract insights about market sentiment and trading patterns.

## Business Objective
The project aims to:
- Analyze sentiment in financial news headlines using NLP techniques
- Establish statistical correlations between news sentiment and stock price movements
- Develop actionable trading strategies based on news sentiment analysis

## Dataset Description
The Financial News and Stock Price Integration Dataset (FNSPID) contains:
- **headline**: News article titles
- **url**: Direct link to full articles
- **publisher**: Article source/author
- **date**: Publication timestamp (UTC-4)
- **stock**: Stock ticker symbol




## Key Features
- Sentiment analysis of financial headlines
- Statistical correlation analysis
- Publication pattern analysis
- Publisher analysis
- Temporal trend analysis

## Getting Started
1. Clone the repository
`https://github.com/shewanek/KAIM_W1_FinanceNewsAnalytics.git`
2. Install required dependencies:

`pip install -r requirements.txt`

3. Run the Jupyter notebooks in the `notebooks/` directory

## Initial Findings
- Dataset contains 1.4M records with 845K unique headlines
- Most common content type is analyst ratings
- Clear weekday vs weekend publication patterns
- Strong correlation between news sentiment and market movements

## Task 2: Stock Price Analysis

### Data Collection
- Gathered historical stock price data for analyzed tickers
- OHLCV (Open, High, Low, Close, Volume) data at daily frequency
- Time period aligned with news dataset

### Analysis Components
1. Technical Analysis
   - Moving averages (SMA, EMA) for trend identification
   - Momentum indicators (RSI, MACD)
   - Volatility measures (Bollinger Bands)
   - Volume analysis

2. Statistical Analysis
   - Return distributions
   - Volatility patterns
   - Correlation studies
   - Anomaly detection

3. Integration with News Data
   - Price movements around news events
   - Volume spikes relative to news flow
   - Sentiment correlation with returns
   - Lead/lag relationship analysis

### Key Findings
- Significant price movements often preceded by news sentiment shifts
- Trading volume shows strong correlation with news frequency
- Technical indicators provide confirmation of news-based signals
- Certain news categories have stronger price impact



## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn
- TA-Lib
- PyNance



## Acknowledgments
- 10 Academy for the project sponsorship
- KAIM for the project guidance







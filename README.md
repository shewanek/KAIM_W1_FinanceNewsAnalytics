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

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn



## Acknowledgments
- 10 Academy for the project sponsorship
- KAIM for the project guidance

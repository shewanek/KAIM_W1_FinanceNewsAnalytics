# Financial Analysis Scripts

This directory contains Python scripts for analyzing financial news data and generating insights.

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

#### exploratory_analysis()
- Analyzes dataset shape and date range
- Shows stock coverage and publishing patterns
- Provides headline length and publisher summaries

#### descriptive_statistics()
- Generates visualizations for headline lengths
- Shows top publishers and article distributions
- Displays monthly article trends
- Provides detailed statistics

#### analyze_sentiment()
- Performs sentiment analysis on headlines
- Categorizes sentiment as positive/neutral/negative
- Visualizes sentiment distribution and trends

#### extract_topics()
- Performs topic modeling using LDA
- Visualizes top words per topic
- Shows topic distributions

#### analyze_key_phrases()
- Analyzes common financial phrase frequencies
- Visualizes phrase distributions
- Tracks key financial terminology

#### analyze_publication_patterns()
- Examines temporal publishing patterns
- Shows daily, hourly, and weekly distributions
- Provides publishing pattern statistics

#### analyze_publishers()
- Analyzes publisher distribution
- Shows top publishers by article count
- Extracts email domains
- Provides publisher content analysis




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class FinancialAnalysis:
    def __init__(self, df=None):
        self.df = df

    def load_data(self, filepath='../data/raw_analyst_ratings/raw_analyst_ratings.csv'):
        """Load data from CSV file into dataframe"""
        self.df = pd.read_csv(filepath)
        return self
    
    def explore_data(self):
        """Perform initial exploratory analysis focusing on key data quality aspects"""
        # Basic dataset info
        print("\nDataset Overview:")
        print("-" * 30)
        print(f"Number of records: {len(self.df)}")
        print(f"Columns: {', '.join(self.df.columns)}")
        
        # Missing values check
        print("\nMissing Values:")
        print("-" * 30)
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Duplicates check
        print("\nDuplicates:")
        print("-" * 30)
        duplicates = self.df.duplicated().sum()
        print(f"Duplicate rows: {duplicates} ({(duplicates/len(self.df))*100:.1f}%)")
        
        # Check duplicates excluding Unnamed: 0 column
     
        duplicates_excl = self.df.drop('Unnamed: 0', axis=1).duplicated().sum()
        print(f"\nDuplicates (excluding Unnamed: 0): {duplicates_excl} ({(duplicates_excl/len(self.df))*100:.1f}%)")

        # Basic statistics
        print("\nNumerical Statistics:")
        print("-" * 30)
        print(self.df.describe())
        
        # Categorical columns summary
        print("\nCategorical Columns:")
        print("-" * 30)
        for col in self.df.select_dtypes(include=['object']).columns:
            print(f"\n{col}:")
            print(f"Unique values: {self.df[col].nunique()}")
            print("Top 3 most common:")
            print(self.df[col].value_counts().head(3))
        
        return self
    
    def clean_data(self):
        """Clean and preprocess the dataset"""
        # Handle missing values first
        self.df = self.df.dropna(subset=['headline', 'stock', 'publisher'])  # Drop rows missing critical info
        
        # Remove duplicate rows while keeping first occurrence
        self.df = self.df.drop_duplicates(subset=['headline', 'stock', 'date'], keep='first')
        
        # Drop unnecessary columns
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop('Unnamed: 0', axis=1)
            
        # Convert and standardize date format
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.df = self.df.dropna(subset=['date'])  # Drop rows with invalid dates
        
        # Clean text fields
        self.df['headline'] = self.df['headline'].apply(lambda x: x.strip() if isinstance(x, str) else x)
        self.df['headline'] = self.df['headline'].str.replace(r'\s+', ' ')  # Remove extra whitespace
        
        self.df['publisher'] = (self.df['publisher'].str.strip()
                                                   .str.title()
                                                   .str.replace(r'\s+', ' '))
        
        self.df['stock'] = (self.df['stock'].str.strip()
                                           .str.upper()
                                           .str.replace(r'[^\w\s]', ''))  # Remove special chars
        
        self.df['url'] = (self.df['url'].str.strip()
                                       .str.lower())  # Standardize URLs to lowercase
        
        # Remove rows with empty strings after cleaning
        self.df = self.df.replace('', pd.NA).dropna()
        
        # Sort chronologically and reset index
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # Add derived features
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        self.df['headline_length'] = self.df['headline'].str.len()
        self.df['hour'] = self.df['date'].dt.hour  # Capture time of day
        
        return self
    def exploratory_analysis(self):
        """Perform initial exploratory analysis after data cleaning"""
        print("\n=== Exploratory Analysis ===")
        
        # Basic dataset info
        print("\nDataset Shape:", self.df.shape)
        print(f"Date Range: {self.df['date'].min()} to {self.df['date'].max()}")
        
        # Stock coverage
        print("\nTop 10 Most Covered Stocks:")
        print(self.df['stock'].value_counts().head(10))
        
        # Publishing patterns
        print("\nArticles by Hour of Day:")
        hour_dist = self.df['hour'].value_counts().sort_index()
        print(hour_dist)
        
        # Content analysis
        print("\nHeadline Length Summary:")
        print(self.df['headline_length'].describe())
        
        # Publisher analysis
        print("\nNumber of Unique Publishers:", self.df['publisher'].nunique())
        print("\nTop 5 Publishers by Article Count:")
        print(self.df['publisher'].value_counts().head())
        
        # Temporal patterns
        print("\nArticles by Day of Week:")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        print(self.df['day_of_week'].value_counts().reindex(day_order))
        
        return self


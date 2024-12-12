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

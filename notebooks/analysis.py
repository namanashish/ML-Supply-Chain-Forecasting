import pandas as pd


df = pd.read_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\predictions\final_predictions.csv")


print(df.head())
print(df.describe())


print("Nulls:", df['PredictedSales'].isnull().sum())
print("Zeros:", (df['PredictedSales'] == 0).sum())

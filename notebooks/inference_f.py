import pandas as pd
import joblib
import numpy as np
from xgboost import XGBRegressor

model=joblib.load(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\models\best_model.pkl")
df_test=pd.read_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\processed\test_encode.csv")

train_features = pd.read_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\processed\train_features.csv")
X_train = train_features.drop(["Sales", "Date"], axis=1)
feature_cols = X_train.columns.tolist()


for col in feature_cols:
    if col not in df_test.columns:
        df_test[col] = 0


X_test_final = df_test[feature_cols]


predictions = model.predict(X_test_final)


output = df_test.copy()
output["PredictedSales"] = predictions

output.to_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\predictions\final_predictions.csv", index=False)

print("Predictions saved")
print(df_test["Open"].value_counts())
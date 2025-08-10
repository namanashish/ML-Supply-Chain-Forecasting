import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
pred = pd.read_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\predictions\final_predictions.csv")
submission=pd.read_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\raw\submission.csv")

submission["PredictedSales"]=pred["PredictedSales"]
submission.to_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\output\final_out.csv")

rmse=np.sqrt(mean_absolute_error(submission["PredictedSales"],submission["Sales"]))
mae=mean_absolute_error(submission["PredictedSales"],submission["Sales"])
mse=mean_squared_error(submission["PredictedSales"],submission["Sales"])
r2=r2_score(submission["PredictedSales"],submission["Sales"])
print(rmse)
print(mae)
print(mse)
print(r2)

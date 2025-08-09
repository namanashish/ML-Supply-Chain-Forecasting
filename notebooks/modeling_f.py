import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib

def safe_mape(y_true,y_pred):
    y_true,y_pred = np.array(y_true),np.array(y_pred)
    mask=y_true!=0
    return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100

df=pd.read_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\processed\train_encoded.csv")
X=df.drop(["Sales","Date"],axis=1)
y=df["Sales"]
print(df.info())
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=54)

#Linear Regression
lr=LinearRegression()
lr.fit(X_train,y_train)
lr_preds=lr.predict(X_test)
lr_rmse=np.sqrt(mean_squared_error(y_test,lr_preds))
lr_mae=mean_absolute_error(y_test,lr_preds)
lr_r2=r2_score(y_test,lr_preds)
lr_mape=safe_mape(y_test,lr_preds)

print(f"Linear Regression:")
print(f"  RMSE: {lr_rmse:.2f}")
print(f"  MAE: {lr_mae:.2f}")
print(f"  R²: {lr_r2:.4f}")
print(f"  MAPE: {lr_mape:.2f}%\n")

#XGBOOST Regressor
xg=XGBRegressor()
xg.fit(X_train,y_train)
xg_preds=xg.predict(X_test)
xg_rmse=np.sqrt(mean_squared_error(y_test,xg_preds))
xg_mae=mean_absolute_error(y_test,xg_preds)
xg_r2=r2_score(y_test,xg_preds)
xg_mape=safe_mape(y_test,xg_preds)

print(f"XGBoost Regressor:")
print(f"  RMSE: {xg_rmse:.2f}")
print(f"  MAE: {xg_mae:.2f}")
print(f"  R²: {xg_r2:.4f}")
print(f"  MAPE: {xg_mape:.2f}%\n")

'''#Random Forest Regressor
rf=RandomForestRegressor()
rf.fit(X_train,y_train)
rf_preds=rf.predict(X_test)
rf_rmse=np.sqrt(mean_squared_error(y_test,rf_preds))
rf_mae=mean_absolute_error(y_test,rf_preds)
rf_r2=r2_score(y_test,rf_preds)
rf_mape=safe_mape(y_test,rf_preds)

print(f"Random Forest:")
print(f"  RMSE: {rf_rmse:.2f}")
print(f"  MAE: {rf_mae:.2f}")
print(f"  R²: {rf_r2:.4f}")
print(f"  MAPE: {rf_mape:.2f}%\n")'''

best_model = xg if xg_rmse < lr_rmse  else lr
joblib.dump(best_model, r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\models\best_model.pkl")

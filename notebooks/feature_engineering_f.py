import pandas as pd

df_test=pd.read_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\processed\cleaned_test.csv",parse_dates=["Date"])
df_train=pd.read_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\processed\cleaned.csv",parse_dates=["Date"])

#FeatureEngineering on Date in Train
df_train["Month"]=df_train["Date"].dt.month
df_train["Day"]=df_train["Date"].dt.day
df_train["Year"]=df_train["Date"].dt.year
df_train["DayOfWeek"]=df_train["Date"].dt.dayofweek
df_train["WeekOfYear"]=df_train["Date"].dt.isocalendar().week.astype(int)
df_train["IsWeekend"]=df_train["Date"].dt.dayofweek.isin([5,6]).astype(int)
df_train["IsMonthStart"]=df_train["Date"].dt.day.isin([1,2,3,4,5,6]).astype(int)
df_train["IsMonthEnd"]=df_train["Date"].dt.day.isin([26,27,28,29,30,31]).astype(int)

#FeatureEngineering on Date in Test
df_test["Month"]=df_test["Date"].dt.month
df_test["Day"]=df_test["Date"].dt.day
df_test["Year"]=df_test["Date"].dt.year
df_test["DayOfWeek"]=df_test["Date"].dt.dayofweek
df_test["WeekOfYear"]=df_test["Date"].dt.isocalendar().week.astype(int)
df_test["IsWeekend"]=df_test["Date"].dt.dayofweek.isin([5,6]).astype(int)
df_test["IsMonthStart"]=df_test["Date"].dt.day.isin([1,2,3,4,5,6]).astype(int)
df_test["IsMonthEnd"]=df_test["Date"].dt.day.isin([26,27,28,29,30,31]).astype(int)


df_train["TimeSince"]=(df_train["Date"]-df_train["Date"].min()).dt.days
df_test["TimeSince"]=(df_test["Date"]-df_test["Date"].min()).dt.days

df_train.to_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\processed\train_timeinc.csv",index=False)
df_test.to_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\processed\test_timeinc.csv",index=False)

#Calculating Time Since Promo & Competition Train
df_train["PromoSince"]=0
df_train.loc[df_train["Promo2"]==1 , "PromoSince"] =(
    12*(df_train["Promo2SinceYear"]-df_train["Promo2SinceYear"].min())+(df_train["Month"]-df_train["Promo2SinceWeek"]//4)
)
df_train["PromoSince"]=df_train["PromoSince"].apply(lambda x : max(x,0))

df_train["CompetitionOpenSince"]=12*(df_train["Year"]-df_train["CompetitionOpenSinceYear"]) + df_train["Month"]-df_train["CompetitionOpenSinceMonth"]
df_train["CompetitionOpenSince"]=df_train["CompetitionOpenSince"].apply(lambda x : max(x,0))
df_train= df_train.sort_values(["Store", "Date"])
for x in [1,7,30]:
    df_train[f"Sales_Lag_{x}"] = df_train.groupby("Store")["Sales"].shift(x)
for window in [7,30]:
    df_train[f"Sales_MA_{window}"]=df_train.groupby("Store")["Sales"].transform(lambda x:x.shift(1).rolling(window=window).mean())
df_train.fillna(0,inplace=True)

#Calculating Time Since Promo & Competition Test
df_test["PromoSince"]=0
df_test.loc[df_test["Promo2"]==1 , "PromoSince"] =(
    12*(df_test["Promo2SinceYear"]-df_test["Promo2SinceYear"].min())+(df_test["Month"]-df_test["Promo2SinceWeek"]//4)
)
df_test["PromoSince"]=df_test["PromoSince"].apply(lambda x : max(x,0))
df_test["CompetitionOpenSince"]=12*(df_test["Year"]-df_test["CompetitionOpenSinceYear"]) + df_test["Month"]-df_test["CompetitionOpenSinceMonth"]
df_test["CompetitionOpenSince"]=df_test["CompetitionOpenSince"].apply(lambda x : max(x,0))


df_train.to_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\processed\train_features.csv",index=False)
df_test.to_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\processed\test_features.csv",index=False)



from sklearn.preprocessing import LabelEncoder
label_cols=["StoreType","Assortment"]
encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col].astype(str))
    df_test[col] = le.transform(df_test[col].astype(str))
    encoders[col] = le

df_train.to_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\processed\train_encoded.csv",index=False)
df_test.to_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\processed\test_encoded.csv",index=False)

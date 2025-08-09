import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


stores=pd.read_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\raw\store.csv")
train=pd.read_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\raw\train.csv",parse_dates=["Date"],low_memory=False)
test=pd.read_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\raw\test.csv",parse_dates=["Date"],low_memory=False)



train=train.drop(columns=["StateHoliday"])
test=test.drop(columns=["StateHoliday"])

storestrain = pd.merge(train,stores,on="Store",how="left")
storestest=pd.merge(test,stores,on="Store",how="left")



storestrain=storestrain.iloc[:-1113]
storestrain.sort_values(["Date","Store"],inplace=True)
storestest.sort_values(["Date","Store"],inplace=True)



storestrain.to_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\processed\storestrain.csv",index=False)
storestest.to_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\processed\storestest.csv",index=False)

#PreProcessing
    #1.CompetitionDist
storestrain["CompetitionDistance"].fillna(storestrain["CompetitionDistance"].mean(),inplace=True)
storestest["CompetitionDistance"].fillna(storestest["CompetitionDistance"].mean(),inplace=True)
    #2.Promo2SinceWeek
storestrain["Promo2SinceWeek"].fillna(0,inplace=True)
storestest["Promo2SinceWeek"].fillna(0,inplace=True)

    #3.PromoInterval
storestrain["PromoInterval"].fillna(0,inplace=True)
storestest["PromoInterval"].fillna(0,inplace=True)

storestrain["PromoInterval"].replace({"Jan,Apr,Jul,Oct" : 1,"Feb,May,Aug,Nov":2,"Mar,Jun,Sept,Dec":3},inplace=True)
storestest["PromoInterval"].replace({"Jan,Apr,Jul,Oct" : 1,"Feb,May,Aug,Nov":2,"Mar,Jun,Sept,Dec":3},inplace=True)

    #4.Promo2SinceYear
storestrain["Promo2SinceYear"].fillna(0,inplace=True)
storestest["Promo2SinceYear"].fillna(0,inplace=True)

    #5.CompetitionOpenSinceMonth
storestrain["CompetitionOpenSinceMonth"].fillna(0,inplace=True)
storestest["CompetitionOpenSinceMonth"].fillna(0,inplace=True)

    #5.CompetitionOpenSinceYear
storestrain["CompetitionOpenSinceYear"].fillna(0,inplace=True)
storestest["CompetitionOpenSinceYear"].fillna(0,inplace=True)

storestrain.to_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\processed\cleaned.csv")
storestest.to_csv(r"C:\Users\naman\Downloads\ML-Driven Supply Chain Forecasting & Optimization\data\processed\cleaned_test.csv")
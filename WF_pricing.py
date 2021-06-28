import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import SVR
from statistics import mean


def get_data():
    filename = "Commodity Cheese Data - Pricing Analyst Interview Case Study.xlsx"
    headers = ['CATEGORY','SUBCATEGORY','BRAND','PRICING LINE','UPC','PRODUCT DESCRIPTION','PACK SIZE','SIZE',
               'UOM', 'PRICE (MAY)','UNIT COST (MAY)','UNITS (MAY)','SALES (MAY)','COMPETITOR PRICE (MAY)','PRICE (JUNE)','UNIT COST (JUNE)','UNITS (JUNE)',
               'SALES (JUNE)','COMPETITOR PRICE (JUNE)', 'SALES (52 WEEKS)']
    df = pd.read_excel(filename, engine='openpyxl',skiprows=2)
    pd.set_option('display.max_columns',30)
    df.columns = headers
    df = df.dropna()
    df.drop(columns=['UPC','CATEGORY','UOM',],inplace=True)

    return df

def data_for_ml(df):
    df['percent price change'] = df['PRICE (JUNE)']/df['PRICE (MAY)']
    df['percent unit change'] = df['UNITS (JUNE)']/df['UNITS (MAY)']
    return df

def mach_learn(df):
    X = df[['percent price change']].values
    y = df[['percent unit change']].values
    x_max = np.round(max(X),2)
    x_min = np.round(min(X),2)
    diff_x = np.arange(x_min, x_max, 0.01)
    diff_x = diff_x.reshape((len(diff_x),1))

    scaler_x = StandardScaler()
    scaler_x.fit(X)
    X_train = scaler_x.transform(X)
    X_test = scaler_x.transform(diff_x)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y)
    # print(X_train,y_train)

    regressor = SVR(kernel='rbf')
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    y_pred_fix = scaler_y.inverse_transform(y_pred)

    plt.figure(figsize=(8,6))
    plt.style.use('seaborn')
    plt.plot(diff_x,y_pred_fix)
    plt.title('Breakdown of Model, Price vs Demand',fontsize=20)
    plt.xlabel('Price Change',fontsize=14)
    plt.ylabel('Demand Change',fontsize=14)
    plt.show()

    return diff_x,y_pred_fix

def find_new_price_values(df,x_test, y_pred):
    new_values = []
    for p in range(len(df)):
        startprice = round(df['PRICE (JUNE)'][p], 2)
        cost = (df['UNIT COST (JUNE)'][p] + df['UNIT COST (MAY)'][p]) / 2
        demand_start = df['UNITS (JUNE)'][p]
        NewPrice = [np.round(startprice * x, 2) for x in x_test]
        NewDemand = [np.round(demand_start * x, 2) for x in y_pred]
        NewProfit = [np.round((NewPrice[x]) * NewDemand[x], 2) for x in range((len(NewPrice) - int(startprice)))]
        PriceOverComp = [np.round(NewPrice[x]-df['COMPETITOR PRICE (JUNE)'][p],2) for x in range(len(NewPrice))]
        
        tableOfPriceInfo = pd.DataFrame(data=[NewPrice,NewDemand,NewProfit,PriceOverComp]).transpose()
        sc1 = StandardScaler()
        sc2 = StandardScaler()
        sc3 = StandardScaler()
        # mms4 = MinMaxScaler()
        
        sc1.fit(tableOfPriceInfo[[0]].values)
        sc2.fit(tableOfPriceInfo[[1]].values)
        sc3.fit(tableOfPriceInfo[[2]].values)
        # mms4.fit(tableOfPriceInfo[[3]].values)
        tableOfPriceInfo[0] = sc1.transform(tableOfPriceInfo[[0]].values)
        tableOfPriceInfo[1] = sc2.transform(tableOfPriceInfo[[1]].values)
        tableOfPriceInfo[2] = sc3.transform(tableOfPriceInfo[[2]].values)
        # tableOfPriceInfo[3] = mms4.transform(tableOfPriceInfo[[3]].values)
        tableOfPriceInfo.dropna(inplace=True)
        tableOfPriceInfo['Optimized'] = tableOfPriceInfo.sum(axis=1)
        
        max_optim_index = tableOfPriceInfo['Optimized'].idxmax()
        tableOfPriceInfo[0] = sc1.inverse_transform(tableOfPriceInfo[[0]].values)
        tableOfPriceInfo[1] = sc2.inverse_transform(tableOfPriceInfo[[1]].values)

        new_values.append([tableOfPriceInfo[0][max_optim_index],tableOfPriceInfo[1][max_optim_index]])
        # new_demand.append(tableOfPriceInfo[1][max_optim_index])

    df_addon = pd.DataFrame(data=new_values)
    print(df_addon)
    
    
    return df_addon

def main():
    df = get_data()
    df = data_for_ml(df)
    x_test, y_pred = mach_learn(df)
    df_addon = find_new_price_values(df, x_test, y_pred)
    df_addon.columns = ['NewPrice', 'NewDemand']

    important_col = ['BRAND','PRICING LINE','PRICE (MAY)','UNIT COST (MAY)',
                    'UNITS (MAY)','COMPETITOR PRICE (MAY)','PRICE (JUNE)',
                    'UNIT COST (JUNE)','UNITS (JUNE)','COMPETITOR PRICE (JUNE)']
    df = df[important_col]
    df = pd.concat([df,df_addon], axis=1)
    print(df)

if __name__ == "__main__":
    main()


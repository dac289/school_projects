import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
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
    # for x in df.columns:
    #     print(x)
    #     print(df[x].isna().sum())
    df = df.dropna()
    df.drop(columns=['UPC','CATEGORY','UOM',],inplace=True)
    # print(len(df.columns))
    # print(len(df))
    return df

def explore_data(df):
    df_prices = df[['BRAND','UNITS (MAY)','UNITS (JUNE)','SALES (MAY)','SALES (JUNE)','SALES (52 WEEKS)']]
    # print(df_prices.groupby('BRAND').sum())
    df['Price Difference'] = df['PRICE (JUNE)'] - df['PRICE (MAY)']
    # print(df[df['BRAND'] == 'HORIZON ORGANIC'][['BRAND','Price Difference']])
    # print(df[df['BRAND'] == '365 EVERYDAY VALUE'][['BRAND', 'Price Difference']])
    # print(df[df['BRAND'] == 'ORGANIC VALLEY'][['BRAND', 'Price Difference']])

    df['Unit Difference'] = df['UNITS (JUNE)'] - df['UNITS (MAY)']
    df['Sale Difference'] = df['SALES (JUNE)'] - df['SALES (MAY)']
    df['Profit (MAY)'] = (df['PRICE (MAY)'] - df['UNIT COST (MAY)']) * df['UNITS (MAY)']
    df['Profit (JUNE)'] = (df['PRICE (JUNE)'] - df['UNIT COST (JUNE)']) * df['UNITS (JUNE)']

    df_365 = df[df['BRAND'] == '365 EVERYDAY VALUE']
    df_horizon = df[df['BRAND'] == 'HORIZON ORGANIC']
    df_orgval = df[df['BRAND'] == 'ORGANIC VALLEY']


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
    print(x_max)
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
    # plt.plot(diff_x,y_pred_fix)
    # plt.show()

    return diff_x,y_pred_fix, regressor

if __name__ == "__main__":
    df = get_data()
    df = explore_data(df)
    x_test, y_pred, regressor = mach_learn(df)
    # print(sum(df['UNIT COST (MAY)']>df['UNIT COST (JUNE)']),sum(df['UNIT COST (MAY)']==df['UNIT COST (JUNE)']),
    #       sum(df['UNIT COST (MAY)']<df['UNIT COST (JUNE)']))
    # print(mean(df['UNIT COST (MAY)'] - df['UNIT COST (JUNE)']))

    data_to_export = pd.DataFrame(x_test,y_pred)
    # print(df['PRICE (JUNE)']-df['COMPETITOR PRICE (JUNE)'])
    # print(df['UNIT COST (JUNE)'] - df['UNIT COST (MAY)'])

    new_values = []
    for p in range(len(df)):
        startprice = round(df['PRICE (MAY)'][p],2)
        cost = (df['UNIT COST (JUNE)'][p] + df['UNIT COST (MAY)'][p]) / 2
        demand_start = df['UNITS (MAY)'][p]
        NewPrice = [np.round(startprice * x,2) for x in x_test]
        NewDemand = [np.round(demand_start * x,2) for x in y_pred]
        NewProfit = [np.round((NewPrice[x] - cost) * NewDemand[x],2) for x in range((len(NewPrice)-int(startprice)))]
        ind = NewProfit.index(max(NewProfit))
        ind_2 = NewDemand.index(max(NewDemand))
        n_price = NewPrice[ind]
        n_demand = NewDemand[ind]
        n_profit = NewProfit[ind]
        n_price2 = NewPrice[ind_2]
        n_demand2 = NewDemand[ind_2]
        n_profit2 = NewProfit[ind_2]
        new_values.append([n_price,n_demand,n_profit,n_price2,n_demand2,n_profit2])

    df_addon = pd.DataFrame(new_values,columns=['Price_4_Profit','Demand_4_Profit','Profit_4_Profit',
                                                'Price_4_Demand','Demand_4_Demand','Profit_4_Demand'])
    print(df_addon)
    graphs_2_print = [9,11, 34, 29, 19,42,50,23]
    for x in graphs_2_print:
        startprice = round(df['PRICE (MAY)'][x], 2)
        cost = (df['UNIT COST (JUNE)'][x] + df['UNIT COST (MAY)'][x]) / 2
        demand_start = df['UNITS (MAY)'][x]
        NewPrice1 = [np.round(startprice * x, 2) for x in x_test]
        NewDemand1 = [np.round(demand_start * x, 2) for x in y_pred]
        NewProfit1 = [np.round((NewPrice1[x] - cost) * NewDemand1[x], 2) for x in range(len(NewPrice1))]
        print(max(NewProfit1))
        ind = NewProfit1.index(max(NewProfit1))
        print(NewProfit1[ind])
        print(NewPrice1[ind])
        plt.plot(NewPrice1,NewDemand1)
        plt.plot(NewPrice1,NewProfit1)
        plt.legend(['NewDemand','NewProfit'])
        plt.title(f"{x}, {df['PRICING LINE'][x]}")
        plt.show()

    df = pd.concat([df,df_addon],axis=1)
    print(df['Price_4_Profit']/df['PRICE (MAY)'])
    print(sum(df['Price_4_Profit'] / df['PRICE (MAY)']<1))
    df_to_excel = df['BRAND','PRICING LINE','PRICE (MAY)','UNIT COST (MAY)','']


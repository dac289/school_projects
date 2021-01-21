import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_error as MSE

def get_data(filename):

    data = pd.read_excel(filename, header=9)
    data = data.melt(id_vars='Year',var_name="Month", value_name="Price")
    month_map = {'Jan':1, 'Feb':2, "Mar":3, "Apr": 4, "May":5, "Jun":6, "Jul":7, "Aug":8, "Sep":9,
                 "Oct": 10, "Nov": 11, "Dec": 12}
    data.Month = data.Month.map(month_map)
    date_col = []
    for index,row in data.iterrows():
        date = '15/' + str(int(row['Month']))+ '/' +str(int(row['Year']))
        date_col.append(date)
    data['Date'] = date_col
    data = data[['Price','Date']]
    data['Date'] = pd.to_datetime(data.Date)
    data.set_index('Date', inplace=True)
    data.sort_index(inplace=True)
    data.dropna(inplace=True)
    data = pd.Series(data['Price'].values, index=pd.date_range('31/1/2010',periods=len(data),freq='M'))
    print(len(data))
    return data
    
def stationary_test(df):

    print("Results for Dickey-Fuller Test:")
    df_test = adfuller(df, autolag='aic')
    df_output = pd.Series(df_test[0:4],
                          index = ["Test Statistic","p_value","# Lages Used",
                                   "Number of Observations Used"])
    '''
    for x in df.values:
        print(x > 0.145)
       ''' 
    df2 = df.iloc[28:]
    df_test2 = adfuller(df2, autolag='aic')
    df_output2 = pd.Series(df_test2[0:4],
                          index = ["Test Statistic","p_value","# Lages Used",
                                   "Number of Observations Used"])
    '''
    plt.figure(figsize=(13,7.5))
    plt.style.use('seaborn')
    plt.plot(df)
    plt.xlabel('Years',fontsize=16)
    plt.ylabel('Texas Electricity Price (USD)',fontsize=16)
    plt.title('Texas Electricity Prices from January 2010 to October 2020',fontsize=20)
    plt.show()
    
    res = seasonal_decompose(df2, freq=12)
    res.plot()
    plt.figure(figsize=(13,7.5))
    plt.show()
    '''
    print(df_output)
    print(df_output2)
    return df2
    
def acf_pacf_plots(df):

    fig = plt.figure(figsize=(13,7.5))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(df, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(df, lags=40, ax=ax2)
    plt.show()

def SARIMAX_model(df):

    my_order = (4,0,1)
    seasonal_order = (0,0,1,12)
    model = SARIMAX(df,order=my_order, seasonal_order = seasonal_order)
    fit_result = model.fit()
    plt.figure(figsize=(13,7.5))
    plt.style.use('seaborn')
    plt.plot(df, label='Orignial Values')
    plt.plot(fit_result.fittedvalues, color='orange', label='Fitted Values')
    plt.legend(fontsize=16)
    plt.xlabel('Years')
    plt.ylabel('Electricity Prices (USD)')
    plt.title('SARIMA fitted model compared to the Original Data', fontsize=20)
    plt.show()
    
    test_mse = MSE(df, fit_result.fittedvalues)
    print(test_mse)
    
    fcast_5y = fit_result.predict(start='30/11/2020',end='31/10/2025')
    plt.figure(figsize=(13,7.5))
    plt.plot(df, label='Original')
    plt.plot(fcast_5y,color='orange',label='Forecasted Values')
    plt.legend(fontsize=16)
    plt.title('Texas Electricity Prices with Forecast Values (November 2020 to October 2025)',fontsize=20)
    plt.xlabel('Years')
    plt.ylabel('Electricity Prices (USD)')
    plt.show()
    
    print(type(fcast_5y))
    acf_pacf_plots(fit_result.fittedvalues)
    fcast_5y.to_csv('Electric_5year_prediction.csv')

def shit_method(series):
    import statsmodels.formula.api as smf
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    from statsmodels.tsa.ar_model import AR

    result = seasonal_decompose(series)
    component_dict = {'Trend': result.trend,'Seasonal': result.seasonal, 'Residual': result.resid}
    prediction_results = []

    for component in ['Seasonal','Trend','Residual']:

        historic = component_dict[component][:int(len(series) * 0.8)].to_list()
        historic = historic[6:]
        test = component_dict[component][int(len(series) * 0.8):].to_list()
        test = test[:-6]
        print((test))
        
        predictions = []
        for i in range(len(test)+66):
            model = AR(historic, missing='drop')
            model_fit = model.fit()
            pred = model_fit.predict(start=len(historic),end=len(historic), dynamic=False)
            predictions.append(pred[0])
            if i < 15:
                historic.append(test[i])
            else:
                historic.append(pred[0])
        #preditions = pd.Series(predictions, name=component, index=pd.date_range('31/5/2020',periods=len(predictions),freq='m')
        prediction_results.append(predictions)
        #test_score = np.sqrt(MSE(test,predictions))
        #print(f'Test for {component} MSE: {test_score}')
        '''
        plt.figure(figsize=(6,5))
        plt.style.use('seaborn')
        plt.plot(test, label='Observed ' + component)
        plt.plot(predictions, color='orange', label='Predictions '+component)
        plt.title(f'Electric: Observed {component} Values vs. Predictions {component} fit')
        plt.legend()
        plt.show()
    '''

    prediction_results = np.array(prediction_results).transpose()
    pred_df = pd.DataFrame(prediction_results, index=pd.date_range('28/2/2019',
                                                                   periods=len(prediction_results),
                                                                   freq='M'))
    pred_df['Prediction'] = pred_df.sum(axis=1)
    pred_ser = pd.Series(data=pred_df['Prediction'],index=pred_df.index)

    pred_for_graph = pred_ser[21:]
    
    # Test Data Graph fit
    test_value = pred_ser[:pred_ser.index.get_loc('30/11/2020')]
    original_for_fit = series[series.index.get_loc('28/2/2019'):]
    
    plt.figure(figsize=(8,5))
    plt.style.use('seaborn')
    plt.plot(original_for_fit, label='Original')
    plt.plot(test_value, color='orange',label='Predicted Values')
    plt.title("Model Fit for Original and Predicted data",fontsize=20)
    plt.xlabel("Years")
    plt.ylabel("Gas Prices (USD)",fontsize=14)
    plt.legend(fontsize=14)
    plt.show()

    # Real Predicted Graph!
    
    plt.figure(figsize=(13,7.5))
    plt.style.use('seaborn')
    plt.plot(series, label='Original')
    plt.plot(pred_for_graph, color='orange',label='Predicted Values')
    plt.title('Texas Electricity prices with Forecast Values (November 2020 to Ocotber 2025)',fontsize=20)
    plt.xlabel('Years',fontsize=16)
    plt.ylabel('Electric Prices (USD)',fontsize=16)
    plt.legend(fontsize=16)
    plt.show()
    
    pred_ser.to_excel('Gas and Electric Predictions.xlsx',sheet_name="Electric",engine='xlsxwriter')
    
def main():

    # Data can be found at https://data.bls.gov/timeseries/APUS37B72610?amp%253bdata_
    # tool=XGtable&output_view=data&include_graphs=true

    df = get_data('SeriesReport-20201119222226_11594c.xlsx')
    df2 = stationary_test(df)
    #acf_pacf_plots(df)
    #SARIMAX_model(df2)
    shit_method(df2)
    
    
if __name__ == "__main__":
    main()

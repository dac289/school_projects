import pandas as pd
import numpy as np
import xlsxwriter

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels as sm
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
        date = '15/' + str(int(row['Month']))+ '/' +str(int(row['Year'])) # dd/mm/yyyy
        date_col.append(date)
    data['Date'] = date_col
    data = data[['Price','Date']]
    data['Date'] = pd.to_datetime(data.Date)
    data.set_index('Date', inplace=True)
    data.sort_index(inplace=True)
    data.dropna(inplace=True)
    data = pd.Series(data['Price'].values, index=pd.date_range('31/1/2010',periods=len(data),freq='m'))
    print(len(data))
    return data
'''
def acf_pacf_plots(df):

    fig = plt.figure(figsize=(13,7.5))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(df,lags=40,ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(df, lags=40,ax=ax2)
    plt.show()

def stationary_test(df):

    print('Results from Dickey-Fuller Test:')
    df_test = adfuller(df, autolag='aic')
    df_output = pd.Series(df_test[0:4], index = ["Test Statistic", 'p_value',
                                                 '# Lags Used','# of Observations Used'])
    plt.figure(figsize=(13,7.5))
    plt.style.use('seaborn')
    plt.plot(df)
    plt.xlabel('Years', fontsize=16)
    plt.ylabel('Gas Prices (USD)',fontsize=16)
    plt.title('Texas Gas Prices (January 2010 to October 2020)',fontsize=20)
    plt.show()
    
    print(df_output)
    res = seasonal_decompose(df, period=12)
    res.plot()
    plt.show()

    
def sarima_model(df):

    my_order = (3,1,0)
    season_order = (2,0,1,12)
    model = SARIMAX(df, order=my_order, seasonal_order = season_order)
    fit_model = model.fit()
    plt.figure(figsize=(13,7.5))
    plt.style.use('seaborn')
    plt.plot(df, label='Original')
    plt.plot(fit_model.fittedvalues, color='orange', label="Fitted Values")
    plt.legend(fontsize=16)
    plt.title('SARIMA fitted model compared to the Original Data', fontsize=20)
    plt.xlabel('Years', fontsize=16)
    plt.ylabel('Gas Prices (USD)',fontsize=16)
    plt.show()

    fcast_5y = fit_model.predict(start='30/11/2020', end='31/10/2025')
    
    plt.figure(figsize=(13,7.5))
    plt.plot(df, label='Original')
    plt.plot(fcast_5y, color='orange', label='Forecast Values')
    plt.title('Texas Gas prices with Forecast Values (November 2020 to Ocotber 2025)',fontsize=20)
    plt.xlabel('Years', fontsize=16)
    plt.ylabel('Gas Prices (USD)', fontsize=16)
    plt.legend(fontsize=16)
    plt.show()
    
    #fcast_5y.to_csv('Gas_5year_prediction.csv')
    '''
def predictive_analysis(series):
    import statsmodels.formula.api as smf
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    from statsmodels.tsa.ar_model import AR

    result = seasonal_decompose(series)
    component_dict = {'Trend': result.trend,'Seasonal': result.seasonal, 'Residual': result.resid}
    prediction_results = []

    for component in ['Seasonal','Trend','Residual']:

        historic = component_dict[component].to_list()[:int(len(series) * 0.8)]
        historic = historic[6:]
        test = component_dict[component].to_list()[int(len(series) * 0.8):]
        test = test[:-6]
        print(len(historic) + len(test))
        
        predictions = []
        
        for i in range(len(test)+66):
            model = AR(historic, missing='drop')
            model_fit = model.fit()
            pred = model_fit.predict(start=len(historic),end=len(historic), dynamic=False)
            predictions.append(pred[0])
            if i < 20:
                historic.append(test[i])
            else:
                historic.append(pred[0])
        #preditions = pd.Series(predictions, name=component, index=pd.date_range('31/5/2020',periods=len(predictions),freq='m')
        prediction_results.append(predictions)
        #test_score = np.sqrt(MSE(test,predictions))
        #print(f'Test for {component} MSE: {test_score}')

        #print(predictions)
        #print(test)
        '''
        plt.figure(figsize=(6,5))
        plt.style.use('seaborn')
        plt.plot(test, label='observed ' + component)
        plt.plot(predictions, color='orange', label='Predictions '+component)
        plt.title(f'Gas: Observed {component} Values vs. Predictions {component} fit')
        plt.legend()
        plt.show()
        
        '''
    prediction_results = np.array(prediction_results).transpose()
    pred_df = pd.DataFrame(prediction_results, index=pd.date_range('30/9/2018',periods=len(prediction_results),freq='m'))
    pred_df['Prediction'] = pred_df.sum(axis=1)
    pred_ser = pd.Series(data=pred_df['Prediction'],index=pred_df.index)

    # Test Data Graph fit
    test_value = pred_ser[:pred_ser.index.get_loc('30/11/2020')]
    original_for_fit = series[series.index.get_loc('30/9/2018'):]

    plt.figure(figsize=(8,5))
    plt.style.use('seaborn')
    plt.plot(original_for_fit, label='Original')
    plt.plot(test_value, color='orange',label='Predicted Values')
    plt.title("Model Fit for Original and Predicted data",fontsize=20)
    plt.xlabel("Years")
    plt.ylabel("Gas Prices (USD)",fontsize=16)
    plt.legend(fontsize=16)
    plt.show()

    # Forecasted Graph!
    '''
    plt.figure(figsize=(13,7.5))
    plt.style.use('seaborn')
    plt.plot(series, label='Original')
    plt.plot(pred_ser, color='orange',label='Predicted Values')
    plt.title('Texas Gas prices with Forecast Values (November 2020 to Ocotber 2025)',fontsize=20)
    plt.xlabel('Years',fontsize=16)
    plt.ylabel('Gas Prices (USD)',fontsize=16)
    plt.legend(fontsize=16)
    plt.show()

    pred_ser.to_excel('Gas_Predictions_2.xlsx',sheet_name="Gas", engine='xlsxwriter')
    '''
def main_gas():

    df = get_data('SeriesReport-20201119222400_e33bad.xlsx')
    
    #stationary_test(df)
    #acf_pacf_plots(df)
    #sarima_model(df)
    #print(df[~df.isnull()])
    predictive_analysis(df)
    

if __name__ = "__main__":
    main()

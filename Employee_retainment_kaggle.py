import os
import pandas as pd
from datetime import datetime,timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split



def get_data():
    main_dir = 'archive/'
    general_data = pd.read_csv(main_dir + 'general_data.csv',index_col='EmployeeID')
    in_time_data = pd.read_csv(main_dir + 'in_time.csv',index_col=0)
    out_time_data = pd.read_csv(main_dir + 'out_time.csv',index_col=0)
    emsur_data = pd.read_csv(main_dir + 'employee_survey_data.csv',index_col='EmployeeID')
    manager_survey_data = pd.read_csv(main_dir + 'manager_survey_data.csv',index_col='EmployeeID')

    start_time = in_time_data.apply(pd.to_datetime)
    finish_time = out_time_data.apply(pd.to_datetime)

    general_data = general_data.join(emsur_data,on='EmployeeID')
    general_data =  general_data.join(manager_survey_data,on='EmployeeID')

    general_data['WorkingHours'] = (finish_time - start_time).mean(axis=1)
    general_data['WorkingHours'] = general_data['WorkingHours']/ np.timedelta64(1,'h')
    print(general_data.columns)

    general_data['Attrition'].replace({'Yes':1, 'No':0},inplace=True)
    general_data['Gender'].replace({'Female': 1, 'Male': 0}, inplace=True)

    general_data['JobLevel'] = general_data['JobLevel'].astype('object')

    value_columns = ['Age','DistanceFromHome','MonthlyIncome','NumCompaniesWorked',
                     'PercentSalaryHike','TotalWorkingYears',
                     'TrainingTimesLastYear','YearsAtCompany','YearsSinceLastPromotion',
                     'YearsWithCurrManager','WorkingHours']
    dummy_columns = ['BusinessTravel','Department','Education','EducationField',"JobLevel",'JobRole','MaritalStatus','StockOptionLevel','EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance','JobInvolvement', 'PerformanceRating']

    print(general_data.nunique())
    general_data.drop(['EmployeeCount','Over18','StandardHours'],axis=1,inplace= True)
    scaler = StandardScaler()
    general_data[value_columns] = scaler.fit(general_data[value_columns]).transform(general_data[value_columns])

    general_data['Education'] = general_data['Education'].replace(
        {1: 'Below College',2: 'College', 3: 'Bachelor',4: 'Master', 5: 'Doctor'})
    general_data['EnvironmentSatisfaction'] = general_data['EnvironmentSatisfaction'].replace(
        {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'})
    general_data['JobInvolvement'] = general_data['JobInvolvement'].replace({1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'})
    general_data['JobSatisfaction'] = general_data['JobSatisfaction'].replace({1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'})

    general_data['PerformanceRating'] = general_data['PerformanceRating'].replace({1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'})
    general_data['WorkLifeBalance'] = general_data['WorkLifeBalance'].replace({1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'})

    dummy_df = pd.get_dummies(general_data[dummy_columns])

    value_df = general_data.filter(value_columns)

    hr_data = pd.concat([value_df,dummy_df],axis=1)

    print(hr_data.columns)
    # sns.heatmap(hr_data.corr())
    # plt.show()

    hr_corr = hr_data.corr()
    corrdf = hr_corr.where(np.triu(np.ones(hr_corr.shape), k=1).astype(np.bool))
    corrdf = corrdf.unstack().reset_index()
    corrdf.columns = ['Var1', 'Var2', 'Correlation']
    corrdf.dropna(subset = ['Correlation'], inplace = True)
    corrdf['Correlation'] = round(corrdf['Correlation'], 2)
    corrdf['Correlation'] = abs(corrdf['Correlation'])
    matrix= corrdf.sort_values(by = 'Correlation', ascending = False).head(50)
    print(matrix)
    y_data = hr_data['Attrition']
    columns_to_drop = list(matrix.Var1[:10].unique())
    columns_to_drop = ['Attrition','PerformanceRating_Outstanding','Department_Sales','YearsWithCurrManager','PerformanceRating_Excellent','BusinessTravel_Travel_Rarely','JobInvolvement_Medium','TotalWorkingYears','WorkLifeBalance_Good','EducationField_Human Resources']
    hr_data.drop(hr_data[columns_to_drop],inplace=True,axis=1)

    X_train, y_train, X_test, y_test = train_test_split(hr_data,y_data,test_size=0.2,shuffle=True)

    

def main():
    get_data()


if __name__=='__main__':
    main()
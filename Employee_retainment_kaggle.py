import os
import pandas as pd
from datetime import datetime,timedelta
import numpy as np
import matplotlib.pyplot as plt


def get_data():
    main_dir = 'archive/'
    general_data = pd.read_csv(main_dir + 'general_data.csv',index_col='EmployeeID')
    in_time_data = pd.read_csv(main_dir + 'in_time.csv',index_col=0)
    out_time_data = pd.read_csv(main_dir + 'out_time.csv',index_col=0)
    employee_survey_data = pd.read_csv(main_dir + 'employee_survey_data.csv',index_col='EmployeeID')
    manager_survey_data = pd.read_csv(main_dir + 'manager_survey_data.csv',index_col='EmployeeID')

    start_time = in_time_data.apply(pd.to_datetime)
    finish_time = out_time_data.apply(pd.to_datetime)

    general_data['WorkingHours'] = (finish_time - start_time).mean(axis=1)
    general_data['WorkingHours'] = general_data['WorkingHours']/ np.timedelta64(1,'h')

    print(len(general_data.columns))

def main():
    get_data()


if __name__=='__main__':
    main()
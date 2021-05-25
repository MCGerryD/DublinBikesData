#Title: Data Analysis and Visualisation on real-world Dataset
#Is Dublin Bikes Usage impacted by Rainfall?
#Gerry Deignan - Data Analytics for Business - March

#import packages
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('classic')
import seaborn as sns
import datatables
from scipy import stats

#Read in the Weather Data to capture Rainfall - Dont need any manipulation on Weather Data so wil Merge at the end
parse_dates_w = ['date']
initial_weather = pd.read_csv("Weather.csv", parse_dates=parse_dates_w, dayfirst=True)
initial_weather['DATE'] = pd.DatetimeIndex(initial_weather['date']).date

#Read in the Dublin Bikes data and use dictionary to force the following two datatypes
dtypes = {'STATION ID':'int', 'TIME':'str'}
#Convert TIME field from OBJECT to DATETIME on the fly
parse_dates = ['TIME']

#Parse the TIME field to a Date field. Need dayfirst argument to avoid treating it as an american date
#Note I am using a pre-Covid timeframe so that data is not impacted by covid restrictions
initial_df = pd.read_csv("dublinbikes_20190101_20190401.csv", dtype=dtypes, parse_dates=parse_dates, dayfirst=True)
print(initial_df.head())
#Extract the Date from TIME so that we can calulate how many bikes were used in the entire day
initial_df['DATE'] = pd.DatetimeIndex(initial_df['TIME']).date

#Sort the dataframe so that I can get the final sum of bike usage for each day (end of day will have the cumulative sum)
initial_df = initial_df.sort_values(['STATION ID', 'TIME'], ascending=(True, True))

#Raw data has the number of bikes that are available each 5 minute interval
#To calculate the number times a bike is taken each day it is necessary
#to subtract the current number of available bikes from the previous one (5 mins before)
#Each time the difference is negative, this signifies the number if bikes taken in that
#5 minute window. Summing these for an entire day willgive the number of bikes taken in that day
initial_df['Interactions']=initial_df.groupby(['STATION ID', 'DATE'])['AVAILABLE BIKES'].diff().fillna(0)
#only count negative differences. The positive differences mean a bike was left there (not taken)
initial_df['Check_Neg'] = np.where(initial_df['Interactions'] < 0,initial_df['Interactions']*-1, 0 )
#Only using this if doing analysis by Station ID but too many for this project so will just do total usage
initial_df['Num_Taken']=initial_df.groupby(['STATION ID', 'DATE'])['Check_Neg'].cumsum().fillna(0)
#Total Usage for all stations by Date
initial_df = initial_df.sort_values(['DATE'], ascending=(True))
initial_df['Total_Num_Taken'] = initial_df.groupby('DATE')['Check_Neg'].cumsum().fillna(0)

#I want to group the data by Day i.e. cumulative sum all 5 minute intervals
# in each day and only keep the last value
def filter_last_timevalue(g):
    return g.groupby('DATE').last().agg({'Total_Num_Taken':'sum'})
#apply the filter
summary_df = initial_df.groupby(['DATE']).apply(filter_last_timevalue)
#reset index values
summary_df = summary_df.reset_index(level=0)
#Merge the summary data with the weather data on DATE and left join to only include values from Bikes data
merged_data = pd.merge(summary_df, initial_weather, how='left', on='DATE')

#Do a trend graph showing the total usage in the period (Matplotlib)
plt.figure(0)
plt.title("Number of Bikes Used per Day")
plt.xlabel("Date")
plt.ylabel("Number Used")
plt.plot('DATE', 'Total_Num_Taken', data=merged_data, )
plt.show()

#Show the correlation of Rainfall to Bike Usage along with printing the R Value (Seaborn)
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_data['rain'],merged_data['Total_Num_Taken'])
plt.figure(1)
chart = sns.regplot(merged_data['rain'],merged_data['Total_Num_Taken'])
chart.set(title='Number of Bikes Taken vs Rainfall(mm)',ylabel='Number Taken',xlabel='Rainfall(mm)')
#Print the R Value
print('The linear coefficient is',r_value)


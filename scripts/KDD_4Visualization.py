#Visualization scripts: These files contain code that generates visualizations, such as scatter plots, histograms, or heatmaps, 
# to explore and analyze the data. They often take the preprocessed data files as input and output the visualizations.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scripts import KDD_3Prediction

# Load data
data_hourly = pd.read_csv("C:\\Users\\Martina\\PycharmProjects\\KDD_Social_Biking\\data\\hour.csv")
data_daily = pd.read_csv("C:\\Users\\Martina\\PycharmProjects\\KDD_Social_Biking\\data\\day.csv")
KDD_3Prediction.data_cleaner(data_hourly)
KDD_3Prediction.outlier_cleaner(data_hourly)

# Compute monthly bike rental counts by year
monthly_bike_rental_counts = data_hourly.groupby(['yr', 'mnth'])['cnt'].mean().reset_index()
monthly_bike_rental_counts['mnth'] = pd.to_datetime(monthly_bike_rental_counts['mnth'], format='%m').dt.month_name().str.slice(stop=3)


# Compute bike rentals by weekday
bike_rentals_by_weekday = data_hourly.groupby('weekday')['cnt'].mean()


# Create first figure and axis for monthly bike rental counts by year
fig1, ax1 = plt.subplots()
ax1.set_title('Monthly Bike Rental Counts by Year')
ax1.set_xlabel('Month')
ax1.set_ylabel('Average Count')
ax1.set_xticks(range(0, 12))
ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax1.plot(monthly_bike_rental_counts[monthly_bike_rental_counts['yr']==0]['mnth'], monthly_bike_rental_counts[monthly_bike_rental_counts['yr']==0]['cnt'], label='2011')
ax1.plot(monthly_bike_rental_counts[monthly_bike_rental_counts['yr']==1]['mnth'], monthly_bike_rental_counts[monthly_bike_rental_counts['yr']==1]['cnt'], label='2012')
ax1.legend()


bike_rentals_by_season = data_hourly.groupby('season')['cnt'].mean()

# Create second figure and axis for bike rentals by season
fig2, ax2 = plt.subplots()
ax2.set_title('Bike Rentals by Season')
ax2.set_xlabel('Season')
ax2.set_ylabel('Total Count')
ax2.bar(bike_rentals_by_season.index, bike_rentals_by_season.values, align='center')
ax2.set_xticks(range(1, 5))
ax2.set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])


#Create a bar plot to show the portion of rentals on holidays vs non-holidays
bike_rental_counts_by_holiday = data_hourly.groupby('holiday')['cnt'].mean()
fig3, ax3 = plt.subplots()
ax3.bar(['Non-holiday', 'Holiday'], bike_rental_counts_by_holiday.values, color=['green', 'red'])
ax3.set_title('Proportion of Rentals on Holiday vs Non-Holidays')
ax3.set_xlabel('Holiday')
ax3.set_ylabel('Average Bike Rental Count')


# Create fourth figure and axis for bike rentals by weekday
fig4, ax4 = plt.subplots()
ax4.set_title('Bike Rentals by Weekday')
ax4.set_xlabel('Weekday')
ax4.set_ylabel('Average Bike Rental Count')
ax4.set_xticks(range(7))
ax4.set_xticklabels(['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
ax4.bar(bike_rentals_by_weekday.index, bike_rentals_by_weekday.values)

#Create a bar plot to show the proportion of rentals on working day vs non-working days
bike_rental_counts_by_workingday = data_hourly.groupby('workingday')['cnt'].mean()
fig5, ax5 = plt.subplots()
ax5.bar(['Non_workingday', 'Workingday'], bike_rental_counts_by_workingday.values, color=['purple', 'pink'])
ax5.set_xlabel('Working Day')
ax5.set_ylabel('Average Bike Rental Count')
ax5.set_title('Proportion of Rentals on Working Days vs Non-working Days')

# Compute bike rentals by weather situation
bike_rentals_by_weather_situation = data_hourly.groupby('weathersit')['cnt'].mean()


# Create a bar plot to show the proportion of rentals during each weather situation
fig6, ax6 = plt.subplots()
ax6.bar(bike_rentals_by_weather_situation.index, bike_rentals_by_weather_situation.values)
ax6.set_title('Bike Rentals by Weather Situation')
ax6.set_xlabel('Weather Situation')
ax6.set_ylabel('Total Count')
ax6.set_xticks(range(1, 5))
ax6.set_xticklabels(['Clear', 'Mist/Cloudy', 'Light Rain/Snow', 'Heavy Rain/Snow'], size=8)
ax6.grid(axis='y')


# Create a scatter plot with temperature and bike rental counts

x = (data_hourly['atemp'] * 39) - 8
x_rounded = np.round(x, decimals=2)
rental_counts = data_hourly.groupby(x)['cnt'].mean()  # Calculate the average rentals instead of count

fig7, ax7 = plt.subplots()
rental_counts.plot(kind='bar', color='steelblue')
ax7.set_title('Average Bike Rentals by Feeling Temperature')
ax7.set_xlabel('Temperature (°C)', fontsize=12)
ax7.set_ylabel('Average Bike Rentals', fontsize=12)
ax7.set_xticklabels([f'{temp:.1f}' for temp in rental_counts.index], fontsize=6)



# Create eighth figure and axis for humidity and bike rental counts
bins = [i for i in range(0, 110, 10)]
labels = ['{}-{}'.format(i, i+10) for i in range(0, 100, 10)]
data_hourly['hum_category'] = pd.cut(data_hourly['hum'], bins=bins, labels=labels)

data_hourly['hum_percent'] = (data_hourly['hum'] * 100).round(0).astype(int)

bike_rentals_by_humidity = data_hourly.groupby('hum_percent')['cnt'].mean().reset_index()


fig8, ax8 = plt.subplots()
sns.barplot(data=bike_rentals_by_humidity, x='hum_percent', y='cnt', alpha=0.7, ax=ax8)
ax8.set_title('Humidity vs Total Bike Rental Counts')
ax8.set_xlabel('Humidity Category')
ax8.set_ylabel('Total Bike Rental Count')
ax8.set_xticks(range(10, 110, 10))
ax8.set_xticklabels(range(10, 110, 10))


data_hourly['windspeed_mps'] = data_hourly['windspeed'] * 67

# Calculate the average bike rentals for each windspeed
average_rentals_by_windspeed = data_hourly.groupby('windspeed_mps')['cnt'].mean().reset_index()

# Create the bar plot
fig9, ax9 = plt.subplots(figsize=(12, 8))
ax9.bar(average_rentals_by_windspeed['windspeed_mps'], average_rentals_by_windspeed['cnt'])
ax9.set_title('Average Bike Rentals by Windspeed', fontsize=18)
ax9.set_xlabel('Windspeed (mps)', fontsize=14)
ax9.set_ylabel('Average Bike Rentals', fontsize=14)

# Rotate x-axis labels for better visibility if needed
plt.xticks(rotation=90)

'''
data_hourly['windspeed_mps'] = data_hourly['windspeed'] * 67
max_windspeed = int(data_hourly['windspeed_mps'].max())

fig9, ax9 = plt.subplots(figsize=(16, 9))
sns.barplot(data=data_hourly, x='windspeed_mps', y='cnt', ax=ax9)
ax9.set_title('Windspeed vs Rental Counts', fontsize=18)
ax9.set_xlabel('Windspeed (mps)', fontsize=14)
ax9.set_ylabel('Rental Counts', fontsize=14)
ax9.set_xlim(0, max_windspeed)
ax9.set_xticks(range(0, max_windspeed+1))
ax9.set_xticklabels(range(0, max_windspeed+1))
'''

#Create a new subplot for the hourly rental counts plot
fig10, ax10 = plt.subplots(figsize=(12,8))
sns.lineplot(data=data_hourly, x='hr', y='cnt')
ax10.set_title('Hourly Rental Counts', fontsize=18)
ax10.set_xlabel('Hour', fontsize= 14)
ax10.set_ylabel('Rental Counts', fontsize=14)

#Create a new subplot for the casual vs registered rentals
total_casual_rentals = data_hourly['casual'].mean()
total_registered_rentals = data_hourly['registered'].mean()
fig11, ax11 = plt.subplots(figsize=(6,6))
ax11.bar(['Casual', 'Registered'], [total_casual_rentals, total_registered_rentals], color=['blue', 'orange'])
ax11.set_title('Rentals by User Type', fontsize=16)
ax11.set_xlabel('User Type', fontsize=14)
ax11.set_ylabel('Rentals', fontsize=14)


# Save the figures
fig1.savefig('1_monthly_bike_rental_counts_by_year.png', dpi=300, bbox_inches='tight')
fig2.savefig('2_bike_rentals_by_season.png', dpi=300, bbox_inches='tight')
fig3.savefig('3_bike_rentals_on_holiday.png', dpi=300, bbox_inches='tight')
fig4.savefig('4_bikes_rentals_weekly.png', dpi=300, bbox_inches='tight')
fig5.savefig('5_bike_rentals_on_workingday.png', dpi=300, bbox_inches='tight')
fig6.savefig('6_bike_rentals_by_weather_situation.png', dpi=300, bbox_inches='tight')
fig7.savefig('7_temperature_vs_bike_rental_counts.png', dpi=300, bbox_inches='tight')
fig8.savefig('8_humidity_vs_bike_rental_counts.png', dpi=300, bbox_inches='tight')
fig9.savefig('9_windspeed_bike_rentals.png', dpi=300, bbox_inches='tight')
fig10.savefig('10_hourly_rental_count.png', dpi=300, bbox_inches='tight')
fig11.savefig('11_casual_vs_registered.png', dpi=300, bbox_inches='tight')

# Show the figures
#plt.show()

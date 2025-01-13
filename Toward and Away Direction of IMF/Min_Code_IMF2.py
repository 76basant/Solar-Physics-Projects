#Full Code:
# The objective of this code to plot the direction of IMF 
# from 1976 to 2018 year and separate them into Toward and  Away



#Required Libraries

import openpyxl
from openpyxl.workbook import Workbook
from openpyxl import load_workbook
import numpy as np
import pandas as pd


# Loading data as .dat file from this link:
#https://omniweb.gsfc.nasa.gov/html/polarity/polarity_tab.html

# read data as table data and using all columns

filepath=r"E:\Manjaro\bassant\A2\Python\Tasks\Cosmic Rays\Task 1\IMF direction original code\Main Folder\IMF.dat"
df1= pd.read_table(filepath, sep="\s+",  usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]  )

#printing data
print (df1)


# Convert rows to one column
stacked_df = df1.stack().reset_index(drop=True)

output_file = "after.xlsx"
stacked_df.to_excel(output_file, index=False)

#cover period cycle 22
#stacked_df= stacked_df.iloc[9266:13328]

#cover period cycle 1/1/1967 to 31/12/2018
stacked_df= stacked_df.iloc[1285:22390]

print(stacked_df.head(20))

#print(stacked_df.tail(21))

df=stacked_df 
#print(df.shape)

# Filter out rows where the Series contains only numbers
df_filtered = df[~df.astype(str).str.isdigit()]

#print(df_filtered)
#print(len(df_filtered ))


# Replace values in the DataFrame column
df_replaced = df_filtered.replace({'B': 'A', 'R': 'T', 'Y': 'M', 'W': 'N'})


#print(df_replaced.head(15))
#print(df_replaced.tail(20))

print(len(df_replaced))



#to make sure the interval is matching with calender
from datetime import datetime

# Define start and end dates
start_date = datetime(1967,1, 1)
end_date = datetime(2018,12,31)

# Calculate the total number of days
total_days = (end_date - start_date).days + 1
print(f"Total expected days: {total_days}")
import pandas as pd

# Assume df_replaced is a Series
df1 = df_replaced.copy()  # Make a copy of the original Series

# Rename the Series
df1.name = 'IMF'

# Convert the Series into a DataFrame (necessary for adding columns)
df3 = df1.reset_index()  # Resets the index and converts the Series into a DataFrame

# Step 1: Generate the date range
start_date = "1967-01-01"
end_date = "2018-12-31"
date_range = pd.date_range(start=start_date, end=end_date, freq='D')  # 'D' for daily frequency

# Step 2: Convert to a DataFrame
date_column = pd.DataFrame(date_range, columns=['Date'])

# Ensure `df3` has the same length as the date range
if len(df3) != len(date_range):
    raise ValueError("Length of data in `df3` does not match the length of the date range!")

# Step 3: Add the Date column to the DataFrame
df3['Date'] = date_column

# Step 4: Reorder columns (if required)
df3 = df3[['Date', 'IMF']]

# Display the resulting DataFrame
print(df3)

output_file = r"E:\Manjaro\bassant\A2\Python\Tasks\Cosmic Rays\Task 1\IMF direction original code\Main Folder\IMF_after.xlsx"
df3.to_excel(output_file, index=False)


#################################################

df=df3
# Count occurrences for each string in the 'IMF' column
df['A_count'] = df['IMF'].apply(lambda x: 1 if x == 'A' else 0)
df['T_count'] = df['IMF'].apply(lambda x: 1 if x == 'T' else 0)
df['N_count'] = df['IMF'].apply(lambda x: 1 if x == 'N' else 0)
df['M_count'] = df['IMF'].apply(lambda x: 1 if x == 'M' else 0)

# Display the resulting DataFrame
print(df)



output_file = r"E:\Manjaro\bassant\A2\Python\Tasks\Cosmic Rays\Task 1\IMF direction original code\Main Folder\IMF_coounts.xlsx"
df.to_excel(output_file, index=False)




# Extract the year from the 'Date' column
df['Year'] = df['Date'].dt.year

# Group by the 'Year' and calculate the sum for each year
yearly_sum = df.groupby('Year')[['A_count', 'T_count', 'N_count', 'M_count']].sum()

# Display the yearly sums
print(yearly_sum)


# Reset the index if 'date' is currently the index
yearly_sum.reset_index( inplace=True)

print(yearly_sum)

df=yearly_sum



####################################
#plotting
#font is used in plotting

font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }

# connecting between points by a curve instead of  line 

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
starting_year=1967
ending_year=2018

cubic_interploation_model = interp1d(df['Year'], df['N_count'], kind = "cubic")

X1_=np.linspace(starting_year, ending_year, 500)
Y1_=cubic_interploation_model(X1_)

plt.scatter(df['Year'], df['N_count'],label='N days',marker='o',color='red')
plt.plot(X1_, Y1_,color='red',linestyle=':')


cubic_interploation_model = interp1d(df['Year'], df['M_count'], kind = "cubic")
X2_=np.linspace(starting_year, ending_year, 500)
Y2_=cubic_interploation_model(X2_)

plt.scatter(df['Year'], df['M_count'],label='M days',marker='o',color='blue')
plt.plot(X2_, Y2_,color='blue',linestyle=':')
plt.legend()

#plt.xlim(1967,2018)
plt.ylim(0,155)
plt.xlabel("Year",font)
plt.ylabel("IMF direction",font)
plt.show()




cubic_interploation_model = interp1d(df['Year'], df['A_count'], kind = "cubic")

X1_=np.linspace(starting_year, ending_year, 500)
Y1_=cubic_interploation_model(X1_)

plt.scatter(df['Year'], df['A_count'],label='A days',marker='o',color='red')
plt.plot(X1_, Y1_,color='red',linestyle=':')


cubic_interploation_model = interp1d(df['Year'], df['T_count'], kind = "cubic")
X2_=np.linspace(starting_year, ending_year, 500)
Y2_=cubic_interploation_model(X2_)

plt.scatter(df['Year'], df['T_count'],label='T days',marker='o',color='blue')
plt.plot(X2_, Y2_,color='blue',linestyle=':')
plt.legend()

plt.ylim(60,220)
plt.xlabel("Year",font)
plt.ylabel("IMF direction",font)
plt.show()


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
df= pd.read_table(filepath, sep="\s+",  usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]  )

#printing data
print (df)



# Add column names manually or generate based on the data's length
df.columns = ["Year", "Days", "Other_Values"] + [f"Col_{i}" for i in range(1, len(df.columns) - 2 )]

# Display the DataFrame to verify it includes all rows
print(df)

# Function to calculate the date based on the year and day of the year
def get_date(year, day_of_year):
    return pd.to_datetime(f'{year}-01-01') + pd.Timedelta(days=day_of_year - 1)

# Extract the year and day of the year for the start row (1963, 306)
start_year = int(df.iloc[0]["Year"])
start_day = int(df.iloc[0]["Days"])

# Extract the year and day of the year for the end row (2024, 328 + 26)
end_year = int(df.iloc[-1]["Year"])
end_day = int(df.iloc[-1]["Days"]) + 26

print(end_year)
# Calculate the start and end dates
start_date = get_date(start_year, start_day)
end_date = get_date(end_year, end_day)

# Generate the date range from the start date to the end date
date_range = pd.date_range(start=start_date, end=end_date)

# Create a new DataFrame with the date range
df_dates = pd.DataFrame(date_range, columns=['Date'])

# Display the resulting DataFrame with the Date column
print(df_dates)



# Convert rows to one column
stacked_df = df.stack().reset_index(drop=True)

print(stacked_df)


df=stacked_df
# Filter out rows where the Series contains only numbers
df_filtered = df[~df.astype(str).str.isdigit()]

print(df_filtered)

df=df_filtered

# Rename the Series
df.name = 'IMF'


# Convert the Series into a DataFrame (necessary for adding columns)
df = df.reset_index()  # Resets the index and converts the Series into a DataFrame

print(df)


df['Date']=df_dates['Date']

df=df.iloc[:,1:]
print(df)



# Reorder columns: Year first, Month second
df = df[['Date', 'IMF']]
print(df)




# Specify the start and end dates
start_date = "1967-01-01"
end_date = "2018-12-31"

# Filter the rows based on the specified date range
filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

# Display the filtered rows
print(filtered_df)

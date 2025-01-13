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

##########################
# to count values of R,W,B,Y indicators of IMF direction in every year
# as summation 

# File name
file_name = filepath

# Initialize a dictionary to store counts grouped by year
counts_by_year = {}

# Open the file and process it
with open(file_name, "r") as file:
    for line in file:
        # Skip empty lines or lines that don't have enough data
        if not line.strip():
            continue
        
        # Skip comment lines
        if line.startswith("#"):
            continue
        
        # Split the line into columns
        columns = line.strip().split()
        
        # Ensure there are enough columns (at least 4: year, metadata, and data)
        if len(columns) < 4:
            print(f"Skipping line due to insufficient data: {line.strip()}")
            continue
        
        # Extract the year and the sequence data
        try:
            year = int(columns[0])  # First column is the year
            sequence_data = columns[3:]  # Data starts from the fourth column
        except ValueError as e:
            print(f"Error parsing line: {line.strip()} -> {e}")
            continue
        
        # Initialize the count dictionary for the year if not already present
        if year not in counts_by_year:
            counts_by_year[year] = {'W': 0, 'B': 0, 'R': 0, 'Y': 0}
        
        # Count occurrences of each letter in the current sequence
        for value in sequence_data:
            if value in counts_by_year[year]:
                counts_by_year[year][value] += 1

# Output the results
print("Counts by Year:")
for year, counts in counts_by_year.items():
    print(f"Year {year}: {counts}")

#######################################################3
#to print all rows related to every year

# File name
file_name = filepath

# Initialize a dictionary to store data grouped by year
data_by_year = {}

# Open the file and process it
with open(file_name, "r") as file:
    for line in file:
        # Skip empty lines or lines that don't have enough data
        if not line.strip():
            continue
        
        # Skip comment lines
        if line.startswith("#"):
            continue
        
        # Split the line into columns
        columns = line.strip().split()
        
        # Ensure there are enough columns (at least 4: year, metadata, and data)
        if len(columns) < 4:
            print(f"Skipping line due to insufficient data: {line.strip()}")
            continue
        
        # Extract the year and the sequence data
        try:
            year = int(columns[0])  # First column is the year
            sequence_data = columns[3:]  # Data starts from the fourth column
        except ValueError as e:
            print(f"Error parsing line: {line.strip()} -> {e}")
            continue
        
        # Add data to the dictionary for the corresponding year
        if year not in data_by_year:
            data_by_year[year] = []
        
        # Append the sequence data to the year
        data_by_year[year].append(sequence_data)

# Output the results
print("Data by Year:")
for year, sequences in data_by_year.items():
    print(f"Year {year}:")
    for sequence in sequences:
        print(" ".join(sequence))  # Join the sequence data for display
    print()  # Add a blank line for better readability



#####################
#to count indicators of IMF for every year separately 

# File name
file_name = filepath

# Initialize a list to store the data for the DataFrame
data = []

# Open the file and process it
with open(file_name, "r") as file:
    for line in file:
        # Skip empty lines or lines that don't have enough data
        if not line.strip():
            continue
        
        # Skip comment lines
        if line.startswith("#"):
            continue
        
        # Split the line into columns
        columns = line.strip().split()
        
        # Ensure there are enough columns (at least 4: year, metadata, and data)
        if len(columns) < 4:
            print(f"Skipping line due to insufficient data: {line.strip()}")
            continue
        
        # Extract the year and the sequence data
        try:
            year = int(columns[0])  # First column is the year
            sequence_data = columns[3:]  # Data starts from the fourth column
        except ValueError as e:
            print(f"Error parsing line: {line.strip()} -> {e}")
            continue
        
        # Count occurrences of each letter in the sequence
        count_B = sequence_data.count("B")
        count_W = sequence_data.count("W")
        count_R = sequence_data.count("R")
        count_Y = sequence_data.count("Y")
        
        # Append the data for this row
        data.append([year, count_B, count_W, count_R, count_Y])

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=["Year", "B_Count", "W_Count", "R_Count", "Y_Count"])

# Display the DataFrame
print(df)

# Optionally, save the DataFrame to a CSV file

filepath1=r"file900.xlsx"
# Step 3: Save the selected columns to a new Excel file (optional)
df.to_excel(filepath1, index=False)


#######################################

# File name
file_name = filepath

# Initialize a dictionary to store counts grouped by year
counts_by_year = {}

# Open the file and process it
with open(file_name, "r") as file:
    for line in file:
        # Skip empty lines or lines that don't have enough data
        if not line.strip():
            continue
        
        # Skip comment lines
        if line.startswith("#"):
            continue
        
        # Split the line into columns
        columns = line.strip().split()
        
        # Ensure there are enough columns (at least 4: year, metadata, and data)
        if len(columns) < 4:
            print(f"Skipping line due to insufficient data: {line.strip()}")
            continue
        
        # Extract the year and the sequence data
        try:
            year = int(columns[0])  # First column is the year
            sequence_data = columns[3:]  # Data starts from the fourth column
        except ValueError as e:
            print(f"Error parsing line: {line.strip()} -> {e}")
            continue
        
        # Initialize the count dictionary for the year if not already present
        if year not in counts_by_year:
            counts_by_year[year] = {'W': 0, 'B': 0, 'R': 0, 'Y': 0}
        
        # Count occurrences of each letter in the current sequence
        for value in sequence_data:
            if value in counts_by_year[year]:
                counts_by_year[year][value] += 1

# Convert the results into a pandas DataFrame
data = {
    "Year": [],
    "count_W": [],
    "count_B": [],
    "count_R": [],
    "count_Y": []
}

for year, counts in sorted(counts_by_year.items()):  # Sort by year for better readability
    data["Year"].append(year)
    data["count_W"].append(counts["W"])
    data["count_B"].append(counts["B"])
    data["count_R"].append(counts["R"])
    data["count_Y"].append(counts["Y"])

df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
output_file = "IMF_direction_counts.xlsx"
df.to_excel(output_file, index=False)

print(f"Data saved to {output_file}")



df=pd.read_excel(r'IMF_direction_counts.xlsx', engine='openpyxl')

############
print(df1)


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
df_replaced = df_filtered.replace({'B': 'A', 'R': 'T', 'Y': np.nan, 'W': np.nan})

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

#####################


#######################
filepath=r"E:\Manjaro\bassant\A2\Python\Tasks\Cosmic Rays\Task 1\IMF direction original code\Main Folder\Solar_Wind.dat"

df2= pd.read_table(filepath, sep="\s+",  usecols=[0,1,2,3]  )

#printing data
#print (df2)

df2=df2.iloc[1096:,:]
print(df2)



# Generate the date range
date_range = pd.date_range(start="1967-01-01", end="2018-12-31", freq="D")

# Ensure the number of rows in df2 matches the number of rows in the date range
if len(df2) != len(date_range):
    raise ValueError("The number of rows in df2 does not match the number of dates in the range!")

# Add Year, Month, and Day columns to df2
df2["Year"] = date_range.year
df2["Month"] = date_range.month
df2["Day"] = date_range.day

# Display the updated DataFrame
#print(df2)

df2=df2.iloc[:,3:]
print(df2)

# Reorder columns: Year first, Month second
df2 = df2[['Year', 'Month', 'Day','SW_Speed']]

print(df2)




# Reset index to default integer range
df1 = df1.reset_index(drop=True)
df2 = df2.reset_index(drop=True)

# Now combine the DataFrames
df_combined = pd.concat([df2, df1], axis=1)

print(df_combined)



df=df_combined 

df['SW_Speed'] = df['SW_Speed'].replace([999.9], np.nan)

print(df)

# Creating SW_A and SW_T columns based on IMF_Polarity
df['SW_A'] = df['SW_Speed'].where(df['IMF'] == 'A')
df['SW_T'] = df['SW_Speed'].where(df['IMF'] == 'T')

# Display the resulting DataFrame
print(df.iloc[:,2:])


output_file = "Solar_after.xlsx"
df.to_excel(output_file, index=False)



# Count the number of NaN values in both columns
nan_count_sw_a = df['SW_A'].isna().sum()
nan_count_sw_t = df['SW_T'].isna().sum()

# Display the DataFrame and the count of NaN values
print("Updated DataFrame:")
print(df)
print(f"\nNumber of NaN values in SW_A: {nan_count_sw_a}")
print(f"\nNumber of NaN values in SW_T: {nan_count_sw_t}")


# Select specific columns by index
df= df.iloc[:, [0, 1,2,5,6]]  # This selects columns at index 0 and 2
#print(df)




#df['SW_A_spline'] = df['SW_A'].interpolate(method='spline', order=3)

#df['SW_T_spline'] = df['SW_T'].interpolate(method='spline', order=3)

#print(df)

output_file = "spline_before.xlsx"
df.to_excel(output_file, index=False)


# Step 1: Combine Year, Month, and Day into a datetime column
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

# Step 2: Set the 'date' column as the index
df.set_index('Date', inplace=True)

# Step 3: Resample to monthly frequency and calculate the average
df_monthly_avg = df.resample('M').mean()

# Display the result
print(df_monthly_avg)

df=df_monthly_avg 
# Step 1: Check which columns contain NaN values
columns_with_nan = df.isnull().any()

# Step 2: Count the number of NaN values in each column
nan_counts = df.isnull().sum()

# Step 3: Filter columns with NaN values and display their counts
nan_columns_and_counts = nan_counts[nan_counts > 0]

# Display the results
print("Columns with NaN values:")
print(nan_columns_and_counts)



# Perform spline interpolation
df['SW_A_interpolated'] = df['SW_A'].interpolate(method='spline', order=3)

# Perform spline interpolation
df['SW_T_interpolated'] = df['SW_T'].interpolate(method='spline', order=3)


# Display the result
print(df)

# Reset the index if 'date' is currently the index
df.reset_index( inplace=True)

# Split 'date' column into 'year' and 'month'
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month


df.drop('Day', axis=1, inplace=True)


print(df)
output_file=r"E:\Manjaro\bassant\A2\Python\Tasks\Cosmic Rays\Task 1\IMF direction original code\Main Folder\monthly.xlsx"
"monthly.xlsx"
df.to_excel(output_file, index=False)

df.set_index('Date', inplace=True)

# Step 2: Resample the data to yearly frequency and calculate the average for each year
df_yearly_avg = df.resample('Y').mean()


df=df_yearly_avg 


# Display the result
print(df)


# Reset the index if 'date' is currently the index
df.reset_index( inplace=True)

df.drop('Month', axis=1, inplace=True)

output_file = "yearly.xlsx"
df.to_excel(output_file, index=False)

df = df.iloc[:, [1, 4, 5]]



print(df)
# Calculate the rolling correlation with window = 13
#rolling_corr = df['SW_A_interpolated'].rolling(window=13).corr(df['SW_T_interpolated'])

#print(rolling_corr )


# Calculate the centered rolling correlation with window=13
rolling_corr = df['SW_A_interpolated'].rolling(window=13,center=True).corr(df['SW_T_interpolated'])

# Display the result
print(rolling_corr)






import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(df['Year'], rolling_corr, label='Rolling Correlation', color='b')
plt.scatter(df['Year'], rolling_corr, label='Rolling Correlation', color='b')
plt.title('Rolling Correlation between value_1 and value_2')
plt.xlabel('Year')
#plt.ylabel('Correlation')
#plt.grid(True)
#plt.legend()
plt.show()








































 






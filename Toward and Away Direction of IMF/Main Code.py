#Full Code:
# The objective of this code to plot the direction of IMF 
# from 1976 to 2018 year and separate them into Toward and Away



#Required Libraries

import openpyxl
from openpyxl.workbook import Workbook
from openpyxl import load_workbook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading data as .dat file from this link:
#https://omniweb.gsfc.nasa.gov/html/polarity/polarity_tab.html

# read data as table data and using all columns

filepath=r"E:\Project 1\IMF.dat"
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

filepath1=r"E:\Project 1\file900.xlsx"
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




cubic_interploation_model = interp1d(df['Year'], df['count_W'], kind = "cubic")

X1_=np.linspace(1964, 2018, 500)
Y2_=cubic_interploation_model(X1_)

plt.scatter(df['Year'], df['count_W'],label='N days',marker='o',color='red')
plt.plot(X1_, Y2_,color='red',linestyle=':')


cubic_interploation_model = interp1d(df['Year'], df['count_Y'], kind = "cubic")
X2_=np.linspace(1964, 2018, 500)
Y2_=cubic_interploation_model(X2_)

plt.scatter(df['Year'], df['count_Y'],label='M days',marker='o',color='blue')
plt.plot(X2_, Y2_,color='blue',linestyle=':')
plt.legend()

plt.xlim(1967,2018)
plt.ylim(0,155)
plt.xlabel("Year",font)
plt.ylabel("IMF direction",font)
plt.show()




cubic_interploation_model = interp1d(df['Year'], df['count_B'], kind = "cubic")

X1_=np.linspace(1964, 2018, 500)
Y2_=cubic_interploation_model(X1_)

plt.scatter(df['Year'], df['count_B'],label='A days',marker='o',color='red')
plt.plot(X1_, Y2_,color='red',linestyle=':')


cubic_interploation_model = interp1d(df['Year'], df['count_R'], kind = "cubic")
X2_=np.linspace(1964, 2018, 500)
Y2_=cubic_interploation_model(X2_)

plt.scatter(df['Year'], df['count_R'],label='T days',marker='o',color='blue')
plt.plot(X2_, Y2_,color='blue',linestyle=':')
plt.legend()

plt.xlim(1967,2018)
plt.ylim(60,220)
plt.xlabel("Year",font)
plt.ylabel("IMF direction",font)
plt.show()


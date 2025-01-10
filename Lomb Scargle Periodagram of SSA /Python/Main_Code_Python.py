#The objective of this code:
#to plot Lomb Scargle periodagram of SSA from cycles 12 to 24

#Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import openpyxl
from openpyxl.workbook import Workbook
from openpyxl import load_workbook
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
############################
#this part to plot monthly average SSA


# Define the input file path
file_path = r'D:\Time Seies Analysis\b10.dat'


# Loading data as .dat file from this link:
#https://solarscience.msfc.nasa.gov/greenwch/sunspot_area.txt
# read data as table data and using all columns
df1= pd.read_table(file_path, sep="\s+",  usecols=[0,1,2]  )
#printing data
print (df1)

#select data from cycles from 21 to 24
df1=df1.loc[1222:1747,:]

# rename the columns, you should use that:
df1.columns= ['Year', 'Month','SSA']
print(df1)

# Calculate monthly averages (if needed)
# For this example, it's assumed the data is already monthly
monthly_data = df1.copy()

# Find rows with NaN values
nan_rows = monthly_data[monthly_data.isnull().any(axis=1)]

# Display the rows with NaN values
print("Rows with NaN values:")
print(nan_rows)
print(monthly_data)

monthly_data['time'] = monthly_data['Year'] + monthly_data['Month'] / 12
print(monthly_data)

#########################################################
#plotting Raw Data of SSA from Cycles 21 to 24 
plt.scatter(monthly_data['time'],monthly_data['SSA'], color='blue', s=3)
#plt.plot( monthly_data['time'],monthly_data['SSA'], color='red')

# Retrieve current y-ticks
yticks = plt.gca().get_yticks()
y_step=yticks[1] - yticks[0]
print("Y-Ticks:", yticks)
print("Y-Step:", y_step)

# Retrieve current x-ticks
xticks = plt.gca().get_xticks()
x_step=xticks[1] - xticks[0]
print("x-Ticks:", xticks)
print("x-Step:", x_step)

# Data: Year and corresponding cycle numbers
years = [2008 + 12 / 12.0, 1996 + 8 / 12.0,1986 +
         9 / 12.0, 1976 + 3 / 12.0]

print(len(years))
cycles = range(24, 11, -1)  # Example cycle numbers: 24, 23, ..., 11
print(list(cycles))
# Plotting vertical lines with annotations
for year, cycle in zip(years, cycles):
    plt.axvline(x=year, color='red', linestyle='--', linewidth=0.5)  # Vertical line
    plt.text(year+1, 2550, f'Cycle {cycle}', color='red', rotation=0, fontsize=12, verticalalignment='bottom')  # Annotation
plt.ylim([0,max(monthly_data['SSA'])+y_step/2])
plt.xlim([min(monthly_data['time']-x_step/5.0),max(monthly_data['time'])+x_step/5.0])
plt.xlabel('Time')
plt.ylabel('SSA ($\mu Hemi$)')
plt.title('SSA Monthly Average Cycles 21-24')
plt.show()

############################################
# save the DataFrame to a Excel file

# Step 3: Save the selected columns to a new Excel file (optional)
monthly_data.to_excel('file1.xlsx', index=False)

#####################
#Object oriented program to plot Lomb Scargle periodogram

# Function to load and preprocess data
def load_data(file_path, date_columns, value_column, start_date):
    df = pd.read_excel(file_path, engine='openpyxl')
    try:
        # Ensure the date columns are integers
        df[date_columns] = df[date_columns].astype(int)
    except ValueError:
        raise ValueError("Ensure the date columns contain only integers.")
    
    # Create a datetime column
    df['Date'] = pd.to_datetime(df[date_columns].assign(DAY=1))
    start_date = pd.to_datetime(start_date)
    
    # Convert time to months since start_date
    df['Time'] = (df['Date'] - start_date) / pd.Timedelta(days=30.44)
    df['Signal'] = df[value_column]  # Rename signal column for consistency
    
    # Remove rows with missing or invalid signal values
    df = df.dropna(subset=['Signal'])
    
    return df[['Time', 'Signal']]

# Function to perform Lomb-Scargle analysis with dynamic frequency range
def lomb_scargle_analysis(time, signal, num_freqs=500, confidence_levels=(95, 99)):
    # Calculate the dynamic frequency range based on the length of the signal
    i = 1 / len(signal)  # Maximum of observed periodicities (min frequency)
    j = 1 / 2           # Minimum of observed periodicities (max frequency)

    # Generate the frequency array dynamically
    frequency = np.linspace(i, j, num_freqs)

    # Perform Lomb-Scargle analysis
    power = LombScargle(time, signal).power(frequency)
    periods = 1 / frequency

    # Find peaks in the power spectrum
    peaks, _ = find_peaks(power, height=np.percentile(power, confidence_levels[0]))
    peak_frequencies = frequency[peaks]
    peak_periods = 1 / peak_frequencies

    # Calculate confidence thresholds
    conf_thresholds = {level: np.percentile(power, level) for level in confidence_levels}

    return {
        'frequency': frequency,
        'power': power,
        'periods': periods,
        'peaks': peaks,
        'peak_frequencies': peak_frequencies,
        'peak_periods': peak_periods,
        'confidence_thresholds': conf_thresholds,
    }

# Function to plot Lomb-Scargle periodogram with only peaks within the desired xlim
def plot_lomb_scargle(results, ax, title="Lomb-Scargle Periodogram", xlabel="Frequency (1/month)", ylabel="Power", xlim=(0, 0.2)):
    # Interpolation for smoother plot
    cubic_interpolation_model = interp1d(results['frequency'], results['power'], kind="cubic")
    interpolated_freq = np.linspace(results['frequency'][0], results['frequency'][-1], 500)
    interpolated_power = cubic_interpolation_model(interpolated_freq)

    # Plot interpolated power spectrum
    ax.plot(interpolated_freq, interpolated_power, color='blue', label='Smoothed Power')

    # Confidence thresholds (95% red, 99% black)
    ax.axhline(y=results['confidence_thresholds'][95], color='red', linestyle='--', label='95% Confidence')
    ax.axhline(y=results['confidence_thresholds'][99], color='black', linestyle='--', label='99% Confidence')

    # Filter peaks that are within the xlim range (0.007, 0.2)
    valid_peaks = [(freq, period) for freq, period in zip(results['peak_frequencies'], results['peak_periods']) if 0.007 <= freq <= 0.2]
    
    # Get y-axis ticks  
    y_ticks = plt.gca().get_yticks()
    print("Y-axis ticks:", y_ticks)

    # Calculate step size (assuming at least two ticks)
    y_step_size = y_ticks[1] - y_ticks[0]
    print("Step size of y-axis:", y_step_size)

    # Highlight peaks with vertical lines
    for freq, period in valid_peaks:
        ax.axvline(x=freq, color='green', linestyle=':', linewidth=1)
        ax.text(freq, max(results['power'])-y_step_size , f'{period / 12.0:.2f} yr', color='red',
                horizontalalignment='right', verticalalignment='center', fontsize=10, rotation=90)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_ylim([0, max(results['power']) + 0.02])
    ax.set_xlim(xlim)  # Set the xlim to the desired range

#######################################################
# Main execution to plot
# cycle 21 and cycle 22 in first figure
# cycle 23 and cycle 24 in second figure

if __name__ == "__main__":
    file_path = r'E:\file1.xlsx'
    date_columns = ['Year', 'Month']
    value_column = 'SSA'  # Change this to the desired column

    # Create subplots for two separate figures, adjusting the size for cycles 21 and 22
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 8))  # Larger size for Figure 1 (cycles 21 and 22)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))  # Figure 2 with standard size for cycles 23 and 24

    # Define the start dates for each cycle and corresponding row/column in the subplot grid
    start_dates = ['1976-03-01', '1986-09-01', '1996-08-01', '2008-12-01']
    cycle_ranges = [(0, 127), (126, 246), (245, 394), (393, None)]  # Define data ranges for each cycle

    # For Figure 1 (cycles 21 and 22)
    for i, (start_date, (start_idx, end_idx)) in enumerate(zip([start_dates[0], start_dates[1]], [cycle_ranges[0], cycle_ranges[1]])):
        # Load and preprocess data for each cycle
        df = load_data(file_path, date_columns, value_column, start_date)
        df = df.iloc[start_idx:end_idx, :]  # Slice data for the current cycle
        
        # Perform Lomb-Scargle analysis with dynamic frequency range
        results = lomb_scargle_analysis(df['Time'].values, df['Signal'].values)

        # Plot results in corresponding subplot for Figure 1
        plot_lomb_scargle(results, axes1[i], title=f"Lomb-Scargle Periodogram - Cycle {21 + i}")

    # For Figure 2 (cycles 23 and 24)
    for i, (start_date, (start_idx, end_idx)) in enumerate(zip([start_dates[2], start_dates[3]], [cycle_ranges[2], cycle_ranges[3]])):
        # Load and preprocess data for each cycle
        df = load_data(file_path, date_columns, value_column, start_date)
        df = df.iloc[start_idx:end_idx, :]  # Slice data for the current cycle
        
        # Perform Lomb-Scargle analysis with dynamic frequency range
        results = lomb_scargle_analysis(df['Time'].values, df['Signal'].values)

        # Plot results in corresponding subplot for Figure 2
        plot_lomb_scargle(results, axes2[i], title=f"Lomb-Scargle Periodogram - Cycle {23 + i}")

    # Display the figures with tight layout
    fig1.tight_layout()
    fig2.tight_layout()
    
    # Show the figures
    plt.show()

#####################
# Main execution for specific Cycle 
if __name__ == "__main__":
    file_path = r'E:\file1.xlsx'
    date_columns = ['Year', 'Month']
    value_column = 'SSA'  # Change this to the desired column

    # Create a subplot for cycle 21
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the size as needed

    # Define the start date and data range for cycle 21
    #Cycle_Number='21'
    #start_date = '1976-03-01'
    #cycle_range = (0, 127)  # Adjust this as needed to match cycle 21's range in your dataset
    
    # Define the start date and data range for cycle 22
    Cycle_Number='22'
    start_date = '1986-09-01'
    cycle_range = (126, 246)  # Adjust this as needed to match cycle 22's range in your dataset
    
    # Define the start date and data range for cycle 23
    #start_date = '1996-08-01'
    #cycle_range = (245, 394)  # Adjust this as needed to match cycle 23's range in your dataset
    
    # Define the start date and data range for cycle 24
    #start_date = '2008-12-01'
    #cycle_range = (393,None)  # Adjust this as needed to match cycle 24's range in your dataset
    

    # Load and preprocess data for cycle 21
    df = load_data(file_path, date_columns, value_column, start_date)
    df = df.iloc[cycle_range[0]:cycle_range[1], :]  # Slice data for cycle 21

    # Perform Lomb-Scargle analysis with dynamic frequency range
    results = lomb_scargle_analysis(df['Time'].values, df['Signal'].values)

    # Plot results for cycle 21
    plot_lomb_scargle(results, ax, title=f'Lomb-Scargle Periodogram - Cycle {Cycle_Number}')

    # Display the figure
    plt.tight_layout()
    plt.show()

###############################

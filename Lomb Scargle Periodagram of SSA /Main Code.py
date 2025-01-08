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
#https://omniweb.gsfc.nasa.gov/html/polarity/polarity_tab.html
# read data as table data and using all columns
df1= pd.read_table(file_path, sep="\s+",  usecols=[0,1,2]  )
#printing data
print (df1)

#select data from cycles from 21 to 24
df1=df1.loc[1222:1747,:]

#if you want to rename the columns, you should use that:
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

#monthly_data=monthly_data.iloc[8:]
print(monthly_data)

monthly_data['time'] = monthly_data['Year'] + monthly_data['Month'] / 12

print(monthly_data)

#plotting Raw Data
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
# Optionally, save the DataFrame to a Excel file

# Step 3: Save the selected columns to a new Excel file (optional)
monthly_data.to_excel('file1.xlsx', index=False)

########################################
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

# Function to plot Lomb-Scargle periodogram with only peaks
def plot_lomb_scargle(results, title="Lomb-Scargle Periodogram", xlabel="Frequency (1/month)", ylabel="Power"):
    plt.figure(figsize=(10, 6))

    # Interpolation for smoother plot
    cubic_interpolation_model = interp1d(results['frequency'], results['power'], kind="cubic")
    interpolated_freq = np.linspace(results['frequency'][0], results['frequency'][-1], 500)
    interpolated_power = cubic_interpolation_model(interpolated_freq)

    
    # Plot interpolated power spectrum
    plt.plot(interpolated_freq, interpolated_power, color='blue', label='Smoothed Power')

    # Confidence thresholds
    for level, threshold in results['confidence_thresholds'].items():
        plt.axhline(y=threshold, linestyle='--', label=f'{level}% Confidence')

    # Highlight peaks with vertical lines
    for freq, period in zip(results['peak_frequencies'], results['peak_periods']):
        plt.axvline(x=freq, color='green', linestyle=':', linewidth=1)
        plt.text(freq, max(results['power']) * 0.9, f'{period / 12.0:.2f} yr', color='red',
                 horizontalalignment='right', verticalalignment='center', fontsize=10, rotation=90)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.ylim([0, max(results['power']) + 0.02])
    plt.xlim([0,0.2])
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    file_path = r'E:\file1.xlsx'
    date_columns = ['Year', 'Month']
    value_column = 'SSA'  # Change this to the desired column
    start_date = '1976-03-01'

    # Load and preprocess data
    df = load_data(file_path, date_columns, value_column, start_date)
    print("Monthly Average SSA from cycles 21 to 24")
    print(df)

    # Perform Lomb-Scargle analysis with dynamic frequency range
    results = lomb_scargle_analysis(df['Time'].values, df['Signal'].values)

    # Plot results
    plot_lomb_scargle(results, title="Lomb-Scargle Periodogram of Signal")

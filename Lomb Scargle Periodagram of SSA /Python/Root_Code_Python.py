#The objective of this Root code:
#to use Lomb Scargle Periodogram for SSAs from Cycles 21 to 24

import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# Specify the full file path
file_path = r'E:\Manjaro\bassant\A2\Matlab\Second Work\Lomb Scargle periodogram\file1.xlsx'

# Load the Excel file
try:
    df = pd.read_excel(file_path, engine='openpyxl')
    print("File loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' does not exist. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")

#select data of cycle 21
Cycle_Number='21'
#df=df.iloc[:127,:]
#start_date = pd.to_datetime('1976-03-01')


#select data of cycle 22
Cycle_Number='22'
#df=df.iloc[126:246,:]
#start_date = pd.to_datetime('1986-09-01')

#select data of cycle 23
Cycle_Number='23'
df=df.iloc[245:394,:]
start_date = pd.to_datetime('1996-08-01')


#select data of cycle 24
#Cycle_Number='24'
#df=df.iloc[393:,:]
#start_date = pd.to_datetime('2008-12-01')

# Calculate the absolute asymmetry
df['Signal'] = df['SSA']


# Ensure the 'Year' and 'Month' columns are integers
df['Year'] = df['Year'].astype(int)
df['Month'] = df['Month'].astype(int)

# Create a time variable in months since the start date
#start_date = pd.to_datetime('1986-01-01')
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
df['Time'] = (df['Date'] - start_date) / pd.Timedelta(days=30.44)  # Approximate months
print(df)

# Extract time and signal for Lomb-Scargle analysis
time = df['Time'].values
signal = df['Signal'].values

# Frequency range for Lomb-Scargle
i = 1 / len(signal)  # Maximum of observed periodicities ( min of frequency)
j = 1 / 2           # Minimum of observed periodicities ( max of frequency)
frequency = np.linspace(i, j, 300)

# Lomb-Scargle periodogram for the absolute asymmetry
lombscargle = LombScargle(time, signal).power(frequency)
#print(lombscargle)

# Calculate the periods (1/frequency)
periods = 1 / frequency
#print(periods)

# Find all peaks in the Lomb-Scargle power spectrum
peaks, _ = find_peaks(lombscargle, height=np.percentile(lombscargle, 95))  # Threshold at 95% confidence level
peak_frequencies = frequency[peaks]
peak_periods = 1 / peak_frequencies

print("Peaks")
print(peaks)
print("peak_frequencies")
print(peak_frequencies)
print("peak_periods")
print(peak_periods)

for freq, period in zip(peak_frequencies, peak_periods):
    print(freq, period/12.0)
    print(freq, np.max(lombscargle)*0.9)

# Confidence levels (95% and 99%)
confidence_95 = np.percentile(lombscargle, 95)
confidence_99 = np.percentile(lombscargle, 99)
############################################
#Plotting 
# Creating the figure for the Lomb-Scargle periodogram
plt.figure(figsize=(10, 6))


#plt.plot(frequency, lombscargle, color='blue', label='Absolute Asymmetry')


cubic_interploation_model = interp1d(frequency, lombscargle, kind = "cubic")
X1_=np.linspace(0.008, 0.2, 500)
Y2_=cubic_interploation_model(X1_)

#plt.scatter(frequency, lombscargle,color='blue')
plt.plot(X1_, Y2_,color='blue')


plt.axhline(y=confidence_95, color='red', linestyle='--', label='95% Confidence')
plt.axhline(y=confidence_99, color='black', linestyle='-.', label='99% Confidence')

# Set the x-axis limit to cut the curve after x = 0.2
plt.xlim([0, 0.2])

plt.xlabel('Frequency (1/month)')
plt.ylabel('Power')
plt.title(f'Lomb-Scargle Periodogram of $SSA$ in cycle {Cycle_Number}')
plt.legend(loc='upper right')
plt.ylim([0, max(lombscargle) + 0.02])
plt.tight_layout()

# Get y-axis ticks
y_ticks = plt.gca().get_yticks()
print("Y-axis ticks:", y_ticks)
# Calculate step size (assuming at least two ticks)
y_step_size = y_ticks[1] - y_ticks[0]
print("Step size of y-axis:", y_step_size)

# Annotate all detected periodicities
for freq, period in zip(peak_frequencies, peak_periods):
    print(freq, period/12.0)
    plt.axvline(x=freq, color='green', linestyle=':', linewidth=1)
    plt.text(freq, np.max(lombscargle), f'{period/12.0:.2f} yr', color='red',
             horizontalalignment='right', verticalalignment='center', fontsize=12, rotation=90)

    
    
    
plt.show()

#The objective of this code:
#Forecast SSA (data from cycles 12 to 24) using LSTM model 20 to 24 cycles

#Required Libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
import openpyxl
from openpyxl.workbook import Workbook
from openpyxl import load_workbook
import os
import seaborn as sns
import tensorflow as tf
import random
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from google.colab import drive

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

drive.mount('/content/drive')

# Change to your desired folder
os.chdir('/content/drive/My Drive/Machine Learning Codes/Time Series Machine Learning/Forecasting Models/')



# Loading data as .dat file from this link:
#https://omniweb.gsfc.nasa.gov/html/polarity/polarity_tab.html
# read data as table data and using all columns
df1= pd.read_table('b10.dat', sep="\s+",  usecols=[0,1,2]  )
#printing data
print (df1.loc[55:1747,:])

df1=df1.loc[55:1747,:]



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
         9 / 12.0, 1976 + 3 / 12.0, 1964 +
         10 / 12.0,1954 + 4 / 12.0, 1944 +
         2 / 12.0, 1933 + 9 / 12.0,1923 +
         8 / 12.0, 1913 + 7 / 12.0, 1902 +
         1 / 12.0, 1890 + 3 / 12.0,1878+12/12]

print(len(years))
cycles = range(24, 11, -1)  # Example cycle numbers: 24, 23, ..., 11
print(list(cycles))
# Plotting vertical lines with annotations
for year, cycle in zip(years, cycles):
    plt.axvline(x=year, color='red', linestyle='--', linewidth=0.5)  # Vertical line
    plt.text(year, 4200, f' {cycle}', color='red', rotation=0, fontsize=12, verticalalignment='bottom')  # Annotation
plt.ylim([0,max(monthly_data['SSA'])+y_step/2])
plt.xlim([min(monthly_data['time']-x_step/5.0),max(monthly_data['time'])+x_step/5.0])
plt.xlabel('Time')
plt.ylabel('SSA ($\mu Hemi$)')
plt.title('SSA Monthly Average Cycles 12-24')
#plt.legend()
#plt.grid(True)
plt.show()
#################################################################

# Calculate 13-month Simple Moving Average (SMA)
window=13
monthly_data['SMA_13'] = monthly_data['SSA'].rolling(window=13).mean()
print(monthly_data.head(13))

monthly_data=monthly_data.iloc[12:]

print(monthly_data)

# Scatter plot of monthly
#plt.figure(figsize=(10, 6))
plt.scatter(monthly_data['time'], monthly_data['SMA_13'], label='Monthly Data', marker='o', color='red',s=3)


# Cubic interpolation
cubic_interpolation_model = interp1d(monthly_data['time'], monthly_data['SMA_13'], kind="cubic")

# Generate interpolated values for the full range of time data
X1_ = np.linspace(monthly_data['time'].min(), monthly_data['time'].max(), 1000)  # Dense range for smooth curve
Y2_ = cubic_interpolation_model(X1_)
# Plot interpolated curve
plt.plot(X1_, Y2_, color='red', linestyle=':', label='Cubic Interpolation')

# Plot formatting
# Retrieve current y-ticks
yticks = plt.gca().get_yticks()
y_step=yticks[1] - yticks[0]
print("Y-Ticks:", yticks)
print("Y-Step:", y_step)

xticks = plt.gca().get_xticks()
x_step=xticks[1] - xticks[0]
print("X-Ticks:", xticks)
print("X-Step:", x_step)

plt.ylim([0,max(monthly_data['SMA_13'])+y_step/5])
plt.xlim([min(monthly_data['time']-x_step/5.0),max(monthly_data['time'])+x_step/5.0])
plt.xlabel('Time')
plt.ylabel('SSA ($\mu Hemi$)')

plt.title('SSA Monthly (Moving Average) with Interpolation')
plt.legend()
#plt.grid(True)
plt.show()
####################################################


###################################################3
# Split the data sequentially (60% training, 40% testing)

print("length of data")
print(len(monthly_data))
data_size=0.6
train_size = int(data_size * len(monthly_data))
x_train = monthly_data[['time']].values[:train_size]
y_train = monthly_data['SMA_13'].values[:train_size]
x_test = monthly_data[['time']].values[train_size:]
y_test = monthly_data['SMA_13'].values[train_size:]

print("x_train:")
print(x_train.shape)
print("y_train:")
print(y_train.shape)
print("x_test:")
print(x_test.shape)
print("y_test:")
print(y_test.shape)

# Linear regression model for trend fitting
model_trend = LinearRegression()
model_trend.fit(x_train, y_train)

# Predict fitted and forecasted values
y_fittedvalue = model_trend.predict(x_train)
y_forecast = model_trend.predict(x_test)

# Plot training data and trend line

plt.scatter(x_train, y_train, color='blue', label='Training Data', s=3)
#plt.plot(x_train, y_train, color='blue')
plt.plot(x_train, y_fittedvalue, color='green', label='Fitted Trend')

# Plot test data and forecast
plt.scatter(x_test, y_test, color='orange', label='Test Data', s=3)
#plt.plot(x_test, y_test, color='orange')
plt.plot(x_test, y_forecast, color='red', label='Forecast')
# Retrieve current y-ticks
yticks = plt.gca().get_yticks()
y_step=yticks[1] - yticks[0]
print("Y-Ticks:", yticks)
print("Y-Step:", y_step)

xticks = plt.gca().get_xticks()
x_step=xticks[1] - xticks[0]
print("X-Ticks:", xticks)
print("X-Step:", x_step)

plt.ylim([0,max(monthly_data['SMA_13'])+y_step/5])
plt.xlim([min(monthly_data['time']-x_step/5.0),max(monthly_data['time'])+x_step/5.0])
plt.xlabel('Time')
plt.ylabel('SSA ($\mu Hemi$)')

plt.title('SSA monthly (Moving Average)  Trend and Forecast')
plt.legend()
#plt.grid(True)
plt.show()
print(monthly_data)
##############################################################


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Generate date range
dates = monthly_data['time']  # Monthly frequency
print(dates)


# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(monthly_data[['SMA_13']])

# Prepare data for LSTM
def create_sequences(data, time_steps):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(x), np.array(y)

time_steps = 1
length_split = 0.6
x, y = create_sequences(scaled_data, time_steps)

# Split data
x_train, y_train = x[:int(length_split * len(x))], y[:int(length_split * len(y))]
x_test, y_test = x[int(length_split * len(x)):], y[int(length_split * len(y)):]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)


# Predict and inverse transform predictions
y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)

# Create a corresponding date range for the test predictions
test_dates = dates[-len(y_test):]
print(test_dates)
# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(dates, monthly_data['SMA_13'], label='Historical Data', color='blue',s=4)
plt.scatter(test_dates, y_pred, label='Forecast', color='red',s=4)

# Retrieve current y-ticks

# Retrieve current y-ticks
yticks = plt.gca().get_yticks()
y_step=yticks[1] - yticks[0]
print("Y-Ticks:", yticks)
print("Y-Step:", y_step)

xticks = plt.gca().get_xticks()
x_step=xticks[1] - xticks[0]
print("X-Ticks:", xticks)
print("X-Step:", x_step)

plt.ylim([0,max(monthly_data['SMA_13'])+y_step/5])
plt.xlim([min(monthly_data['time']-x_step/5.0),max(monthly_data['time'])+x_step/5.0])
plt.xlabel('Time')
plt.ylabel('SSA ($\mu Hemi$)')

plt.title('LSTM Forecast for SSA')
plt.legend()
#plt.grid(True)
plt.show()




##########################################################
print(monthly_data)
# Plot results
#plt.figure(figsize=(12, 6))
plt.scatter(dates, monthly_data['SMA_13'], label='Historical Data', color='blue',s=10)
plt.scatter(test_dates, y_pred, label='Forecast', color='red',s=3)
# Adding a vertical line at x=5
#plt.axvline(x=2019+12/12, color='red', linestyle='--')
#plt.axvline(x=2008+12/12.0, color='red', linestyle='--')
#plt.axvline(x=1996+8/12.0, color='red', linestyle='--')
#plt.axvline(x=1986+9/12.0, color='red', linestyle='--')
#plt.axvline(x=1976+3/12.0, color='red', linestyle='--')
#plt.axvline(x=1964+10/12.0, color='red', linestyle='--')
#plt.axvline(x=1954+4/12.0, color='red', linestyle='--')
#plt.axvline(x=1944+2/12.0, color='red', linestyle='--')
#plt.axvline(x=1933+9/12.0, color='red', linestyle='--')
#plt.axvline(x=1923+8/12.0, color='red', linestyle='--')
#plt.axvline(x=1913+7/12.0, color='red', linestyle='--')
#plt.axvline(x=1902+1/12.0, color='red', linestyle='--')
#plt.axvline(x=1890+3/12.0, color='red', linestyle='--')
#plt.axvline(x=1878+12/12.0, color='red', linestyle='--')



# Data: Year and corresponding cycle numbers
years = [2008 + 12 / 12.0, 1996 + 8 / 12.0,1986 +
         9 / 12.0, 1976 + 3 / 12.0, 1964 +
         10 / 12.0,1954 + 4 / 12.0, 1944 +
         2 / 12.0, 1933 + 9 / 12.0,1923 +
         8 / 12.0, 1913 + 7 / 12.0, 1902 +
         1 / 12.0, 1890 + 3 / 12.0, 1878+12/12]
print(len(years))
cycles = range(24, 11, -1)  # Example cycle numbers: 24, 23, ..., 11
print(list(cycles))
# Plotting vertical lines with annotations
for year, cycle in zip(years, cycles):
    plt.axvline(x=year, color='red', linestyle='--')  # Vertical line
    plt.text(year, 2200, f' {cycle}', color='blue', rotation=0, fontsize=12, verticalalignment='bottom')  # Annotation


##plt.text(1880+3/12.0, 2200, f' {13}', color='blue', rotation=0, fontsize=12, verticalalignment='bottom')  # Annotation

plt.xlabel('Date')
plt.ylabel('SSA monthly Average')
plt.title('LSTM Forecast for SSA')
#plt.xlim([0,2020])
plt.ylim([0,2500])
plt.legend()
plt.grid(True)
plt.show()
#####################################################

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Ensure y_test is inverse-transformed
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))


#create datframe
df = pd.DataFrame({'Actual': y_test_inverse.flatten(), 'Predicted': y_pred.flatten()})
print(df)
# Calculate R² score
r2 = r2_score(y_test_inverse, y_pred)
print(f"R² Score: {r2}")


# Scatter plot for comparison
plt.figure(figsize=(8, 8))
plt.scatter(y_test_inverse, y_pred, color='blue', label='Predictions vs. True Data', s=10)
y_min, y_max = min(y_test_inverse), max(y_test_inverse)
plt.plot([y_min, y_max], [y_min, y_max], color='red', linestyle='--', label='Perfect Prediction Line')


# Retrieve current y-ticks
yticks = plt.gca().get_yticks()
y_step=yticks[1] - yticks[0]
print("Y-Ticks:", yticks)
print("Y-Step:", y_step)

xticks = plt.gca().get_xticks()
x_step=xticks[1] - xticks[0]
print("X-Ticks:", xticks)
print("X-Step:", x_step)

plt.ylim([min(y_pred),max(y_pred)+y_step/5])
plt.xlim([min(y_test_inverse),max(y_test_inverse)+x_step/5.0])
plt.xlabel('Time')
plt.ylabel('SSA ($\mu Hemi$)')


plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs True Values (LSTM Model)')
plt.legend()
#plt.grid(True)
plt.show()
##########################################################


# Define a range of epoch values to test
epoch_values = [10, 20, 50, 100, 200]
mse_values = []
r2_values = []

for epochs in epoch_values:
    # Build a new model for each epoch configuration
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model with the current epoch value
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    # Predict on test data
    y_pred = model.predict(x_test)

    # Inverse scale predictions and true values for comparison
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inverse = scaler.inverse_transform(y_pred)

    # Calculate MSE and R² score
    mse = mean_squared_error(y_test_inverse, y_pred_inverse)
    r2 = r2_score(y_test_inverse, y_pred_inverse)
    mse_values.append(mse)
    r2_values.append(r2)
    print(f"Epochs: {epochs}, MSE: {mse:.4f}, R²: {r2:.4f}")

# Plot MSE and R² vs. Epochs
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot MSE
ax1.set_xlabel('Number of Epochs')
ax1.set_ylabel('Mean Squared Error (MSE)', color='blue')
ax1.plot(epoch_values, mse_values, marker='o', linestyle='-', color='blue', label='MSE')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis for R²
ax2 = ax1.twinx()
ax2.set_ylabel('R² Score', color='green')
ax2.plot(epoch_values, r2_values, marker='o', linestyle='--', color='green', label='R²')
ax2.tick_params(axis='y', labelcolor='green')

plt.title('Model Performance: MSE and R² vs. Training Epochs')
fig.tight_layout()
plt.show()

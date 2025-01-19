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

###################################################3
# Split the data sequentially (60% training, 40% testing)

print("length of data")
print(len(monthly_data))
data_size=0.6
train_size = int(data_size * len(monthly_data))
x_train = monthly_data[['time']].values[:train_size]
y_train = monthly_data['SSA'].values[:train_size]
x_test = monthly_data[['time']].values[train_size:]
y_test = monthly_data['SSA'].values[train_size:]

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

plt.ylim([0,max(monthly_data['SSA'])+y_step/5])
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


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import Adam, RMSprop
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# Load and preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(monthly_data[['SSA']])

def create_sequences(data, time_steps):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(x), np.array(y)

# Define configurations to test
configurations = [
    {"time_steps": 12, "neurons": 50, "dropout": 0.2, "optimizer": Adam(learning_rate=0.001)},
    {"time_steps": 24, "neurons": 100, "dropout": 0.3, "optimizer": RMSprop(learning_rate=0.001)},
    {"time_steps": 36, "neurons": 150, "dropout": 0.4, "optimizer": Adam(learning_rate=0.0005)}
]

results = {}

for idx, config in enumerate(configurations):
    print(f"\nTesting Configuration {idx + 1}: {config}")

    time_steps = config['time_steps']
    x, y = create_sequences(scaled_data, time_steps)

    # Split data
    split_ratio = 0.5
    train_size = int(split_ratio * len(x))
    x_train, y_train = x[:train_size], y[:train_size]
    x_test, y_test = x[train_size:], y[train_size:]

    # Build model
    model = Sequential([
        LSTM(config['neurons'], return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        Dropout(config['dropout']),
        Bidirectional(LSTM(config['neurons'])),
        Dense(1)
    ])
    model.compile(optimizer=config['optimizer'], loss='mean_squared_error')

    # Train model
    model.fit(x_train, y_train, epochs=50, batch_size=64, verbose=1)

    # Make predictions
    y_pred = model.predict(x_test)
    y_pred_inverse = scaler.inverse_transform(y_pred)
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Store results
    results[idx] = {
        "predictions": y_pred_inverse,
        "true_values": y_test_inverse,
        "dates": dates[-len(y_test):]
    }

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.scatter(dates, monthly_data['SSA'], label='Historical Data', color='blue', s=10)
    plt.scatter(results[idx]["dates"], results[idx]["predictions"], label='Forecast', color='red', s=10)
    plt.title(f'LSTM Forecast Configuration {idx + 1}')
    plt.xlabel('Time')
    plt.ylabel('SSA ($\mu Hemi$)')
    plt.legend()
    plt.show()

    # Calculate metrics
    mse = mean_squared_error(results[idx]["true_values"], results[idx]["predictions"])
    r2 = r2_score(results[idx]["true_values"], results[idx]["predictions"])

    # Store metrics
    results[idx]["mse"] = mse
    results[idx]["r2"] = r2

    print(f"Configuration {idx + 1} MSE: {mse}")
    print(f"Configuration {idx + 1} R²: {r2}")

    # Plot predicted vs true values with Linear Regression
    plt.figure(figsize=(12, 6))
    plt.scatter(results[idx]["true_values"], results[idx]["predictions"], color='blue', label='Predicted vs True')

    # Fit linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(results[idx]["true_values"], results[idx]["predictions"])
    y_lin_reg = lin_reg.predict(results[idx]["true_values"])

    # Plot linear regression line
    plt.plot(results[idx]["true_values"], y_lin_reg, color='red', linewidth=2, label='Linear Regression Line')

    plt.title(f'Predicted vs True Values with Linear Regression (Config {idx + 1})')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.xlim([0, 4100])
    plt.ylim([0, 2600])
    plt.legend()
    plt.show()

# Compare all configurations based on the metrics
print("\nComparison of configurations based on MSE, and R²:")

for idx in results:
    print(f"\nConfiguration {idx + 1}:")
    print(f"  MSE: {results[idx]['mse']}")
    print(f"  R²: {results[idx]['r2']}")

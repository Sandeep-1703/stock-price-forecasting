import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


import yfinance as yf
import datetime


# List of popular stock ticker symbols for users to choose from
stock_list = {
    "Apple Inc.": "AAPL",
    "Microsoft Corporation": "MSFT",
    "Amazon.com, Inc.": "AMZN",
    "Alphabet Inc. (Google)": "GOOGL",
    "Meta Platforms, Inc. (Facebook)": "META",
    "NVIDIA Corporation": "NVDA",
    "Tesla, Inc.": "TSLA",
    "Netflix, Inc.": "NFLX",
    "Intel Corporation": "INTC",
    "Adobe Inc.": "ADBE",
    "JPMorgan Chase & Co.": "JPM",
    "Bank of America Corporation": "BAC",
    "Coca-Cola Company": "KO",
    "PepsiCo, Inc.": "PEP",
    "Nike, Inc.": "NKE",
    "Walmart Inc.": "WMT",
    "The Home Depot, Inc.": "HD",
    "Exxon Mobil Corporation": "XOM",
    "Chevron Corporation": "CVX",
    "Berkshire Hathaway Inc.": "BRK.B",
    "McDonald's Corporation": "MCD",
    "Adani Power": "ADANIPOWER.NS",
    "Tata Motors": "TATAMOTORS.NS"
}

# Display the list of companies to the user and allow them to pick a stock
print("Select a stock by entering the corresponding number:")
for idx, company in enumerate(stock_list.keys(), start=1):
    print(f"{idx}. {company}")

# Ask the user for their choice
choice = int(input("\nEnter the corresponding stock number you'd like to analyze: "))

# Validate user input
if 1 <= choice <= len(stock_list):
    selected_stock = list(stock_list.values())[choice - 1]
    print(f"\nYou selected: {list(stock_list.keys())[choice - 1]} (Ticker: {selected_stock})")
else:
    print("Invalid selection! Please restart and choose a valid stock number.")
    exit()

start_date = '2010-01-01'
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

# Fetch data using yfinance for the selected stock
tickerData = yf.Ticker(selected_stock)
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)

# If you want to convert it into a DataFrame explicitly (though it's already a DataFrame)
df = pd.DataFrame(tickerDf)

# Use 'Close' column for prediction
data = df.filter(['Close'])

# 2. Scale the data to the range [0, 1] using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# 3. Function to create sequences (time steps)
def create_sequences(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]  # Select the sequence of `time_step` size
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])  # The value to predict is the next time step
    return np.array(dataX), np.array(dataY)

# 4. Create training sequences
time_step = 100  # We'll look back at the last 100 days for predictions
X, y = create_sequences(scaled_data, time_step)

# 5. Reshape X for LSTM [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# 6. Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# 7. Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, batch_size=16, epochs=15, verbose=1)

# 8. Prepare data for prediction: take the last `time_step` days as input
last_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)


# 9. Predict the next 5 days
predicted_values = []
for _ in range(future_days):
    prediction = model.predict(last_sequence)  # Predict the next day
    predicted_values.append(prediction[0, 0])  # Store the prediction
    # Update the input for the next prediction
    last_sequence = np.append(last_sequence[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

# 10. Convert the predictions back to the original scale (inverse transform)
predicted_values_actual = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))

# 11. Convert the past data back to the original scale (inverse transform)
past_data_actual = scaler.inverse_transform(scaled_data[-time_step:].reshape(-1, 1))

# 12. Plot the results
# Combine the past data with the predicted values
all_data_actual = np.concatenate((past_data_actual, predicted_values_actual))

# X-axis for the past 100 days and the next 5 predicted days
x_values = np.arange(time_step + future_days)

# Plot the past data and the predicted future values
plt.figure(figsize=(10, 6))
plt.plot(x_values[:time_step], past_data_actual, label="Past Data", color='blue')
plt.plot(x_values[time_step:], predicted_values_actual, label=f"Predicted Next {future_days} Days", color='red')

# Customize plot
plt.title(f"Past Data and Predicted Next {future_days} Days for {list(stock_list.keys())[choice - 1]}")
plt.xlabel('Time (Days)')
plt.ylabel('Value')
# Reduce number of ticks by showing every 10th tick
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=7))  # Automatically adjust the number of ticks

plt.legend()
plt.grid(True)
plt.show()





# Function to ask the user for the number of future days
def get_future_days():
    while True:
        try:
            # Asking user to input the number of future days for prediction
            future_days = int(input("Enter the number of future days for which you want a prediction: "))
            
            # Ensure the input is a positive number
            if future_days > 0:
                return future_days
            else:
                print("Please enter a positive integer greater than 0.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Now, let's call the function to get the input
future_days = get_future_days()

print(f"You've requested predictions for {future_days} future days.")

# 8. Prepare data for prediction: take the last `time_step` days as input
last_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)

# 9. Predict the next 5 days
predicted_values = []
for _ in range(future_days):
    prediction = model.predict(last_sequence)  # Predict the next day
    predicted_values.append(prediction[0, 0])  # Store the prediction
    # Update the input for the next prediction
    last_sequence = np.append(last_sequence[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

# 10. Convert the predictions back to the original scale (inverse transform)
predicted_values_actual = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))

# 11. Convert the past data back to the original scale (inverse transform)
past_data_actual = scaler.inverse_transform(scaled_data[-time_step:].reshape(-1, 1))

# 12. Plot the results
# Combine the past data with the predicted values
all_data_actual = np.concatenate((past_data_actual, predicted_values_actual))

# X-axis for the past 100 days and the next 5 predicted days
x_values = np.arange(time_step + future_days)

# Plot the past data and the predicted future values
plt.figure(figsize=(10, 6))
plt.plot(x_values[:time_step], past_data_actual, label="Past Data", color='blue')
plt.plot(x_values[time_step:], predicted_values_actual, label=f"Predicted Next {future_days} Days", color='red')

# Customize plot
plt.title(f"Past Data and Predicted Next {future_days} Days for {list(stock_list.keys())[choice - 1]}")
plt.xlabel('Time (Days)')
plt.ylabel('Value')
# Reduce number of ticks by showing every 10th tick
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=7))  # Automatically adjust the number of ticks

plt.legend()
plt.grid(True)
plt.show()

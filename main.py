# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from datetime import datetime

# Fetch stock data
start_date = "2015-01-01"
end_date = "2023-01-01"
stock_symbol = "AAPL"

df = DataReader(stock_symbol, 'yahoo', start_date, end_date)
df = df[['Close']]  # Focus on 'Close' price
df.reset_index(inplace=True)
df.set_index('Date', inplace=True)

# Plot the stock closing prices
plt.figure(figsize=(10, 6))
plt.plot(df['Close'], label=f"{stock_symbol} Closing Prices")
plt.title(f"{stock_symbol} Stock Prices")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Split into training and testing datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences for LSTM
def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(x), np.array(y)

sequence_length = 60  # 60 days of historical data to predict the next day
x_train, y_train = create_sequences(train_data, sequence_length)
x_test, y_test = create_sequences(test_data, sequence_length)

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train the LSTM model
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Reverse scale the true prices for comparison
true_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Visualize the predictions vs actual prices
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(true_prices):], true_prices, label='True Prices')
plt.plot(df.index[-len(predicted_prices):], predicted_prices, label='LSTM Predictions', linestyle='dashed')
plt.title(f"{stock_symbol} Stock Price Prediction")
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Forecast future prices
last_sequence = scaled_data[-sequence_length:]
last_sequence = last_sequence.reshape(1, sequence_length, 1)

future_steps = 30  # Forecast for 30 days
forecast = []

for _ in range(future_steps):
    prediction = model.predict(last_sequence)
    forecast.append(prediction[0, 0])
    last_sequence = np.append(last_sequence[:, 1:, :], [[prediction]], axis=1)

# Scale back the forecast to original prices
forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
forecast_dates = pd.date_range(df.index[-1], periods=future_steps + 1, freq='B')[1:]

# Plot the forecasted future prices
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Historical Prices')
plt.plot(forecast_dates, forecast, label='Forecasted Prices', color='red')
plt.title(f"{stock_symbol} 30-Day Stock Price Forecast")
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Save the LSTM model
model.save('lstm_stock_model.h5')

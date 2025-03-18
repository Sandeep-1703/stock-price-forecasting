# Stock Price Prediction Using LSTM

## Overview
This project uses an LSTM (Long Short-Term Memory) model to predict stock prices based on historical data. The script allows users to select a stock, fetches its historical price data using Yahoo Finance, and trains an LSTM model to predict future stock prices.

## Features
- Allows users to select a stock from a predefined list.
- Fetches historical stock price data using `yfinance`.
- Normalizes data using `MinMaxScaler`.
- Creates time-series sequences for LSTM input.
- Builds and trains an LSTM-based deep learning model.
- Predicts future stock prices for a user-defined number of days.
- Visualizes past stock prices along with predicted prices.

## Requirements
Ensure you have the following Python libraries installed:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow yfinance
```

## Usage
1. Run the script:

   ```bash
   python stock_prediction.py
   ```

2. Select a stock by entering the corresponding number from the displayed list.
3. Enter the number of future days for which you want predictions.
4. The script will fetch the stock data, train the model, and plot past stock prices along with predicted values.

## File Breakdown
- `yfinance`: Fetches historical stock data.
- `MinMaxScaler`: Normalizes stock prices to a range between 0 and 1.
- `LSTM Model`: Trained to predict future prices based on historical data.
- `Matplotlib`: Used for visualizing stock price trends.

## How It Works
1. **Fetch Data**: The script collects stock prices starting from 2010.
2. **Preprocessing**: Data is scaled and split into sequences.
3. **Model Training**: The LSTM model is trained on historical data.
4. **Prediction**: The trained model predicts future stock prices.
5. **Visualization**: The script plots actual stock prices along with predicted values.

## Example Output
After running the script, you will see a plot showing:
- The past 100 days of stock price data (in blue).
- The predicted stock prices for the next `n` days (in red).

## Notes
- The model is trained using a simple architecture and may require hyperparameter tuning for improved accuracy.
- Data is fetched dynamically from Yahoo Finance, so availability may vary.

## License
This project is for educational purposes and does not constitute financial advice. Use at your own risk.


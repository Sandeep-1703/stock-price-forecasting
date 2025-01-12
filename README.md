# Stock Price Forecasting using LSTM

This project demonstrates how to use a Long Short-Term Memory (LSTM) model for stock price forecasting. It involves preprocessing stock data, training the LSTM model, and making predictions for both historical test data and future stock prices.

---

## Features
- Fetches stock price data from Yahoo Finance.
- Uses LSTM to predict stock prices based on historical data.
- Visualizes the predictions versus actual prices.
- Forecasts future stock prices for the next 30 days.

---

## Requirements

Before running the code, ensure the following Python libraries are installed:

```bash
pip install numpy pandas matplotlib scikit-learn keras pandas-datareader
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-price-forecasting-lstm.git
   cd stock-price-forecasting-lstm
   ```

2. Open the `lstm_stock_forecasting.py` script.

3. Set your parameters:
   - **Stock Symbol**: Replace `AAPL` with your desired stock symbol.
   - **Start and End Dates**: Modify `start_date` and `end_date` to your desired range.

4. Run the script:
   ```bash
   python lstm_stock_forecasting.py
   ```

5. Outputs:
   - Historical stock prices vs. predictions.
   - Forecasted stock prices for the next 30 days.

---

## Model Architecture

- **LSTM Layers**: Captures temporal dependencies in stock prices.
- **Dropout Layers**: Reduces overfitting during training.
- **Dense Layers**: Outputs the final predicted price.

### Model Summary
- Input Shape: `(60, 1)` (60 days of historical data per sample)
- Hidden Layers:
  - LSTM (50 units) with return sequences.
  - Dropout (20%).
  - LSTM (50 units) without return sequences.
  - Dropout (20%).
- Output Layer: Dense (1 unit).

---

## Data Flow

1. **Data Collection**:
   - Stock price data is fetched using `pandas-datareader` from Yahoo Finance.

2. **Data Preprocessing**:
   - Normalize prices using `MinMaxScaler`.
   - Create sequences of 60 days for training/testing.

3. **Model Training**:
   - Train on 80% of the data.
   - Validate on 20% of the data.

4. **Prediction and Forecasting**:
   - Predict prices for test data.
   - Forecast prices for the next 30 days using the trained model.

---

## Visualizations

1. **Historical Prices**:
   - Plot of historical stock prices.

2. **Predicted vs Actual Prices**:
   - Comparison of predicted and actual prices for the test set.

3. **Forecasted Prices**:
   - Predicted prices for the next 30 days.

---

## Example Plots

1. **Historical Prices**:
   ![Historical Prices](assets/historical_prices.png)

2. **Predicted vs Actual**:
   ![Predictions vs Actual](assets/predictions_vs_actual.png)

3. **Forecasted Prices**:
   ![Forecasted Prices](assets/forecasted_prices.png)

---

## Future Improvements

1. **Include Additional Features**:
   - Add technical indicators or volume data.

2. **Optimize Hyperparameters**:
   - Use grid search or Bayesian optimization for tuning.

3. **Use Attention Mechanisms**:
   - Enhance LSTM performance by focusing on relevant time steps.

---

## References
- [Keras Documentation](https://keras.io/)
- [Yahoo Finance API](https://finance.yahoo.com/)
- [LSTM in Deep Learning](https://en.wikipedia.org/wiki/Long_short-term_memory)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Feel free to fork and improve this project! Contributions are welcome.


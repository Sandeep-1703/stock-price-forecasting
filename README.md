Stock Price Prediction using LSTM

Overview

This project implements a Long Short-Term Memory (LSTM) model to predict stock prices using historical stock market data. It fetches stock prices from Yahoo Finance and allows users to select a stock from a predefined list. The model then predicts future stock prices based on past trends.

Features

Fetches stock price data from Yahoo Finance.

Allows users to select a stock from a predefined list.

Uses MinMaxScaler to normalize the data for better LSTM performance.

Creates sequences of past stock prices as input for prediction.

Implements an LSTM model to predict future stock prices.

Plots past stock data along with predicted future prices.

Requirements

To run this project, you need the following Python packages:

pip install numpy pandas matplotlib scikit-learn tensorflow yfinance

How to Run

Run the script.

Select a stock from the displayed list by entering the corresponding number.

Enter the number of future days for which you want a prediction.

The model will train using historical stock data.

The predicted stock prices for the next few days will be displayed and plotted.

Model Details

The LSTM model consists of:

Two LSTM layers with 50 units each.

A Dense output layer with one neuron.

Mean Squared Error (MSE) loss function.

Adam optimizer.

The model is trained for 15 epochs with a batch size of 16.

Data Preprocessing

Data is fetched using yfinance.

Only the closing price is used for training.

The data is normalized using MinMaxScaler to fit in the range [0,1].

Time-step sequences of 100 days are created for training.

Prediction & Visualization

The model predicts future stock prices based on the last 100 days of data.

The results are plotted using Matplotlib.

The blue curve represents past stock prices, while the red curve shows the predicted values for future days.

Notes

The model does not guarantee accurate predictions since stock prices depend on various factors beyond historical trends.

It is recommended to use this project for learning purposes rather than financial decisions.

Future Improvements

Enhance the model with additional features like volume, moving averages, and technical indicators.

Implement different neural network architectures such as GRU or Transformer models.

Fine-tune hyperparameters for better performance.

License

This project is for educational purposes only. Use at your own risk.


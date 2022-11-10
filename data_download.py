import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
from sklearn.metrics import mean_squared_error as MSE
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpdates
from statistics import mean
import pandas as pd
import numpy as np
import yfinance as yf

#1: Defining the function to retrieve date and stock inputs from the keyboard

stock_ticker = input ("Please enter the stock ticker: ")
start_d = input ("Please enter the start date (YYYY-MM-DD): ")
end_d = input ("Please enter the end date (YYYY-MM-DD): ")

def stocks (stock, start_date, end_date):
    ticker = stock
    tickerdata = yf.Ticker (ticker)
    tickerdf = tickerdata.history (period = '1d', start = start_date, end = end_date)
    return tickerdf
stock2 = stocks(stock_ticker, start_d, end_d)
stock2.reset_index (level = 'Date')
#stock3 = stock2.drop(columns = ['Dividends', 'Stock Splits', 'Open', 'Volume'])
print (stock2.head(10))

stock2.to_csv('amzn1.csv')


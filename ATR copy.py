'''
Average True Range:
- measures market volatility by decomposing the entire price range of an asset price for that period

Steps to calculate:
- True range indicator: the greatest out of:
> current high  - current low
> |current high - previous close|
> |current low - previous close|
- ATR: Moving avg. of the true ranges (Usually simple moving average)

- A stock with a higher volatility has a higher ATR and vice versa

'''
# Importing the necessary libraries
from sre_constants import CH_LOCALE
from tracemalloc import start
from cProfile import label
import numpy as np
import pandas as pd
from statistics import mean
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from zmq import MAXMSGSIZE

'''
# Prompting the user to enter the date range and stock ticker of interest
stock_ticker = input ("Please enter the stock ticker: ")
start_d = input ("Please enter the start date (YYYY-MM-DD): ")
end_d = input ("Please enter the end date (YYYY-MM-DD): ")

# Defining the stock data retrieval function
def stock_data (stock, start_date, end_date):
    # The ticker
    ticker = stock
    tickerdata = yf.Ticker (ticker)
    tickerdf = tickerdata.history(period = '1d', start = start_date, end = end_date)
    return tickerdf
stock2 = stock_data (stock_ticker, start_d, end_d)
stock2.reset_index (level = 'Date')
stock3 = stock2.drop(columns = ['Dividends', 'Stock Splits', 'Volume'])
#print (stock3.head(10))
'''
stock2 = pd.read_csv("amzn1.csv")
def ATR(data):
    high_low = data['High']- data ['Low']
    #data['High - Low'] = high_low
    high_close = np.abs(data['High'] - data['Close'].shift())
    #data['|High - Previous Close|'] = high_close
    low_close = np.abs(data['Low'] - data ['Close'].shift())
    #data ["|Low - Previous Close|"] = low_close
    true_range = np.amax(np.vstack(((high_low).to_numpy(), high_close.to_numpy(), low_close.to_numpy())).T, axis = 1)
    # The Line above converts the object types to numpy and then uses the axis max (amax) feature of numpy.
    #data['True Range'] = true_range
    atr = pd.Series(true_range).rolling(14).mean().to_numpy()
    data["ATR"] = atr


ATR(stock2)
#stock4 = stock2.dropna()
print(stock2.head(10))

# Plotting RSI and the graph of closing stock prices in one graph

figure, axis = plt.subplots(2)
figure.suptitle('Stock Close Price and ATR Chart')
axis[0].plot(stock2['Close'])
axis[0].set_title('Closing Price Chart')

axis[1].plot(stock2['ATR'])
axis[1].set_title('ATR Chart')

figure.tight_layout(pad = 1.0)
plt.show()

#stock3 = stock2.drop (columns = 'True Range')
#print (stock3.head(10))

stock2.to_csv("amzn2.csv", index = False)



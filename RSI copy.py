# Relative Strength Index Programe: Using the data on stock prices from Yahoo Finance to calculate the RSI
# RSI: A technical oscillator that uses clsong price data for identifying overbought and oversold signals
# Also used to spot divergences warning of a trend change in price

# Formula:
# RSI_stepone = 100 - [100/{1+(avg. gain / avg. loss)}]
# Typically we use the average gain and average loss over 14 periods

#1: Importing the necessary libraries
from cProfile import label
from statistics import mean
from tracemalloc import start
from matplotlib.lines import lineStyles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
import yfinance as yf
from statistics import mean
from datetime import datetime
from datetime import timedelta
'''
#2: Defining the data retrieval function
stock_ticker = input("Please enter the stock ticker to generate Closing price and RSI graphs: ")
start_d = input("Please enter the start date (YYYY-MM-DD): ")
end_d = input ("Please enter the end date (YYYY-MM-DD): ")

def stocks (stock, start_date, end_date):
    # The ticker
    ticker = stock
    tickerdata = yf.Ticker (ticker)
    tickerdf = tickerdata.history(period = '1d', start = start_date, end = end_date)
    return tickerdf
stock2 = stocks (stock_ticker, start_d, end_d)
stock2.reset_index (level = 'Date')
stock3 = stock2.drop(columns = ['Dividends', 'Stock Splits', 'Open', 'High', 'Low', 'Volume'])
print (stock3.head(10))
'''

stock3 = pd.read_csv("amzn3.csv")
# stock3 is the main data source we will use for the RSI Calculation
def RSI_fn(data, periods, rolling = True):
    close_delta = data['Close'].diff() # The difference between the price on the ith day and the i-1 th day
    up = close_delta.clip (lower = 0)  # the .clip limits the values stored to those that are greater than 0
    dn = - 1 * close_delta.clip (upper = 0) # the upper = 0 conditions limits the stroed values to those that are lower than 0
    # The down values are multiplied by -1 so that the program stotes them as positives
    ma_up = up.rolling(window = periods-1).mean()
    ma_dn = dn.rolling(window = periods-1).mean()
    rsi = ma_up / ma_dn
    rsi = 100 - (100 / (1+rsi))
    new_df1 = pd.DataFrame()
    new_df1 = stock3 [periods:]
    new_df1['RSI'] = rsi
    return rsi, new_df1
Relative_Strength_Index, new_df1 = RSI_fn(data = stock3, periods = 14)
print (new_df1.head(5))

stock3['MFI'] = np.nan
print (stock3.head(5))
stock4 = stock3.iloc[:14:,:]
stock5 = pd.concat([stock4, new_df1])
#stock6 = stock5.drop (columns=['Unnamed: 0', 'Dividends'])
print (stock5.head(16))
stock5.to_csv("amzn3.csv", index = False)



'''
# Plotting RSI and the graph of closing stock prices in one graph

figure, axis = plt.subplots(2)
figure.suptitle('Stock Close Price and RSI Chart')
axis[0].plot(new_df1['Close'])
axis[0].set_title('Closing Price Chart')

axis[1].plot(new_df1['RSI'])
axis[1].set_title('RSI Chart')

os_one = axis[1].axhline(0, linestyle = '--', color = 'green', label = "Oversold 1")
os_two = axis[1].axhline (20, linestyle = '--', color = 'blue', label = "Oversold 2")
os_three = axis[1].axhline(30, linestyle = '--', label = "Oversold 3")

# Significant level: Overbought
ob_one = axis[1].axhline (70, linestyle = '--', label = "Overbought 1")
ob_two = axis[1].axhline(80, linestyle = '--', color = 'red', label = "Overbought 2")
ob_three = axis[1].axhline(100, linestyle = '--', color = 'yellow', label = "Overbought 3")
legend = axis[1].legend(handles = [ob_one, ob_two, ob_three, os_one, os_two, os_three], loc = "center left",
                        bbox_to_anchor = (1,0.5) ,ncol = 1, fontsize = "small")
figure.tight_layout(pad = 1.0)
plt.show()

'''
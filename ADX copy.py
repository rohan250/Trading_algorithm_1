from statistics import mean
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
from sklearn.metrics import mean_squared_error as MSE
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpdates

#1: Defining the function to retrieve date and stock inputs from the keyboard
'''
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
stock3 = stock2.drop(columns = ['Dividends', 'Stock Splits', 'Open', 'Volume'])
print (stock3.head(10))
'''
stock2 = pd.read_csv("amzn1.csv")

#2: Writing the ADX Function

def ADX(data, periods):
    #A: Calculating +DM, -DM, and true range for each period:
    # +DM:
    PDM = data['High'] - data['High'].shift()
    PDM[PDM < 0] = 0
    #data['+DM'] = PDM
    # -DM:
    NDM = data['Low'] - data ['Low'].shift()
    NDM[NDM > 0] = 0
    #data['-DM'] = NDM
    # True Range:
    high_low = data['High']- data ['Low']
    #data['High - Low'] = high_low # can be used to add the data to the dataframe and view it for checking
    high_close = np.abs(data['High'] - data['Close'].shift())
    #data['|High - Previous Close|'] = high_close
    low_close = np.abs(data['Low'] - data ['Close'].shift())
    #data ["|Low - Previous Close|"] = low_close
    true_range = np.amax(np.vstack(((high_low).to_numpy(), high_close.to_numpy(), low_close.to_numpy())).T, axis = 1)
    #data['TR'] = true_range
    # Using the exponential moving average ATR rather than the custom Wiles formula
    atr = pd.Series(true_range).ewm(com = periods - 1, adjust = True, min_periods = periods).mean().to_numpy()
    #data['ATR'] = atr
    # + DI :
    PDM_EMA = pd.Series(PDM).ewm(com = periods - 1).mean().to_numpy()
    PDI = 100 * (PDM_EMA / atr)
    data ['+DI'] = PDI
    # -DI:
    NDM_EMA = np.abs(pd.Series(NDM).ewm(com = periods - 1).mean().to_numpy())
    NDI = 100 * (NDM_EMA / atr)
    data ['-DI'] = NDI
    DX = 100 * (np.abs(PDI - NDI)/np.abs(PDI + NDI))
    #data['DX'] = DX
    ADX = ((pd.Series(DX).shift(1)*(periods -1))+pd.Series(DX))/periods
    adx_smooth = pd.Series(ADX).ewm (com = periods - 1).mean().to_numpy()
    #data ['ADX'] = ADX
    data ['ADX smooth'] = adx_smooth



ADX(stock2, periods = 14)
#print (stock2.head(40))
# Plotting RSI and the graph of closing stock prices in one graph

#Simple Code without candlestick
figure, axis = plt.subplots(2)
cl_p = axis[0].plot(stock2 ['Close'], color = 'blue', linewidth = 1, label = "Closing price")
axis[0].set_xlabel('Date')
axis[0].set_ylabel('Closing stock price')
#axis.plot(stock3['+DI'])
#axis.plot(stock3['-DI'])

axis2 = axis[0].twinx()
adx_line  = axis2.plot(stock2['ADX smooth'], color = 'grey', linewidth = 0.7, alpha = 0.8, label = "ADX Smooth")
PDI_line = axis[1].plot(stock2['+DI'], color = 'green', linewidth = 0.7, alpha = 0.7, label = "+DI line")
NDI_line = axis[1].plot(stock2['-DI'], color = 'red', linewidth = 0.7, alpha = 0.7, label = "-DI line")
axis2.set_ylabel('ADX')
axis[1].set_ylabel('+DI, -DI')
axis[1].set_xlabel ('Date')
axis2.legend(loc = 0)
axis[1].legend (loc = 0)
plt.show()
figure.tight_layout (pad = 0.5)

stock3 = stock2.drop (columns = ['Stock Splits', '+DI', '-DI'])
print (stock3.head(5))

stock3.to_csv("amzn2.csv", index = False)


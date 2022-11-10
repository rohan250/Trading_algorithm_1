# Money Flow Index Programe: Using the data on stock prices from Yahoo Finance to calculate the MFI
# MFI: A technical oscillator that uses price and volume data for identifying overbought and oversold signals
# Also used to spot divergences warning of a trend change in price

# Formula:
# MFI = 100 - [100/ (1 + MFR)]
# MFR = [(14 period +ve money flow)/ (14 period -ve money flow )]
# Raw Money flow = Typical price * volume
# Typical price = [(High + Low + Close) / 3]

#1: Importing the necessary libraries
from cProfile import label
from tracemalloc import start
from matplotlib.lines import lineStyles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
import yfinance as yf
'''
#2: Defining the data retrieval function
stock_ticker = input("Please enter the stock ticker to generate Closing price and MFI graphs: ")

def stocks (stock, start_date, end_date):
    # The ticker
    ticker = stock
    tickerdata = yf.Ticker (ticker)
    tickerdf = tickerdata.history(period = '1d', start = start_date, end = end_date)
    return tickerdf
stock2 = stocks (stock_ticker, "2017-10-17", "2022-10-17")
stock2.drop (columns = ["Dividends", "Stock Splits"], inplace = True)
print (stock2.head())
'''
stock2 = pd.read_csv("amzn2.csv")
stock2['MFI'] = np.nan
print (stock2.head(5))


def MFI(data, period):
    typ_pr = (data['Close'] + data['Low'] + data ['High']) /3
    money_flow = typ_pr * stock2 ["Volume"]
    pos_fl = []
    neg_fl = []

    for i in range (1, len(typ_pr)):
        if typ_pr[i]> typ_pr[i-1]:
            pos_fl.append (typ_pr[i-1])
            neg_fl.append(0)
        elif typ_pr[i-1]>typ_pr[i]:
            pos_fl.append(0)
            neg_fl.append(typ_pr[i-1])
        else:
            pos_fl.append(0)
            neg_fl.append(0)
    pos_mf = []
    neg_mf = []
    for i in range (period-1, len(pos_fl)):
        pos_mf.append(sum(pos_fl[i+1-period: i +1]))
    for i in range (period-1, len(neg_fl)):
        neg_mf.append(sum(neg_fl[i+1-period : i + 1]))
    MFI = 100 * (np.array(pos_mf) / (np.array (pos_mf) + np.array(neg_mf)))
    new_df = pd.DataFrame()
    new_df = stock2[period:]
    new_df ['MFI'] = MFI
    return MFI, new_df
Money_Flow_Index, new_df = MFI (stock2, 14)
print (new_df.head(5))


#df1 = new_df.drop(columns = ['Open', 'High', 'Low', 'Volume'])
print (new_df.head(5))
# Plotting RSi and the candlestick graph of stock prices in one graph


figure, axis = plt.subplots(2)
figure.suptitle('Stock Close Price and MFI Chart')
axis[0].plot(new_df['Close'])
axis[0].set_title('Closing Price Chart')

axis[1].plot(new_df['MFI'])
axis[1].set_title('MFI Chart')

os_one = axis[1].axhline(0, linestyle = '--', color = 'green', label = "Oversold line 1")
os_two = axis[1].axhline (20, linestyle = '--', color = 'blue', label = "Oversold line 2")
os_three = axis[1].axhline(30, linestyle = '--', label = "Oversold line 3")

# Significant level: Overbought
ob_one = axis[1].axhline (70, linestyle = '--', label = "Overbought line 1")
ob_two = axis[1].axhline(80, linestyle = '--', color = 'red', label = "Overbought line 2")
ob_three = axis[1].axhline(100, linestyle = '--', color = 'yellow', label = "Overbought line 3")
legend = axis[1].legend(handles = [ob_one, ob_two, ob_three, os_one, os_two, os_three], loc = "center left",
                        bbox_to_anchor = (1,0.5) ,ncol = 1)
figure.tight_layout(pad = 1.0)
plt.show()

stock2['MFI'] = np.nan
print (stock2.head(5))
stock3 = stock2.iloc[:14:,:]
stock4 = pd.concat([stock3, new_df])
stock5 = stock4.drop (columns=['Unnamed: 0', 'Dividends'])
print (stock5.head(16))
stock5.to_csv("amzn3.csv", index = False)

'''

#A: Plotting the closing price
plt.figure(figsize = (10,4))
plt.title ('MFI and stock price close chart')
plt.plot(df1['Close'])
plt.title("Closing Price chart")

#Plotting corresponding MFI values and significance levels
plt.figure(figsize = (10,4))
plt.title("MFI Chart")
plt.plot(df1['MFI'])

# Significant level: Oversold
os_one = plt.axhline(0, linestyle = '--', color = 'green', label = "Oversold line 1")
os_two = plt.axhline (20, linestyle = '--', color = 'blue', label = "Oversold line 2")
os_three = plt.axhline(30, linestyle = '--', label = "Oversold line 3")

# Significant level: Overbought
ob_one = plt.axhline (70, linestyle = '--', label = "Overbought line 1")
ob_two = plt.axhline(80, linestyle = '--', color = 'red', label = "Overbought line 2")
ob_three = plt.axhline(100, linestyle = '--', color = 'yellow', label = "Overbought line 3")
legend = plt.legend(handles = [ob_one, ob_two, ob_three, os_one, os_two, os_three], loc = "upper center", ncol = 3)
plt.show()

'''
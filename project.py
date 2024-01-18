import matplotlib.pyplot as plt
import csv
from math import sqrt

class StockData:
    def __init__(self, date, open_price, high, low, close, volume):
        self.__date = date  # private attribute
        self.open = open_price  # public attribute
        self.high = high  # public attribute
        self.low = low
        self.close = close  # public attribute
        self.volume = volume  # public attribute
        
    def __repr__(self):
        return f"StockData({self.__date}, {self.open}, {self.high}, ...)"
        
    def __calculate_daily_change(self):  # private method
        return self.close - self.open
    
    def get_daily_change(self):  # public method
        return self.__calculate_daily_change()
    
    def get_date(self):
        return self.__date
    

# Adding functions for ATR and Bollinger Bands to the provided code

# Function to calculate Average True Range (ATR)
def calculate_atr(stock_list, period=14):
    tr_list = []
    for i in range(1, len(stock_list)):
        current_high = stock_list[i].high
        current_low = stock_list[i].low
        previous_close = stock_list[i - 1].close

        tr = max(current_high - current_low, 
                 abs(current_high - previous_close), 
                 abs(current_low - previous_close))
        tr_list.append(tr)

    # Calculate moving average of TR
    atr_list = [sum(tr_list[:period]) / period]
    for i in range(period, len(tr_list)):
        atr = (atr_list[-1] * (period - 1) + tr_list[i]) / period
        atr_list.append(atr)

    return atr_list

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(stock_list, period=20):
    moving_avg_list = []
    upper_band_list = []
    lower_band_list = []

    for i in range(period - 1, len(stock_list)):
        close_prices = [stock.close for stock in stock_list[i - period + 1:i + 1]]
        moving_avg = sum(close_prices) / period
        moving_avg_list.append(moving_avg)

        std_dev = sqrt(sum([(p - moving_avg) ** 2 for p in close_prices]) / period)
        upper_band = moving_avg + 2 * std_dev
        lower_band = moving_avg - 2 * std_dev

        upper_band_list.append(upper_band)
        lower_band_list.append(lower_band)

    return moving_avg_list, upper_band_list, lower_band_list


def read_stock_data(file_path):
    stock_list = []
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            date = row['Date']
            open_price = float(row['Open'])
            high = float(row['High'])
            low = float(row['Low'])
            close = float(row['Close'])
            volume = int(row['Volume'])
            stock = StockData(date, open_price, high, low, close, volume)
            stock_list.append(stock)
    return stock_list


# Function to perform analysis
def analyze_stock_data(stock_list, rsi_values, macd_line, signal_line):
    buy_signals = []
    sell_signals = []
    overbought = []
    oversold = []
    print(len(stock_list))
    print(len(macd_line))
    print(len(signal_line))

    start_index = len(stock_list) - len(macd_line)
    print(start_index)

    for i in range(start_index, len(stock_list)):
        macd_index = i - start_index  # Adjust index for macd_line and signal_line
        
        
        if i < len(rsi_values):  
            # Calculate MACD difference
            macd_diff = abs(macd_line[macd_index] - signal_line[macd_index])

            # Buy signals
            if macd_line[macd_index] > signal_line[macd_index] or rsi_values[i] <= 30:
                buy_signals.append((stock_list[i].get_date(), stock_list[i].close, macd_diff, abs(30 - rsi_values[i])))

            # Sell signals
            if macd_line[macd_index] < signal_line[macd_index] or rsi_values[i] >= 70:
                sell_signals.append((stock_list[i].get_date(), stock_list[i].close, macd_diff, abs(70 - rsi_values[i])))

            # Overbought
            if rsi_values[i] >= 70:
                overbought.append((stock_list[i].get_date(), stock_list[i].close, abs(70 - rsi_values[i])))

            # Oversold
            if rsi_values[i] <= 30:
                oversold.append((stock_list[i].get_date(), stock_list[i].close, abs(30 - rsi_values[i])))

    # Write all signals to a text file
    with open('all_signals.txt', 'w') as file:
        file.write("All Buy Signals:\n")
        for signal in buy_signals:
            file.write(f"Date: {signal[0]}, Close: {signal[1]}, MACD Diff: {signal[2]}\n")

        file.write("\nAll Sell Signals:\n")
        for signal in sell_signals:
            file.write(f"Date: {signal[0]}, Close: {signal[1]}, MACD Diff: {signal[2]}\n")

    # Sort and select top 10 buy and sell signals based on MACD difference primarily
    top_buy_signals = sorted(buy_signals, key=lambda x: x[2], reverse=True)[:10]
    top_sell_signals = sorted(sell_signals, key=lambda x: x[2], reverse=True)[:10]
    # Sort and select top 10 overbought and oversold signals
    top_overbought = sorted(overbought, key=lambda x: x[2], reverse=True)[:10]
    top_oversold = sorted(oversold, key=lambda x: x[2], reverse=True)[:10]

    # Print the analysis results
    print("Top 10 Buy Signals:")
    for signal in top_buy_signals:
        print(signal)

    print("Top 10 Sell Signals:")
    for signal in top_sell_signals:
        print(signal)

    # Print the analysis results for overbought and oversold
    print("Top 10 Overbought Times:")
    for time in top_overbought:
        print(time)

    print("Top 10 Oversold Times:")
    for time in top_oversold:
        print(time)


# tests
def test_stock_data():
    # Create a sample StockData object with sample data
    stock = StockData('2023-01-01', 100, 110, 90, 105, 10000)
    # Assert statements to test methods
    assert stock.get_daily_change() == (105 - 100)

# Helper function to calculate EMA
def calculate_ema(prices, period, smoothing=2):
    ema = [sum(prices[:period]) / period]
    multiplier = smoothing / (1 + period)
    for price in prices[period:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])
    return ema

# Function to calculate RSI
def calculate_rsi(prices, period=14):
    gains = losses = 0
    for i in range(1, period + 1):
        delta = prices[i] - prices[i - 1]
        if delta > 0:
            gains += delta
        else:
            losses -= delta
    average_gain = gains / period
    average_loss = losses / period if losses != 0 else 1

    rsis = []
    for i in range(period, len(prices)):
        delta = prices[i] - prices[i - 1]
        gain = max(delta, 0)
        loss = -min(delta, 0)
        average_gain = (average_gain * (period - 1) + gain) / period
        average_loss = (average_loss * (period - 1) + loss) / period
        rs = average_gain / average_loss if average_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        rsis.append(rsi)
    return rsis

# Function to calculate MACD
def calculate_macd(prices):
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
   
    macd_line = [ema12[i + 14] - ema26[i] for i in range(len(ema26) - 14)]
    signal_line = calculate_ema(macd_line, 9)
    return macd_line, signal_line

# Function to plot the closing prices, RSI, and MACD
def plot_stock_analysis(closing_prices, rsi_values, macd_line, signal_line, atr_values, dates):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(10, 8), sharex=True)

    # Plot closing prices
    ax1.plot(dates, closing_prices, label='Closing Prices')
    ax1.set_title('Stock Prices')
    ax1.legend()

    # Plot RSI
    ax2.plot(dates, rsi_values, label='RSI')
    ax2.axhline(70, color='r', linestyle='dashed')
    ax2.axhline(30, color='g', linestyle='dashed')
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.legend()

    # Plot MACD
    ax3.plot(dates, macd_line, label='MACD Line')
    ax3.plot(dates, signal_line, label='Signal Line', linestyle='--')
    ax3.set_title('Moving Average Convergence Divergence (MACD)')
    ax3.legend()

    # Plot ATR
    ax4.plot(dates, atr_values, label='ATR')
    ax4.set_title('Average True Range (ATR)')
    ax4.legend()

    # Improve layout and show the plot
    plt.tight_layout()
    plt.show()

# Main program
def main():
    closing_prices = []
    stock_list = []
    
        # Read the closing prices from the CSV
    with open('stock.csv', 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            closing_prices.append(float(row['Close']))
        stock_list = read_stock_data('stock.csv')    
        
        # Calculate RSI and MACD
        rsi_values = calculate_rsi(closing_prices)
        macd_line, signal_line = calculate_macd(closing_prices)
        atr_values = calculate_atr(stock_list, period=14)

        # Adjust start index for alignment based on the length of the shortest array (signal_line)
        start_index = len(stock_list) - len(signal_line)

        aligned_dates = [stock.get_date() for stock in stock_list[-len(signal_line):]]
        aligned_closing_prices = closing_prices[-len(signal_line):]
        aligned_rsi_values = rsi_values[-len(signal_line):]
        aligned_macd_line = macd_line[-len(signal_line):]
        aligned_signal_line = signal_line  # Already the shortest
        aligned_atr_values = atr_values[-len(signal_line):]

       
        print("Lengths after alignment:", len(aligned_dates), len(aligned_closing_prices), len(aligned_rsi_values), len(aligned_macd_line), len(aligned_signal_line), len(aligned_atr_values))

        plot_stock_analysis(aligned_closing_prices, aligned_rsi_values, aligned_macd_line, aligned_signal_line, aligned_atr_values, aligned_dates)

        analyze_stock_data(stock_list, rsi_values, macd_line, signal_line)
        

    

if __name__ == "__main__":
    main()
    #test_stock_data()
    
    


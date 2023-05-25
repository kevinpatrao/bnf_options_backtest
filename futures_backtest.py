import pandas as pd
import numpy as np
from pydantic import BaseModel
from datetime import datetime

class BankniftyData(BaseModel):
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


def load_banknifty_data(file_path):
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

    # Convert the filtered DataFrame into a list of BankniftyData objects using Pydantic
    data_list = []
    for row in df.itertuples(index=False):
        data = BankniftyData(date=row.date, open=row.open, high=row.high,
                             low=row.low, close=row.close, volume=row.volume)
        data_list.append(data)

    return data_list

def backtest_strategy(data_list):
    # Define periods for moving averages
    fma_period = 50
    sma_period = 200

    # Variables to store trade-related data
    trade_data = []
    position = "None"
    entry_price = 0.0
    entry_time = None

    # Function to save trade data
    def save_trade_data(entry_type, entry_time, entry_price, stop_loss, exit_time, exit_price, exit_type):
        trade_data.append({
            "Entry Type": entry_type,
            "Entry Time": entry_time,
            "Entry Price": entry_price,
            "Stop Loss": stop_loss,
            "Exit Time": exit_time,
            "Exit Price": exit_price,
            "Exit Type": exit_type
        })

    # Loop through each record in the futures data
    for i in range(201, len(data_list)):
        current_candle = data_list[i]
        prev_candle = data_list[i - 1]

        # Calculate moving averages
        prev_fma = sum(
            data.close for data in data_list[i - fma_period:i]) / fma_period
        prev_sma = sum(
            data.close for data in data_list[i - sma_period:i]) / sma_period
        prev2_fma = sum(
            data.close for data in data_list[i - fma_period - 1:i - 1]) / fma_period
        prev2_sma = sum(
            data.close for data in data_list[i - sma_period - 1:i - 1]) / sma_period

        # Define long and short conditions
        long_condition = prev_fma > prev_sma and prev2_fma < prev2_sma
        short_condition = prev_fma < prev_sma and prev2_fma > prev2_sma

        # Trading logic
        if position == "None":           
            if long_condition:
                position = "Long"
                entry_price = current_candle.close
                entry_time = current_candle.date
                stop_loss = min(prev_sma, entry_price - entry_price * 0.001)
            elif short_condition:
                position = "Short"
                entry_price = current_candle.close
                entry_time = current_candle.date
                stop_loss = max(prev_sma, entry_price + entry_price * 0.001)

        elif position == "Long":            
            if current_candle.low < stop_loss and current_candle.open > stop_loss:
                exit_price = current_candle.close
                exit_time = current_candle.date
                save_trade_data(position, entry_time, entry_price, stop_loss, exit_time, exit_price, "SL")
                position = "None"
            elif short_condition:
                exit_price = current_candle.close
                exit_time = current_candle.date
                save_trade_data(position, entry_time, entry_price, stop_loss, exit_time, exit_price, "New")   
                position = "Short"
                entry_price = current_candle.close 
                entry_time = current_candle.date 
                stop_loss = max(prev_sma, entry_price + entry_price * 0.001)                
            elif current_candle.low < stop_loss and current_candle.open < stop_loss:
                stop_loss = current_candle.low

        elif position == "Short": 
            if current_candle.high > stop_loss and current_candle.open < stop_loss:
                exit_price = current_candle.close
                exit_time = current_candle.date
                save_trade_data(position, entry_time, entry_price, stop_loss, exit_time, exit_price, "SL")
                position = "None"          
            elif long_condition:
                exit_price = current_candle.close
                exit_time = current_candle.date
                save_trade_data(position, entry_time, entry_price, stop_loss, exit_time, exit_price, "New")   
                position = "Long"
                entry_price = current_candle.close
                entry_time = current_candle.date
                stop_loss = min(prev_sma, entry_price - entry_price * 0.001)               
            elif current_candle.high > stop_loss and current_candle.open > stop_loss:
                stop_loss = current_candle.high

    # Convert trade_data list into a DataFrame
    df_trades = pd.DataFrame(trade_data)
    return df_trades


# Run the code
file_path = "../Data/underlying_data/bnf_spot.csv"
banknifty_data = load_banknifty_data(file_path)
tradebook = backtest_strategy(banknifty_data)
print(tradebook)
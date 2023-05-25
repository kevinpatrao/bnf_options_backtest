import os
import datetime
import calendar
import math
import pandas as pd
from pydantic import BaseModel
from typing import List


# Define the data model for option data
class OptionData(BaseModel):
    ticker: str
    date: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: int


# Class for analyzing options data
class OptionsAnalyzer:
    def __init__(self):
        self.option_data_list = []

    def calculate_ATM_strike(self, underlying_price: float, position: str) -> tuple:
        # Calculate the nearest ATM strike based on the underlying price and position
        strike_difference = 100
        nearest_strike = round(
            underlying_price / strike_difference) * strike_difference

        if position == 'Long':
            if underlying_price < nearest_strike:
                atm_strike = nearest_strike
            else:
                atm_strike = nearest_strike + strike_difference
            option_type = 'CE'  # Call Option
        elif position == 'Short':
            if underlying_price > nearest_strike:
                atm_strike = nearest_strike
            else:
                atm_strike = nearest_strike - strike_difference
            option_type = 'PE'  # Put Option
            
        return atm_strike, option_type

    def get_option_contract_path(self, timestamp: datetime.datetime, underlying_price: float, position_type: str) -> str:
        # Determine the option contract path based on timestamp, underlying price, and position type
        year = timestamp.strftime('%y')
        month_names = ['January', 'February', 'March']
        month = timestamp.strftime('%B')
        month_folder = month_names[timestamp.month - 1]

        # Calculate the last Thursday of the month
        last_day = calendar.monthrange(timestamp.year, timestamp.month)[1]
        last_thursday = last_day - \
            ((calendar.weekday(timestamp.year, timestamp.month, last_day) - 3) % 7)

        # Determine month to be passed for folder and file path generation
        if timestamp.day > last_thursday:
            option_month = month_names[(timestamp.month + 1) % 12 - 1]
            option_month_initials = month_names[(
                timestamp.month + 1) % 12 - 1][:3].upper()
            folder_path = os.path.join(
                '..', 'data', 'options_data', month_folder)
        else:
            option_month = month
            option_month_initials = month[:3].upper()
            folder_path = os.path.join(
                '..', 'data', 'options_data', option_month)

        atm_strike, option_type = self.calculate_ATM_strike(
            underlying_price, position_type)
        option_contract = f'BANKNIFTY{year}{option_month_initials}{atm_strike}{option_type}'
        file_path = os.path.join(folder_path, option_contract + '.csv')

        return file_path

    # Load options data from a CSV file into a list of OptionData objects
    def load_options_data(self, file_path: str) -> list:
        df = pd.read_csv(file_path)
        df.columns = [column.replace('<', '').replace(
            '>', '').replace('o/i ', 'oi') for column in df.columns]
        df['date'] = pd.to_datetime(
            df['date'] + ' ' + df['time'], format='%m/%d/%Y %H:%M:%S')

        # Convert the filtered DataFrame into a list of OptionData objects using Pydantic
        data_list = []
        for row in df.itertuples(index=False):
            data = OptionData(
                ticker=row.ticker, date=row.date, open=row.open,
                high=row.high, low=row.low, close=row.close,
                volume=row.volume, oi=row.oi
            )
            data_list.append(data)
            
        return data_list

    # Get the options prices based on the given options data and timestamp
    def get_options_price(self, timestamp: datetime.datetime, underlying_price: float, position_type: str) -> tuple:

        options_file_path = self.get_option_contract_path(
            timestamp, underlying_price, position_type)

        option_data_list = self.load_options_data(options_file_path)
        target_date = timestamp.date()
        target_timestamp = timestamp

        # Iterate up to 5 minutes (including the original timestamp)
        for _ in range(6):
            for option_data in option_data_list:
                if option_data.date == target_timestamp:
                    return option_data.date, option_data.close, option_data.ticker

            target_timestamp += datetime.timedelta(minutes=1)

        # If no matching timestamp is found within 5 minutes, return the closest available timestamp
        return min(option_data_list, key=lambda x: abs(x.date - target_timestamp)).date, None, None



# Define the data model for bank Nifty data
class BankniftyData(BaseModel):
    date: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

# Class for performing moving average crossover backtesting
class MovingAverageCrossoverBacktester:
    # Load bank Nifty data from a CSV file into a list of BankniftyData objects
    def load_spot_data(self, file_path: str) -> List[BankniftyData]:  
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

        start_timestamp = pd.to_datetime('2021-01-01 09:15:00')
        end_timestamp = pd.to_datetime('2021-03-25 03:30:00')

        df = df[(df['date'] >= start_timestamp)
                & (df['date'] <= end_timestamp)]

        # Convert the filtered DataFrame into a list of BankniftyData objects using Pydantic
        data_list = []
        for row in df.itertuples(index=False):
            data = BankniftyData(date=row.date, open=row.open, high=row.high,
                                 low=row.low, close=row.close, volume=row.volume)
            data_list.append(data)

        return data_list

    # Perform the moving average crossover backtesting
    def backtest(self, data_list: List[BankniftyData]) -> pd.DataFrame:
        # Define constants for moving averages
        fma_period = 50
        sma_period = 200

        # Variables to store trade-related data
        trade_data = []
        position = "None"
        stop_loss = 0.0
        underlying_price_at_entry = 0.0

        # Function to save trade data
        def save_trade_data(ticker, entry_type, entry_time, entry_price, stop_loss, exit_time, exit_price, exit_type):
            trade_data.append({
                "Ticker": ticker,
                "Entry Type": entry_type,
                "Entry Time": entry_time,
                "Entry Price": entry_price,
                "Stop Loss": stop_loss,
                "Exit Time": exit_time,
                "Exit Price": exit_price,
                "Exit Type": exit_type
            })

        # Loop through records in the data
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
                    entry_time, entry_price, ticker = OptionsAnalyzer().get_options_price(
                        current_candle.date, prev_candle.close, position)
                    # print('Long Entry Passed: ', current_candle.date, prev_candle.close, position)
                    stop_loss = min(prev_sma, prev_candle.close -
                                    prev_candle.close * 0.001)
                    underlying_price_at_entry = prev_candle.close

                elif short_condition:
                    position = "Short"
                    entry_time, entry_price, ticker = OptionsAnalyzer().get_options_price(
                        current_candle.date, prev_candle.close, position)
                    # print('Short Entry Passed: ', current_candle.date, prev_candle.close, position)
                    stop_loss = max(prev_sma, prev_candle.close +
                                    prev_candle.close * 0.001)
                    underlying_price_at_entry = prev_candle.close

            elif position == "Long":
                if current_candle.low < stop_loss and current_candle.open > stop_loss:
                    exit_time, exit_price, ticker = OptionsAnalyzer().get_options_price(
                        current_candle.date, underlying_price_at_entry, position)
                    # print('L SL Exit Passed: ', current_candle.date, underlying_price_at_entry, position)
                    save_trade_data(ticker, position, entry_time, entry_price,
                                    stop_loss, exit_time, exit_price, "SL")
                    # print('Exit L SL Saving: ', ticker, position, entry_time, entry_price, stop_loss, exit_time, exit_price, "SL")
                    # print('----------------------------------------------------------------------')
                    position = "None"

                elif short_condition:
                    exit_time, exit_price, ticker = OptionsAnalyzer().get_options_price(
                        current_candle.date, underlying_price_at_entry, position)
                    #print('L S Exit Passed: ', current_candle.date, underlying_price_at_entry, position)
                    save_trade_data(ticker, position, entry_time, entry_price,
                                    stop_loss, exit_time, exit_price, "New Trade")
                    # print('Exit L S Saving: ', ticker, position, entry_time, entry_price, stop_loss, exit_time, exit_price, "New")
                    # print('----------------------------------------------------------------------')
                    position = "Short"
                    entry_time, entry_price, ticker = OptionsAnalyzer().get_options_price(
                        current_candle.date, current_candle.close, position)
                    #print('L S Entry Passed: ', current_candle.date, current_candle.close, position)
                    stop_loss = max(prev_sma, current_candle.close +
                                    current_candle.close * 0.001)
                elif current_candle.low < stop_loss and current_candle.open < stop_loss:
                    stop_loss = current_candle.low

            elif position == "Short":
                if current_candle.high > stop_loss and current_candle.open < stop_loss:
                    exit_time, exit_price, ticker = OptionsAnalyzer().get_options_price(
                        current_candle.date, underlying_price_at_entry, position)
                    # print('S SL Exit Passed: ', current_candle.date, underlying_price_at_entry, position)
                    save_trade_data(ticker, position, entry_time, entry_price,
                                    stop_loss, exit_time, exit_price, "SL")
                    # print('Exit S SL Saving: ', ticker, position, entry_time, entry_price, stop_loss, exit_time, exit_price, "SL")
                    # print('----------------------------------------------------------------------')
                    position = "None"
                elif long_condition:
                    exit_time, exit_price, ticker = OptionsAnalyzer().get_options_price(
                        current_candle.date, underlying_price_at_entry, position)
                    # print('S L Exit Passed: ', current_candle.date, underlying_price_at_entry, position)
                    save_trade_data(ticker, position, entry_time, entry_price,
                                    stop_loss, exit_time, exit_price, "New Trade")
                    # print('Exit S L Saving: ', ticker, position, entry_time, entry_price, stop_loss, exit_time, exit_price, "New")
                    # print('----------------------------------------------------------------------')
                    position = "Long"
                    entry_time, entry_price, ticker = OptionsAnalyzer().get_options_price(
                        current_candle.date, current_candle.close, position)
                    # print('S L Entry Passed: ', current_candle.date, current_candle.close, position)
                    stop_loss = min(prev_sma, current_candle.close -
                                    current_candle.close * 0.001)
                elif current_candle.high > stop_loss and current_candle.open > stop_loss:
                    stop_loss = current_candle.high

        # Convert trade_data list into a DataFrame
        df_trades = pd.DataFrame(trade_data)
        return df_trades



# Define the data model for a trade
class Trade(BaseModel):
    ticker: str
    entry_type: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_type: str

# Class for calculating performance metrics
class MetricsCalculator:
    def __init__(self, tradebook: pd.DataFrame):
        self.tradebook = tradebook
    
    # Calculate the Sharpe ratio based on the given trades
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0, frequency: int = 252) -> float:
        excess_returns = returns - risk_free_rate
        annualized_returns = excess_returns.mean() * frequency
        annualized_volatility = excess_returns.std() * math.sqrt(frequency)
        sharpe_ratio = annualized_returns / annualized_volatility
        return sharpe_ratio
    
    # Calculate the Calmar ratio based on the given trades
    def calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        calmar_ratio = returns.mean() / abs(max_drawdown)
        return calmar_ratio
    
    # Calculate the maximum drawdown based on the given trades
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        equity_curve.index = pd.to_datetime(equity_curve.index)
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        return max_drawdown
    
    # Calculate the compound annual growth rate (CAGR) based on the given trades
    def calculate_cagr(self, equity_curve: pd.Series, time_period: int) -> float:
        start_equity = equity_curve.iloc[0]
        end_equity = equity_curve.iloc[-1]
        cagr = (end_equity / start_equity) ** (1 / time_period) - 1
        return cagr

    # Calculate the total returns based on the given trades
    def calculate_total_returns(self, equity_curve: pd.Series) -> float:
        total_returns = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        return total_returns

    # Calculate the drawdown duration for the given trades
    def calculate_dd_duration(self, equity_curve: pd.Series) -> int:
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        dd_duration = drawdown[drawdown < 0].index.to_series().diff().dt.total_seconds() / 60
        max_dd_duration = dd_duration.max()
        return max_dd_duration

    # Calculate the hit ratio for the given trades
    def calculate_hit_ratio(self, tradebook: pd.DataFrame) -> float:
        num_profitable_trades = (tradebook['Exit Price'] - tradebook['Entry Price'] > 0).sum()
        total_trades = len(tradebook)
        hit_ratio = num_profitable_trades / total_trades
        return hit_ratio

    # Calculate the average pnl for trades
    def calculate_avg_profit_loss_per_trade(self, tradebook: pd.DataFrame) -> tuple:
        profit_trades = tradebook[tradebook['Exit Price'] > tradebook['Entry Price']]
        loss_trades = tradebook[tradebook['Exit Price'] < tradebook['Entry Price']]
        avg_profit_per_trade = profit_trades['Exit Price'].mean() - profit_trades['Entry Price'].mean()
        avg_loss_per_trade = loss_trades['Exit Price'].mean() - loss_trades['Entry Price'].mean()
        return avg_profit_per_trade, avg_loss_per_trade

    # Generate the metrics report
    def calculate_metrics_report(self, risk_free_rate: float = 0.0, frequency: int = 252) -> dict:
        # Calculate returns
        returns = (self.tradebook['Exit Price'] / self.tradebook['Entry Price']) - 1

        # Metric calculations
        sharpe_ratio = self.calculate_sharpe_ratio(returns, risk_free_rate, frequency)
        equity_curve = (returns + 1).cumprod()
        max_drawdown = self.calculate_max_drawdown(equity_curve)
        calmar_ratio = self.calculate_calmar_ratio(returns, max_drawdown)
        cagr = self.calculate_cagr(equity_curve, len(self.tradebook) / frequency / 60)
        total_returns = self.calculate_total_returns(equity_curve)
        max_dd_duration = self.calculate_dd_duration(equity_curve)
        hit_ratio = self.calculate_hit_ratio(self.tradebook)
        avg_profit_per_trade, avg_loss_per_trade = self.calculate_avg_profit_loss_per_trade(self.tradebook)

        # Create metrics report structure
        # Create DataFrame for metrics report
        metrics_report = pd.DataFrame({
            'Metric': ['Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio', 'CAGR', 'Total Returns',
                       'Max DD Duration', 'Hit Ratio', 'Avg Profit per Trade', 'Avg Loss per Trade'],
            'Value': [sharpe_ratio, max_drawdown, calmar_ratio, cagr, total_returns,
                      max_dd_duration, hit_ratio, avg_profit_per_trade, avg_loss_per_trade]
        })
        
        # Format the Value column to display upto fixed decimal places
        metrics_report['Value'] = metrics_report['Value'].round(2)

        return metrics_report



# Initialise backtester and provide path to data files
file_path = "../Data/underlying_data/bnf_spot.csv"
backtester = MovingAverageCrossoverBacktester()
data = backtester.load_spot_data(file_path)

# Backtest and generate tradebook and metrics report
tradebook = backtester.backtest(data)
metrics_calculator = MetricsCalculator(tradebook)
metrics_report = metrics_calculator.calculate_metrics_report(risk_free_rate=0.02, frequency=252)

# Save trade details as CSV file
tradebook.to_csv('../Data/tradebook.csv')
metrics_report.to_csv('../Data/metrics_report.csv')

# Display tradebook and metrics report
print(tradebook)
print(metrics_report)
import pandas as pd
import numpy as np
from config import CONFIG
from logger_config import analysis_logger


def calculate_portfolio_value(df, signals, cost_rate=0.0, initial_capital=1.0):
    """
    计算投资组合价值
    
    参数:
        df: 数据框，必须包含'close'列
        signals: 交易信号列表 (0=不持仓, 1=持仓)
        cost_rate: 交易成本率 (默认0.0)
        initial_capital: 初始资本 (默认1.0)
    
    返回:
        包含每日投资组合价值的列表
    """
    if 'close' not in df.columns:
        raise ValueError("数据框必须包含 'close' 列")
    
    if len(signals) != len(df):
        raise ValueError(f"信号数量 ({len(signals)}) 必须与数据框行数 ({len(df)}) 相同")
    
    capital = initial_capital
    position = 0
    values = []
    
    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        
        if i == 0:
            if signals[i] == 1:
                position = capital / (current_price * (1 + cost_rate))
                capital = 0
            values.append(initial_capital)
            continue
        
        prev_signal = signals[i-1]
        current_signal = signals[i]
        
        if prev_signal == 0 and current_signal == 1:
            # 买入信号
            position = capital / (current_price * (1 + cost_rate))
            capital = 0
            
        elif prev_signal == 1 and current_signal == 0:
            # 卖出信号
            capital = position * current_price * (1 - cost_rate)
            position = 0
        
        if position > 0:
            values.append(position * current_price)
        else:
            values.append(capital)
    
    return values


def buy_hold_strategy(df):
    """
    买入并持有策略 - 从指定日期开始持仓
    
    参数:
        df: 数据框，必须包含'date'列
    
    返回:
        signals 列表
    """
    if 'date' not in df.columns:
        raise ValueError("数据框必须包含 'date' 列")
    
    signals = []
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    start_date = pd.to_datetime(CONFIG['ANALYSIS_START_DATE'])
    
    try:
        start_idx = df_copy[df_copy['date'] >= start_date].index[0]
    except IndexError:
        analysis_logger.warning(f"未找到 {CONFIG['ANALYSIS_START_DATE']} 或之后的日期，使用第一个日期")
        start_idx = 0
    
    for i in range(len(df)):
        if i < start_idx:
            signals.append(0)
        else:
            signals.append(1)
    
    return signals
    
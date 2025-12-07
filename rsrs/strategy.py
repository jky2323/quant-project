import numpy as np
import pandas as pd
from config import CONFIG
from logger_config import analysis_logger


def _get_start_index(df):
    """获取分析开始索引"""
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    start_date = pd.to_datetime(CONFIG['ANALYSIS_START_DATE'])
    try:
        return df_copy[df_copy['date'] >= start_date].index[0]
    except IndexError:
        analysis_logger.warning(f"未找到 {CONFIG['ANALYSIS_START_DATE']} 或之后的日期")
        return 0


def _generate_slope_signals(df, buy_threshold, sell_threshold):
    """
    生成斜率信号
    
    参数:
        df: 包含 rsrs_slope 列的 DataFrame
        buy_threshold: 买入阈值
        sell_threshold: 卖出阈值
    
    返回:
        signals 列表
    """
    signals = []
    position = 0
    start_idx = _get_start_index(df)
    
    for i in range(len(df)):
        if i < start_idx:
            signals.append(0)
            continue
        
        slope = df['rsrs_slope'].iloc[i]
        
        if np.isnan(slope):
            signals.append(position)
            continue
        
        if position == 0 and slope > buy_threshold:
            position = 1
        elif position == 1 and slope < sell_threshold:
            position = 0
        
        signals.append(position)
    
    return signals


def _generate_score_signals(df, score_column, threshold, buy_threshold=1.0, sell_threshold=0.8):
    """
    生成分数信号
    
    参数:
        df: 包含分数列和斜率列的 DataFrame
        score_column: 分数列名称
        threshold: 分数阈值
        buy_threshold: 斜率买入阈值（用作回退）
        sell_threshold: 斜率卖出阈值（用作回退）
    
    返回:
        signals 列表
    """
    signals = []
    position = 0
    
    start_idx = _get_start_index(df)
    first_score_idx = df[score_column].first_valid_index()
    
    for i in range(len(df)):
        if i < start_idx:
            signals.append(0)
            continue
        
        score = df[score_column].iloc[i]
        slope = df['rsrs_slope'].iloc[i]
        
        # 分数计算未完成时，使用斜率作为回退
        if i < first_score_idx or pd.isna(score):
            if np.isnan(slope):
                signals.append(position)
                continue
            
            if position == 0 and slope > buy_threshold:
                position = 1
            elif position == 1 and slope < sell_threshold:
                position = 0
        else:
            # 使用分数信号
            if position == 0 and score > threshold:
                position = 1
            elif position == 1 and score < -threshold:
                position = 0
        
        signals.append(position)
    
    return signals


def backtest_slope_strategy(df, buy_threshold=None, sell_threshold=None):
    """
    RSRS 斜率策略回测
    
    参数:
        df: 数据框，必须包含 rsrs_slope 列
        buy_threshold: 买入阈值
        sell_threshold: 卖出阈值
    
    返回:
        signals 列表
    """
    buy_threshold = buy_threshold or CONFIG['SLOPE_BUY_THRESHOLD']
    sell_threshold = sell_threshold or CONFIG['SLOPE_SELL_THRESHOLD']
    
    return _generate_slope_signals(df, buy_threshold, sell_threshold)


def backtest_standard_score_strategy(df, threshold=None):
    """
    标准分策略
    
    参数:
        df: 数据框，必须包含 standard_score 列
        threshold: 分数阈值
    
    返回:
        (signals, df) 元组
    """
    threshold = threshold or CONFIG['STANDARD_SCORE_PARAMS']['threshold']
    signals = _generate_score_signals(df, 'standard_score', threshold)
    return signals, df


def backtest_modified_standard_score_strategy(df, threshold=None):
    """
    修正标准分策略
    
    参数:
        df: 数据框，必须包含 modified_standard_score 列
        threshold: 分数阈值
    
    返回:
        (signals, df) 元组
    """
    threshold = threshold or CONFIG['MODIFIED_SCORE_PARAMS']['threshold']
    signals = _generate_score_signals(df, 'modified_standard_score', threshold)
    return signals, df


def backtest_right_skewed_standard_score_strategy(df, threshold=None):
    """
    右偏标准分策略
    
    参数:
        df: 数据框，必须包含 right_skewed_standard_score 列
        threshold: 分数阈值
    
    返回:
        (signals, df) 元组
    """
    threshold = threshold or CONFIG['RIGHT_SKEWED_SCORE_PARAMS']['threshold']
    signals = _generate_score_signals(df, 'right_skewed_standard_score', threshold)
    return signals, df

def _generate_volume_optimized_signals(df, base_score_column, threshold):
    """
    生成交易量优化信号
    
    参数:
        df: 包含基础分数列和volume_correlation列的DataFrame
        base_score_column: 基础分数列名称
        threshold: 分数阈值
    
    返回:
        signals 列表
    """
    signals = []
    position = 0
    
    start_idx = _get_start_index(df)
    first_score_idx = df[base_score_column].first_valid_index()
    
    for i in range(len(df)):
        if i < start_idx:
            signals.append(0)
            continue
        
        score = df[base_score_column].iloc[i]
        volume_corr = df['volume_correlation'].iloc[i]
        slope = df['rsrs_slope'].iloc[i]  # 需要斜率作为回退
        
        # 分数计算未完成时，使用斜率作为回退（与原策略保持一致）
        if i < first_score_idx or pd.isna(score) or pd.isna(volume_corr):
            if np.isnan(slope):
                signals.append(position)
                continue
            
            # 使用斜率回退逻辑
            if position == 0 and slope > CONFIG['SLOPE_BUY_THRESHOLD']:
                position = 1
            elif position == 1 and slope < CONFIG['SLOPE_SELL_THRESHOLD']:
                position = 0
        else:
            # 使用分数信号 + 交易量过滤
            if position == 0 and score > threshold and volume_corr > 0:
            # if position == 0 and score > threshold:
                position = 1
            elif position == 1 and score < -threshold:
                position = 0
        
        signals.append(position)
    
    return signals

def backtest_volume_optimized_standard_score_strategy(df, threshold=None):
    """交易量优化的标准分策略"""
    threshold = threshold or CONFIG['STANDARD_SCORE_PARAMS']['threshold']
    signals = _generate_volume_optimized_signals(df, 'standard_score', threshold)
    return signals, df

def backtest_volume_optimized_modified_score_strategy(df, threshold=None):
    """交易量优化的修正标准分策略"""
    threshold = threshold or CONFIG['MODIFIED_SCORE_PARAMS']['threshold']
    signals = _generate_volume_optimized_signals(df, 'modified_standard_score', threshold)
    return signals, df

def backtest_volume_optimized_right_skewed_strategy(df, threshold=None):
    """交易量优化的右偏标准分策略"""
    threshold = threshold or CONFIG['RIGHT_SKEWED_SCORE_PARAMS']['threshold']
    signals = _generate_volume_optimized_signals(df, 'right_skewed_standard_score', threshold)
    return signals, df


def _generate_price_optimized_signals(df, base_score_column, threshold, ma_window=None, compare_days=None):
    """
    生成价格优化信号
    
    参数:
        df: 包含基础分数列的DataFrame
        base_score_column: 基础分数列名称
        threshold: 分数阈值
        ma_window: 均线窗口
        compare_days: 比较间隔天数
    
    返回:
        signals 列表
    """
    ma_window = ma_window or CONFIG['PRICE_TREND_WINDOW']
    compare_days = compare_days or CONFIG['PRICE_COMPARE_DAYS']
    
    signals = []
    position = 0
    
    start_idx = _get_start_index(df)
    first_score_idx = df[base_score_column].first_valid_index()
    
    # 预先计算20日均线
    df_copy = df.copy()
    df_copy['ma_20'] = df_copy['close'].rolling(window=ma_window).mean()
    
    for i in range(len(df)):
        if i < start_idx:
            signals.append(0)
            continue
        
        score = df_copy[base_score_column].iloc[i]
        slope = df_copy['rsrs_slope'].iloc[i]
        ma_20_current = df_copy['ma_20'].iloc[i]
        
        # 检查是否有足够的均线数据进行比较
        has_ma_data = (i >= ma_window + compare_days - 1 and 
                      not pd.isna(df_copy['ma_20'].iloc[i-1]) and 
                      not pd.isna(df_copy['ma_20'].iloc[i-compare_days]))
        
        # 趋势判断：前一日MA20 > 前三日MA20
        is_uptrend = False
        if has_ma_data:
            ma_prev = df_copy['ma_20'].iloc[i-1]      # 前一日MA20
            ma_prev_2 = df_copy['ma_20'].iloc[i-2]
            ma_prev_3 = df_copy['ma_20'].iloc[i-compare_days]  # 前三日MA20
            ma_prev_4 = df_copy['ma_20'].iloc[i-compare_days-1]
            is_uptrend = ma_prev > ma_prev_3
            # is_uptrend = ma_20_current > ma_prev and ma_20_current > ma_prev_2 and ma_20_current > ma_prev_3
            # is_uptrend = ma_20_current > ma_prev_3
            # is_uptrend = ma_20_current > ma_prev and ma_prev > ma_prev_2 and ma_prev_2 > ma_prev_3
            # is_uptrend = ma_prev > ma_prev_2 and ma_prev_2 > ma_prev_3 and ma_prev_3 > ma_prev_4
        
        # 分数计算未完成时，使用斜率作为回退（不应用趋势过滤）
        if i < first_score_idx or pd.isna(score):
            if np.isnan(slope):
                signals.append(position)
                continue
            
            if position == 0 and slope > CONFIG['SLOPE_BUY_THRESHOLD'] and (not has_ma_data or is_uptrend):
                position = 1
            elif position == 1 and slope < CONFIG['SLOPE_SELL_THRESHOLD']:
                position = 0
        else:
            # 使用分数信号 + 价格趋势过滤
            if position == 0 and score > threshold and (not has_ma_data or is_uptrend):
                position = 1
            elif position == 1 and score < -threshold:
                position = 0
        
        signals.append(position)
    
    return signals

def backtest_price_optimized_standard_score_strategy(df, threshold=None):
    """价格优化的标准分策略"""
    threshold = threshold or CONFIG['STANDARD_SCORE_PARAMS']['threshold']
    signals = _generate_price_optimized_signals(df, 'standard_score', threshold)
    return signals, df

def backtest_price_optimized_modified_score_strategy(df, threshold=None):
    """价格优化的修正标准分策略"""
    threshold = threshold or CONFIG['MODIFIED_SCORE_PARAMS']['threshold']
    signals = _generate_price_optimized_signals(df, 'modified_standard_score', threshold)
    return signals, df

def backtest_price_optimized_right_skewed_strategy(df, threshold=None):
    """价格优化的右偏标准分策略"""
    threshold = threshold or CONFIG['RIGHT_SKEWED_SCORE_PARAMS']['threshold']
    signals = _generate_price_optimized_signals(df, 'right_skewed_standard_score', threshold)
    return signals, df
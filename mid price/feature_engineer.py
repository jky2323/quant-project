import pandas as pd
import numpy as np

# ===================================================================
# 1. 标签生成逻辑
# ===================================================================

def phi(x, alpha):
    """
    根据价格变化 x 和阈值 alpha，将结果映射到标签 0, 1, 2。
    0: 下跌, 1: 不变, 2: 上涨
    """
    if x < -alpha:
        return 0  # 下跌
    elif x > alpha:
        return 2  # 上涨
    else:
        return 1  # 不变

def generate_labels(df):
    """
    根据 n_midprice 的未来变化，为DataFrame生成5个价格走势标签。

    Args:
        df (pd.DataFrame): 包含 'n_midprice' 和 'unique_id' 的已清洗数据。
    """
    if df.empty or 'n_midprice' not in df.columns or 'unique_id' not in df.columns:
        print("❌ Error: DataFrame is empty or missing required columns ('n_midprice' or 'unique_id').")
        return df
        
    alpha_settings = {
        5: 0.0005,
        10: 0.0005,
        20: 0.001,
        40: 0.001,
        60: 0.001
    }
    
    print("⏳ Starting label generation (label5 to label60)...")

    for N, alpha in alpha_settings.items():
        label_col = f'label{N}'
        
        # 1. 计算未来 N 个 tick 的 n_midprice (n_midprice_{t+N})
        # 确保 shift(-N) 操作只在当前交易时段内进行
        df[label_col + '_future_midprice'] = df.groupby('unique_id')['n_midprice'].transform(
            lambda x: x.shift(-N) 
        )
        
        # 2. 计算价格变化 (price_diff)
        price_diff = df[label_col + '_future_midprice'] - df['n_midprice']
        
        # 3. 应用 phi 函数生成最终标签
        df[label_col] = price_diff.apply(lambda x: phi(x, alpha) if pd.notna(x) else np.nan)
        
        # 删除临时列
        df.drop(columns=[label_col + '_future_midprice'], inplace=True)
        
        print(f"✅ Generated {label_col}.")

    return df

# ===================================================================
# 2. 衍生特征计算逻辑 (因子库)
# ===================================================================

def calculate_factors_for_group(group_df):
    """
    计算单个交易时段 (unique_id) 内的所有因子。
    """
    factors = {}
    
    # 辅助变量，用于简化计算
    midprice = group_df['n_midprice']
    amount_delta = group_df['amount_delta']
    close = group_df['n_close']
    
    # --- 1. 价格动量 ---
    for period in [5, 10, 20]:
        factors[f'price_change_{period}'] = midprice.pct_change(period).fillna(0)
        factors[f'close_change_{period}'] = close.pct_change(period).fillna(0)
    
    factors['price_acceleration'] = midprice.diff().diff().fillna(0) 
    factors['price_volatility_10'] = midprice.rolling(10, min_periods=1).std().fillna(0)
    factors['price_volatility_20'] = midprice.rolling(20, min_periods=1).std().fillna(0)

    # --- 2. 订单簿不平衡 ---
    ask1, bid1 = group_df['n_ask1'], group_df['n_bid1']
    asize1, bsize1 = group_df['n_asize1'], group_df['n_bsize1']
    
    factors['bid_ask_spread'] = ask1 - bid1
    factors['relative_spread'] = factors['bid_ask_spread'] / ((ask1 + bid1) / 2).replace(0, 1e-6)

    # OIB1
    factors['order_imbalance_1'] = (bsize1 - asize1) / (bsize1 + asize1).replace(0, 1e-6)
    
    # OIB5
    bid_depth_sum = group_df[[f'n_bsize{i}' for i in range(1, 6)]].sum(axis=1)
    ask_depth_sum = group_df[[f'n_asize{i}' for i in range(1, 6)]].sum(axis=1)
    factors['order_imbalance_5'] = (bid_depth_sum - ask_depth_sum) / (bid_depth_sum + ask_depth_sum).replace(0, 1e-6)

    # 价格压力因子
    pressure_num = bid1 * bsize1 - ask1 * asize1
    pressure_den = bid1 * bsize1 + ask1 * asize1
    factors['price_pressure'] = pressure_num / pressure_den.replace(0, 1e-6)

    # 订单簿深度
    factors['bid_depth'] = bid_depth_sum
    factors['ask_depth'] = ask_depth_sum
    factors['depth_ratio'] = factors['bid_depth'] / factors['ask_depth'].replace(0, 1e-6)
    
    # 增强因子：加权平均价格 (WAP)
    WAP_num = bid1 * asize1 + ask1 * bsize1
    WAP_den = bsize1 + asize1
    factors['WAP'] = WAP_num / WAP_den.replace(0, 1e-6)
    factors['log_WAP_midprice_ratio'] = np.log(factors['WAP'] / midprice.replace(0, 1e-6))
    
    # --- 3. 成交量因子 ---
    vol_sum_5 = amount_delta.rolling(5, min_periods=1).sum()
    vol_sum_10 = amount_delta.rolling(10, min_periods=1).sum()
    factors['volume_momentum_5'] = vol_sum_5.pct_change().fillna(0)
    factors['volume_momentum_10'] = vol_sum_10.pct_change().fillna(0)
    
    # VWAP (简化的中价*量和/量和)
    vol_w_midprice_5 = (midprice * amount_delta).rolling(5, min_periods=1).sum()
    factors['vwap_5'] = vol_w_midprice_5 / vol_sum_5.replace(0, 1e-6)
    
    factors['volume_volatility_10'] = amount_delta.rolling(10, min_periods=1).std().fillna(0)
    factors['volume_volatility_20'] = amount_delta.rolling(20, min_periods=1).std().fillna(0)

    # 异常成交量
    vol_mean_20 = amount_delta.rolling(20, min_periods=1).mean()
    vol_std_20 = amount_delta.rolling(20, min_periods=1).std().replace(0, 1e-6)
    factors['volume_anomaly'] = (amount_delta - vol_mean_20) / vol_std_20
    factors['volume_anomaly'] = factors['volume_anomaly'].replace([np.inf, -np.inf], 0).fillna(0)

    # --- 4. 微观结构因子 ---
    midprice_diff = midprice.diff()
    amount_delta_safe = amount_delta.replace(0, 1e-6)

    factors['price_efficiency'] = midprice_diff.abs() / amount_delta_safe.fillna(1e-6)
    factors['liquidity_measure'] = amount_delta / factors['bid_ask_spread'].replace(0, 1e-6)

    factors['order_flow_imbalance'] = (
        (bid1.diff() > 0).astype(int) - (ask1.diff() < 0).astype(int)
    ).fillna(0)
    
    factors['price_impact'] = midprice_diff / amount_delta_safe.fillna(1e-6)

    # --- 5. 技术形态因子 ---
    factors['ma_5'] = midprice.rolling(5, min_periods=1).mean()
    factors['ma_10'] = midprice.rolling(10, min_periods=1).mean()
    factors['ma_20'] = midprice.rolling(20, min_periods=1).mean()

    factors['ma_cross_5_10'] = factors['ma_5'] - factors['ma_10']
    factors['ma_cross_5_20'] = factors['ma_5'] - factors['ma_20']

    vol_20 = midprice.rolling(20, min_periods=1).std()
    factors['bollinger_upper'] = factors['ma_20'] + 2 * vol_20
    factors['bollinger_lower'] = factors['ma_20'] - 2 * vol_20
    
    boll_range = factors['bollinger_upper'] - factors['bollinger_lower']
    factors['bollinger_position'] = (midprice - factors['bollinger_lower']) / boll_range.replace(0, 1e-6)

    # RSI 类似指标
    returns = midprice.diff()
    gain = returns.where(returns > 0, 0).rolling(14, min_periods=1).mean()
    loss = (-returns).where(returns < 0, 0).rolling(14, min_periods=1).mean()
    factors['rsi_like'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-6)))

    # --- 6. 统计因子 ---
    factors['lag_1'] = midprice.shift(1).fillna(0)
    factors['lag_5'] = midprice.shift(5).fillna(0)
    
    factors['price_skew_10'] = midprice.rolling(10, min_periods=1).skew().fillna(0)
    factors['price_kurtosis_10'] = midprice.rolling(10, min_periods=1).kurt().fillna(0)

    factors['price_quantile_25'] = midprice.rolling(20, min_periods=1).quantile(0.25).fillna(0)
    factors['price_quantile_75'] = midprice.rolling(20, min_periods=1).quantile(0.75).fillna(0)
    
    # --- 7. 时间序列因子 ---
    if 'time' in group_df.columns:
        group_df['time_seconds'] = pd.to_timedelta(group_df['time']).dt.total_seconds()
        total_seconds = 23400 # 6.5 小时交易时间
        factors['time_of_day_sin'] = np.sin(2 * np.pi * group_df['time_seconds'] / total_seconds) 
        factors['time_of_day_cos'] = np.cos(2 * np.pi * group_df['time_seconds'] / total_seconds) 

        factors['minutes_from_open'] = (group_df['time_seconds'] - group_df['time_seconds'].min()) / 60

    # 返回所有计算出的因子
    return pd.DataFrame(factors, index=group_df.index).fillna(0)


# 确保 def 关键字在文件的最左侧，没有缩进
def create_all_features(df):
    """
    构造所有因子的主函数，安全地在每个 unique_id 内计算滚动因子。
    """
    if df.empty or 'unique_id' not in df.columns:
        print("❌ Error: DataFrame is empty or missing 'unique_id' column.")
        return df

    print("⏳ Starting advanced feature engineering grouped by unique_id...")
    
    # 使用 groupby 和 apply 在每个交易时段内计算因子
    all_features_df = df.groupby('unique_id', group_keys=False).apply(calculate_factors_for_group)
    
    # 将新的衍生特征合并回原始 DataFrame
    existing_cols = set(df.columns)
    new_features = {}
    for col in all_features_df.columns:
        # 仅选择新的列名进行合并
        if col not in existing_cols:
            new_features[col] = all_features_df[col]
            
    df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

    print(f"✅ Feature engineering complete. Total columns: {len(df.columns)}")
    return df
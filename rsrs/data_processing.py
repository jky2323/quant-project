import pandas as pd
import numpy as np
import statsmodels.api as sm
from config import CONFIG
from logger_config import analysis_logger


def calculate_rsrs_slope(df, n=None):
    """
    计算RSRS斜率
    
    参数:
        df: 数据框
        n: 回归窗口大小
    
    返回:
        包含rsrs_slope和r_squared列的数据框
    """
    n = n or CONFIG['DEFAULT_N']
    
    slopes = []
    r_squareds = []
    
    for i in range(len(df)):
        if i < n - 1:
            slopes.append(np.nan)
            r_squareds.append(np.nan)
        else:
            high_n = df['high'].iloc[i-n+1:i+1]
            low_n = df['low'].iloc[i-n+1:i+1]
            
            try:
                X = sm.add_constant(low_n)
                model = sm.OLS(high_n, X).fit()
                
                slopes.append(model.params[1])
                r_squareds.append(model.rsquared)
            except Exception as e:
                analysis_logger.warning(f"计算索引 {i} 处的RSRS斜率时出错: {e}")
                slopes.append(np.nan)
                r_squareds.append(np.nan)
    
    result_df = df.copy()
    result_df['rsrs_slope'] = slopes
    result_df['r_squared'] = r_squareds
    
    return result_df


def calculate_standard_score(df, m=None):
    """
    计算标准分
    
    参数:
        df: 包含rsrs_slope列的数据框
        m: 统计窗口大小
    
    返回:
        包含standard_score列的数据框
    """
    m = m or CONFIG['DEFAULT_M']
    
    if 'rsrs_slope' not in df.columns:
        raise ValueError("数据框必须包含 'rsrs_slope' 列")
    
    standard_scores = []
    
    for i in range(len(df)):
        if i < m - 1:
            standard_scores.append(np.nan)
        else:
            slopes_m = df['rsrs_slope'].iloc[i-m+1:i+1]
            
            current_slope = df['rsrs_slope'].iloc[i]
            mean_slope = slopes_m.mean()
            std_slope = slopes_m.std()
            
            if std_slope > 0:
                z_score = (current_slope - mean_slope) / std_slope
            else:
                z_score = 0
            
            standard_scores.append(z_score)
    
    result_df = df.copy()
    result_df['standard_score'] = standard_scores
    
    return result_df


def calculate_modified_standard_score(df):
    """
    计算修正标准分 = 标准分 * R²
    
    参数:
        df: 包含standard_score和r_squared列的数据框
    
    返回:
        包含modified_standard_score列的数据框
    """
    if 'standard_score' not in df.columns or 'r_squared' not in df.columns:
        raise ValueError("数据框必须包含 'standard_score' 和 'r_squared' 列")
    
    modified_scores = []
    
    for i in range(len(df)):
        if pd.isna(df['standard_score'].iloc[i]) or pd.isna(df['r_squared'].iloc[i]):
            modified_scores.append(np.nan)
        else:
            modified_score = df['standard_score'].iloc[i] * df['r_squared'].iloc[i]
            modified_scores.append(modified_score)
    
    result_df = df.copy()
    result_df['modified_standard_score'] = modified_scores
    return result_df


def calculate_right_skewed_standard_score(df):
    """
    计算右偏标准分 = 修正标准分 * 斜率
    
    参数:
        df: 包含modified_standard_score和rsrs_slope列的数据框
    
    返回:
        包含right_skewed_standard_score列的数据框
    """
    if 'modified_standard_score' not in df.columns or 'rsrs_slope' not in df.columns:
        raise ValueError("数据框必须包含 'modified_standard_score' 和 'rsrs_slope' 列")
    
    right_skewed_scores = []
    
    for i in range(len(df)):
        if pd.isna(df['modified_standard_score'].iloc[i]) or pd.isna(df['rsrs_slope'].iloc[i]):
            right_skewed_scores.append(np.nan)
        else:
            right_skewed_score = df['modified_standard_score'].iloc[i] * df['rsrs_slope'].iloc[i]
            right_skewed_scores.append(right_skewed_score)
    
    result_df = df.copy()
    result_df['right_skewed_standard_score'] = right_skewed_scores
    return result_df


def calculate_volume_correlation(df, correlation_window=None):
    """
    计算交易量与修正标准分的相关性 - 健壮版本
    """
    correlation_window = correlation_window or CONFIG['VOLUME_CORRELATION_WINDOW']
    
    volume_correlations = []
    
    for i in range(len(df)):
        if i < correlation_window - 1:
            volume_correlations.append(np.nan)
            continue
            
        volume_data = df['volume'].iloc[i-correlation_window+1:i+1]
        score_data = df['modified_standard_score'].iloc[i-correlation_window+1:i+1]
        
        # 严格的数据有效性检查
        valid_data = (
            len(volume_data) == correlation_window and
            len(score_data) == correlation_window and
            not volume_data.isna().any() and
            not score_data.isna().any() and
            volume_data.std() > 1e-10 and  # 避免浮点精度问题
            score_data.std() > 1e-10
        )
        
        if not valid_data:
            volume_correlations.append(np.nan)
            continue
        
        # 手动计算相关性，避免numpy警告
        volume_mean = volume_data.mean()
        score_mean = score_data.mean()
        
        numerator = ((volume_data - volume_mean) * (score_data - score_mean)).sum()
        denominator = (np.sqrt(((volume_data - volume_mean)**2).sum()) * 
                      np.sqrt(((score_data - score_mean)**2).sum()))
        
        if denominator < 1e-10:  # 避免除以零
            volume_correlations.append(np.nan)
        else:
            correlation = numerator / denominator
            # 确保相关性在合理范围内
            correlation = np.clip(correlation, -1.0, 1.0)
            volume_correlations.append(correlation)
    
    result_df = df.copy()
    result_df['volume_correlation'] = volume_correlations
    
    # 记录有效相关性计算的比例
    valid_count = len([x for x in volume_correlations if not pd.isna(x)])
    analysis_logger.info(f"交易量相关性计算完成，有效数据点: {valid_count}/{len(df)} ({valid_count/len(df)*100:.1f}%)")
    
    return result_df
CONFIG = {
    # 时间配置
    'ANALYSIS_START_DATE': '2005-02-18',
    'STATISTICS_START_DATE': '2005-03-01',
    
    # RSRS 斜率参数
    'DEFAULT_N': 18,
    'SLOPE_BUY_THRESHOLD': 1.0,
    'SLOPE_SELL_THRESHOLD': 0.8,
    
    # 标准分参数
    'DEFAULT_M': 600,
    'DEFAULT_THRESHOLD': 0.7,
    
    # 策略参数
    'STANDARD_SCORE_PARAMS': {
        'n': 18,
        'm': 600,
        'threshold': 0.7,
    },
    'MODIFIED_SCORE_PARAMS': {
        'n': 18,
        'm': 600,
        'threshold': 0.7,
    },
    'RIGHT_SKEWED_SCORE_PARAMS': {
        'n': 16,
        'm': 300,
        'threshold': 0.7,
    },
    # 'RIGHT_SKEWED_SCORE_PARAMS': {
    #     'n': 18,
    #     'm': 600,
    #     'threshold': 0.7,
    # },
    
    # 分析参数
    'FORWARD_DAYS': 10,
    'BIN_WIDTH_PROBABILITY': 0.1,
    'BIN_WIDTH_RETURN': 0.2,
    'PARAMETER_SENSITIVITY_N_RANGE': range(14, 25),
    
    # 文件和目录
    'DATA_FILE': 'hs300_data.csv',
    'PICTURE_DIR': 'picture',
    'LOG_DIR': 'logs',
    
    # 成本率
    'COST_RATES': [0.0, 0.002, 0.003],
    'COST_LABELS': ['No cost', '0.4% round-trip cost', '0.6% round-trip cost'],

    # 新增：价格优化参数
    'PRICE_TREND_WINDOW': 20,  # 均线窗口
    'PRICE_COMPARE_DAYS': 3,   # 比较间隔天数
    
    # 价格优化策略参数
    'PRICE_OPTIMIZED_PARAMS': {
        'ma_window': 20,
        'compare_days': 3,
        'threshold': 0.7,  # 沿用原有的阈值
    },
    
    # 新增：交易量优化参数
    'VOLUME_CORRELATION_WINDOW': 10,  # 相关性计算窗口
    'VOLUME_COLUMN': 'volume',        # 交易量列名
    
    # 优化策略参数
    'VOLUME_OPTIMIZED_PARAMS': {
        'correlation_window': 10,
        'threshold': 0.7,  # 沿用原有的阈值
    },

    # 多市场配置
    'MARKETS': {
        'hs300': {
            'data_file': 'hs300_data.csv',
            'name': 'HS300',
            'benchmark_name': 'HS300'
        },
        'sh50': {
            'data_file': 'sz50_data.csv', 
            'name': 'SSE50',
            'benchmark_name': 'SSE50'
        },
        'zz500': {
            'data_file': 'zz500_data.csv',
            'name': 'CSI500',
            'benchmark_name': 'CSI500'
        }
    },
    
    # 默认市场
    'DEFAULT_MARKET': 'hs300'
}
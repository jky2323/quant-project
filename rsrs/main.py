import os
import pandas as pd
from config import CONFIG
from logger_config import analysis_logger

from data_processing import *
from plot.indicators import *
from plot.strategy_performance import *
from plot.cost_analysis import *
from plot.parameter_sensitivity import *
from plot.score_analysis import *
from plot.price_optimized_strategies import *
def _setup_environment():
    """设置环境和目录"""
    os.makedirs(CONFIG['PICTURE_DIR'], exist_ok=True)
    os.makedirs(CONFIG['LOG_DIR'], exist_ok=True)
    analysis_logger.info("="*80)
    analysis_logger.info("开始RSRS策略分析")
    analysis_logger.info("="*80)


def _load_data():
    """加载数据"""
    try:
        if not os.path.exists(CONFIG['DATA_FILE']):
            raise FileNotFoundError(f"数据文件不存在: {CONFIG['DATA_FILE']}")
        
        df = pd.read_csv(CONFIG['DATA_FILE'])
        
        if df.empty:
            raise ValueError("CSV 文件为空")
        
        required_columns = ['date', 'close', 'high', 'low']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要列: {missing_columns}")
        
        analysis_logger.info(f"成功加载数据文件: {CONFIG['DATA_FILE']}")
        analysis_logger.info(f"数据行数: {len(df)}")
        
        return df
    
    except Exception as e:
        analysis_logger.error(f"加载数据失败: {e}")
        raise


def _prepare_all_scores(df):
    """
    一次性计算所有标准分（分别处理不同参数）
    
    参数:
        df: 原始数据框
    
    返回:
        包含所有标准分的数据框
    """
    analysis_logger.info("\n=== 计算所有标准分 ===")
    
    # ========== 标准分和修正标准分（n=18, m=600） ==========
    df_result = calculate_rsrs_slope(df, n=CONFIG['DEFAULT_N'])
    analysis_logger.info(f"RSRS 斜率计算完成 (n={CONFIG['DEFAULT_N']})")
    
    df_result = calculate_standard_score(df_result, m=CONFIG['DEFAULT_M'])
    analysis_logger.info(f"标准分计算完成 (m={CONFIG['DEFAULT_M']})")
    
    df_result = calculate_modified_standard_score(df_result)
    analysis_logger.info("修正标准分计算完成")
    
    # ========== 右偏标准分（n=16, m=300） ==========
    # 需要用不同的参数重新计算
    n_right = CONFIG['RIGHT_SKEWED_SCORE_PARAMS']['n']
    m_right = CONFIG['RIGHT_SKEWED_SCORE_PARAMS']['m']
    
    df_right_skewed = calculate_rsrs_slope(df, n=n_right)
    analysis_logger.info(f"RSRS 斜率计算完成 (n={n_right}) - 用于右偏标准分")
    
    df_right_skewed = calculate_standard_score(df_right_skewed, m=m_right)
    analysis_logger.info(f"标准分计算完成 (m={m_right}) - 用于右偏标准分")
    
    df_right_skewed = calculate_modified_standard_score(df_right_skewed)
    analysis_logger.info("修正标准分计算完成 - 用于右偏标准分")
    
    df_right_skewed = calculate_right_skewed_standard_score(df_right_skewed)
    analysis_logger.info("右偏标准分计算完成")
    
    # 将右偏标准分列合并到主数据框
    df_result['right_skewed_standard_score'] = df_right_skewed['right_skewed_standard_score']
    
    analysis_logger.info(f"包含列: {', '.join(df_result.columns.tolist())}")
    
    return df_result


# def _prepare_all_scores(df):
#     """
#     一次性计算所有标准分（现在全部使用相同参数 n=18, m=600）
#     """
#     analysis_logger.info("\n=== 计算所有标准分（统一参数 n=18, m=600）===")
    
#     # 统一使用 n=18, m=600 计算所有分数
#     df_result = calculate_rsrs_slope(df, n=18)
#     df_result = calculate_standard_score(df_result, m=600)
#     df_result = calculate_modified_standard_score(df_result)
#     df_result = calculate_right_skewed_standard_score(df_result)
    
#     analysis_logger.info("所有标准分计算完成（统一参数 n=18, m=600）")
#     return df_result


def run_basic_analysis(df):
    """运行基础分析"""
    analysis_logger.info("\n=== 运行基础分析 ===")
    
    try:
        plot_slope_histogram(df)
        plot_slope_mean(df)
        
        plot_standard_score_distribution(df)
        
        plot_modified_standard_score_distribution(df)
        
        plot_right_skewed_score_distribution(df)
        
        plot_strategy_performance(df)
        
        analysis_logger.info("基础分析完成")
    
    except Exception as e:
        analysis_logger.error(f"基础分析失败: {e}")
        raise


def run_cost_analysis(df):
    """运行成本分析"""
    analysis_logger.info("\n=== 运行成本分析 ===")
    
    try:
        plot_slope_strategy_with_costs(df)
        plot_standard_score_strategy_with_costs(df)
        plot_right_skewed_strategy_with_costs(df)
        analysis_logger.info("成本分析完成")
    
    except Exception as e:
        analysis_logger.error(f"成本分析失败: {e}")
        raise


def run_parameter_analysis(df):
    """运行参数敏感性分析"""
    analysis_logger.info("\n=== 运行参数敏感性分析 ===")
    
    try:
        from plot.parameter_sensitivity import (
            plot_parameter_sensitivity_strategy_curves,
            plot_parameter_sensitivity_n,
            plot_optimized_strategies_parameter_sensitivity_m
        )
        
        # 原有的N参数敏感性分析
        n_range = CONFIG['PARAMETER_SENSITIVITY_N_RANGE']
        plot_parameter_sensitivity_strategy_curves(df, n_range=n_range)
        sensitivity_results_n = plot_parameter_sensitivity_n(df, n_range=n_range)
        
        # 新增：M参数敏感性分析（优化策略）
        m_range = [450, 500, 550, 600, 650, 700, 750, 800]
        sensitivity_results_m = plot_optimized_strategies_parameter_sensitivity_m(df, m_range=m_range)
        
        analysis_logger.info("参数敏感性分析完成")
        return sensitivity_results_n, sensitivity_results_m
    
    except Exception as e:
        analysis_logger.error(f"参数敏感性分析失败: {e}")
        raise


def run_correlation_analysis(df):
    """运行相关性分析"""
    analysis_logger.info("\n=== 运行相关性分析 ===")
    
    try:
        forward_days = CONFIG['FORWARD_DAYS']
        
        # 标准分相关性
        plot_score_vs_up_probability(df, forward_days=forward_days)
        plot_score_vs_expected_return(df, forward_days=forward_days)
        
        # 修正标准分相关性
        plot_modified_score_vs_up_probability(df, forward_days=forward_days)
        plot_modified_score_vs_expected_return(df, forward_days=forward_days)
        
        # 右偏标准分相关性
        plot_right_skewed_score_vs_up_probability(df, forward_days=forward_days)
        plot_right_skewed_score_vs_expected_return(df, forward_days=forward_days)
        
        analysis_logger.info("相关性分析完成")
    
    except Exception as e:
        analysis_logger.error(f"相关性分析失败: {e}")
        raise


def run_strategy_statistics(df):
    """运行策略统计分析"""
    analysis_logger.info("\n=== 计算详细统计信息 ===")
    
    try:
        # plot_different_score_strategies_comparison(df, compute_stats=True)
        strategies_data = plot_all_strategies_comparison(df)
        analysis_logger.info("策略统计分析完成")
    
    except Exception as e:
        analysis_logger.error(f"策略统计分析失败: {e}")
        raise


def run_volume_optimized_analysis(df):
    """运行交易量优化分析"""
    analysis_logger.info("\n=== 运行交易量优化分析 ===")
    
    try:
        # 首先计算交易量相关性
        from data_processing import calculate_volume_correlation
        df = calculate_volume_correlation(df)
        analysis_logger.info("交易量相关性计算完成")
        
        # 绘制交易量优化策略对比
        from plot.volume_optimized_strategies import plot_volume_optimized_strategies_comparison
        plot_volume_optimized_strategies_comparison(df)
        
        analysis_logger.info("交易量优化分析完成")
    
    except Exception as e:
        analysis_logger.error(f"交易量优化分析失败: {e}")
        raise


def run_price_optimized_analysis(df):
    """运行价格优化分析"""
    analysis_logger.info("\n=== 运行价格优化分析 ===")
    
    try:
        plot_price_optimized_strategies_comparison(df)
        plot_price_optimized_right_skewed_with_costs(df)
        
        analysis_logger.info("价格优化分析完成")
    
    except Exception as e:
        analysis_logger.error(f"价格优化分析失败: {e}")
        raise


def run_multi_market_analysis():
    """运行多市场分析"""
    analysis_logger.info("\n=== 运行多市场分析 ===")
    
    try:
        from plot.multi_market_analysis import plot_multi_market_strategies
        results = plot_multi_market_strategies()
        analysis_logger.info("多市场分析完成")
        return results
    
    except Exception as e:
        analysis_logger.error(f"多市场分析失败: {e}")
        raise


def main(analysis_config=None):
    """
    主程序 - 可配置的分析流程
    
    参数:
        analysis_config: 分析配置字典，格式如下:
        {
            'basic': True,           # 运行基础分析
            'cost': True,            # 运行成本分析
            'parameter': True,       # 运行参数敏感性分析
            'correlation': True,     # 运行相关性分析
            'statistics': True,      # 计算详细统计信息
        }
    """
    # 默认配置
    default_config = {
        'basic': True,
        'cost': True,
        'parameter': True,
        'correlation': True,
        'statistics': True,
        'volume_optimized': True,
    }
    
    config = analysis_config or default_config
    
    try:
        # 设置环境
        _setup_environment()
        
        # 加载数据
        analysis_logger.info("正在加载数据...")
        df = _load_data()
        
        # 一次性计算所有标准分（分别处理不同参数）
        df = _prepare_all_scores(df)
        
        # # 基础分析
        # if config['basic']:
        #     run_basic_analysis(df)
        
        # # 成本分析
        # if config['cost']:
        #     run_cost_analysis(df)
        
        # # 参数敏感性分析
        # if config['parameter']:
        #     sensitivity_results = run_parameter_analysis(df)
        
        # # 相关性分析
        # if config['correlation']:
        #     run_correlation_analysis(df)
        
        # 策略统计分析
        if config['statistics']:
            run_strategy_statistics(df)

        # if config.get('price_optimized', True):
        #     run_price_optimized_analysis(df)
     
        # # 在适当位置添加交易量优化分析
        # if config.get('volume_optimized', True):  # 可以配置是否运行
        #     run_volume_optimized_analysis(df)

        # run_multi_market_analysis()


        # # 完成
        # analysis_logger.info("\n所有分析完成！")
        # analysis_logger.info("="*80)
        # analysis_logger.info(f"日志文件位置: {CONFIG['LOG_DIR']}/")
    
    except Exception as e:
        analysis_logger.error(f"分析失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # 可选：自定义分析配置
    # custom_config = {
    #     'basic': True,
    #     'cost': False,
    #     'parameter': False,
    #     'correlation': True,
    #     'statistics': True,
    # }
    # main(analysis_config=custom_config)
    
    main()




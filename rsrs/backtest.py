import pandas as pd
import numpy as np
from logger_config import analysis_logger
from utils import calculate_portfolio_value

def calculate_strategy_statistics(df, signals, strategy_name):
    """
    计算策略统计指标（仅计算，不输出）
    
    参数:
        df: 数据框
        signals: 交易信号列表
        strategy_name: 策略名称
    
    返回:
        统计指标字典
    """
    values = calculate_portfolio_value(df, signals)
    returns = pd.Series(values).pct_change().dropna()
    
    # 持仓统计
    total_days = len([s for s in signals if s == 1])
    trade_days = len([i for i in range(1, len(signals)) if signals[i] != signals[i-1]])
    
    # 日级统计
    holding_returns = [returns.iloc[i] for i in range(len(returns)) if signals[i+1] == 1]
    
    win_days = len([r for r in holding_returns if r > 0])
    loss_days = len([r for r in holding_returns if r < 0])
    win_rate_daily = win_days / len(holding_returns) if holding_returns else 0
    
    avg_win_daily = np.mean([r for r in holding_returns if r > 0]) if any(r > 0 for r in holding_returns) else 0
    avg_loss_daily = np.mean([r for r in holding_returns if r < 0]) if any(r < 0 for r in holding_returns) else 0
    profit_loss_ratio_daily = abs(avg_win_daily / avg_loss_daily) if avg_loss_daily != 0 else 0
    
    # 交易级统计
    trade_returns = []
    entry_price = 0
    in_trade = False
    
    for i in range(1, len(signals)):
        if signals[i] == 1 and signals[i-1] == 0:
            entry_price = df['close'].iloc[i]
            in_trade = True
        elif signals[i] == 0 and signals[i-1] == 1 and in_trade:
            exit_price = df['close'].iloc[i]
            trade_return = (exit_price - entry_price) / entry_price
            trade_returns.append(trade_return)
            in_trade = False
    
    if in_trade:
        exit_price = df['close'].iloc[-1]
        trade_return = (exit_price - entry_price) / entry_price
        trade_returns.append(trade_return)
    
    win_trades = len([r for r in trade_returns if r > 0])
    loss_trades = len([r for r in trade_returns if r < 0])
    win_rate_trade = win_trades / len(trade_returns) if trade_returns else 0
    
    avg_win_trade = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
    avg_loss_trade = np.mean([r for r in trade_returns if r < 0]) if any(r < 0 for r in trade_returns) else 0
    profit_loss_ratio_trade = abs(avg_win_trade / avg_loss_trade) if avg_loss_trade != 0 else 0
    
    # 收益统计
    total_return = values[-1] - 1
    years = len(df) / 252
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # 风险指标
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    cumulative = pd.Series(values)
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 持仓周期
    holding_periods = []
    current_hold = 0
    for signal in signals:
        if signal == 1:
            current_hold += 1
        elif current_hold > 0:
            holding_periods.append(current_hold)
            current_hold = 0
    if current_hold > 0:
        holding_periods.append(current_hold)
    
    avg_holding_days = np.mean(holding_periods) if holding_periods else 0
    
    stats = {
        'strategy_name': strategy_name,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_days': total_days,
        'trade_days': trade_days,
        'win_days': win_days,
        'loss_days': loss_days,
        'win_rate_daily': win_rate_daily,
        'avg_win_daily': avg_win_daily,
        'avg_loss_daily': avg_loss_daily,
        'profit_loss_ratio_daily': profit_loss_ratio_daily,
        'win_trades': win_trades,
        'loss_trades': loss_trades,
        'win_rate_trade': win_rate_trade,
        'avg_win_trade': avg_win_trade,
        'avg_loss_trade': avg_loss_trade,
        'profit_loss_ratio_trade': profit_loss_ratio_trade,
        'avg_holding_days': avg_holding_days,
        'max_win_trade': max(trade_returns) if trade_returns else 0,
        'min_win_trade': min(trade_returns) if trade_returns else 0,
    }
    
    return stats


def log_strategy_statistics(stats):
    """
    将统计信息写入日志文件
    
    参数:
        stats: 统计字典（由 calculate_strategy_statistics 返回）
    """
    analysis_logger.info(f"\n{'='*60}")
    analysis_logger.info(f"{stats['strategy_name']} 统计表现")
    analysis_logger.info(f"{'='*60}")
    
    analysis_logger.info(f"年化收益率: {stats['annual_return']*100:.2f}%")
    analysis_logger.info(f"夏普比率: {stats['sharpe_ratio']:.2f}")
    analysis_logger.info(f"最大回撤: {stats['max_drawdown']*100:.2f}%")
    analysis_logger.info(f"持仓总天数: {int(stats['total_days'])}")
    analysis_logger.info(f"交易次数: {stats['trade_days']}")
    analysis_logger.info(f"平均持仓天数: {stats['avg_holding_days']:.2f}")
    analysis_logger.info(f"获利天数: {stats['win_days']}")
    analysis_logger.info(f"亏损天数: {stats['loss_days']}")
    analysis_logger.info(f"胜率（按天）: {stats['win_rate_daily']*100:.2f}%")
    analysis_logger.info(f"平均盈利率（按天）: {stats['avg_win_daily']*100:.2f}%")
    analysis_logger.info(f"平均亏损率（按天）: {stats['avg_loss_daily']*100:.2f}%")
    analysis_logger.info(f"平均盈亏比（按天）: {stats['profit_loss_ratio_daily']:.2f}")
    analysis_logger.info(f"盈利次数: {stats['win_trades']}")
    analysis_logger.info(f"亏损次数: {stats['loss_trades']}")
    analysis_logger.info(f"胜率（按次）: {stats['win_rate_trade']*100:.2f}%")
    analysis_logger.info(f"平均盈利率（按次）: {stats['avg_win_trade']*100:.2f}%")
    analysis_logger.info(f"平均亏损率（按次）: {stats['avg_loss_trade']*100:.2f}%")
    analysis_logger.info(f"平均盈亏比（按次）: {stats['profit_loss_ratio_trade']:.2f}")
    analysis_logger.info(f"单次最大盈利: {stats['max_win_trade']*100:.2f}%")
    analysis_logger.info(f"单次最大亏损: {stats['min_win_trade']*100:.2f}%")
    analysis_logger.info("-" * 60)
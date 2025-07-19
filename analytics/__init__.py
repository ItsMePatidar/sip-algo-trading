"""
Analytics module for SIP Algorithmic Trading System

This module provides comprehensive performance analytics, risk metrics,
and visualization tools for strategy analysis and backtesting.
"""

from .performance import (
    PerformanceAnalyzer,
    calculate_returns,
    calculate_cagr,
    calculate_total_return,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_information_ratio,
    calculate_max_drawdown,
    calculate_var,
    calculate_cvar
)

from .risk_metrics import (
    RiskAnalyzer,
    calculate_downside_deviation,
    calculate_upside_capture,
    calculate_downside_capture,
    calculate_beta,
    calculate_alpha,
    calculate_tracking_error,
    calculate_correlation,
    calculate_rolling_volatility,
    calculate_rolling_sharpe,
    stress_test_portfolio
)

from .visualization import (
    PerformanceVisualizer,
    plot_cumulative_returns,
    plot_rolling_sharpe,
    plot_drawdown,
    plot_returns_distribution,
    plot_risk_return_scatter,
    plot_correlation_heatmap,
    create_performance_dashboard,
    plot_strategy_comparison
)

__all__ = [
    'PerformanceAnalyzer',
    'RiskAnalyzer', 
    'PerformanceVisualizer',
    'calculate_returns',
    'calculate_cagr',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'plot_cumulative_returns',
    'create_performance_dashboard'
]
"""
Backtesting Framework for SIP Algorithmic Trading System

This module provides comprehensive backtesting capabilities for testing
SIP strategies against historical market data.

Key Components:
- BacktestEngine: Main backtesting orchestrator
- Portfolio: Portfolio tracking and management
- PerformanceMetrics: Comprehensive performance analysis
"""

from .backtest_engine import BacktestEngine, BacktestConfig, BacktestResults
from .portfolio import Portfolio, Position, Transaction
from .metrics import PerformanceMetrics, RiskMetrics, PerformanceReport

__all__ = [
    'BacktestEngine',
    'BacktestConfig', 
    'BacktestResults',
    'Portfolio',
    'Position',
    'Transaction',
    'PerformanceMetrics',
    'RiskMetrics',
    'PerformanceReport'
]
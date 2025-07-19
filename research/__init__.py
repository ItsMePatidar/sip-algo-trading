"""
Research Framework for SIP Algorithmic Trading System

This module provides comprehensive research and analysis tools for:
- Strategy development and testing
- Market analysis and regime detection
- Performance attribution and factor analysis
- Risk analysis and scenario testing

Key Components:
- StrategyResearcher: Strategy development and backtesting tools
- MarketAnalyzer: Market analysis and regime detection
- FactorAnalysis: Performance attribution analysis
- ScenarioTester: Stress testing and scenario analysis
"""

from .strategy_research import (
    StrategyResearcher, 
    StrategyComparison, 
    ParameterSensitivity,
    StrategyOptimizer,
    ResearchReport
)
from .market_analysis import (
    MarketAnalyzer,
    MarketRegime,
    TechnicalIndicators,
    CorrelationAnalysis,
    VolatilityAnalysis,
    MarketResearchReport
)

__all__ = [
    'StrategyResearcher',
    'StrategyComparison', 
    'ParameterSensitivity',
    'StrategyOptimizer',
    'ResearchReport',
    'MarketAnalyzer',
    'MarketRegime',
    'TechnicalIndicators',
    'CorrelationAnalysis',
    'VolatilityAnalysis',
    'MarketResearchReport'
]
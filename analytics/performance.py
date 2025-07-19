import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies
    """
    
    def __init__(self, benchmark_symbol: str = '^NSEI', risk_free_rate: float = 0.06):
        """
        Initialize performance analyzer
        
        Args:
            benchmark_symbol: Benchmark for comparison (default: Nifty 50)
            risk_free_rate: Risk-free rate for calculations (default: 6%)
        """
        self.benchmark_symbol = benchmark_symbol
        self.risk_free_rate = risk_free_rate
        
    def analyze_strategy_performance(self, 
                                   portfolio_values: pd.Series,
                                   benchmark_values: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Comprehensive performance analysis of a strategy
        
        Args:
            portfolio_values: Time series of portfolio values
            benchmark_values: Time series of benchmark values (optional)
            
        Returns:
            Dictionary of performance metrics
        """
        returns = calculate_returns(portfolio_values)
        
        # Basic return metrics
        total_return = calculate_total_return(portfolio_values)
        cagr = calculate_cagr(portfolio_values)
        
        # Risk metrics
        volatility = calculate_annualized_volatility(returns)
        max_dd = calculate_max_drawdown(portfolio_values)
        var_95 = calculate_var(returns, confidence_level=0.95)
        cvar_95 = calculate_cvar(returns, confidence_level=0.95)
        
        # Risk-adjusted returns
        sharpe = calculate_sharpe_ratio(returns, self.risk_free_rate)
        sortino = calculate_sortino_ratio(returns, self.risk_free_rate)
        calmar = calculate_calmar_ratio(returns, max_dd)
        
        # Win/Loss metrics
        win_rate = self._calculate_win_rate(returns)
        avg_win, avg_loss = self._calculate_avg_win_loss(returns)
        profit_factor = self._calculate_profit_factor(returns)
        
        # Time-based metrics
        best_month = returns.resample('M').sum().max() if len(returns) > 30 else returns.max()
        worst_month = returns.resample('M').sum().min() if len(returns) > 30 else returns.min()
        
        metrics = {
            # Return metrics
            'total_return': total_return,
            'cagr': cagr,
            'annualized_volatility': volatility,
            
            # Risk metrics
            'max_drawdown': max_dd,
            'var_95': var_95,
            'cvar_95': cvar_95,
            
            # Risk-adjusted metrics
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            
            # Win/Loss metrics
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            
            # Time-based metrics
            'best_month': best_month,
            'worst_month': worst_month,
            
            # Additional metrics
            'total_periods': len(portfolio_values),
            'positive_periods': len(returns[returns > 0]),
            'negative_periods': len(returns[returns < 0])
        }
        
        # Add benchmark comparison if provided
        if benchmark_values is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_values)
            metrics.update(benchmark_metrics)
        
        return metrics
    
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate percentage of positive return periods"""
        if len(returns) == 0:
            return 0.0
        return len(returns[returns > 0]) / len(returns)
    
    def _calculate_avg_win_loss(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate average win and loss amounts"""
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0
        
        return avg_win, avg_loss
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor (total wins / total losses)"""
        total_wins = returns[returns > 0].sum()
        total_losses = abs(returns[returns < 0].sum())
        
        if total_losses == 0:
            return float('inf') if total_wins > 0 else 1.0
        
        return total_wins / total_losses
    
    def _calculate_benchmark_metrics(self, 
                                   strategy_returns: pd.Series,
                                   benchmark_values: pd.Series) -> Dict[str, float]:
        """Calculate metrics relative to benchmark"""
        benchmark_returns = calculate_returns(benchmark_values)
        
        # Align returns
        aligned_strategy, aligned_benchmark = strategy_returns.align(benchmark_returns, join='inner')
        
        if len(aligned_strategy) == 0:
            return {}
        
        # Calculate relative metrics
        beta = self._calculate_beta(aligned_strategy, aligned_benchmark)
        alpha = self._calculate_alpha(aligned_strategy, aligned_benchmark, beta)
        correlation = aligned_strategy.corr(aligned_benchmark)
        tracking_error = (aligned_strategy - aligned_benchmark).std() * np.sqrt(252)
        information_ratio = calculate_information_ratio(aligned_strategy, aligned_benchmark)
        
        return {
            'beta': beta,
            'alpha': alpha,
            'correlation': correlation,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }
    
    def _calculate_beta(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta (systematic risk)"""
        if len(strategy_returns) < 2 or benchmark_returns.var() == 0:
            return 0.0
        
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = benchmark_returns.var()
        
        return covariance / benchmark_variance
    
    def _calculate_alpha(self, strategy_returns: pd.Series, 
                        benchmark_returns: pd.Series, beta: float) -> float:
        """Calculate alpha (excess return over expected return)"""
        if len(strategy_returns) == 0:
            return 0.0
        
        strategy_mean = strategy_returns.mean() * 252  # Annualized
        benchmark_mean = benchmark_returns.mean() * 252  # Annualized
        
        expected_return = self.risk_free_rate + beta * (benchmark_mean - self.risk_free_rate)
        alpha = strategy_mean - expected_return
        
        return alpha

# Performance calculation functions
def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    Calculate returns from price series
    
    Args:
        prices: Time series of prices
        method: 'simple' or 'log' returns
        
    Returns:
        Time series of returns
    """
    if method == 'log':
        return np.log(prices / prices.shift(1)).dropna()
    else:
        return (prices / prices.shift(1) - 1).dropna()

def calculate_total_return(prices: pd.Series) -> float:
    """Calculate total return over the period"""
    if len(prices) < 2:
        return 0.0
    return (prices.iloc[-1] / prices.iloc[0]) - 1

def calculate_cagr(prices: pd.Series) -> float:
    """Calculate Compound Annual Growth Rate"""
    if len(prices) < 2:
        return 0.0
    
    total_return = calculate_total_return(prices)
    start_date = prices.index[0]
    end_date = prices.index[-1]
    years = (end_date - start_date).days / 365.25
    
    if years <= 0:
        return 0.0
    
    return (1 + total_return) ** (1 / years) - 1

def calculate_annualized_volatility(returns: pd.Series) -> float:
    """Calculate annualized volatility"""
    if len(returns) < 2:
        return 0.0
    return returns.std() * np.sqrt(252)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252
    if excess_returns.std() == 0:
        return 0.0
    
    return (excess_returns.mean() * 252) / (returns.std() * np.sqrt(252))

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
    """Calculate Sortino ratio (downside deviation)"""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    downside_std = downside_returns.std() * np.sqrt(252)
    return (excess_returns.mean() * 252) / downside_std

def calculate_calmar_ratio(returns: pd.Series, max_drawdown: float) -> float:
    """Calculate Calmar ratio (CAGR / Max Drawdown)"""
    if max_drawdown == 0:
        return float('inf') if returns.mean() > 0 else 0.0
    
    annual_return = returns.mean() * 252
    return annual_return / abs(max_drawdown)

def calculate_information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate Information ratio (active return / tracking error)"""
    active_returns = strategy_returns - benchmark_returns
    
    if len(active_returns) < 2 or active_returns.std() == 0:
        return 0.0
    
    return (active_returns.mean() * 252) / (active_returns.std() * np.sqrt(252))

def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown"""
    if len(prices) < 2:
        return 0.0
    
    cumulative = (1 + calculate_returns(prices)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()

def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Value at Risk"""
    if len(returns) < 2:
        return 0.0
    
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)"""
    if len(returns) < 2:
        return 0.0
    
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()
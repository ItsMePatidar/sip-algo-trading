import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import math

class RiskMetrics:
    """Calculate various risk metrics for portfolio performance"""
    
    @staticmethod
    def volatility(returns: List[float], annualize: bool = True) -> float:
        """Calculate volatility (standard deviation of returns)"""
        if not returns or len(returns) < 2:
            return 0.0
        
        vol = np.std(returns, ddof=1)
        if annualize:
            vol *= np.sqrt(252)  # Assuming 252 trading days per year
        
        return vol
    
    @staticmethod
    def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        excess_returns = [r - risk_free_rate/252 for r in returns]  # Daily risk-free rate
        avg_excess_return = np.mean(excess_returns)
        vol = np.std(excess_returns, ddof=1)
        
        if vol == 0:
            return 0.0
        
        return (avg_excess_return / vol) * np.sqrt(252)
    
    @staticmethod
    def sortino_ratio(returns: List[float], risk_free_rate: float = 0.05, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio (uses downside deviation instead of total volatility)"""
        if not returns or len(returns) < 2:
            return 0.0
        
        excess_returns = [r - risk_free_rate/252 for r in returns]
        avg_excess_return = np.mean(excess_returns)
        
        # Calculate downside deviation
        downside_returns = [min(0, r - target_return/252) for r in returns]
        downside_deviation = np.sqrt(np.mean([r**2 for r in downside_returns]))
        
        if downside_deviation == 0:
            return 0.0
        
        return (avg_excess_return / downside_deviation) * np.sqrt(252)
    
    @staticmethod
    def maximum_drawdown(values: List[float]) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown
        
        Returns:
            (max_drawdown, start_index, end_index)
        """
        if not values or len(values) < 2:
            return 0.0, 0, 0
        
        peak = values[0]
        max_drawdown = 0.0
        start_idx = 0
        end_idx = 0
        temp_start = 0
        
        for i, value in enumerate(values):
            if value > peak:
                peak = value
                temp_start = i
            
            drawdown = (peak - value) / peak if peak > 0 else 0
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                start_idx = temp_start
                end_idx = i
        
        return max_drawdown, start_idx, end_idx
    
    @staticmethod
    def calmar_ratio(returns: List[float], values: List[float]) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if not returns or not values:
            return 0.0
        
        annual_return = (np.mean(returns) * 252) if returns else 0.0
        max_dd, _, _ = RiskMetrics.maximum_drawdown(values)
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / max_dd
    
    @staticmethod
    def value_at_risk(returns: List[float], confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR) at given confidence level"""
        if not returns:
            return 0.0
        
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def expected_shortfall(returns: List[float], confidence_level: float = 0.05) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        if not returns:
            return 0.0
        
        var = RiskMetrics.value_at_risk(returns, confidence_level)
        shortfall_returns = [r for r in returns if r <= var]
        
        return np.mean(shortfall_returns) if shortfall_returns else 0.0
    
    @staticmethod
    def beta(portfolio_returns: List[float], benchmark_returns: List[float]) -> float:
        """Calculate portfolio beta relative to benchmark"""
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return 1.0
        
        covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
        benchmark_variance = np.var(benchmark_returns, ddof=1)
        
        if benchmark_variance == 0:
            return 1.0
        
        return covariance / benchmark_variance
    
    @staticmethod
    def alpha(portfolio_returns: List[float], benchmark_returns: List[float], risk_free_rate: float = 0.05) -> float:
        """Calculate portfolio alpha"""
        if len(portfolio_returns) != len(benchmark_returns):
            return 0.0
        
        portfolio_return = np.mean(portfolio_returns) * 252
        benchmark_return = np.mean(benchmark_returns) * 252
        beta = RiskMetrics.beta(portfolio_returns, benchmark_returns)
        
        return portfolio_return - risk_free_rate - beta * (benchmark_return - risk_free_rate)

class PerformanceMetrics:
    """Calculate comprehensive performance metrics"""
    
    def __init__(self, portfolio_values: List[Tuple[datetime, float]], 
                 benchmark_returns: List[float] = None,
                 risk_free_rate: float = 0.05):
        
        self.portfolio_values = portfolio_values
        self.benchmark_returns = benchmark_returns or []
        self.risk_free_rate = risk_free_rate
        
        # Calculate returns
        self.portfolio_returns = self._calculate_returns()
        
        # Initialize risk metrics calculator
        self.risk_metrics = RiskMetrics()
    
    def _calculate_returns(self) -> List[float]:
        """Calculate daily returns from portfolio values"""
        if len(self.portfolio_values) < 2:
            return []
        
        returns = []
        for i in range(1, len(self.portfolio_values)):
            prev_value = self.portfolio_values[i-1][1]
            curr_value = self.portfolio_values[i][1]
            
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)
        
        return returns
    
    def total_return(self) -> float:
        """Calculate total return percentage"""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        start_value = self.portfolio_values[0][1]
        end_value = self.portfolio_values[-1][1]
        
        if start_value <= 0:
            return 0.0
        
        return ((end_value - start_value) / start_value) * 100
    
    def annualized_return(self) -> float:
        """Calculate annualized return"""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        start_date = self.portfolio_values[0][0]
        end_date = self.portfolio_values[-1][0]
        days = (end_date - start_date).days
        
        if days <= 0:
            return 0.0
        
        start_value = self.portfolio_values[0][1]
        end_value = self.portfolio_values[-1][1]
        
        if start_value <= 0:
            return 0.0
        
        years = days / 365.25
        return (((end_value / start_value) ** (1/years)) - 1) * 100
    
    def volatility(self, annualized: bool = True) -> float:
        """Calculate portfolio volatility"""
        return self.risk_metrics.volatility(self.portfolio_returns, annualized) * 100
    
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        return self.risk_metrics.sharpe_ratio(self.portfolio_returns, self.risk_free_rate)
    
    def sortino_ratio(self) -> float:
        """Calculate Sortino ratio"""
        return self.risk_metrics.sortino_ratio(self.portfolio_returns, self.risk_free_rate)
    
    def maximum_drawdown(self) -> Tuple[float, datetime, datetime]:
        """Calculate maximum drawdown with dates"""
        values = [v[1] for v in self.portfolio_values]
        max_dd, start_idx, end_idx = self.risk_metrics.maximum_drawdown(values)
        
        start_date = self.portfolio_values[start_idx][0] if start_idx < len(self.portfolio_values) else None
        end_date = self.portfolio_values[end_idx][0] if end_idx < len(self.portfolio_values) else None
        
        return max_dd * 100, start_date, end_date
    
    def calmar_ratio(self) -> float:
        """Calculate Calmar ratio"""
        values = [v[1] for v in self.portfolio_values]
        return self.risk_metrics.calmar_ratio(self.portfolio_returns, values)
    
    def value_at_risk(self, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        return self.risk_metrics.value_at_risk(self.portfolio_returns, confidence_level) * 100
    
    def expected_shortfall(self, confidence_level: float = 0.05) -> float:
        """Calculate Expected Shortfall"""
        return self.risk_metrics.expected_shortfall(self.portfolio_returns, confidence_level) * 100
    
    def beta(self) -> float:
        """Calculate portfolio beta"""
        if not self.benchmark_returns:
            return 1.0
        
        return self.risk_metrics.beta(self.portfolio_returns, self.benchmark_returns)
    
    def alpha(self) -> float:
        """Calculate portfolio alpha"""
        if not self.benchmark_returns:
            return 0.0
        
        return self.risk_metrics.alpha(self.portfolio_returns, self.benchmark_returns, self.risk_free_rate) * 100
    
    def information_ratio(self) -> float:
        """Calculate information ratio"""
        if not self.benchmark_returns or len(self.portfolio_returns) != len(self.benchmark_returns):
            return 0.0
        
        excess_returns = [p - b for p, b in zip(self.portfolio_returns, self.benchmark_returns)]
        
        if not excess_returns:
            return 0.0
        
        mean_excess = np.mean(excess_returns)
        tracking_error = np.std(excess_returns, ddof=1)
        
        if tracking_error == 0:
            return 0.0
        
        return (mean_excess / tracking_error) * np.sqrt(252)
    
    def win_rate(self) -> float:
        """Calculate percentage of positive return days"""
        if not self.portfolio_returns:
            return 0.0
        
        positive_days = sum(1 for r in self.portfolio_returns if r > 0)
        return (positive_days / len(self.portfolio_returns)) * 100
    
    def average_win_loss_ratio(self) -> float:
        """Calculate ratio of average winning day to average losing day"""
        if not self.portfolio_returns:
            return 0.0
        
        wins = [r for r in self.portfolio_returns if r > 0]
        losses = [r for r in self.portfolio_returns if r < 0]
        
        if not wins or not losses:
            return 0.0
        
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        return avg_win / avg_loss if avg_loss > 0 else 0.0

@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    
    # Basic metrics
    total_return: float
    annualized_return: float
    volatility: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk metrics
    maximum_drawdown: float
    max_drawdown_start: datetime
    max_drawdown_end: datetime
    value_at_risk_5: float
    expected_shortfall_5: float
    
    # Benchmark comparison
    beta: float
    alpha: float
    information_ratio: float
    
    # Trading metrics
    win_rate: float
    avg_win_loss_ratio: float
    
    # Portfolio statistics
    start_date: datetime
    end_date: datetime
    total_days: int
    trading_days: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            'total_return_pct': self.total_return,
            'annualized_return_pct': self.annualized_return,
            'volatility_pct': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'maximum_drawdown_pct': self.maximum_drawdown,
            'max_drawdown_start': self.max_drawdown_start.isoformat() if self.max_drawdown_start else None,
            'max_drawdown_end': self.max_drawdown_end.isoformat() if self.max_drawdown_end else None,
            'value_at_risk_5_pct': self.value_at_risk_5,
            'expected_shortfall_5_pct': self.expected_shortfall_5,
            'beta': self.beta,
            'alpha_pct': self.alpha,
            'information_ratio': self.information_ratio,
            'win_rate_pct': self.win_rate,
            'avg_win_loss_ratio': self.avg_win_loss_ratio,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'total_days': self.total_days,
            'trading_days': self.trading_days
        }
    
    def __str__(self) -> str:
        """String representation of the report"""
        return f"""
Performance Report
==================
Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')} ({self.trading_days} trading days)

Returns:
  Total Return: {self.total_return:.2f}%
  Annualized Return: {self.annualized_return:.2f}%
  Volatility: {self.volatility:.2f}%

Risk-Adjusted Returns:
  Sharpe Ratio: {self.sharpe_ratio:.3f}
  Sortino Ratio: {self.sortino_ratio:.3f}
  Calmar Ratio: {self.calmar_ratio:.3f}

Risk Metrics:
  Maximum Drawdown: {self.maximum_drawdown:.2f}%
  Value at Risk (5%): {self.value_at_risk_5:.2f}%
  Expected Shortfall (5%): {self.expected_shortfall_5:.2f}%

Benchmark Comparison:
  Beta: {self.beta:.3f}
  Alpha: {self.alpha:.2f}%
  Information Ratio: {self.information_ratio:.3f}

Trading Statistics:
  Win Rate: {self.win_rate:.1f}%
  Avg Win/Loss Ratio: {self.avg_win_loss_ratio:.2f}
        """.strip()
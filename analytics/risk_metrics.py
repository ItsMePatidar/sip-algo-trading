import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class RiskAnalyzer:
    """
    Comprehensive risk analysis for trading strategies
    """
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        """
        Initialize risk analyzer
        
        Args:
            confidence_levels: List of confidence levels for VaR calculations
        """
        self.confidence_levels = confidence_levels
    
    def analyze_portfolio_risk(self, 
                             portfolio_returns: pd.Series,
                             benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Comprehensive risk analysis of portfolio
        
        Args:
            portfolio_returns: Time series of portfolio returns
            benchmark_returns: Time series of benchmark returns (optional)
            
        Returns:
            Dictionary of risk metrics
        """
        risk_metrics = {}
        
        # Basic risk metrics
        risk_metrics['volatility'] = calculate_annualized_volatility(portfolio_returns)
        risk_metrics['downside_deviation'] = calculate_downside_deviation(portfolio_returns)
        
        # VaR and CVaR
        for conf_level in self.confidence_levels:
            var_key = f'var_{int(conf_level * 100)}'
            cvar_key = f'cvar_{int(conf_level * 100)}'
            
            risk_metrics[var_key] = calculate_var(portfolio_returns, conf_level)
            risk_metrics[cvar_key] = calculate_cvar(portfolio_returns, conf_level)
        
        # Skewness and Kurtosis
        risk_metrics['skewness'] = portfolio_returns.skew()
        risk_metrics['kurtosis'] = portfolio_returns.kurtosis()
        
        # Rolling metrics
        rolling_vol = calculate_rolling_volatility(portfolio_returns, window=30)
        risk_metrics['avg_rolling_vol'] = rolling_vol.mean()
        risk_metrics['max_rolling_vol'] = rolling_vol.max()
        risk_metrics['min_rolling_vol'] = rolling_vol.min()
        
        # Maximum consecutive losses
        risk_metrics['max_consecutive_losses'] = self._calculate_max_consecutive_losses(portfolio_returns)
        
        # Benchmark comparison (if provided)
        if benchmark_returns is not None:
            benchmark_metrics = self._calculate_benchmark_risk_metrics(portfolio_returns, benchmark_returns)
            risk_metrics.update(benchmark_metrics)
        
        return risk_metrics
    
    def _calculate_max_consecutive_losses(self, returns: pd.Series) -> int:
        """Calculate maximum number of consecutive negative returns"""
        if len(returns) == 0:
            return 0
        
        consecutive_losses = 0
        max_consecutive = 0
        
        for ret in returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return max_consecutive
    
    def _calculate_benchmark_risk_metrics(self, 
                                        portfolio_returns: pd.Series,
                                        benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics relative to benchmark"""
        # Align returns
        aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
        
        if len(aligned_portfolio) == 0:
            return {}
        
        metrics = {}
        
        # Beta and correlation
        metrics['beta'] = calculate_beta(aligned_portfolio, aligned_benchmark)
        metrics['correlation'] = calculate_correlation(aligned_portfolio, aligned_benchmark)
        
        # Tracking error
        metrics['tracking_error'] = calculate_tracking_error(aligned_portfolio, aligned_benchmark)
        
        # Upside/Downside capture
        metrics['upside_capture'] = calculate_upside_capture(aligned_portfolio, aligned_benchmark)
        metrics['downside_capture'] = calculate_downside_capture(aligned_portfolio, aligned_benchmark)
        
        return metrics
    
    def stress_test(self, 
                   portfolio_returns: pd.Series,
                   stress_scenarios: Optional[Dict[str, Dict]] = None) -> Dict[str, float]:
        """
        Perform stress testing on portfolio
        
        Args:
            portfolio_returns: Time series of portfolio returns
            stress_scenarios: Custom stress scenarios (optional)
            
        Returns:
            Dictionary of stress test results
        """
        if stress_scenarios is None:
            stress_scenarios = self._get_default_stress_scenarios()
        
        results = {}
        
        for scenario_name, scenario in stress_scenarios.items():
            scenario_result = self._apply_stress_scenario(portfolio_returns, scenario)
            results[scenario_name] = scenario_result
        
        return results
    
    def _get_default_stress_scenarios(self) -> Dict[str, Dict]:
        """Get default stress test scenarios"""
        return {
            'market_crash': {
                'type': 'shock',
                'magnitude': -0.20,  # 20% drop
                'description': '20% market crash'
            },
            'high_volatility': {
                'type': 'volatility_shock',
                'multiplier': 2.0,  # Double volatility
                'description': 'Volatility doubles'
            },
            'interest_rate_shock': {
                'type': 'rate_shock',
                'rate_change': 0.02,  # 2% increase
                'description': '2% interest rate increase'
            }
        }
    
    def _apply_stress_scenario(self, returns: pd.Series, scenario: Dict) -> float:
        """Apply stress scenario to returns"""
        if scenario['type'] == 'shock':
            # Apply direct shock to portfolio
            stressed_portfolio_value = (1 + returns.sum()) * (1 + scenario['magnitude'])
            return (stressed_portfolio_value - 1) - returns.sum()
        
        elif scenario['type'] == 'volatility_shock':
            # Scale returns by volatility multiplier
            vol_multiplier = scenario['multiplier']
            mean_return = returns.mean()
            stressed_returns = mean_return + (returns - mean_return) * vol_multiplier
            return stressed_returns.sum() - returns.sum()
        
        else:
            # Default: return original performance
            return 0.0

# Risk calculation functions
def calculate_downside_deviation(returns: pd.Series, target_return: float = 0.0) -> float:
    """Calculate downside deviation"""
    downside_returns = returns[returns < target_return] - target_return
    if len(downside_returns) == 0:
        return 0.0
    return np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)

def calculate_upside_capture(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate upside capture ratio"""
    # Align returns
    aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
    
    # Get periods where benchmark was positive
    up_periods = aligned_benchmark > 0
    
    if up_periods.sum() == 0:
        return 0.0
    
    portfolio_up = aligned_portfolio[up_periods].mean()
    benchmark_up = aligned_benchmark[up_periods].mean()
    
    if benchmark_up == 0:
        return 0.0
    
    return portfolio_up / benchmark_up

def calculate_downside_capture(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate downside capture ratio"""
    # Align returns
    aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
    
    # Get periods where benchmark was negative
    down_periods = aligned_benchmark < 0
    
    if down_periods.sum() == 0:
        return 0.0
    
    portfolio_down = aligned_portfolio[down_periods].mean()
    benchmark_down = aligned_benchmark[down_periods].mean()
    
    if benchmark_down == 0:
        return 0.0
    
    return portfolio_down / benchmark_down

def calculate_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate portfolio beta"""
    # Align returns
    aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
    
    if len(aligned_portfolio) < 2 or aligned_benchmark.var() == 0:
        return 0.0
    
    covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
    benchmark_variance = aligned_benchmark.var()
    
    return covariance / benchmark_variance

def calculate_alpha(portfolio_returns: pd.Series, 
                   benchmark_returns: pd.Series,
                   risk_free_rate: float = 0.06) -> float:
    """Calculate portfolio alpha"""
    beta = calculate_beta(portfolio_returns, benchmark_returns)
    
    # Align returns
    aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
    
    if len(aligned_portfolio) == 0:
        return 0.0
    
    portfolio_return = aligned_portfolio.mean() * 252
    benchmark_return = aligned_benchmark.mean() * 252
    
    expected_return = risk_free_rate + beta * (benchmark_return - risk_free_rate)
    
    return portfolio_return - expected_return

def calculate_tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate tracking error"""
    # Align returns
    aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
    
    if len(aligned_portfolio) < 2:
        return 0.0
    
    active_returns = aligned_portfolio - aligned_benchmark
    return active_returns.std() * np.sqrt(252)

def calculate_correlation(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate correlation with benchmark"""
    # Align returns
    aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
    
    if len(aligned_portfolio) < 2:
        return 0.0
    
    return aligned_portfolio.corr(aligned_benchmark)

def calculate_rolling_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    """Calculate rolling volatility"""
    return returns.rolling(window=window).std() * np.sqrt(252)

def calculate_rolling_sharpe(returns: pd.Series, 
                           window: int = 30, 
                           risk_free_rate: float = 0.06) -> pd.Series:
    """Calculate rolling Sharpe ratio"""
    excess_returns = returns - risk_free_rate / 252
    rolling_mean = excess_returns.rolling(window=window).mean() * 252
    rolling_std = returns.rolling(window=window).std() * np.sqrt(252)
    
    return rolling_mean / rolling_std

def stress_test_portfolio(returns: pd.Series, scenarios: Dict[str, float]) -> Dict[str, float]:
    """
    Stress test portfolio against various scenarios
    
    Args:
        returns: Portfolio returns
        scenarios: Dictionary of stress scenarios {name: shock_magnitude}
        
    Returns:
        Dictionary of stress test results
    """
    results = {}
    base_value = (1 + returns).prod()
    
    for scenario_name, shock in scenarios.items():
        stressed_value = base_value * (1 + shock)
        stress_impact = (stressed_value / base_value) - 1
        results[scenario_name] = stress_impact
    
    return results
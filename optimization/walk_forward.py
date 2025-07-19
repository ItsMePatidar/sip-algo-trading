# optimization/walk_forward.py - Walk-Forward Analysis Implementation

"""
Walk-Forward Analysis for SIP Strategy Optimization

Walk-forward analysis is a robust method for testing trading strategies that:
1. Divides historical data into multiple periods
2. Optimizes parameters on training data
3. Tests on out-of-sample data
4. Rolls forward through time
5. Provides realistic performance estimates

This helps prevent overfitting and gives more realistic expectations
of strategy performance in live trading.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import json

from strategies.base_strategy import BaseSIPStrategy, StrategyConfig
from strategies import create_strategy
from backtesting.backtest_engine import BacktestEngine, BacktestResult
from optimization.grid_search import GridSearchOptimizer, OptimizationResult

logger = logging.getLogger(__name__)

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis"""
    training_period_months: int = 24  # Months of data for training
    testing_period_months: int = 6    # Months of data for testing
    step_size_months: int = 3         # How much to advance each step
    min_training_periods: int = 12    # Minimum periods needed for training
    
    # Optimization settings
    optimization_metric: str = 'sharpe_ratio'  # Metric to optimize
    max_workers: int = 4              # Parallel processing workers
    
    # Validation settings
    require_positive_returns: bool = True
    min_trades_per_period: int = 5
    max_drawdown_threshold: float = 0.3  # 30%

@dataclass
class WalkForwardPeriod:
    """Represents a single walk-forward period"""
    period_id: int
    training_start: datetime
    training_end: datetime
    testing_start: datetime
    testing_end: datetime
    
    # Optimization results
    best_parameters: Dict[str, Any] = field(default_factory=dict)
    optimization_result: OptimizationResult = None
    
    # Testing results
    testing_result: BacktestResult = None
    
    # Performance metrics
    training_performance: Dict[str, float] = field(default_factory=dict)
    testing_performance: Dict[str, float] = field(default_factory=dict)

@dataclass
class WalkForwardResult:
    """Results of walk-forward analysis"""
    config: WalkForwardConfig
    strategy_name: str
    symbols: List[str]
    periods: List[WalkForwardPeriod]
    
    # Aggregate statistics
    total_periods: int = 0
    successful_periods: int = 0
    
    # Performance metrics
    aggregate_performance: Dict[str, float] = field(default_factory=dict)
    period_performances: List[Dict[str, float]] = field(default_factory=list)
    
    # Parameter stability
    parameter_stability: Dict[str, float] = field(default_factory=dict)
    parameter_frequency: Dict[str, Dict[str, int]] = field(default_factory=dict)

class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis implementation for SIP strategies
    """
    
    def __init__(self, 
                 backtest_engine: BacktestEngine,
                 grid_optimizer: GridSearchOptimizer,
                 config: WalkForwardConfig = None):
        """
        Initialize walk-forward analyzer
        
        Args:
            backtest_engine: Backtesting engine
            grid_optimizer: Grid search optimizer
            config: Walk-forward configuration
        """
        self.backtest_engine = backtest_engine
        self.grid_optimizer = grid_optimizer
        self.config = config or WalkForwardConfig()
        
    def run_walk_forward_analysis(self,
                                  strategy_name: str,
                                  strategy_config: StrategyConfig,
                                  parameter_grid: Dict[str, List[Any]],
                                  market_data: Dict[str, pd.DataFrame],
                                  start_date: datetime = None,
                                  end_date: datetime = None) -> WalkForwardResult:
        """
        Run complete walk-forward analysis
        
        Args:
            strategy_name: Name of strategy to test
            strategy_config: Base strategy configuration
            parameter_grid: Parameter grid for optimization
            market_data: Historical market data
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            Walk-forward analysis results
        """
        logger.info(f"Starting walk-forward analysis for {strategy_name}")
        
        # Determine date range
        if start_date is None or end_date is None:
            start_date, end_date = self._determine_date_range(market_data)
        
        # Generate walk-forward periods
        periods = self._generate_periods(start_date, end_date)
        logger.info(f"Generated {len(periods)} walk-forward periods")
        
        # Initialize result
        result = WalkForwardResult(
            config=self.config,
            strategy_name=strategy_name,
            symbols=strategy_config.symbols,
            periods=periods,
            total_periods=len(periods)
        )
        
        # Process each period
        successful_periods = 0
        
        for i, period in enumerate(periods):
            logger.info(f"Processing period {i+1}/{len(periods)}: "
                       f"{period.training_start.date()} to {period.testing_end.date()}")
            
            try:
                # Run optimization on training data
                training_data = self._extract_period_data(
                    market_data, period.training_start, period.training_end
                )
                
                if not self._validate_training_data(training_data):
                    logger.warning(f"Insufficient training data for period {i+1}")
                    continue
                
                # Optimize parameters
                optimization_result = self._optimize_period(
                    strategy_name, strategy_config, parameter_grid, 
                    training_data, period
                )
                
                if optimization_result is None:
                    logger.warning(f"Optimization failed for period {i+1}")
                    continue
                
                period.optimization_result = optimization_result
                period.best_parameters = optimization_result.best_parameters
                period.training_performance = optimization_result.best_performance
                
                # Test on out-of-sample data
                testing_data = self._extract_period_data(
                    market_data, period.testing_start, period.testing_end
                )
                
                if not self._validate_testing_data(testing_data):
                    logger.warning(f"Insufficient testing data for period {i+1}")
                    continue
                
                # Run backtest with optimized parameters
                testing_result = self._test_period(
                    strategy_name, strategy_config, period.best_parameters,
                    testing_data, period
                )
                
                if testing_result is None:
                    logger.warning(f"Testing failed for period {i+1}")
                    continue
                
                period.testing_result = testing_result
                period.testing_performance = testing_result.performance_metrics
                
                successful_periods += 1
                logger.info(f"Period {i+1} completed successfully")
                
            except Exception as e:
                logger.error(f"Error processing period {i+1}: {str(e)}")
                continue
        
        result.successful_periods = successful_periods
        
        # Calculate aggregate statistics
        self._calculate_aggregate_statistics(result)
        
        # Analyze parameter stability
        self._analyze_parameter_stability(result)
        
        logger.info(f"Walk-forward analysis completed: {successful_periods}/{len(periods)} periods successful")
        
        return result
    
    def _generate_periods(self, start_date: datetime, end_date: datetime) -> List[WalkForwardPeriod]:
        """Generate walk-forward periods"""
        periods = []
        period_id = 0
        
        current_start = start_date
        
        while True:
            # Calculate training period
            training_start = current_start
            training_end = training_start + timedelta(days=30 * self.config.training_period_months)
            
            # Calculate testing period
            testing_start = training_end + timedelta(days=1)
            testing_end = testing_start + timedelta(days=30 * self.config.testing_period_months)
            
            # Check if we have enough data
            if testing_end > end_date:
                break
            
            period = WalkForwardPeriod(
                period_id=period_id,
                training_start=training_start,
                training_end=training_end,
                testing_start=testing_start,
                testing_end=testing_end
            )
            
            periods.append(period)
            period_id += 1
            
            # Advance to next period
            current_start += timedelta(days=30 * self.config.step_size_months)
        
        return periods
    
    def _determine_date_range(self, market_data: Dict[str, pd.DataFrame]) -> Tuple[datetime, datetime]:
        """Determine available date range from market data"""
        start_dates = []
        end_dates = []
        
        for symbol, data in market_data.items():
            if not data.empty:
                start_dates.append(data.index[0])
                end_dates.append(data.index[-1])
        
        if not start_dates:
            raise ValueError("No market data available")
        
        # Use the common date range
        start_date = max(start_dates)
        end_date = min(end_dates)
        
        # Ensure we have enough data
        min_required_days = (self.config.training_period_months + self.config.testing_period_months) * 30
        if (end_date - start_date).days < min_required_days:
            raise ValueError(f"Insufficient data: need at least {min_required_days} days")
        
        return start_date, end_date
    
    def _extract_period_data(self, 
                           market_data: Dict[str, pd.DataFrame], 
                           start_date: datetime, 
                           end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Extract data for a specific period"""
        period_data = {}
        
        for symbol, data in market_data.items():
            # Filter data for the period
            mask = (data.index >= start_date) & (data.index <= end_date)
            period_data[symbol] = data[mask].copy()
        
        return period_data
    
    def _validate_training_data(self, training_data: Dict[str, pd.DataFrame]) -> bool:
        """Validate training data quality"""
        for symbol, data in training_data.items():
            if len(data) < self.config.min_training_periods:
                logger.warning(f"Insufficient training data for {symbol}: {len(data)} periods")
                return False
            
            # Check for missing data
            if data.isnull().sum().sum() > len(data) * 0.1:  # More than 10% missing
                logger.warning(f"Too much missing data for {symbol}")
                return False
        
        return True
    
    def _validate_testing_data(self, testing_data: Dict[str, pd.DataFrame]) -> bool:
        """Validate testing data quality"""
        for symbol, data in testing_data.items():
            if len(data) < 5:  # Minimum 5 data points for testing
                logger.warning(f"Insufficient testing data for {symbol}: {len(data)} periods")
                return False
        
        return True
    
    def _optimize_period(self,
                        strategy_name: str,
                        strategy_config: StrategyConfig,
                        parameter_grid: Dict[str, List[Any]],
                        training_data: Dict[str, pd.DataFrame],
                        period: WalkForwardPeriod) -> OptimizationResult:
        """Optimize parameters for a specific period"""
        try:
            # Create temporary config for this period
            temp_config = StrategyConfig(
                name=f"{strategy_config.name}_period_{period.period_id}",
                symbols=strategy_config.symbols,
                base_amount=strategy_config.base_amount,
                frequency=strategy_config.frequency,
                start_date=period.training_start,
                end_date=period.training_end,
                parameters=strategy_config.parameters.copy()
            )
            
            # Run optimization
            result = self.grid_optimizer.optimize(
                strategy_name=strategy_name,
                base_config=temp_config,
                parameter_grid=parameter_grid,
                market_data=training_data,
                optimization_metric=self.config.optimization_metric
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed for period {period.period_id}: {str(e)}")
            return None
    
    def _test_period(self,
                    strategy_name: str,
                    strategy_config: StrategyConfig,
                    best_parameters: Dict[str, Any],
                    testing_data: Dict[str, pd.DataFrame],
                    period: WalkForwardPeriod) -> BacktestResult:
        """Test optimized parameters on out-of-sample data"""
        try:
            # Create config with optimized parameters
            test_config = StrategyConfig(
                name=f"{strategy_config.name}_test_{period.period_id}",
                symbols=strategy_config.symbols,
                base_amount=strategy_config.base_amount,
                frequency=strategy_config.frequency,
                start_date=period.testing_start,
                end_date=period.testing_end,
                parameters=best_parameters
            )
            
            # Create strategy
            strategy = create_strategy(strategy_name, test_config)
            
            # Run backtest
            result = self.backtest_engine.run_backtest(
                strategy=strategy,
                market_data=testing_data,
                start_date=period.testing_start,
                end_date=period.testing_end
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Testing failed for period {period.period_id}: {str(e)}")
            return None
    
    def _calculate_aggregate_statistics(self, result: WalkForwardResult):
        """Calculate aggregate performance statistics"""
        if result.successful_periods == 0:
            return
        
        # Collect all testing performances
        testing_performances = []
        for period in result.periods:
            if period.testing_result is not None:
                testing_performances.append(period.testing_performance)
                result.period_performances.append(period.testing_performance)
        
        if not testing_performances:
            return
        
        # Calculate aggregate metrics
        metrics_to_aggregate = [
            'total_return', 'annualized_return', 'volatility', 
            'sharpe_ratio', 'max_drawdown', 'win_rate'
        ]
        
        for metric in metrics_to_aggregate:
            values = [perf.get(metric, 0) for perf in testing_performances if perf.get(metric) is not None]
            if values:
                result.aggregate_performance[f'{metric}_mean'] = np.mean(values)
                result.aggregate_performance[f'{metric}_std'] = np.std(values)
                result.aggregate_performance[f'{metric}_median'] = np.median(values)
                result.aggregate_performance[f'{metric}_min'] = np.min(values)
                result.aggregate_performance[f'{metric}_max'] = np.max(values)
        
        # Calculate consistency metrics
        returns = [perf.get('total_return', 0) for perf in testing_performances]
        if returns:
            positive_periods = sum(1 for r in returns if r > 0)
            result.aggregate_performance['positive_periods_pct'] = positive_periods / len(returns)
            result.aggregate_performance['periods_analyzed'] = len(returns)
    
    def _analyze_parameter_stability(self, result: WalkForwardResult):
        """Analyze parameter stability across periods"""
        if result.successful_periods == 0:
            return
        
        # Collect parameter values across periods
        parameter_values = {}
        
        for period in result.periods:
            if period.best_parameters:
                for param_name, param_value in period.best_parameters.items():
                    if param_name not in parameter_values:
                        parameter_values[param_name] = []
                    parameter_values[param_name].append(param_value)
        
        # Calculate stability metrics
        for param_name, values in parameter_values.items():
            if len(values) > 1:
                # For numerical parameters
                if all(isinstance(v, (int, float)) for v in values):
                    values_array = np.array(values)
                    cv = np.std(values_array) / np.mean(values_array) if np.mean(values_array) != 0 else 0
                    result.parameter_stability[param_name] = 1.0 - min(cv, 1.0)  # Stability score
                
                # Count frequency of each value
                if param_name not in result.parameter_frequency:
                    result.parameter_frequency[param_name] = {}
                
                for value in values:
                    value_str = str(value)
                    result.parameter_frequency[param_name][value_str] = result.parameter_frequency[param_name].get(value_str, 0) + 1

class WalkForwardReporter:
    """Generate reports from walk-forward analysis results"""
    
    @staticmethod
    def generate_summary_report(result: WalkForwardResult) -> str:
        """Generate a summary report"""
        report = []
        report.append("WALK-FORWARD ANALYSIS SUMMARY")
        report.append("=" * 50)
        report.append(f"Strategy: {result.strategy_name}")
        report.append(f"Symbols: {', '.join(result.symbols)}")
        report.append(f"Total Periods: {result.total_periods}")
        report.append(f"Successful Periods: {result.successful_periods}")
        report.append(f"Success Rate: {result.successful_periods/result.total_periods*100:.1f}%")
        report.append("")
        
        if result.aggregate_performance:
            report.append("AGGREGATE PERFORMANCE METRICS")
            report.append("-" * 30)
            
            key_metrics = ['total_return_mean', 'annualized_return_mean', 'sharpe_ratio_mean', 'max_drawdown_mean']
            for metric in key_metrics:
                if metric in result.aggregate_performance:
                    value = result.aggregate_performance[metric]
                    report.append(f"{metric.replace('_', ' ').title()}: {value:.4f}")
            
            if 'positive_periods_pct' in result.aggregate_performance:
                pct = result.aggregate_performance['positive_periods_pct'] * 100
                report.append(f"Positive Periods: {pct:.1f}%")
            
            report.append("")
        
        if result.parameter_stability:
            report.append("PARAMETER STABILITY")
            report.append("-" * 20)
            for param, stability in result.parameter_stability.items():
                report.append(f"{param}: {stability:.3f}")
            report.append("")
        
        return "\n".join(report)
    
    @staticmethod
    def generate_detailed_report(result: WalkForwardResult) -> str:
        """Generate a detailed report"""
        report = [WalkForwardReporter.generate_summary_report(result)]
        
        report.append("PERIOD-BY-PERIOD RESULTS")
        report.append("=" * 50)
        
        for i, period in enumerate(result.periods):
            if period.testing_result is not None:
                report.append(f"Period {i+1}: {period.testing_start.date()} to {period.testing_end.date()}")
                
                if period.testing_performance:
                    for metric, value in period.testing_performance.items():
                        if isinstance(value, (int, float)):
                            report.append(f"  {metric}: {value:.4f}")
                
                if period.best_parameters:
                    report.append("  Best Parameters:")
                    for param, value in period.best_parameters.items():
                        report.append(f"    {param}: {value}")
                
                report.append("")
        
        return "\n".join(report)
    
    @staticmethod
    def save_results(result: WalkForwardResult, filepath: str):
        """Save results to file"""
        # Convert to serializable format
        data = {
            'config': result.config.__dict__,
            'strategy_name': result.strategy_name,
            'symbols': result.symbols,
            'total_periods': result.total_periods,
            'successful_periods': result.successful_periods,
            'aggregate_performance': result.aggregate_performance,
            'parameter_stability': result.parameter_stability,
            'parameter_frequency': result.parameter_frequency,
            'periods': []
        }
        
        for period in result.periods:
            period_data = {
                'period_id': period.period_id,
                'training_start': period.training_start.isoformat(),
                'training_end': period.training_end.isoformat(),
                'testing_start': period.testing_start.isoformat(),
                'testing_end': period.testing_end.isoformat(),
                'best_parameters': period.best_parameters,
                'training_performance': period.training_performance,
                'testing_performance': period.testing_performance
            }
            data['periods'].append(period_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
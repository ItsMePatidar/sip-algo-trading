import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from strategies.base_strategy import BaseSIPStrategy, StrategyConfig
from strategies import create_strategy, list_available_strategies
from backtesting.backtest_engine import BacktestEngine, BacktestConfig, BacktestResults
from backtesting.metrics import PerformanceMetrics, PerformanceReport

logger = logging.getLogger(__name__)

@dataclass
class StrategyComparison:
    """Results from comparing multiple strategies"""
    strategies: List[str]
    results: Dict[str, BacktestResults]
    comparison_metrics: Dict[str, Dict[str, float]]
    rankings: Dict[str, List[str]]  # Rankings by different metrics
    statistical_tests: Dict[str, Any]

@dataclass
class ParameterSensitivity:
    """Results from parameter sensitivity analysis"""
    strategy_name: str
    parameter_name: str
    parameter_values: List[float]
    performance_metrics: Dict[str, List[float]]
    optimal_value: float
    sensitivity_score: float

@dataclass
class ResearchReport:
    """Comprehensive research report"""
    title: str
    summary: str
    methodology: str
    key_findings: List[str]
    recommendations: List[str]
    strategy_comparisons: Optional[StrategyComparison]
    parameter_analysis: Optional[List[ParameterSensitivity]]
    charts: Dict[str, Any]
    raw_data: Dict[str, Any]
    generated_date: datetime

class StrategyResearcher:
    """
    Comprehensive strategy research and development toolkit
    
    Features:
    - Strategy backtesting and comparison
    - Parameter optimization and sensitivity analysis
    - Performance attribution analysis
    - Risk analysis and scenario testing
    - Automated report generation
    """
    
    def __init__(self, 
                 initial_cash: float = 100000.0,
                 commission_rate: float = 0.001,
                 tax_rate: float = 0.001,
                 risk_free_rate: float = 0.05):
        
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self.risk_free_rate = risk_free_rate
        
        # Research cache
        self.backtest_cache = {}
        self.analysis_cache = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def compare_strategies(self,
                          strategies: List[Tuple[str, Dict[str, Any]]],
                          market_data: Dict[str, pd.DataFrame],
                          start_date: datetime,
                          end_date: datetime,
                          benchmark_data: Optional[pd.DataFrame] = None) -> StrategyComparison:
        """
        Compare multiple strategies on the same dataset
        
        Args:
            strategies: List of (strategy_name, parameters) tuples
            market_data: Historical market data
            start_date: Backtest start date
            end_date: Backtest end date
            benchmark_data: Benchmark data for comparison
            
        Returns:
            StrategyComparison object with detailed results
        """
        logger.info(f"Comparing {len(strategies)} strategies from {start_date} to {end_date}")
        
        # Create backtest configuration
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_cash=self.initial_cash,
            commission_rate=self.commission_rate,
            tax_rate=self.tax_rate,
            risk_free_rate=self.risk_free_rate
        )
        
        results = {}
        comparison_metrics = {}
        
        # Run backtests for each strategy
        for strategy_name, parameters in strategies:
            try:
                logger.info(f"Backtesting strategy: {strategy_name}")
                
                # Create strategy configuration
                strategy_config = StrategyConfig(
                    name=f"{strategy_name}_research",
                    symbols=list(market_data.keys()),
                    base_amount=parameters.get('base_amount', 5000.0),
                    frequency=parameters.get('frequency', 'monthly'),
                    parameters=parameters
                )
                
                # Create and run strategy
                strategy = create_strategy(strategy_name, strategy_config)
                engine = BacktestEngine(config)
                result = engine.run_backtest(strategy, market_data, benchmark_data)
                
                results[strategy_name] = result
                
                # Extract key metrics
                comparison_metrics[strategy_name] = {
                    'total_return': result.performance_report.total_return,
                    'annualized_return': result.performance_report.annualized_return,
                    'volatility': result.performance_report.volatility,
                    'sharpe_ratio': result.performance_report.sharpe_ratio,
                    'sortino_ratio': result.performance_report.sortino_ratio,
                    'calmar_ratio': result.performance_report.calmar_ratio,
                    'maximum_drawdown': result.performance_report.maximum_drawdown,
                    'win_rate': result.performance_report.win_rate,
                    'alpha': result.performance_report.alpha,
                    'beta': result.performance_report.beta
                }
                
                logger.info(f"✓ {strategy_name}: {result.performance_report.total_return:.2f}% return, "
                           f"{result.performance_report.sharpe_ratio:.3f} Sharpe")
                
            except Exception as e:
                logger.error(f"Error backtesting {strategy_name}: {str(e)}")
                continue
        
        # Calculate rankings
        rankings = self._calculate_rankings(comparison_metrics)
        
        # Perform statistical tests
        statistical_tests = self._perform_statistical_tests(results)
        
        return StrategyComparison(
            strategies=list(results.keys()),
            results=results,
            comparison_metrics=comparison_metrics,
            rankings=rankings,
            statistical_tests=statistical_tests
        )
    
    def analyze_parameter_sensitivity(self,
                                    strategy_name: str,
                                    base_parameters: Dict[str, Any],
                                    parameter_ranges: Dict[str, List[float]],
                                    market_data: Dict[str, pd.DataFrame],
                                    start_date: datetime,
                                    end_date: datetime,
                                    metric: str = 'sharpe_ratio') -> List[ParameterSensitivity]:
        """
        Analyze sensitivity of strategy performance to parameter changes
        
        Args:
            strategy_name: Name of strategy to analyze
            base_parameters: Base parameter configuration
            parameter_ranges: Dictionary of parameter names to test values
            market_data: Historical market data
            start_date: Backtest start date
            end_date: Backtest end date
            metric: Performance metric to optimize
            
        Returns:
            List of ParameterSensitivity objects
        """
        logger.info(f"Analyzing parameter sensitivity for {strategy_name}")
        
        sensitivity_results = []
        
        for param_name, param_values in parameter_ranges.items():
            logger.info(f"Testing parameter: {param_name} with {len(param_values)} values")
            
            performance_metrics = {
                'total_return': [],
                'sharpe_ratio': [],
                'sortino_ratio': [],
                'maximum_drawdown': [],
                'volatility': []
            }
            
            for param_value in param_values:
                # Create modified parameters
                test_parameters = base_parameters.copy()
                test_parameters[param_name] = param_value
                
                try:
                    # Run backtest
                    result = self._run_single_backtest(
                        strategy_name, test_parameters, market_data, start_date, end_date
                    )
                    
                    # Record metrics
                    performance_metrics['total_return'].append(result.performance_report.total_return)
                    performance_metrics['sharpe_ratio'].append(result.performance_report.sharpe_ratio)
                    performance_metrics['sortino_ratio'].append(result.performance_report.sortino_ratio)
                    performance_metrics['maximum_drawdown'].append(result.performance_report.maximum_drawdown)
                    performance_metrics['volatility'].append(result.performance_report.volatility)
                    
                except Exception as e:
                    logger.warning(f"Failed backtest for {param_name}={param_value}: {str(e)}")
                    # Fill with zeros for failed backtests
                    for metric_name in performance_metrics:
                        performance_metrics[metric_name].append(0.0)
            
            # Find optimal value and calculate sensitivity
            if performance_metrics[metric]:
                optimal_idx = np.argmax(performance_metrics[metric])
                optimal_value = param_values[optimal_idx]
                
                # Calculate sensitivity score (coefficient of variation)
                metric_values = performance_metrics[metric]
                sensitivity_score = np.std(metric_values) / np.mean(metric_values) if np.mean(metric_values) != 0 else 0
            else:
                optimal_value = param_values[0]
                sensitivity_score = 0.0
            
            sensitivity_results.append(ParameterSensitivity(
                strategy_name=strategy_name,
                parameter_name=param_name,
                parameter_values=param_values,
                performance_metrics=performance_metrics,
                optimal_value=optimal_value,
                sensitivity_score=sensitivity_score
            ))
        
        return sensitivity_results
    
    def optimize_strategy_parameters(self,
                                   strategy_name: str,
                                   parameter_grid: Dict[str, List[float]],
                                   market_data: Dict[str, pd.DataFrame],
                                   start_date: datetime,
                                   end_date: datetime,
                                   optimization_metric: str = 'sharpe_ratio',
                                   max_workers: int = 4) -> Tuple[Dict[str, Any], BacktestResults]:
        """
        Optimize strategy parameters using grid search
        
        Args:
            strategy_name: Name of strategy to optimize
            parameter_grid: Grid of parameters to test
            market_data: Historical market data
            start_date: Backtest start date
            end_date: Backtest end date
            optimization_metric: Metric to optimize
            max_workers: Number of parallel workers
            
        Returns:
            Tuple of (best_parameters, best_results)
        """
        logger.info(f"Optimizing {strategy_name} parameters using grid search")
        
        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        best_score = -float('inf')
        best_parameters = None
        best_results = None
        
        # Use parallel processing for optimization
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_params = {}
            for combination in param_combinations:
                params = dict(zip(param_names, combination))
                future = executor.submit(
                    self._run_single_backtest,
                    strategy_name, params, market_data, start_date, end_date
                )
                future_to_params[future] = params
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                
                try:
                    result = future.result()
                    
                    # Get optimization metric value
                    if optimization_metric == 'sharpe_ratio':
                        score = result.performance_report.sharpe_ratio
                    elif optimization_metric == 'total_return':
                        score = result.performance_report.total_return
                    elif optimization_metric == 'sortino_ratio':
                        score = result.performance_report.sortino_ratio
                    elif optimization_metric == 'calmar_ratio':
                        score = result.performance_report.calmar_ratio
                    else:
                        score = result.performance_report.sharpe_ratio
                    
                    # Check if this is the best so far
                    if score > best_score:
                        best_score = score
                        best_parameters = params
                        best_results = result
                    
                    completed += 1
                    if completed % 10 == 0:
                        logger.info(f"Completed {completed}/{len(param_combinations)} optimizations")
                
                except Exception as e:
                    logger.warning(f"Optimization failed for params {params}: {str(e)}")
        
        logger.info(f"Optimization completed. Best {optimization_metric}: {best_score:.4f}")
        logger.info(f"Best parameters: {best_parameters}")
        
        return best_parameters, best_results
    
    def walk_forward_analysis(self,
                            strategy_name: str,
                            parameters: Dict[str, Any],
                            market_data: Dict[str, pd.DataFrame],
                            start_date: datetime,
                            end_date: datetime,
                            train_period_months: int = 12,
                            test_period_months: int = 3,
                            reoptimize: bool = True) -> Dict[str, Any]:
        """
        Perform walk-forward analysis to test strategy robustness
        
        Args:
            strategy_name: Name of strategy to test
            parameters: Base strategy parameters
            market_data: Historical market data
            start_date: Analysis start date
            end_date: Analysis end date
            train_period_months: Training period length in months
            test_period_months: Testing period length in months
            reoptimize: Whether to reoptimize parameters for each period
            
        Returns:
            Dictionary with walk-forward analysis results
        """
        logger.info(f"Performing walk-forward analysis for {strategy_name}")
        
        results = {
            'periods': [],
            'in_sample_results': [],
            'out_of_sample_results': [],
            'parameters_used': [],
            'cumulative_returns': [],
            'rolling_metrics': {}
        }
        
        current_date = start_date
        period_num = 0
        
        while current_date + timedelta(days=30*train_period_months + 30*test_period_months) <= end_date:
            period_num += 1
            
            # Define training and testing periods
            train_start = current_date
            train_end = current_date + timedelta(days=30*train_period_months)
            test_start = train_end
            test_end = train_end + timedelta(days=30*test_period_months)
            
            logger.info(f"Period {period_num}: Train {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}, "
                       f"Test {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
            
            try:
                # Training phase
                if reoptimize:
                    # Optimize parameters on training data
                    param_grid = self._create_parameter_grid(strategy_name, parameters)
                    best_params, train_result = self.optimize_strategy_parameters(
                        strategy_name, param_grid, market_data, train_start, train_end
                    )
                    used_parameters = best_params
                else:
                    # Use fixed parameters
                    train_result = self._run_single_backtest(
                        strategy_name, parameters, market_data, train_start, train_end
                    )
                    used_parameters = parameters
                
                # Testing phase
                test_result = self._run_single_backtest(
                    strategy_name, used_parameters, market_data, test_start, test_end
                )
                
                # Store results
                results['periods'].append({
                    'period': period_num,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end
                })
                results['in_sample_results'].append(train_result)
                results['out_of_sample_results'].append(test_result)
                results['parameters_used'].append(used_parameters)
                
                logger.info(f"Period {period_num} completed: IS Return {train_result.performance_report.total_return:.2f}%, "
                           f"OOS Return {test_result.performance_report.total_return:.2f}%")
                
            except Exception as e:
                logger.error(f"Error in period {period_num}: {str(e)}")
                continue
            
            # Move to next period
            current_date = test_start
        
        # Calculate aggregate metrics
        results['summary'] = self._calculate_walk_forward_summary(results)
        
        return results
    
    def generate_research_report(self,
                               title: str,
                               strategy_comparison: Optional[StrategyComparison] = None,
                               parameter_analysis: Optional[List[ParameterSensitivity]] = None,
                               walk_forward_results: Optional[Dict[str, Any]] = None,
                               custom_analysis: Optional[Dict[str, Any]] = None) -> ResearchReport:
        """
        Generate comprehensive research report
        
        Args:
            title: Report title
            strategy_comparison: Strategy comparison results
            parameter_analysis: Parameter sensitivity analysis
            walk_forward_results: Walk-forward analysis results
            custom_analysis: Additional custom analysis
            
        Returns:
            ResearchReport object
        """
        logger.info(f"Generating research report: {title}")
        
        # Generate summary
        summary = self._generate_summary(strategy_comparison, parameter_analysis, walk_forward_results)
        
        # Generate key findings
        key_findings = self._generate_key_findings(strategy_comparison, parameter_analysis, walk_forward_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(strategy_comparison, parameter_analysis)
        
        # Generate charts
        charts = self._generate_charts(strategy_comparison, parameter_analysis, walk_forward_results)
        
        # Compile raw data
        raw_data = {
            'strategy_comparison': strategy_comparison,
            'parameter_analysis': parameter_analysis,
            'walk_forward_results': walk_forward_results,
            'custom_analysis': custom_analysis
        }
        
        return ResearchReport(
            title=title,
            summary=summary,
            methodology="Systematic backtesting and statistical analysis using historical market data",
            key_findings=key_findings,
            recommendations=recommendations,
            strategy_comparisons=strategy_comparison,
            parameter_analysis=parameter_analysis,
            charts=charts,
            raw_data=raw_data,
            generated_date=datetime.now()
        )
    
    def _run_single_backtest(self,
                           strategy_name: str,
                           parameters: Dict[str, Any],
                           market_data: Dict[str, pd.DataFrame],
                           start_date: datetime,
                           end_date: datetime) -> BacktestResults:
        """Run a single backtest with given parameters"""
        
        # Create cache key
        cache_key = f"{strategy_name}_{hash(str(sorted(parameters.items())))}_{start_date}_{end_date}"
        
        if cache_key in self.backtest_cache:
            return self.backtest_cache[cache_key]
        
        # Create strategy configuration
        strategy_config = StrategyConfig(
            name=f"{strategy_name}_test",
            symbols=list(market_data.keys()),
            base_amount=parameters.get('base_amount', 5000.0),
            frequency=parameters.get('frequency', 'monthly'),
            parameters=parameters
        )
        
        # Create backtest configuration
        backtest_config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_cash=self.initial_cash,
            commission_rate=self.commission_rate,
            tax_rate=self.tax_rate,
            risk_free_rate=self.risk_free_rate
        )
        
        # Run backtest
        strategy = create_strategy(strategy_name, strategy_config)
        engine = BacktestEngine(backtest_config)
        result = engine.run_backtest(strategy, market_data)
        
        # Cache result
        self.backtest_cache[cache_key] = result
        
        return result
    
    def _calculate_rankings(self, comparison_metrics: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """Calculate strategy rankings by different metrics"""
        rankings = {}
        
        for metric in ['total_return', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio']:
            if all(metric in metrics for metrics in comparison_metrics.values()):
                strategy_scores = [(strategy, metrics[metric]) for strategy, metrics in comparison_metrics.items()]
                strategy_scores.sort(key=lambda x: x[1], reverse=True)
                rankings[metric] = [strategy for strategy, _ in strategy_scores]
        
        return rankings
    
    def _perform_statistical_tests(self, results: Dict[str, BacktestResults]) -> Dict[str, Any]:
        """Perform statistical tests on strategy returns"""
        tests = {}
        
        # Extract returns for each strategy
        strategy_returns = {}
        for name, result in results.items():
            strategy_returns[name] = result.daily_returns
        
        # Perform pairwise t-tests (simplified)
        try:
            from scipy import stats
            
            strategy_names = list(strategy_returns.keys())
            for i, strategy1 in enumerate(strategy_names):
                for strategy2 in strategy_names[i+1:]:
                    returns1 = strategy_returns[strategy1]
                    returns2 = strategy_returns[strategy2]
                    
                    if len(returns1) == len(returns2) and len(returns1) > 1:
                        t_stat, p_value = stats.ttest_rel(returns1, returns2)
                        tests[f"{strategy1}_vs_{strategy2}"] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
        except ImportError:
            logger.warning("scipy not available for statistical tests")
        
        return tests
    
    def _create_parameter_grid(self, strategy_name: str, base_parameters: Dict[str, Any]) -> Dict[str, List[float]]:
        """Create parameter grid for optimization"""
        # Default parameter grids for different strategies
        if strategy_name == 'value_averaging':
            return {
                'target_growth_rate': [0.08, 0.10, 0.12, 0.14, 0.16],
                'max_investment_multiplier': [1.5, 2.0, 2.5, 3.0],
                'min_investment_multiplier': [0.1, 0.2, 0.3, 0.5]
            }
        elif strategy_name == 'momentum_sip':
            return {
                'momentum_period': [30, 45, 60, 90, 120],
                'momentum_threshold': [0.02, 0.05, 0.08, 0.10],
                'momentum_multiplier': [1.2, 1.5, 1.8, 2.0]
            }
        elif strategy_name == 'volatility_sip':
            return {
                'volatility_period': [20, 30, 45, 60],
                'high_vol_threshold': [0.20, 0.25, 0.30, 0.35],
                'high_vol_multiplier': [1.3, 1.5, 1.8, 2.0]
            }
        else:
            # Default grid
            return {'base_amount': [3000, 4000, 5000, 6000, 7000]}
    
    def _calculate_walk_forward_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for walk-forward analysis"""
        if not results['out_of_sample_results']:
            return {}
        
        oos_returns = [r.performance_report.total_return for r in results['out_of_sample_results']]
        oos_sharpe = [r.performance_report.sharpe_ratio for r in results['out_of_sample_results']]
        
        return {
            'average_oos_return': np.mean(oos_returns),
            'oos_return_std': np.std(oos_returns),
            'average_oos_sharpe': np.mean(oos_sharpe),
            'oos_sharpe_std': np.std(oos_sharpe),
            'positive_periods': sum(1 for r in oos_returns if r > 0),
            'total_periods': len(oos_returns),
            'win_rate': sum(1 for r in oos_returns if r > 0) / len(oos_returns) * 100
        }
    
    def _generate_summary(self, strategy_comparison, parameter_analysis, walk_forward_results) -> str:
        """Generate executive summary"""
        summary_parts = []
        
        if strategy_comparison:
            best_strategy = max(strategy_comparison.comparison_metrics.items(), 
                              key=lambda x: x[1]['sharpe_ratio'])[0]
            summary_parts.append(f"Strategy comparison shows {best_strategy} performed best with "
                                f"{strategy_comparison.comparison_metrics[best_strategy]['total_return']:.2f}% total return.")
        
        if parameter_analysis:
            summary_parts.append(f"Parameter sensitivity analysis conducted on {len(parameter_analysis)} parameters.")
        
        if walk_forward_results:
            wf_summary = walk_forward_results.get('summary', {})
            if wf_summary:
                summary_parts.append(f"Walk-forward analysis shows {wf_summary['win_rate']:.1f}% positive periods "
                                    f"with average return of {wf_summary['average_oos_return']:.2f}%.")
        
        return " ".join(summary_parts) if summary_parts else "Comprehensive analysis completed."
    
    def _generate_key_findings(self, strategy_comparison, parameter_analysis, walk_forward_results) -> List[str]:
        """Generate key findings list"""
        findings = []
        
        if strategy_comparison:
            # Best performing strategy
            best_strategy = max(strategy_comparison.comparison_metrics.items(), 
                              key=lambda x: x[1]['sharpe_ratio'])[0]
            findings.append(f"{best_strategy} strategy achieved the highest risk-adjusted returns")
            
            # Risk analysis
            safest_strategy = min(strategy_comparison.comparison_metrics.items(), 
                                key=lambda x: x[1]['maximum_drawdown'])[0]
            findings.append(f"{safest_strategy} strategy had the lowest maximum drawdown")
        
        if parameter_analysis:
            # Most sensitive parameter
            most_sensitive = max(parameter_analysis, key=lambda x: x.sensitivity_score)
            findings.append(f"{most_sensitive.parameter_name} is the most sensitive parameter for optimization")
        
        if walk_forward_results:
            wf_summary = walk_forward_results.get('summary', {})
            if wf_summary and wf_summary['win_rate'] > 60:
                findings.append("Strategy shows robust out-of-sample performance")
            elif wf_summary:
                findings.append("Strategy performance may be overfitted to historical data")
        
        return findings
    
    def _generate_recommendations(self, strategy_comparison, parameter_analysis) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if strategy_comparison:
            best_strategy = max(strategy_comparison.comparison_metrics.items(), 
                              key=lambda x: x[1]['sharpe_ratio'])[0]
            recommendations.append(f"Consider implementing {best_strategy} strategy for live trading")
            
            # Risk management
            recommendations.append("Implement position sizing based on maximum drawdown analysis")
            recommendations.append("Consider portfolio allocation across multiple strategies")
        
        if parameter_analysis:
            stable_params = [p for p in parameter_analysis if p.sensitivity_score < 0.1]
            if stable_params:
                recommendations.append("Focus optimization on less sensitive parameters for robustness")
            
            recommendations.append("Implement adaptive parameter adjustment based on market conditions")
        
        recommendations.append("Conduct regular walk-forward analysis to validate strategy performance")
        
        return recommendations
    
    def _generate_charts(self, strategy_comparison, parameter_analysis, walk_forward_results) -> Dict[str, Any]:
        """Generate visualization charts"""
        charts = {}
        
        if strategy_comparison:
            charts.update(self._create_comparison_charts(strategy_comparison))
        
        if parameter_analysis:
            charts.update(self._create_sensitivity_charts(parameter_analysis))
        
        if walk_forward_results:
            charts.update(self._create_walk_forward_charts(walk_forward_results))
        
        return charts
    
    def _create_comparison_charts(self, comparison: StrategyComparison) -> Dict[str, Any]:
        """Create strategy comparison charts"""
        charts = {}
        
        # Performance comparison bar chart
        metrics = ['total_return', 'sharpe_ratio', 'maximum_drawdown', 'volatility']
        strategies = list(comparison.comparison_metrics.keys())
        
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=('Total Return (%)', 'Sharpe Ratio', 
                                         'Maximum Drawdown (%)', 'Volatility (%)'))
        
        for i, metric in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            values = [comparison.comparison_metrics[s][metric] for s in strategies]
            
            fig.add_trace(
                go.Bar(x=strategies, y=values, name=metric.replace('_', ' ').title()),
                row=row, col=col
            )
        
        fig.update_layout(height=600, showlegend=False)
        charts['strategy_comparison'] = fig
        
        # Cumulative returns chart
        fig_returns = go.Figure()
        
        for strategy_name, result in comparison.results.items():
            dates = [date for date, _ in result.daily_values]
            values = [value for _, value in result.daily_values]
            
            # Calculate cumulative returns
            initial_value = values[0] if values else 100000
            cumulative_returns = [(v - initial_value) / initial_value * 100 for v in values]
            
            fig_returns.add_trace(
                go.Scatter(x=dates, y=cumulative_returns, mode='lines', 
                          name=strategy_name, line=dict(width=2))
            )
        
        fig_returns.update_layout(
            title='Cumulative Returns Comparison',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            height=400
        )
        charts['cumulative_returns'] = fig_returns
        
        return charts
    
    def _create_sensitivity_charts(self, parameter_analysis: List[ParameterSensitivity]) -> Dict[str, Any]:
        """Create parameter sensitivity charts"""
        charts = {}
        
        for analysis in parameter_analysis:
            # Parameter sensitivity line chart
            fig = go.Figure()
            
            for metric_name, metric_values in analysis.performance_metrics.items():
                if metric_name in ['sharpe_ratio', 'total_return', 'sortino_ratio']:
                    fig.add_trace(
                        go.Scatter(
                            x=analysis.parameter_values,
                            y=metric_values,
                            mode='lines+markers',
                            name=metric_name.replace('_', ' ').title(),
                            line=dict(width=2)
                        )
                    )
            
            # Mark optimal value
            optimal_idx = analysis.parameter_values.index(analysis.optimal_value)
            optimal_sharpe = analysis.performance_metrics['sharpe_ratio'][optimal_idx]
            
            fig.add_trace(
                go.Scatter(
                    x=[analysis.optimal_value],
                    y=[optimal_sharpe],
                    mode='markers',
                    name='Optimal',
                    marker=dict(size=12, color='red', symbol='star')
                )
            )
            
            fig.update_layout(
                title=f'Parameter Sensitivity: {analysis.parameter_name}',
                xaxis_title=analysis.parameter_name,
                yaxis_title='Performance Metric',
                height=400
            )
            
            charts[f'sensitivity_{analysis.parameter_name}'] = fig
        
        return charts
    
    def _create_walk_forward_charts(self, walk_forward_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create walk-forward analysis charts"""
        charts = {}
        
        if 'out_of_sample_results' not in walk_forward_results:
            return charts
        
        # Out-of-sample performance over time
        periods = range(1, len(walk_forward_results['out_of_sample_results']) + 1)
        oos_returns = [r.performance_report.total_return for r in walk_forward_results['out_of_sample_results']]
        oos_sharpe = [r.performance_report.sharpe_ratio for r in walk_forward_results['out_of_sample_results']]
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Out-of-Sample Returns (%)', 'Out-of-Sample Sharpe Ratio'))
        
        fig.add_trace(
            go.Scatter(x=list(periods), y=oos_returns, mode='lines+markers', 
                      name='OOS Returns', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=list(periods), y=oos_sharpe, mode='lines+markers', 
                      name='OOS Sharpe', line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=False)
        fig.update_xaxes(title_text="Period", row=2, col=1)
        
        charts['walk_forward_performance'] = fig
        
        return charts

class StrategyOptimizer:
    """Advanced strategy optimization using multiple algorithms"""
    
    def __init__(self, researcher: StrategyResearcher):
        self.researcher = researcher
    
    def genetic_algorithm_optimization(self,
                                     strategy_name: str,
                                     parameter_bounds: Dict[str, Tuple[float, float]],
                                     market_data: Dict[str, pd.DataFrame],
                                     start_date: datetime,
                                     end_date: datetime,
                                     population_size: int = 20,
                                     generations: int = 10,
                                     mutation_rate: float = 0.1) -> Dict[str, Any]:
        """
        Optimize strategy parameters using genetic algorithm
        
        Args:
            strategy_name: Name of strategy to optimize
            parameter_bounds: Dictionary of parameter names to (min, max) bounds
            market_data: Historical market data
            start_date: Backtest start date
            end_date: Backtest end date
            population_size: Size of population
            generations: Number of generations
            mutation_rate: Mutation rate
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting genetic algorithm optimization for {strategy_name}")
        
        param_names = list(parameter_bounds.keys())
        bounds = list(parameter_bounds.values())
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = []
            for min_val, max_val in bounds:
                individual.append(np.random.uniform(min_val, max_val))
            population.append(individual)
        
        best_fitness = -float('inf')
        best_individual = None
        fitness_history = []
        
        for generation in range(generations):
            logger.info(f"Generation {generation + 1}/{generations}")
            
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                params = dict(zip(param_names, individual))
                
                try:
                    result = self.researcher._run_single_backtest(
                        strategy_name, params, market_data, start_date, end_date
                    )
                    fitness = result.performance_report.sharpe_ratio
                except:
                    fitness = -10  # Penalty for failed backtests
                
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            fitness_history.append(max(fitness_scores))
            
            # Selection, crossover, mutation
            new_population = []
            
            # Keep best individuals (elitism)
            elite_count = population_size // 10
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # Generate offspring
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutate(child1, bounds, mutation_rate)
                child2 = self._mutate(child2, bounds, mutation_rate)
                
                new_population.extend([child1, child2])
            
            population = new_population[:population_size]
        
        # Return best solution
        best_params = dict(zip(param_names, best_individual))
        
        return {
            'best_parameters': best_params,
            'best_fitness': best_fitness,
            'fitness_history': fitness_history,
            'generations': generations
        }
    
    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Tournament selection for genetic algorithm"""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1, parent2, crossover_rate=0.8):
        """Single-point crossover"""
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        crossover_point = np.random.randint(1, len(parent1))
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def _mutate(self, individual, bounds, mutation_rate):
        """Gaussian mutation"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                min_val, max_val = bounds[i]
                # Gaussian mutation with 10% of range as standard deviation
                std_dev = (max_val - min_val) * 0.1
                mutated[i] += np.random.normal(0, std_dev)
                # Ensure within bounds
                mutated[i] = np.clip(mutated[i], min_val, max_val)
        
        return mutated
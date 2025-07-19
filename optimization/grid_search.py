import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import itertools
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import json
import pickle
from pathlib import Path

from strategies.base_strategy import BaseSIPStrategy, StrategyConfig
from strategies import create_strategy
from backtesting.backtest_engine import BacktestEngine, BacktestConfig, BacktestResults
from backtesting.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

@dataclass
class GridSearchResults:
    """Results from grid search optimization"""
    strategy_name: str
    parameter_grid: Dict[str, List[Any]]
    optimization_metric: str
    total_combinations: int
    completed_combinations: int
    best_parameters: Dict[str, Any]
    best_score: float
    best_result: BacktestResults
    all_results: List[Tuple[Dict[str, Any], float, BacktestResults]]
    optimization_time: float
    convergence_data: List[Tuple[int, float]]  # (iteration, best_score_so_far)
    
    def get_parameter_sensitivity(self, parameter_name: str) -> Dict[str, Any]:
        """Get sensitivity analysis for a specific parameter"""
        if parameter_name not in self.parameter_grid:
            raise ValueError(f"Parameter {parameter_name} not in grid")
        
        # Group results by parameter value
        param_performance = {}
        for params, score, _ in self.all_results:
            param_value = params.get(parameter_name)
            if param_value is not None:
                if param_value not in param_performance:
                    param_performance[param_value] = []
                param_performance[param_value].append(score)
        
        # Calculate statistics for each parameter value
        sensitivity_data = {}
        for param_value, scores in param_performance.items():
            sensitivity_data[param_value] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'count': len(scores)
            }
        
        return sensitivity_data
    
    def get_top_results(self, n: int = 10) -> List[Tuple[Dict[str, Any], float, BacktestResults]]:
        """Get top N performing parameter combinations"""
        sorted_results = sorted(self.all_results, key=lambda x: x[1], reverse=True)
        return sorted_results[:n]
    
    def save_results(self, filepath: str):
        """Save optimization results to file"""
        # Create a serializable version
        save_data = {
            'strategy_name': self.strategy_name,
            'parameter_grid': self.parameter_grid,
            'optimization_metric': self.optimization_metric,
            'total_combinations': self.total_combinations,
            'completed_combinations': self.completed_combinations,
            'best_parameters': self.best_parameters,
            'best_score': self.best_score,
            'optimization_time': self.optimization_time,
            'convergence_data': self.convergence_data,
            # Store simplified results (without full backtest objects)
            'simplified_results': [(params, score) for params, score, _ in self.all_results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        logger.info(f"Grid search results saved to {filepath}")

class GridSearchOptimizer:
    """
    Grid Search Parameter Optimization
    
    Systematically tests all combinations of parameters in a predefined grid.
    Provides comprehensive analysis of parameter sensitivity and interactions.
    
    Features:
    - Exhaustive parameter space exploration
    - Parallel processing support
    - Progress tracking and resumption
    - Parameter sensitivity analysis
    - Statistical significance testing
    """
    
    def __init__(self,
                 initial_cash: float = 100000.0,
                 commission_rate: float = 0.001,
                 tax_rate: float = 0.001,
                 risk_free_rate: float = 0.05,
                 n_workers: int = 4,
                 use_multiprocessing: bool = False,
                 cache_results: bool = True,
                 cache_dir: str = "optimization_cache"):
        
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self.risk_free_rate = risk_free_rate
        self.n_workers = n_workers
        self.use_multiprocessing = use_multiprocessing
        self.cache_results = cache_results
        self.cache_dir = Path(cache_dir)
        
        # Create cache directory
        if self.cache_results:
            self.cache_dir.mkdir(exist_ok=True)
        
        # Results cache
        self.results_cache = {}
        
        logger.info(f"GridSearchOptimizer initialized with {n_workers} workers")
    
    def optimize(self,
                strategy_name: str,
                parameter_grid: Dict[str, List[Any]],
                market_data: Dict[str, pd.DataFrame],
                start_date: datetime,
                end_date: datetime,
                optimization_metric: str = 'sharpe_ratio',
                base_parameters: Optional[Dict[str, Any]] = None,
                validation_split: float = 0.0,
                early_stopping_patience: int = None,
                progress_callback: Optional[Callable] = None) -> GridSearchResults:
        """
        Perform grid search optimization
        
        Args:
            strategy_name: Name of strategy to optimize
            parameter_grid: Grid of parameters to search
            market_data: Historical market data
            start_date: Optimization start date
            end_date: Optimization end date
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            base_parameters: Base parameters not in grid
            validation_split: Fraction for validation (0.0 = no validation)
            early_stopping_patience: Stop if no improvement for N iterations
            progress_callback: Optional callback for progress updates
            
        Returns:
            GridSearchResults object
        """
        start_time = datetime.now()
        logger.info(f"Starting grid search optimization for {strategy_name}")
        logger.info(f"Parameter grid: {parameter_grid}")
        logger.info(f"Optimization metric: {optimization_metric}")
        
        # Generate parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        total_combinations = len(param_combinations)
        logger.info(f"Total parameter combinations: {total_combinations}")
        
        # Split data for validation if requested
        if validation_split > 0:
            split_date = start_date + timedelta(days=int((end_date - start_date).days * (1 - validation_split)))
            train_start, train_end = start_date, split_date
            val_start, val_end = split_date, end_date
            logger.info(f"Using validation split: Train {train_start} to {train_end}, Validate {val_start} to {val_end}")
        else:
            train_start, train_end = start_date, end_date
            val_start, val_end = None, None
        
        # Initialize tracking variables
        all_results = []
        best_score = -float('inf')
        best_parameters = None
        best_result = None
        convergence_data = []
        early_stopping_counter = 0
        
        # Process combinations
        if self.n_workers == 1:
            # Single-threaded execution
            for i, combination in enumerate(param_combinations):
                params = dict(zip(param_names, combination))
                result = self._evaluate_parameters(
                    strategy_name, params, base_parameters, market_data,
                    train_start, train_end, optimization_metric
                )
                
                if result is not None:
                    _, score, backtest_result = result
                    all_results.append(result)
                    
                    # Update best
                    if score > best_score:
                        best_score = score
                        best_parameters = params
                        best_result = backtest_result
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                    
                    convergence_data.append((i + 1, best_score))
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(i + 1, total_combinations, best_score, params)
                    
                    # Early stopping
                    if early_stopping_patience and early_stopping_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {i + 1} iterations")
                        break
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Completed {i + 1}/{total_combinations} combinations. Best score: {best_score:.4f}")
        
        else:
            # Multi-threaded/multiprocessing execution
            executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
            
            with executor_class(max_workers=self.n_workers) as executor:
                # Submit all jobs
                future_to_params = {}
                for combination in param_combinations:
                    params = dict(zip(param_names, combination))
                    future = executor.submit(
                        self._evaluate_parameters,
                        strategy_name, params, base_parameters, market_data,
                        train_start, train_end, optimization_metric
                    )
                    future_to_params[future] = params
                
                # Collect results
                completed = 0
                for future in as_completed(future_to_params):
                    params = future_to_params[future]
                    
                    try:
                        result = future.result()
                        if result is not None:
                            _, score, backtest_result = result
                            all_results.append(result)
                            
                            # Update best
                            if score > best_score:
                                best_score = score
                                best_parameters = params
                                best_result = backtest_result
                                early_stopping_counter = 0
                            else:
                                early_stopping_counter += 1
                            
                            completed += 1
                            convergence_data.append((completed, best_score))
                            
                            # Progress callback
                            if progress_callback:
                                progress_callback(completed, total_combinations, best_score, params)
                            
                            if completed % 10 == 0:
                                logger.info(f"Completed {completed}/{total_combinations} combinations. Best score: {best_score:.4f}")
                    
                    except Exception as e:
                        logger.error(f"Error evaluating parameters {params}: {str(e)}")
                        completed += 1
        
        # Validation if requested
        if validation_split > 0 and best_parameters is not None:
            logger.info("Running validation on best parameters...")
            val_result = self._evaluate_parameters(
                strategy_name, best_parameters, base_parameters, market_data,
                val_start, val_end, optimization_metric
            )
            if val_result:
                _, val_score, _ = val_result
                logger.info(f"Validation score: {val_score:.4f}")
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        results = GridSearchResults(
            strategy_name=strategy_name,
            parameter_grid=parameter_grid,
            optimization_metric=optimization_metric,
            total_combinations=total_combinations,
            completed_combinations=len(all_results),
            best_parameters=best_parameters,
            best_score=best_score,
            best_result=best_result,
            all_results=all_results,
            optimization_time=optimization_time,
            convergence_data=convergence_data
        )
        
        logger.info(f"Grid search completed in {optimization_time:.2f} seconds")
        logger.info(f"Best parameters: {best_parameters}")
        logger.info(f"Best {optimization_metric}: {best_score:.4f}")
        
        return results
    
    def _evaluate_parameters(self,
                           strategy_name: str,
                           parameters: Dict[str, Any],
                           base_parameters: Optional[Dict[str, Any]],
                           market_data: Dict[str, pd.DataFrame],
                           start_date: datetime,
                           end_date: datetime,
                           optimization_metric: str) -> Optional[Tuple[Dict[str, Any], float, BacktestResults]]:
        """Evaluate a single parameter combination"""
        
        # Combine parameters
        full_parameters = {}
        if base_parameters:
            full_parameters.update(base_parameters)
        full_parameters.update(parameters)
        
        # Check cache
        cache_key = self._generate_cache_key(strategy_name, full_parameters, start_date, end_date)
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]
        
        try:
            # Create strategy configuration
            strategy_config = StrategyConfig(
                name=f"{strategy_name}_optimization",
                symbols=list(market_data.keys()),
                base_amount=full_parameters.get('base_amount', 5000.0),
                frequency=full_parameters.get('frequency', 'monthly'),
                parameters=full_parameters
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
            
            # Extract optimization metric
            if optimization_metric == 'sharpe_ratio':
                score = result.performance_report.sharpe_ratio
            elif optimization_metric == 'total_return':
                score = result.performance_report.total_return
            elif optimization_metric == 'sortino_ratio':
                score = result.performance_report.sortino_ratio
            elif optimization_metric == 'calmar_ratio':
                score = result.performance_report.calmar_ratio
            elif optimization_metric == 'information_ratio':
                score = result.performance_report.information_ratio
            elif optimization_metric == 'max_drawdown':
                score = -result.performance_report.maximum_drawdown  # Negative because we want to minimize
            else:
                score = result.performance_report.sharpe_ratio  # Default
            
            # Handle invalid scores
            if np.isnan(score) or np.isinf(score):
                score = -999.0
            
            eval_result = (parameters.copy(), score, result)
            
            # Cache result
            if self.cache_results:
                self.results_cache[cache_key] = eval_result
            
            return eval_result
            
        except Exception as e:
            logger.warning(f"Failed to evaluate parameters {parameters}: {str(e)}")
            return None
    
    def _generate_cache_key(self,
                          strategy_name: str,
                          parameters: Dict[str, Any],
                          start_date: datetime,
                          end_date: datetime) -> str:
        """Generate cache key for parameter combination"""
        param_str = "_".join([f"{k}_{v}" for k, v in sorted(parameters.items())])
        return f"{strategy_name}_{param_str}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    
    def analyze_parameter_interactions(self,
                                     results: GridSearchResults,
                                     parameter_pairs: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        Analyze interactions between parameters
        
        Args:
            results: Grid search results
            parameter_pairs: Specific parameter pairs to analyze
            
        Returns:
            Dictionary with interaction analysis
        """
        if not results.all_results:
            return {}
        
        # Get all parameter combinations and scores
        all_params = [params for params, _, _ in results.all_results]
        all_scores = [score for _, score, _ in results.all_results]
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(all_params)
        df['score'] = all_scores
        
        interactions = {}
        
        # Analyze correlations between parameters and score
        param_columns = [col for col in df.columns if col != 'score']
        correlations = df[param_columns + ['score']].corr()['score'].drop('score')
        interactions['parameter_correlations'] = correlations.to_dict()
        
        # Analyze specific parameter pairs if provided
        if parameter_pairs:
            for param1, param2 in parameter_pairs:
                if param1 in df.columns and param2 in df.columns:
                    # Group by parameter combinations and calculate mean score
                    grouped = df.groupby([param1, param2])['score'].agg(['mean', 'std', 'count'])
                    interactions[f'{param1}_vs_{param2}'] = grouped.to_dict()
        
        return interactions
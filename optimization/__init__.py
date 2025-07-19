"""
Parameter Optimization Framework for SIP Algorithmic Trading System

This module provides advanced optimization algorithms for finding optimal
strategy parameters through systematic and intelligent search methods.

Key Components:
- GridSearchOptimizer: Exhaustive grid search optimization
- GeneticOptimizer: Evolutionary algorithm optimization
- WalkForwardOptimizer: Time-series aware optimization and validation
- BayesianOptimizer: Bayesian optimization for efficient search
- OptimizationResults: Comprehensive results tracking
"""

from .grid_search import GridSearchOptimizer, GridSearchResults
from .genetic_optimizer import GeneticOptimizer, GeneticResults, Individual
from .walk_forward import WalkForwardOptimizer, WalkForwardResults, PeriodResult
from .bayesian_optimizer import BayesianOptimizer, BayesianResults

__all__ = [
    'GridSearchOptimizer',
    'GridSearchResults',
    'GeneticOptimizer', 
    'GeneticResults',
    'Individual',
    'WalkForwardOptimizer',
    'WalkForwardResults',
    'PeriodResult',
    'BayesianOptimizer',
    'BayesianResults'
]
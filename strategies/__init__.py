from .base_strategy import BaseSIPStrategy, StrategyConfig, InvestmentDecision
from .value_averaging import ValueAveragingStrategy
from .momentum_sip import MomentumSIPStrategy
from .volatility_sip import VolatilitySIPStrategy
from .dollar_cost_avg import DollarCostAveragingStrategy
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

# Strategy Registry
STRATEGY_REGISTRY = {
    'value_averaging': ValueAveragingStrategy,
    'momentum_sip': MomentumSIPStrategy,
    'volatility_sip': VolatilitySIPStrategy,
    'dollar_cost_averaging': DollarCostAveragingStrategy,
}

def create_strategy(strategy_name: str, config: StrategyConfig) -> BaseSIPStrategy:
    """
    Factory function to create strategy instances
    
    Args:
        strategy_name: Name of the strategy
        config: Strategy configuration
        
    Returns:
        Strategy instance
        
    Raises:
        ValueError: If strategy name is not recognized
    """
    if strategy_name not in STRATEGY_REGISTRY:
        available_strategies = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available strategies: {available_strategies}")
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    return strategy_class(config)

def list_available_strategies() -> List[str]:
    """Get list of available strategy names"""
    return list(STRATEGY_REGISTRY.keys())

def get_strategy_info(strategy_name: str) -> Dict[str, Any]:
    """Get information about a specific strategy"""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{strategy_name}'")
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    
    # Create a dummy config to get strategy info
    dummy_config = StrategyConfig(
        name=strategy_name,
        symbols=['DUMMY'],
        base_amount=1000.0
    )
    
    strategy = strategy_class(dummy_config)
    
    return {
        'name': strategy_name,
        'class': strategy_class.__name__,
        'description': strategy_class.__doc__.strip() if strategy_class.__doc__ else "No description available",
        'parameters': getattr(strategy, 'parameters', {}),
    }
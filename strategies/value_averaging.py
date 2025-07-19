import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from .base_strategy import BaseSIPStrategy, StrategyConfig, InvestmentDecision
logger = logging.getLogger(__name__)

class ValueAveragingStrategy(BaseSIPStrategy):
    """
    Value Averaging Strategy
    
    This strategy aims to reach a target portfolio value by investing more
    when the portfolio is below target and less when above target.
    
    Key Principles:
    - Set a target growth rate for the portfolio
    - Invest amount needed to reach target value
    - Automatically buy more during market downturns
    - Buy less (or even sell) during market upturns
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # Strategy-specific parameters
        self.target_growth_rate = self.parameters.get('target_growth_rate', 0.12)  # 12% annual
        self.max_investment_multiplier = self.parameters.get('max_investment_multiplier', 3.0)
        self.min_investment_multiplier = self.parameters.get('min_investment_multiplier', 0.0)
        self.rebalance_threshold = self.parameters.get('rebalance_threshold', 0.1)  # 10%
        
        # Track target values
        self.target_values = {symbol: 0.0 for symbol in self.symbols}
        self.execution_count = 0
        
    def calculate_investment_amount(self, 
                                   symbol: str, 
                                   current_price: float, 
                                   market_data: pd.DataFrame,
                                   portfolio_state: Dict[str, Any],
                                   execution_date: datetime) -> float:
        """Calculate investment amount using value averaging logic"""
        
        # Get current portfolio value for this symbol
        current_holdings = portfolio_state.get('holdings', {}).get(symbol, 0)
        current_value = current_holdings * current_price
        
        # Calculate target value
        self.execution_count += 1
        
        # Target value grows by target growth rate each period
        monthly_growth_rate = (1 + self.target_growth_rate) ** (1/12) - 1
        allocation = self._get_symbol_allocation(symbol)
        
        target_value = self.base_amount * allocation * self.execution_count * (1 + monthly_growth_rate) ** (self.execution_count - 1)
        
        # Calculate required investment
        required_investment = target_value - current_value
        
        # Apply investment limits
        max_investment = self.base_amount * allocation * self.max_investment_multiplier
        min_investment = self.base_amount * allocation * self.min_investment_multiplier
        
        # Constrain investment amount
        investment_amount = max(min_investment, min(max_investment, required_investment))
        
        logger.info(f"Value Averaging - {symbol}: Target=${target_value:.2f}, "
                   f"Current=${current_value:.2f}, Investment=${investment_amount:.2f}")
        
        return max(0, investment_amount)  # Never invest negative amounts
    
    def should_execute(self, 
                      current_date: datetime, 
                      last_execution_date: Optional[datetime]) -> bool:
        """Execute monthly on the first trading day"""
        if last_execution_date is None:
            return True
        
        if self.frequency == 'monthly':
            # Execute if it's been more than 25 days (approximately monthly)
            return (current_date - last_execution_date).days >= 25
        elif self.frequency == 'weekly':
            return (current_date - last_execution_date).days >= 7
        elif self.frequency == 'daily':
            return (current_date - last_execution_date).days >= 1
        
        return False
    
    def _get_symbol_allocation(self, symbol: str) -> float:
        """Get allocation percentage for a symbol"""
        # Equal allocation by default
        return 1.0 / len(self.symbols)
    
    def _calculate_confidence(self, 
                            symbol: str, 
                            market_data: pd.DataFrame, 
                            portfolio_state: Dict[str, Any]) -> float:
        """Calculate confidence based on market volatility"""
        if len(market_data) < 20:
            return 0.8  # Lower confidence with limited data
        
        # Higher confidence during volatile periods (when value averaging is most effective)
        returns = market_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Confidence increases with volatility (value averaging works better in volatile markets)
        confidence = min(1.0, 0.5 + volatility)
        return confidence
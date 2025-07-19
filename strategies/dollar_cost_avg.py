import logging
from .base_strategy import BaseSIPStrategy, StrategyConfig, InvestmentDecision
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
logger = logging.getLogger(__name__)
class DollarCostAveragingStrategy(BaseSIPStrategy):
    """
    Traditional Dollar Cost Averaging Strategy
    
    This is the baseline strategy that invests a fixed amount regularly,
    regardless of market conditions.
    
    Key Principles:
    - Invest the same amount every period
    - Ignore market conditions
    - Simple and disciplined approach
    - Good benchmark for other strategies
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # Strategy-specific parameters
        self.adjustment_factor = self.parameters.get('adjustment_factor', 1.0)
        
    def calculate_investment_amount(self, 
                                   symbol: str, 
                                   current_price: float, 
                                   market_data: pd.DataFrame,
                                   portfolio_state: Dict[str, Any],
                                   execution_date: datetime) -> float:
        """Calculate fixed investment amount"""
        
        base_investment = self.base_amount * self._get_symbol_allocation(symbol)
        investment_amount = base_investment * self.adjustment_factor
        
        logger.info(f"Dollar Cost Averaging - {symbol}: Investment=${investment_amount:.2f}")
        
        return investment_amount
    
    def should_execute(self, 
                      current_date: datetime, 
                      last_execution_date: Optional[datetime]) -> bool:
        """Execute based on frequency setting"""
        if last_execution_date is None:
            return True
        
        if self.frequency == 'monthly':
            return (current_date - last_execution_date).days >= 25
        elif self.frequency == 'weekly':
            return (current_date - last_execution_date).days >= 7
        elif self.frequency == 'daily':
            return (current_date - last_execution_date).days >= 1
        
        return False
    
    def _get_symbol_allocation(self, symbol: str) -> float:
        """Get allocation percentage for a symbol"""
        return 1.0 / len(self.symbols)
    
    def _calculate_confidence(self, 
                            symbol: str, 
                            market_data: pd.DataFrame, 
                            portfolio_state: Dict[str, Any]) -> float:
        """DCA always has consistent confidence"""
        return 1.0  # DCA is always confident as it doesn't depend on market conditions
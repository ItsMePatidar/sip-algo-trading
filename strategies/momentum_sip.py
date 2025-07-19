import logging
from .base_strategy import BaseSIPStrategy, StrategyConfig, InvestmentDecision
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
logger = logging.getLogger(__name__)
class MomentumSIPStrategy(BaseSIPStrategy):
    """
    Momentum-Based SIP Strategy
    
    This strategy adjusts investment amounts based on recent price momentum.
    
    Key Principles:
    - Increase investment during positive momentum
    - Decrease investment during negative momentum
    - Use multiple timeframes for momentum calculation
    - Include momentum strength in decision making
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # Strategy-specific parameters
        self.momentum_period = self.parameters.get('momentum_period', 60)  # 60 days
        self.momentum_threshold = self.parameters.get('momentum_threshold', 0.05)  # 5%
        self.momentum_multiplier = self.parameters.get('momentum_multiplier', 1.5)
        self.anti_momentum_multiplier = self.parameters.get('anti_momentum_multiplier', 0.7)
        self.max_multiplier = self.parameters.get('max_multiplier', 2.0)
        self.min_multiplier = self.parameters.get('min_multiplier', 0.5)
        
    def calculate_investment_amount(self, 
                                   symbol: str, 
                                   current_price: float, 
                                   market_data: pd.DataFrame,
                                   portfolio_state: Dict[str, Any],
                                   execution_date: datetime) -> float:
        """Calculate investment amount based on momentum"""
        
        base_investment = self.base_amount * self._get_symbol_allocation(symbol)
        
        # Calculate momentum
        momentum = self._calculate_momentum(market_data)
        
        # Determine multiplier based on momentum
        if momentum > self.momentum_threshold:
            # Positive momentum - invest more
            multiplier = self.momentum_multiplier
        elif momentum < -self.momentum_threshold:
            # Negative momentum - invest less
            multiplier = self.anti_momentum_multiplier
        else:
            # Neutral momentum - normal investment
            multiplier = 1.0
        
        # Apply momentum strength
        momentum_strength = abs(momentum)
        adjusted_multiplier = 1.0 + (multiplier - 1.0) * momentum_strength
        
        # Constrain multiplier
        final_multiplier = max(self.min_multiplier, min(self.max_multiplier, adjusted_multiplier))
        
        investment_amount = base_investment * final_multiplier
        
        logger.info(f"Momentum SIP - {symbol}: Momentum={momentum:.3f}, "
                   f"Multiplier={final_multiplier:.2f}, Investment=${investment_amount:.2f}")
        
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
    
    def _calculate_momentum(self, market_data: pd.DataFrame) -> float:
        """Calculate price momentum"""
        if len(market_data) < self.momentum_period:
            return 0.0
        
        # Calculate momentum as percentage change over period
        current_price = market_data.iloc[-1]['Close']
        past_price = market_data.iloc[-self.momentum_period]['Close']
        
        momentum = (current_price - past_price) / past_price
        
        return momentum
    
    def _get_symbol_allocation(self, symbol: str) -> float:
        """Get allocation percentage for a symbol"""
        return 1.0 / len(self.symbols)
    
    def _calculate_confidence(self, 
                            symbol: str, 
                            market_data: pd.DataFrame, 
                            portfolio_state: Dict[str, Any]) -> float:
        """Calculate confidence based on momentum consistency"""
        if len(market_data) < self.momentum_period:
            return 0.6
        
        # Calculate momentum over different periods
        short_momentum = self._calculate_momentum_period(market_data, 20)
        medium_momentum = self._calculate_momentum_period(market_data, 40)
        long_momentum = self._calculate_momentum_period(market_data, 60)
        
        # Higher confidence when momentum is consistent across timeframes
        momentum_consistency = 1.0 - abs(short_momentum - long_momentum)
        confidence = max(0.3, min(1.0, momentum_consistency))
        
        return confidence
    
    def _calculate_momentum_period(self, market_data: pd.DataFrame, period: int) -> float:
        """Calculate momentum for a specific period"""
        if len(market_data) < period:
            return 0.0
        
        current_price = market_data.iloc[-1]['Close']
        past_price = market_data.iloc[-period]['Close']
        
        return (current_price - past_price) / past_price
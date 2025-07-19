import logging
from .base_strategy import BaseSIPStrategy, StrategyConfig, InvestmentDecision
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
logger = logging.getLogger(__name__)

class VolatilitySIPStrategy(BaseSIPStrategy):
    """
    Volatility-Based SIP Strategy
    
    This strategy increases investment during high volatility periods,
    taking advantage of market uncertainty for better long-term returns.
    
    Key Principles:
    - Invest more during high volatility
    - Invest less during low volatility
    - Use realized volatility for decision making
    - Consider volatility persistence
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # Strategy-specific parameters
        self.volatility_period = self.parameters.get('volatility_period', 30)  # 30 days
        self.high_vol_threshold = self.parameters.get('high_vol_threshold', 0.25)  # 25% annualized
        self.low_vol_threshold = self.parameters.get('low_vol_threshold', 0.15)   # 15% annualized
        self.high_vol_multiplier = self.parameters.get('high_vol_multiplier', 1.5)
        self.low_vol_multiplier = self.parameters.get('low_vol_multiplier', 0.8)
        self.max_multiplier = self.parameters.get('max_multiplier', 2.0)
        self.min_multiplier = self.parameters.get('min_multiplier', 0.5)
        
    def calculate_investment_amount(self, 
                                   symbol: str, 
                                   current_price: float, 
                                   market_data: pd.DataFrame,
                                   portfolio_state: Dict[str, Any],
                                   execution_date: datetime) -> float:
        """Calculate investment amount based on volatility"""
        
        base_investment = self.base_amount * self._get_symbol_allocation(symbol)
        
        # Calculate volatility
        volatility = self._calculate_volatility(market_data)
        
        # Determine multiplier based on volatility
        if volatility > self.high_vol_threshold:
            # High volatility - invest more
            multiplier = self.high_vol_multiplier
        elif volatility < self.low_vol_threshold:
            # Low volatility - invest less
            multiplier = self.low_vol_multiplier
        else:
            # Medium volatility - normal investment
            multiplier = 1.0
        
        # Scale multiplier based on volatility intensity
        vol_intensity = self._calculate_volatility_intensity(volatility)
        adjusted_multiplier = 1.0 + (multiplier - 1.0) * vol_intensity
        
        # Constrain multiplier
        final_multiplier = max(self.min_multiplier, min(self.max_multiplier, adjusted_multiplier))
        
        investment_amount = base_investment * final_multiplier
        
        logger.info(f"Volatility SIP - {symbol}: Volatility={volatility:.3f}, "
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
    
    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate annualized volatility"""
        if len(market_data) < self.volatility_period:
            return 0.2  # Default volatility
        
        # Calculate daily returns
        returns = market_data['Close'].pct_change().dropna()
        
        # Get recent returns
        recent_returns = returns.tail(self.volatility_period)
        
        # Calculate volatility (annualized)
        volatility = recent_returns.std() * np.sqrt(252)
        
        return volatility
    
    def _calculate_volatility_intensity(self, volatility: float) -> float:
        """Calculate volatility intensity (0 to 1)"""
        # Normalize volatility to 0-1 scale
        min_vol = 0.1
        max_vol = 0.5
        
        intensity = (volatility - min_vol) / (max_vol - min_vol)
        return max(0.0, min(1.0, intensity))
    
    def _get_symbol_allocation(self, symbol: str) -> float:
        """Get allocation percentage for a symbol"""
        return 1.0 / len(self.symbols)
    
    def _calculate_confidence(self, 
                            symbol: str, 
                            market_data: pd.DataFrame, 
                            portfolio_state: Dict[str, Any]) -> float:
        """Calculate confidence based on volatility persistence"""
        if len(market_data) < self.volatility_period * 2:
            return 0.7
        
        # Calculate volatility over different periods
        current_vol = self._calculate_volatility_period(market_data, self.volatility_period)
        past_vol = self._calculate_volatility_period(market_data, self.volatility_period, offset=self.volatility_period)
        
        # Higher confidence when volatility is persistent
        vol_persistence = 1.0 - abs(current_vol - past_vol) / max(current_vol, past_vol, 0.1)
        confidence = max(0.4, min(1.0, vol_persistence))
        
        return confidence
    
    def _calculate_volatility_period(self, market_data: pd.DataFrame, period: int, offset: int = 0) -> float:
        """Calculate volatility for a specific period with optional offset"""
        start_idx = max(0, len(market_data) - period - offset)
        end_idx = len(market_data) - offset
        
        if end_idx <= start_idx:
            return 0.2
        
        period_data = market_data.iloc[start_idx:end_idx]
        returns = period_data['Close'].pct_change().dropna()
        
        if len(returns) < 5:
            return 0.2
        
        return returns.std() * np.sqrt(252)
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StrategyConfig:
    """Configuration for SIP strategies"""
    name: str
    symbols: List[str]
    base_amount: float
    frequency: str = 'monthly'  # 'monthly', 'weekly', 'daily'
    start_date: datetime = None
    end_date: datetime = None
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.start_date is None:
            self.start_date = datetime.now()

@dataclass
class InvestmentDecision:
    """Represents an investment decision made by a strategy"""
    date: datetime
    symbol: str
    amount: float
    price: float
    quantity: int
    reason: str = ""
    confidence: float = 1.0  # 0.0 to 1.0

class BaseSIPStrategy(ABC):
    """Abstract base class for all SIP strategies"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
        self.symbols = config.symbols
        self.base_amount = config.base_amount
        self.frequency = config.frequency
        self.parameters = config.parameters
        self.execution_history = []
        self.portfolio_value_history = []
        
    @abstractmethod
    def calculate_investment_amount(self, 
                                   symbol: str, 
                                   current_price: float, 
                                   market_data: pd.DataFrame,
                                   portfolio_state: Dict[str, Any],
                                   execution_date: datetime) -> float:
        """
        Calculate investment amount for a given symbol
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            market_data: Historical market data
            portfolio_state: Current portfolio state
            execution_date: Date of execution
            
        Returns:
            Investment amount in currency
        """
        pass
    
    @abstractmethod
    def should_execute(self, 
                      current_date: datetime, 
                      last_execution_date: Optional[datetime]) -> bool:
        """
        Determine if strategy should execute on given date
        
        Args:
            current_date: Current date
            last_execution_date: Date of last execution
            
        Returns:
            True if should execute, False otherwise
        """
        pass
    
    def make_investment_decisions(self, 
                                 market_data: Dict[str, pd.DataFrame],
                                 portfolio_state: Dict[str, Any],
                                 execution_date: datetime) -> List[InvestmentDecision]:
        """
        Make investment decisions for all symbols
        
        Args:
            market_data: Dictionary of market data for each symbol
            portfolio_state: Current portfolio state
            execution_date: Date of execution
            
        Returns:
            List of investment decisions
        """
        decisions = []
        
        for symbol in self.symbols:
            if symbol not in market_data:
                logger.warning(f"No market data available for {symbol}")
                continue
                
            # Get current price
            current_data = market_data[symbol]
            if current_data.empty:
                logger.warning(f"Empty market data for {symbol}")
                continue
                
            current_price = current_data.iloc[-1]['Close']
            
            # Calculate investment amount
            investment_amount = self.calculate_investment_amount(
                symbol, current_price, current_data, portfolio_state, execution_date
            )
            
            if investment_amount > 0:
                quantity = int(investment_amount / current_price)
                if quantity > 0:
                    decision = InvestmentDecision(
                        date=execution_date,
                        symbol=symbol,
                        amount=investment_amount,
                        price=current_price,
                        quantity=quantity,
                        reason=f"{self.name} strategy execution",
                        confidence=self._calculate_confidence(symbol, current_data, portfolio_state)
                    )
                    decisions.append(decision)
                    
        return decisions
    
    def _calculate_confidence(self, 
                            symbol: str, 
                            market_data: pd.DataFrame, 
                            portfolio_state: Dict[str, Any]) -> float:
        """Calculate confidence score for investment decision"""
        # Default implementation - can be overridden by specific strategies
        return 1.0
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'symbols': self.symbols,
            'base_amount': self.base_amount,
            'frequency': self.frequency,
            'parameters': self.parameters,
            'execution_count': len(self.execution_history)
        }
    
    def record_execution(self, decisions: List[InvestmentDecision]):
        """Record strategy execution"""
        self.execution_history.extend(decisions)
        logger.info(f"{self.name} strategy executed: {len(decisions)} decisions made")
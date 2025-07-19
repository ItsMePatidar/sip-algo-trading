import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import pandas as pd

from data.market_data import MarketDataProvider
from data.storage import DatabaseManager
from .base_strategy import BaseSIPStrategy, StrategyConfig, InvestmentDecision
from . import create_strategy

logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Manages strategy execution and portfolio state
    """
    
    def __init__(self, db_manager: DatabaseManager, market_data_provider: MarketDataProvider):
        self.db_manager = db_manager
        self.market_data_provider = market_data_provider
        self.active_strategies = {}
        self.portfolio_state = {
            'holdings': {},  # {symbol: quantity}
            'cash': 100000.0,  # Starting cash
            'total_invested': 0.0,
            'last_execution_dates': {}  # {strategy_name: last_execution_date}
        }
    
    def add_strategy(self, strategy: BaseSIPStrategy):
        """Add a strategy to the manager"""
        self.active_strategies[strategy.name] = strategy
        self.portfolio_state['last_execution_dates'][strategy.name] = None
        logger.info(f"Added strategy: {strategy.name}")
    
    def remove_strategy(self, strategy_name: str):
        """Remove a strategy from the manager"""
        if strategy_name in self.active_strategies:
            del self.active_strategies[strategy_name]
            if strategy_name in self.portfolio_state['last_execution_dates']:
                del self.portfolio_state['last_execution_dates'][strategy_name]
            logger.info(f"Removed strategy: {strategy_name}")
    
    def execute_strategies(self, execution_date: datetime = None) -> Dict[str, List[InvestmentDecision]]:
        """
        Execute all active strategies
        
        Args:
            execution_date: Date to execute strategies (default: now)
            
        Returns:
            Dictionary mapping strategy names to their investment decisions
        """
        if execution_date is None:
            execution_date = datetime.now()
        
        all_decisions = {}
        
        for strategy_name, strategy in self.active_strategies.items():
            try:
                # Check if strategy should execute
                last_execution = self.portfolio_state['last_execution_dates'].get(strategy_name)
                
                if not strategy.should_execute(execution_date, last_execution):
                    logger.info(f"Strategy {strategy_name} skipped execution")
                    continue
                
                # Get market data for strategy symbols
                market_data = self._get_market_data_for_strategy(strategy)
                
                if not market_data:
                    logger.warning(f"No market data available for strategy {strategy_name}")
                    continue
                
                # Execute strategy
                decisions = strategy.make_investment_decisions(
                    market_data, self.portfolio_state, execution_date
                )
                
                if decisions:
                    # Update portfolio state
                    self._apply_investment_decisions(decisions)
                    
                    # Record execution
                    strategy.record_execution(decisions)
                    self.portfolio_state['last_execution_dates'][strategy_name] = execution_date
                    
                    all_decisions[strategy_name] = decisions
                    
                    logger.info(f"Strategy {strategy_name} executed: {len(decisions)} decisions")
                else:
                    logger.info(f"Strategy {strategy_name} made no investment decisions")
                    
            except Exception as e:
                logger.error(f"Error executing strategy {strategy_name}: {str(e)}")
                continue
        
        return all_decisions
    
    def _get_market_data_for_strategy(self, strategy: BaseSIPStrategy) -> Dict[str, pd.DataFrame]:
        """Get market data for all symbols in a strategy"""
        market_data = {}
        
        for symbol in strategy.symbols:
            try:
                # Get historical data (last 252 trading days)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                
                data = self.db_manager.get_market_data(symbol, start_date, end_date)
                
                if data.empty:
                    # Try to fetch fresh data
                    logger.info(f"Fetching fresh data for {symbol}")
                    data = self.market_data_provider.fetch_historical_data(symbol, period='1y')
                
                if not data.empty:
                    market_data[symbol] = data
                else:
                    logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error getting market data for {symbol}: {str(e)}")
        
        return market_data
    
    def _apply_investment_decisions(self, decisions: List[InvestmentDecision]):
        """Apply investment decisions to portfolio state"""
        for decision in decisions:
            # Update holdings
            current_holdings = self.portfolio_state['holdings'].get(decision.symbol, 0)
            self.portfolio_state['holdings'][decision.symbol] = current_holdings + decision.quantity
            
            # Update cash and total invested
            self.portfolio_state['cash'] -= decision.amount
            self.portfolio_state['total_invested'] += decision.amount
            
            logger.info(f"Applied decision: {decision.symbol} +{decision.quantity} units, "
                       f"Amount: ${decision.amount:.2f}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        total_value = self.portfolio_state['cash']
        holdings_value = 0.0
        
        # Calculate holdings value
        for symbol, quantity in self.portfolio_state['holdings'].items():
            try:
                current_price = self.market_data_provider.get_current_price(symbol)
                if current_price:
                    symbol_value = quantity * current_price
                    holdings_value += symbol_value
                    total_value += symbol_value
            except Exception as e:
                logger.warning(f"Could not get current price for {symbol}: {str(e)}")
        
        return {
            'total_value': total_value,
            'cash': self.portfolio_state['cash'],
            'holdings_value': holdings_value,
            'total_invested': self.portfolio_state['total_invested'],
            'unrealized_pnl': total_value - 100000.0,  # Assuming 100k starting capital
            'holdings': self.portfolio_state['holdings'].copy(),
            'active_strategies': list(self.active_strategies.keys())
        }
    
    def get_strategy_performance(self, strategy_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific strategy"""
        if strategy_name not in self.active_strategies:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        strategy = self.active_strategies[strategy_name]
        
        # Calculate basic metrics
        total_decisions = len(strategy.execution_history)
        total_invested = sum(d.amount for d in strategy.execution_history)
        
        # Calculate current value of strategy investments
        current_value = 0.0
        for decision in strategy.execution_history:
            try:
                current_price = self.market_data_provider.get_current_price(decision.symbol)
                if current_price:
                    current_value += decision.quantity * current_price
            except Exception:
                pass
        
        return {
            'strategy_name': strategy_name,
            'total_executions': total_decisions,
            'total_invested': total_invested,
            'current_value': current_value,
            'unrealized_pnl': current_value - total_invested,
            'average_confidence': sum(d.confidence for d in strategy.execution_history) / max(total_decisions, 1),
            'last_execution': self.portfolio_state['last_execution_dates'].get(strategy_name),
            'strategy_info': strategy.get_strategy_info()
        }
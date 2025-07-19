import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class Transaction:
    """Represents a single transaction"""
    date: datetime
    symbol: str
    transaction_type: str  # 'BUY', 'SELL'
    quantity: int
    price: float
    amount: float
    commission: float = 0.0
    tax: float = 0.0
    total_cost: float = field(init=False)
    
    def __post_init__(self):
        self.total_cost = self.amount + self.commission + self.tax

@dataclass
class Position:
    """Represents a position in a security"""
    symbol: str
    quantity: int = 0
    average_price: float = 0.0
    total_cost: float = 0.0
    current_price: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def market_value(self) -> float:
        """Current market value of the position"""
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss"""
        return self.market_value - self.total_cost
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L percentage"""
        if self.total_cost == 0:
            return 0.0
        return (self.unrealized_pnl / self.total_cost) * 100

class Portfolio:
    """
    Portfolio tracking and management for backtesting
    
    Tracks positions, cash, transactions, and calculates performance metrics
    """
    
    def __init__(self, 
                 initial_cash: float = 100000.0,
                 commission_rate: float = 0.001,  # 0.1% commission
                 tax_rate: float = 0.001):        # 0.1% tax
        
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.transactions: List[Transaction] = []
        
        # Performance tracking
        self.daily_values: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []
        self.benchmark_returns: List[float] = []
        
        # Statistics
        self.total_commission_paid = 0.0
        self.total_tax_paid = 0.0
        self.total_invested = 0.0
        
    def buy(self, 
            symbol: str, 
            quantity: int, 
            price: float, 
            date: datetime,
            force: bool = False) -> bool:
        """
        Buy shares of a security
        
        Args:
            symbol: Security symbol
            quantity: Number of shares to buy
            price: Price per share
            date: Transaction date
            force: Force transaction even if insufficient cash
            
        Returns:
            True if transaction successful, False otherwise
        """
        # Calculate costs
        gross_amount = quantity * price
        commission = gross_amount * self.commission_rate
        tax = gross_amount * self.tax_rate
        total_cost = gross_amount + commission + tax
        
        # Check if sufficient cash
        if not force and total_cost > self.cash:
            logger.warning(f"Insufficient cash for {symbol} purchase: Need ${total_cost:.2f}, Have ${self.cash:.2f}")
            return False
        
        # Execute transaction
        transaction = Transaction(
            date=date,
            symbol=symbol,
            transaction_type='BUY',
            quantity=quantity,
            price=price,
            amount=gross_amount,
            commission=commission,
            tax=tax
        )
        
        # Update cash
        self.cash -= total_cost
        
        # Update position
        if symbol in self.positions:
            position = self.positions[symbol]
            # Calculate new average price
            total_quantity = position.quantity + quantity
            total_cost_new = position.total_cost + total_cost
            
            position.quantity = total_quantity
            position.total_cost = total_cost_new
            position.average_price = total_cost_new / total_quantity if total_quantity > 0 else 0
            position.last_updated = date
        else:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                average_price=price,
                total_cost=total_cost,
                current_price=price,
                last_updated=date
            )
        
        # Record transaction
        self.transactions.append(transaction)
        self.total_commission_paid += commission
        self.total_tax_paid += tax
        self.total_invested += gross_amount
        
        logger.debug(f"Bought {quantity} shares of {symbol} at ${price:.2f} on {date.strftime('%Y-%m-%d')}")
        return True
    
    def sell(self, 
             symbol: str, 
             quantity: int, 
             price: float, 
             date: datetime,
             force: bool = False) -> bool:
        """
        Sell shares of a security
        
        Args:
            symbol: Security symbol
            quantity: Number of shares to sell
            price: Price per share
            date: Transaction date
            force: Force transaction even if insufficient shares
            
        Returns:
            True if transaction successful, False otherwise
        """
        # Check if position exists and has sufficient shares
        if symbol not in self.positions:
            logger.warning(f"No position in {symbol} to sell")
            return False
        
        position = self.positions[symbol]
        if not force and quantity > position.quantity:
            logger.warning(f"Insufficient shares to sell: Want {quantity}, Have {position.quantity}")
            return False
        
        # Calculate proceeds
        gross_amount = quantity * price
        commission = gross_amount * self.commission_rate
        tax = gross_amount * self.tax_rate
        net_proceeds = gross_amount - commission - tax
        
        # Execute transaction
        transaction = Transaction(
            date=date,
            symbol=symbol,
            transaction_type='SELL',
            quantity=quantity,
            price=price,
            amount=gross_amount,
            commission=commission,
            tax=tax
        )
        
        # Update cash
        self.cash += net_proceeds
        
        # Update position
        position.quantity -= quantity
        if position.quantity <= 0:
            # Close position
            del self.positions[symbol]
        else:
            # Adjust total cost proportionally
            remaining_ratio = position.quantity / (position.quantity + quantity)
            position.total_cost *= remaining_ratio
            position.last_updated = date
        
        # Record transaction
        self.transactions.append(transaction)
        self.total_commission_paid += commission
        self.total_tax_paid += tax
        
        logger.debug(f"Sold {quantity} shares of {symbol} at ${price:.2f} on {date.strftime('%Y-%m-%d')}")
        return True
    
    def update_prices(self, prices: Dict[str, float], date: datetime):
        """Update current prices for all positions"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
                self.positions[symbol].last_updated = date
    
    def get_total_value(self) -> float:
        """Get total portfolio value (cash + positions)"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_positions_value(self) -> float:
        """Get total value of all positions"""
        return sum(pos.market_value for pos in self.positions.values())
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def get_realized_pnl(self) -> float:
        """Calculate realized P&L from completed transactions"""
        realized_pnl = 0.0
        position_tracker = {}  # Track average cost basis
        
        for transaction in self.transactions:
            symbol = transaction.symbol
            
            if symbol not in position_tracker:
                position_tracker[symbol] = {'quantity': 0, 'total_cost': 0.0}
            
            if transaction.transaction_type == 'BUY':
                position_tracker[symbol]['quantity'] += transaction.quantity
                position_tracker[symbol]['total_cost'] += transaction.total_cost
            elif transaction.transaction_type == 'SELL':
                if position_tracker[symbol]['quantity'] > 0:
                    avg_cost = position_tracker[symbol]['total_cost'] / position_tracker[symbol]['quantity']
                    realized_pnl += (transaction.price - avg_cost) * transaction.quantity
                    
                    # Update position tracker
                    sold_cost = avg_cost * transaction.quantity
                    position_tracker[symbol]['quantity'] -= transaction.quantity
                    position_tracker[symbol]['total_cost'] -= sold_cost
        
        return realized_pnl
    
    def record_daily_value(self, date: datetime, benchmark_return: float = 0.0):
        """Record daily portfolio value for performance tracking"""
        total_value = self.get_total_value()
        self.daily_values.append((date, total_value))
        
        # Calculate daily return
        if len(self.daily_values) > 1:
            prev_value = self.daily_values[-2][1]
            daily_return = (total_value - prev_value) / prev_value if prev_value > 0 else 0.0
            self.daily_returns.append(daily_return)
            self.benchmark_returns.append(benchmark_return)
    
    def get_portfolio_summary(self) -> Dict[str, any]:
        """Get comprehensive portfolio summary"""
        total_value = self.get_total_value()
        positions_value = self.get_positions_value()
        unrealized_pnl = self.get_unrealized_pnl()
        realized_pnl = self.get_realized_pnl()
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'initial_cash': self.initial_cash,
            'total_invested': self.total_invested,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'total_pnl': unrealized_pnl + realized_pnl,
            'total_return_pct': ((total_value - self.initial_cash) / self.initial_cash) * 100,
            'positions_count': len(self.positions),
            'transactions_count': len(self.transactions),
            'total_commission_paid': self.total_commission_paid,
            'total_tax_paid': self.total_tax_paid,
            'cash_ratio': self.cash / total_value if total_value > 0 else 0,
            'positions': {symbol: {
                'quantity': pos.quantity,
                'average_price': pos.average_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl_pct
            } for symbol, pos in self.positions.items()}
        }
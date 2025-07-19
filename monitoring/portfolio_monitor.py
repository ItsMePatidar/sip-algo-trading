# =============================================================================
# monitoring/portfolio_monitor.py - Real-time Portfolio Monitoring
# =============================================================================

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import time
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Individual position details"""
    symbol: str
    quantity: int
    average_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    day_change: float
    day_change_percent: float
    weight: float
    last_updated: datetime

@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot at a point in time"""
    timestamp: datetime
    total_value: float
    cash_value: float
    invested_value: float
    total_pnl: float
    total_pnl_percent: float
    day_pnl: float
    day_pnl_percent: float
    positions: Dict[str, Position]
    sector_allocation: Dict[str, float]
    top_gainers: List[str]
    top_losers: List[str]

@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float

class PortfolioMonitor:
    """
    Real-time Portfolio Monitoring System
    
    Provides comprehensive portfolio tracking including:
    - Real-time position monitoring
    - P&L calculations
    - Performance metrics
    - Risk monitoring
    - Historical tracking
    - Alert generation
    """
    
    def __init__(self, db_manager, market_data_provider, alert_manager=None):
        self.db_manager = db_manager
        self.market_data_provider = market_data_provider
        self.alert_manager = alert_manager
        
        # Portfolio state
        self.current_positions = {}
        self.cash_balance = 0.0
        self.initial_capital = 100000.0  # Default initial capital
        self.portfolio_history = deque(maxlen=1000)  # Last 1000 snapshots
        self.daily_snapshots = {}
        
        # Performance tracking
        self.inception_date = datetime.now()
        self.peak_value = self.initial_capital
        self.valley_value = self.initial_capital
        
        # Monitoring settings
        self.monitoring_active = False
        self.update_interval = 5  # seconds
        self.monitor_thread = None
        
        # Alert thresholds
        self.alert_thresholds = {
            'position_loss': -0.10,      # 10% position loss
            'daily_loss': -0.05,         # 5% daily loss
            'drawdown': -0.15,           # 15% drawdown
            'concentration': 0.25        # 25% position concentration
        }
        
        logger.info("Portfolio Monitor initialized")
    
    def start_monitoring(self):
        """Start real-time portfolio monitoring"""
        if self.monitoring_active:
            logger.warning("Portfolio monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Portfolio monitoring started")
    
    def stop_monitoring(self):
        """Stop portfolio monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("Portfolio monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self.update_portfolio()
                self._check_alerts()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.update_interval * 2)  # Wait longer on error
    
    def update_portfolio(self):
        """Update portfolio with latest market data"""
        try:
            # Load current positions from database
            self._load_positions_from_db()
            
            # Get latest market prices
            symbols = list(self.current_positions.keys())
            if symbols:
                current_prices = self.market_data_provider.get_multiple_prices(symbols)
                self._update_position_values(current_prices)
            
            # Calculate portfolio metrics
            portfolio_snapshot = self._create_portfolio_snapshot()
            
            # Store snapshot
            self.portfolio_history.append(portfolio_snapshot)
            
            # Store daily snapshot (once per day)
            today = datetime.now().date()
            if today not in self.daily_snapshots:
                self.daily_snapshots[today] = portfolio_snapshot
            
            logger.debug(f"Portfolio updated: ${portfolio_snapshot.total_value:,.2f}")
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {str(e)}")
    
    def _load_positions_from_db(self):
        """Load current positions from database"""
        try:
            # This would integrate with your database
            # For now, we'll use a simplified approach
            
            # Get positions from database (placeholder implementation)
            positions_data = self._get_positions_from_database()
            
            self.current_positions = {}
            for pos_data in positions_data:
                symbol = pos_data['symbol']
                self.current_positions[symbol] = {
                    'quantity': pos_data['quantity'],
                    'average_price': pos_data['average_price'],
                    'current_price': pos_data.get('current_price', pos_data['average_price']),
                    'last_updated': datetime.now()
                }
                
        except Exception as e:
            logger.error(f"Error loading positions from DB: {str(e)}")
    
    def _get_positions_from_database(self):
        """Get positions from database (placeholder)"""
        # This would query your actual database
        # For demo purposes, return sample data
        return [
            {'symbol': 'RELIANCE', 'quantity': 10, 'average_price': 2600.0},
            {'symbol': 'TCS', 'quantity': 5, 'average_price': 3500.0},
            {'symbol': 'HDFCBANK', 'quantity': 8, 'average_price': 1650.0}
        ]
    
    def _update_position_values(self, current_prices: Dict[str, float]):
        """Update position values with current market prices"""
        for symbol, position in self.current_positions.items():
            if symbol in current_prices:
                position['current_price'] = current_prices[symbol]
                position['last_updated'] = datetime.now()
    
    def _create_portfolio_snapshot(self) -> PortfolioSnapshot:
        """Create a complete portfolio snapshot"""
        positions = {}
        total_market_value = 0.0
        total_cost_basis = 0.0
        sector_allocation = defaultdict(float)
        
        # Calculate position details
        for symbol, pos_data in self.current_positions.items():
            quantity = pos_data['quantity']
            avg_price = pos_data['average_price']
            current_price = pos_data['current_price']
            
            market_value = quantity * current_price
            cost_basis = quantity * avg_price
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_percent = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
            
            # Calculate day change (simplified - would need previous day's close)
            day_change = market_value * 0.01  # Placeholder
            day_change_percent = 1.0  # Placeholder
            
            position = Position(
                symbol=symbol,
                quantity=quantity,
                average_price=avg_price,
                current_price=current_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_percent=unrealized_pnl_percent,
                day_change=day_change,
                day_change_percent=day_change_percent,
                weight=0.0,  # Will be calculated below
                last_updated=pos_data['last_updated']
            )
            
            positions[symbol] = position
            total_market_value += market_value
            total_cost_basis += cost_basis
            
            # Sector allocation (simplified)
            sector = self._get_sector_for_symbol(symbol)
            sector_allocation[sector] += market_value
        
        # Calculate weights
        for position in positions.values():
            position.weight = (position.market_value / total_market_value) * 100 if total_market_value > 0 else 0
        
        # Portfolio totals
        total_value = total_market_value + self.cash_balance
        total_pnl = total_market_value - total_cost_basis
        total_pnl_percent = (total_pnl / total_cost_basis) * 100 if total_cost_basis > 0 else 0
        
        # Day P&L (simplified calculation)
        day_pnl = self._calculate_day_pnl()
        day_pnl_percent = (day_pnl / total_value) * 100 if total_value > 0 else 0
        
        # Convert sector allocation to percentages
        sector_allocation_pct = {
            sector: (value / total_market_value) * 100 
            for sector, value in sector_allocation.items()
        } if total_market_value > 0 else {}
        
        # Top gainers and losers
        top_gainers, top_losers = self._get_top_movers(positions)
        
        return PortfolioSnapshot(
            timestamp=datetime.now(),
            total_value=total_value,
            cash_value=self.cash_balance,
            invested_value=total_market_value,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            day_pnl=day_pnl,
            day_pnl_percent=day_pnl_percent,
            positions=positions,
            sector_allocation=sector_allocation_pct,
            top_gainers=top_gainers,
            top_losers=top_losers
        )
    
    def _get_sector_for_symbol(self, symbol: str) -> str:
        """Get sector for a symbol (simplified mapping)"""
        sector_mapping = {
            'RELIANCE': 'Energy',
            'TCS': 'Technology',
            'INFY': 'Technology',
            'HDFCBANK': 'Banking',
            'ICICIBANK': 'Banking',
            'HINDUNILVR': 'Consumer Goods',
            'BAJFINANCE': 'Financial Services'
        }
        return sector_mapping.get(symbol, 'Other')
    
    def _calculate_day_pnl(self) -> float:
        """Calculate day P&L (simplified)"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        current_value = self.portfolio_history[-1].total_value
        previous_value = self.portfolio_history[-2].total_value
        
        return current_value - previous_value
    
    def _get_top_movers(self, positions: Dict[str, Position]) -> Tuple[List[str], List[str]]:
        """Get top gainers and losers"""
        sorted_positions = sorted(
            positions.values(), 
            key=lambda p: p.unrealized_pnl_percent, 
            reverse=True
        )
        
        top_gainers = [p.symbol for p in sorted_positions[:3] if p.unrealized_pnl_percent > 0]
        top_losers = [p.symbol for p in sorted_positions[-3:] if p.unrealized_pnl_percent < 0]
        top_losers.reverse()  # Show worst performers first
        
        return top_gainers, top_losers
    
    def _check_alerts(self):
        """Check for alert conditions"""
        if not self.alert_manager or not self.portfolio_history:
            return
        
        current_snapshot = self.portfolio_history[-1]
        
        # Check position loss alerts
        for symbol, position in current_snapshot.positions.items():
            if position.unrealized_pnl_percent <= self.alert_thresholds['position_loss'] * 100:
                self.alert_manager.send_alert(
                    level='WARNING',
                    title=f'Position Loss Alert - {symbol}',
                    message=f'{symbol} is down {position.unrealized_pnl_percent:.1f}%',
                    data={'symbol': symbol, 'loss_percent': position.unrealized_pnl_percent}
                )
        
        # Check daily loss alert
        if current_snapshot.day_pnl_percent <= self.alert_thresholds['daily_loss'] * 100:
            self.alert_manager.send_alert(
                level='WARNING',
                title='Daily Loss Alert',
                message=f'Portfolio is down {current_snapshot.day_pnl_percent:.1f}% today',
                data={'day_pnl_percent': current_snapshot.day_pnl_percent}
            )
        
        # Check concentration alert
        for symbol, position in current_snapshot.positions.items():
            if position.weight >= self.alert_thresholds['concentration'] * 100:
                self.alert_manager.send_alert(
                    level='WARNING',
                    title=f'Concentration Alert - {symbol}',
                    message=f'{symbol} represents {position.weight:.1f}% of portfolio',
                    data={'symbol': symbol, 'weight': position.weight}
                )
    
    def get_current_snapshot(self) -> Optional[PortfolioSnapshot]:
        """Get the most recent portfolio snapshot"""
        return self.portfolio_history[-1] if self.portfolio_history else None
    
    def get_historical_snapshots(self, days: int = 30) -> List[PortfolioSnapshot]:
        """Get historical snapshots for specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            snapshot for snapshot in self.portfolio_history 
            if snapshot.timestamp >= cutoff_date
        ]
    
    def calculate_performance_metrics(self, days: int = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not self.portfolio_history:
            return self._get_empty_metrics()
        
        # Get snapshots for calculation
        if days:
            snapshots = self.get_historical_snapshots(days)
        else:
            snapshots = list(self.portfolio_history)
        
        if len(snapshots) < 2:
            return self._get_empty_metrics()
        
        # Calculate returns
        values = [s.total_value for s in snapshots]
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        
        # Performance calculations
        total_return = (values[-1] - values[0]) / values[0] if values[0] > 0 else 0
        
        # Annualized return
        days_elapsed = (snapshots[-1].timestamp - snapshots[0].timestamp).days
        annualized_return = ((1 + total_return) ** (365.25 / max(days_elapsed, 1))) - 1 if days_elapsed > 0 else 0
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252) if returns else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown calculations
        peak_value = values[0]
        max_drawdown = 0
        current_drawdown = 0
        
        for value in values:
            if value > peak_value:
                peak_value = value
            
            drawdown = (peak_value - value) / peak_value
            max_drawdown = max(max_drawdown, drawdown)
        
        current_drawdown = (peak_value - values[-1]) / peak_value
        
        # Win rate
        winning_periods = sum(1 for r in returns if r > 0)
        win_rate = winning_periods / len(returns) if returns else 0
        
        # Profit factor
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Sortino ratio
        negative_returns = [r for r in returns if r < 0]
        downside_deviation = np.std(negative_returns) * np.sqrt(252) if negative_returns else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio
        )
    
    def _get_empty_metrics(self) -> PerformanceMetrics:
        """Get empty performance metrics"""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0
        )
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        current_snapshot = self.get_current_snapshot()
        performance_metrics = self.calculate_performance_metrics()
        
        if not current_snapshot:
            return {'error': 'No portfolio data available'}
        
        return {
            'portfolio_value': {
                'total_value': current_snapshot.total_value,
                'cash_value': current_snapshot.cash_value,
                'invested_value': current_snapshot.invested_value,
                'day_pnl': current_snapshot.day_pnl,
                'day_pnl_percent': current_snapshot.day_pnl_percent,
                'total_pnl': current_snapshot.total_pnl,
                'total_pnl_percent': current_snapshot.total_pnl_percent
            },
            'positions': {
                symbol: asdict(position) 
                for symbol, position in current_snapshot.positions.items()
            },
            'allocation': {
                'sector_allocation': current_snapshot.sector_allocation,
                'cash_percentage': (current_snapshot.cash_value / current_snapshot.total_value) * 100,
                'invested_percentage': (current_snapshot.invested_value / current_snapshot.total_value) * 100
            },
            'performance': asdict(performance_metrics),
            'top_movers': {
                'gainers': current_snapshot.top_gainers,
                'losers': current_snapshot.top_losers
            },
            'last_updated': current_snapshot.timestamp.isoformat()
        }
    
    def add_trade(self, symbol: str, side: str, quantity: int, price: float):
        """Add a new trade to the portfolio"""
        if symbol not in self.current_positions:
            self.current_positions[symbol] = {
                'quantity': 0,
                'average_price': 0.0,
                'current_price': price,
                'last_updated': datetime.now()
            }
        
        position = self.current_positions[symbol]
        
        if side.upper() == 'BUY':
            # Calculate new average price
            total_cost = (position['quantity'] * position['average_price']) + (quantity * price)
            total_quantity = position['quantity'] + quantity
            
            position['quantity'] = total_quantity
            position['average_price'] = total_cost / total_quantity if total_quantity > 0 else 0
            position['current_price'] = price
            
            # Reduce cash
            self.cash_balance -= quantity * price
            
        elif side.upper() == 'SELL':
            # Reduce position
            position['quantity'] = max(0, position['quantity'] - quantity)
            position['current_price'] = price
            
            # Increase cash
            self.cash_balance += quantity * price
            
            # Remove position if quantity is 0
            if position['quantity'] == 0:
                del self.current_positions[symbol]
        
        position['last_updated'] = datetime.now()
        
        logger.info(f"Trade added: {side} {quantity} {symbol} @ {price}")
    
    def set_cash_balance(self, cash_amount: float):
        """Set cash balance"""
        self.cash_balance = cash_amount
        logger.info(f"Cash balance set to ${cash_amount:,.2f}")
    
    def set_initial_capital(self, capital: float):
        """Set initial capital"""
        self.initial_capital = capital
        self.cash_balance = capital
        logger.info(f"Initial capital set to ${capital:,.2f}")
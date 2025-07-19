import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base_strategy import BaseSIPStrategy
from .portfolio import Portfolio, Transaction
from .metrics import PerformanceMetrics, PerformanceReport

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: datetime
    end_date: datetime
    initial_cash: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    tax_rate: float = 0.001         # 0.1%
    benchmark_symbol: str = 'NIFTY50'
    risk_free_rate: float = 0.05    # 5% annual
    rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly'

@dataclass 
class BacktestResults:
    """Results from backtesting"""
    config: BacktestConfig
    strategy_name: str
    portfolio_summary: Dict[str, Any]
    performance_report: PerformanceReport
    daily_values: List[Tuple[datetime, float]]
    daily_returns: List[float]
    benchmark_returns: List[float]
    transactions: List[Transaction]
    execution_log: List[str]

class BacktestEngine:
    """
    Main backtesting engine for SIP strategies
    
    Features:
    - Historical data simulation
    - Transaction cost modeling
    - Multiple rebalancing frequencies
    - Comprehensive performance analysis
    - Benchmark comparison
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio = Portfolio(
            initial_cash=config.initial_cash,
            commission_rate=config.commission_rate,
            tax_rate=config.tax_rate
        )
        self.execution_log = []
        
    def run_backtest(self, 
                    strategy: BaseSIPStrategy,
                    market_data: Dict[str, pd.DataFrame],
                    benchmark_data: Optional[pd.DataFrame] = None) -> BacktestResults:
        """
        Run backtest for a given strategy
        
        Args:
            strategy: SIP strategy to test
            market_data: Historical market data for all symbols
            benchmark_data: Benchmark data for comparison
            
        Returns:
            BacktestResults object with comprehensive results
        """
        logger.info(f"Starting backtest for strategy: {strategy.name}")
        logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
        
        # Validate data
        if not self._validate_market_data(market_data):
            raise ValueError("Invalid market data provided")
        
        # Get trading dates
        trading_dates = self._get_trading_dates(market_data)
        
        # Initialize benchmark returns
        benchmark_returns = self._calculate_benchmark_returns(benchmark_data, trading_dates)
        
        # Track strategy execution dates
        last_execution_date = None
        
        # Main simulation loop
        for current_date in trading_dates:
            try:
                # Update portfolio with current prices
                current_prices = self._get_current_prices(market_data, current_date)
                self.portfolio.update_prices(current_prices, current_date)
                
                # Record daily portfolio value
                benchmark_return = benchmark_returns.get(current_date, 0.0)
                self.portfolio.record_daily_value(current_date, benchmark_return)
                
                # Check if strategy should execute
                if strategy.should_execute(current_date, last_execution_date):
                    self._execute_strategy(strategy, market_data, current_date)
                    last_execution_date = current_date
                
            except Exception as e:
                logger.error(f"Error processing date {current_date}: {str(e)}")
                self.execution_log.append(f"ERROR on {current_date}: {str(e)}")
                continue
        
        # Generate results
        results = self._generate_results(strategy, benchmark_returns)
        
        logger.info(f"Backtest completed for {strategy.name}")
        logger.info(f"Total Return: {results.performance_report.total_return:.2f}%")
        logger.info(f"Sharpe Ratio: {results.performance_report.sharpe_ratio:.3f}")
        
        return results
    
    def _validate_market_data(self, market_data: Dict[str, pd.DataFrame]) -> bool:
        """Validate market data integrity"""
        if not market_data:
            logger.error("No market data provided")
            return False
        
        for symbol, data in market_data.items():
            if data.empty:
                logger.warning(f"Empty data for symbol {symbol}")
                continue
            
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.error(f"Missing columns for {symbol}: {missing_columns}")
                return False
            
            # Check for data within backtest period
            data_start = data.index.min()
            data_end = data.index.max()
            
            if data_end < self.config.start_date or data_start > self.config.end_date:
                logger.warning(f"Data for {symbol} doesn't overlap with backtest period")
        
        return True
    
    def _get_trading_dates(self, market_data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """Get list of trading dates within backtest period"""
        all_dates = set()
        
        for symbol, data in market_data.items():
            # Filter data to backtest period
            mask = (data.index >= self.config.start_date) & (data.index <= self.config.end_date)
            filtered_data = data[mask]
            all_dates.update(filtered_data.index)
        
        # Convert to sorted list
        trading_dates = sorted(list(all_dates))
        
        logger.info(f"Found {len(trading_dates)} trading dates")
        return trading_dates
    
    def _get_current_prices(self, market_data: Dict[str, pd.DataFrame], current_date: datetime) -> Dict[str, float]:
        """Get current prices for all symbols on given date"""
        current_prices = {}
        
        for symbol, data in market_data.items():
            try:
                # Get price for current date
                if current_date in data.index:
                    current_prices[symbol] = data.loc[current_date, 'Close']
                else:
                    # Use forward fill to get last available price
                    available_data = data[data.index <= current_date]
                    if not available_data.empty:
                        current_prices[symbol] = available_data.iloc[-1]['Close']
            except Exception as e:
                logger.warning(f"Could not get price for {symbol} on {current_date}: {str(e)}")
        
        return current_prices
    
    def _calculate_benchmark_returns(self, 
                                   benchmark_data: Optional[pd.DataFrame], 
                                   trading_dates: List[datetime]) -> Dict[datetime, float]:
        """Calculate benchmark returns for each trading date"""
        benchmark_returns = {}
        
        if benchmark_data is None or benchmark_data.empty:
            # Return zero returns if no benchmark data
            return {date: 0.0 for date in trading_dates}
        
        # Calculate daily returns
        benchmark_data = benchmark_data.sort_index()
        daily_returns = benchmark_data['Close'].pct_change().fillna(0)
        
        for date in trading_dates:
            if date in daily_returns.index:
                benchmark_returns[date] = daily_returns.loc[date]
            else:
                benchmark_returns[date] = 0.0
        
        return benchmark_returns
    
    def _execute_strategy(self, 
                         strategy: BaseSIPStrategy, 
                         market_data: Dict[str, pd.DataFrame], 
                         execution_date: datetime):
        """Execute strategy on given date"""
        try:
            # Prepare market data for strategy (historical data up to execution date)
            strategy_market_data = {}
            
            for symbol in strategy.symbols:
                if symbol in market_data:
                    # Get historical data up to execution date
                    historical_data = market_data[symbol][market_data[symbol].index <= execution_date]
                    if not historical_data.empty:
                        strategy_market_data[symbol] = historical_data
            
            if not strategy_market_data:
                self.execution_log.append(f"No market data available for strategy execution on {execution_date}")
                return
            
            # Get current portfolio state
            portfolio_state = {
                'holdings': {symbol: pos.quantity for symbol, pos in self.portfolio.positions.items()},
                'cash': self.portfolio.cash,
                'total_invested': self.portfolio.total_invested
            }
            
            # Make investment decisions
            decisions = strategy.make_investment_decisions(
                strategy_market_data, portfolio_state, execution_date
            )
            
            # Execute decisions
            for decision in decisions:
                success = self.portfolio.buy(
                    symbol=decision.symbol,
                    quantity=decision.quantity,
                    price=decision.price,
                    date=decision.date
                )
                
                if success:
                    log_msg = f"Executed: {decision.symbol} +{decision.quantity} @ ${decision.price:.2f}"
                    self.execution_log.append(log_msg)
                    logger.debug(log_msg)
                else:
                    log_msg = f"Failed: {decision.symbol} +{decision.quantity} @ ${decision.price:.2f} (insufficient cash)"
                    self.execution_log.append(log_msg)
                    logger.warning(log_msg)
            
            # Record strategy execution
            strategy.record_execution(decisions)
            
        except Exception as e:
            error_msg = f"Strategy execution failed on {execution_date}: {str(e)}"
            self.execution_log.append(error_msg)
            logger.error(error_msg)
    
    def _generate_results(self, strategy: BaseSIPStrategy, benchmark_returns: Dict[datetime, float]) -> BacktestResults:
        """Generate comprehensive backtest results"""
        
        # Get portfolio summary
        portfolio_summary = self.portfolio.get_portfolio_summary()
        
        # Calculate performance metrics
        benchmark_returns_list = [benchmark_returns.get(date, 0.0) for date, _ in self.portfolio.daily_values]
        
        performance_metrics = PerformanceMetrics(
            portfolio_values=self.portfolio.daily_values,
            benchmark_returns=benchmark_returns_list,
            risk_free_rate=self.config.risk_free_rate
        )
        
        # Generate performance report
        max_dd, dd_start, dd_end = performance_metrics.maximum_drawdown()
        
        performance_report = PerformanceReport(
            total_return=performance_metrics.total_return(),
            annualized_return=performance_metrics.annualized_return(),
            volatility=performance_metrics.volatility(),
            sharpe_ratio=performance_metrics.sharpe_ratio(),
            sortino_ratio=performance_metrics.sortino_ratio(),
            calmar_ratio=performance_metrics.calmar_ratio(),
            maximum_drawdown=max_dd,
            max_drawdown_start=dd_start,
            max_drawdown_end=dd_end,
            value_at_risk_5=performance_metrics.value_at_risk(),
            expected_shortfall_5=performance_metrics.expected_shortfall(),
            beta=performance_metrics.beta(),
            alpha=performance_metrics.alpha(),
            information_ratio=performance_metrics.information_ratio(),
            win_rate=performance_metrics.win_rate(),
            avg_win_loss_ratio=performance_metrics.average_win_loss_ratio(),
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            total_days=(self.config.end_date - self.config.start_date).days,
            trading_days=len(self.portfolio.daily_values)
        )
        
        return BacktestResults(
            config=self.config,
            strategy_name=strategy.name,
            portfolio_summary=portfolio_summary,
            performance_report=performance_report,
            daily_values=self.portfolio.daily_values,
            daily_returns=self.portfolio.daily_returns,
            benchmark_returns=benchmark_returns_list,
            transactions=self.portfolio.transactions,
            execution_log=self.execution_log
        )
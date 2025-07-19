#!/usr/bin/env python3
"""
main.py - SIP Algorithmic Trading System Main Application

This is the main entry point for the SIP Algorithmic Trading System.
It initializes all components and runs the core trading logic.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import time
import signal
import logging

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from config.settings import settings
from data.storage import DatabaseManager
from data.market_data import MarketDataProvider
from brokers.zerodha_broker import ZerodhaBroker
from orders.order_manager import OrderManager
from utils.helpers import setup_logging, is_market_open, calculate_sip_amount
from controls.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, TriggerCondition, TriggerType, CircuitBreakerState
from controls.emergency_stop import EmergencyStopManager, StopLevel, StopReason
from controls.manual_override import ManualOverride, OverrideLevel
logger = logging.getLogger(__name__)
class SIPTradingSystem:
    def __init__(self):
        # ... existing initialization ...
        
        # Initialize emergency controls
        self.setup_emergency_controls()
    
    def setup_emergency_controls(self):
        """Setup emergency control systems"""
        
        # 1. Circuit Breaker
        cb_config = CircuitBreakerConfig(
            name="main_system_breaker",
            triggers=[
                TriggerCondition(TriggerType.PORTFOLIO_LOSS, 0.15, timedelta(minutes=5)),
                TriggerCondition(TriggerType.DAILY_LOSS, 0.05, timedelta(hours=1)),
                TriggerCondition(TriggerType.DRAWDOWN_LIMIT, 0.20, timedelta(minutes=10))
            ],
            cooldown_period=timedelta(minutes=30),
            auto_reset=True
        )
        self.circuit_breaker = CircuitBreaker(cb_config)
        
        # 2. Emergency Stop
        self.emergency_stop_manager = EmergencyStopManager()
        self.emergency_stop = self.emergency_stop_manager.register_system("SIP_TRADING")
        
        # 3. Manual Override
        self.manual_override = ManualOverride("SIP_TRADING")
        self.manual_override.add_user_permission("admin", "Administrator", OverrideLevel.EMERGENCY_CONTROL)
        # Add callbacks
        self.circuit_breaker.add_trigger_callback(self._on_circuit_breaker_trigger)
        self.emergency_stop.add_stop_callback(self._on_emergency_stop)
        
        logger.info("Emergency controls initialized successfully")
    def _on_circuit_breaker_trigger(self, event):
       """Handle circuit breaker trigger"""
       logger.critical(f"Circuit breaker triggered: {event.trigger_type.value}")
       
       # Stop all trading activities
       self.stop_all_strategies()
       
       # Send alerts
       self.send_critical_alert(f"Circuit breaker triggered: {event.trigger_type.value}")
   
    def _on_emergency_stop(self, stop_event):
        """Handle emergency stop"""
        logger.critical(f"Emergency stop activated: {stop_event.reason.value}")
        
        # Immediate actions based on stop level
        if stop_event.level == StopLevel.HARD_STOP:
            self.cancel_all_pending_orders()
            self.stop_all_strategies()
        elif stop_event.level == StopLevel.PANIC_STOP:
            self.initiate_emergency_liquidation()
        
        # Send critical alerts
        self.send_emergency_alert(stop_event)
    
    def check_safety_conditions(self, portfolio_state, market_state, system_state):
        """Check all safety conditions before trading"""
        
        # Check emergency stop
        if self.emergency_stop.is_stopped:
            logger.warning("Trading blocked: Emergency stop active")
            return False
        
        # Check circuit breaker
        should_trigger = self.circuit_breaker.evaluate_conditions(
            portfolio_state, market_state, system_state
        )
        
        if should_trigger or self.circuit_breaker.state == CircuitBreakerState.TRIGGERED:
            logger.warning("Trading blocked: Circuit breaker triggered")
            return False
        
        return True
class SIPTradingSystem:
    """Main SIP Trading System class"""
    
    def __init__(self):
        """Initialize the SIP Trading System"""
        self.logger = None
        self.db_manager = None
        self.market_data = None
        self.broker = None
        self.order_manager = None
        self.is_running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        if self.logger:
            self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.is_running = False
    
    def initialize(self):
        """Initialize all system components"""
        try:
            # Setup logging
            setup_logging()
            self.logger = logging.getLogger(__name__)
            
            self.logger.info("=" * 60)
            self.logger.info("SIP Algorithmic Trading System - Starting Up")
            self.logger.info("=" * 60)
            
            # Initialize database
            self.logger.info("Initializing database...")
            self.db_manager = DatabaseManager()
            self.logger.info("✓ Database initialized successfully")
            
            # Initialize market data provider
            self.logger.info("Initializing market data provider...")
            self.market_data = MarketDataProvider(self.db_manager)
            self.logger.info("✓ Market data provider initialized")
            
            # Initialize broker
            self.logger.info("Initializing broker connection...")
            broker_config = {
                'api_key': settings.broker.api_key,
                'api_secret': settings.broker.api_secret
            }
            self.broker = ZerodhaBroker(broker_config)
            
            # Connect to broker
            if self.broker.connect():
                self.logger.info("✓ Broker connected successfully")
            else:
                self.logger.error("✗ Failed to connect to broker")
                return False
            
            # Initialize order manager
            self.logger.info("Initializing order manager...")
            self.order_manager = OrderManager(self.db_manager, self.broker)
            self.logger.info("✓ Order manager initialized")
            
            # Get account information
            try:
                balance = self.broker.get_balance()
                self.logger.info(f"Account balance: ₹{balance['cash']:.2f}")
                
                positions = self.broker.get_positions()
                self.logger.info(f"Current positions: {len(positions)}")
                
            except Exception as e:
                self.logger.warning(f"Could not retrieve account info: {str(e)}")
            
            self.logger.info("✓ System initialization completed successfully")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"✗ System initialization failed: {str(e)}")
            else:
                print(f"✗ System initialization failed: {str(e)}")
            return False
    
    def update_market_data(self, symbols):
        """Update market data for given symbols"""
        self.logger.info(f"Updating market data for {len(symbols)} symbols...")
        
        try:
            for symbol in symbols:
                self.logger.info(f"Fetching data for {symbol}...")
                data = self.market_data.fetch_historical_data(symbol, period='5d')
                
                if not data.empty:
                    self.logger.info(f"✓ Updated {len(data)} records for {symbol}")
                else:
                    self.logger.warning(f"⚠ No data updated for {symbol}")
                
                time.sleep(0.5)  # Rate limiting
            
            self.logger.info("✓ Market data update completed")
            
        except Exception as e:
            self.logger.error(f"✗ Market data update failed: {str(e)}")
    
    def get_current_prices(self, symbols):
        """Get current prices for symbols"""
        self.logger.info(f"Fetching current prices for {len(symbols)} symbols...")
        
        try:
            prices = self.market_data.get_multiple_prices(symbols)
            
            for symbol, price in prices.items():
                self.logger.info(f"  {symbol}: ₹{price:.2f}")
            
            return prices
            
        except Exception as e:
            self.logger.error(f"✗ Failed to get current prices: {str(e)}")
            return {}
    
    def execute_sip_strategy(self, sip_config):
        """Execute SIP investment strategy"""
        self.logger.info("Executing SIP strategy...")
        
        try:
            # Check if market is open
            if not is_market_open():
                self.logger.info("Market is closed, skipping SIP execution")
                return False
            
            # Get current prices
            current_prices = self.get_current_prices(sip_config['symbols'])
            
            if not current_prices:
                self.logger.error("Could not get current prices, aborting SIP execution")
                return False
            
            # Calculate investment amounts
            orders_to_place = []
            total_investment = 0
            
            for i, symbol in enumerate(sip_config['symbols']):
                if symbol in current_prices:
                    # Calculate investment amount for this symbol
                    base_amount = sip_config['total_amount'] * sip_config['allocation'][i]
                    
                    # Apply market condition adjustment (if configured)
                    market_condition = self._assess_market_condition(symbol, current_prices[symbol])
                    invest_amount = calculate_sip_amount(base_amount, market_condition)
                    
                    # Calculate quantity
                    quantity = int(invest_amount / current_prices[symbol])
                    
                    if quantity > 0:
                        orders_to_place.append({
                            'symbol': symbol,
                            'quantity': quantity,
                            'price': current_prices[symbol],
                            'amount': invest_amount
                        })
                        total_investment += invest_amount
                        
                        self.logger.info(f"  {symbol}: {quantity} units @ ₹{current_prices[symbol]:.2f} = ₹{invest_amount:.2f}")
            
            # Place orders
            if orders_to_place:
                self.logger.info(f"Placing {len(orders_to_place)} SIP orders (Total: ₹{total_investment:.2f})")
                
                orders_placed = []
                for order in orders_to_place:
                    order_id = self.order_manager.place_market_order(
                        order['symbol'], 'BUY', order['quantity']
                    )
                    
                    if order_id:
                        orders_placed.append(order_id)
                        self.logger.info(f"✓ Order placed for {order['symbol']}: {order_id[:8]}...")
                    else:
                        self.logger.error(f"✗ Failed to place order for {order['symbol']}")
                
                self.logger.info(f"✓ SIP execution completed: {len(orders_placed)} orders placed")
                return True
            else:
                self.logger.warning("No orders to place")
                return False
                
        except Exception as e:
            self.logger.error(f"✗ SIP execution failed: {str(e)}")
            return False
    
    def _assess_market_condition(self, symbol, current_price):
        """Assess market condition for a symbol (simplified)"""
        try:
            # Get historical data for trend analysis
            historical_data = self.market_data.fetch_historical_data(symbol, period='1mo')
            
            if not historical_data.empty:
                # Simple moving average comparison
                recent_avg = historical_data['Close'].tail(5).mean()
                
                if current_price > recent_avg * 1.05:
                    return 'bull'  # Price is 5% above recent average
                elif current_price < recent_avg * 0.95:
                    return 'bear'  # Price is 5% below recent average
                else:
                    return 'neutral'
            
            return 'neutral'
            
        except Exception as e:
            self.logger.warning(f"Could not assess market condition for {symbol}: {str(e)}")
            return 'neutral'
    
    def monitor_orders(self):
        """Monitor active orders"""
        try:
            self.order_manager.update_order_statuses()
            active_orders = self.order_manager.get_active_orders()
            
            if active_orders:
                self.logger.info(f"Monitoring {len(active_orders)} active orders")
                
                for order in active_orders:
                    self.logger.info(f"  {order['symbol']} {order['side']} {order['quantity']} - {order['status']}")
            else:
                self.logger.debug("No active orders to monitor")
                
        except Exception as e:
            self.logger.error(f"Error monitoring orders: {str(e)}")
    
    def run_demo_mode(self):
        """Run system in demonstration mode"""
        self.logger.info("Running in demonstration mode...")
        
        # Demo SIP configuration
        demo_sip_config = {
            'symbols': ['NIFTY50', 'RELIANCE', 'TCS'],
            'allocation': [0.6, 0.25, 0.15],
            'total_amount': 5000.0,
            'frequency': 'monthly'
        }
        
        try:
            # 1. Update market data
            self.update_market_data(demo_sip_config['symbols'])
            
            # 2. Get current prices
            current_prices = self.get_current_prices(demo_sip_config['symbols'])
            
            # 3. Execute SIP strategy
            if current_prices:
                self.execute_sip_strategy(demo_sip_config)
            
            # 4. Monitor orders
            self.monitor_orders()
            
            # 5. Display portfolio summary
            self.display_portfolio_summary()
            
        except Exception as e:
            self.logger.error(f"Demo mode failed: {str(e)}")
    
    def display_portfolio_summary(self):
        """Display portfolio summary"""
        self.logger.info("Portfolio Summary:")
        self.logger.info("-" * 40)
        
        try:
            # Get account balance
            balance = self.broker.get_balance()
            self.logger.info(f"Cash Balance: ₹{balance['cash']:.2f}")
            self.logger.info(f"Used Margin: ₹{balance['used']:.2f}")
            
            # Get positions
            positions = self.broker.get_positions()
            if positions:
                self.logger.info(f"Holdings: {len(positions)} positions")
                for position in positions:
                    self.logger.info(f"  {position}")
            else:
                self.logger.info("No current positions")
            
            # Get recent orders
            active_orders = self.order_manager.get_active_orders()
            if active_orders:
                self.logger.info(f"Active Orders: {len(active_orders)}")
                for order in active_orders:
                    self.logger.info(f"  {order['symbol']} {order['side']} {order['quantity']} - {order['status']}")
            else:
                self.logger.info("No active orders")
                
        except Exception as e:
            self.logger.error(f"Error displaying portfolio summary: {str(e)}")
    
    def run_interactive_mode(self):
        """Run system in interactive mode"""
        self.logger.info("Starting interactive mode...")
        self.logger.info("Commands: 'sip', 'portfolio', 'prices', 'orders', 'quit'")
        
        self.is_running = True
        
        while self.is_running:
            try:
                command = input("\nSIP Trading> ").strip().lower()
                
                if command == 'quit' or command == 'q':
                    self.logger.info("Shutting down...")
                    break
                elif command == 'sip':
                    self.run_demo_mode()
                elif command == 'portfolio':
                    self.display_portfolio_summary()
                elif command == 'prices':
                    symbols = ['NIFTY50', 'RELIANCE', 'TCS', 'HDFCBANK']
                    self.get_current_prices(symbols)
                elif command == 'orders':
                    self.monitor_orders()
                elif command == 'help':
                    print("Available commands:")
                    print("  sip       - Execute SIP strategy")
                    print("  portfolio - Show portfolio summary")
                    print("  prices    - Get current prices")
                    print("  orders    - Monitor active orders")
                    print("  quit      - Exit the system")
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                self.logger.error(f"Error in interactive mode: {str(e)}")
    
    def run(self, mode='demo'):
        """Run the SIP Trading System"""
        if not self.initialize():
            return False
        
        try:
            if mode == 'demo':
                self.run_demo_mode()
            elif mode == 'interactive':
                self.run_interactive_mode()
            else:
                self.logger.error(f"Unknown mode: {mode}")
                return False
            
            self.logger.info("✓ System execution completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ System execution failed: {str(e)}")
            return False
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup system resources"""
        if self.logger:
            self.logger.info("Cleaning up system resources...")
        
        # Close database connections, logout from broker, etc.
        # Add cleanup code here as needed
        
        if self.logger:
            self.logger.info("✓ Cleanup completed")

def main():
    """Main application entry point"""
    print("SIP Algorithmic Trading System")
    print("=" * 50)
    
    # Parse command line arguments
    mode = 'demo'  # default mode
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['demo', 'interactive']:
            mode = arg
        elif arg in ['--help', '-h']:
            print("Usage: sip-trading [mode]")
            print("Modes:")
            print("  demo        - Run demonstration mode (default)")
            print("  interactive - Run interactive mode")
            print("  --help      - Show this help message")
            return
        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for usage information")
            return
    
    # Create and run the system
    system = SIPTradingSystem()
    
    try:
        success = system.run(mode)
        if success:
            print("\n✓ System completed successfully!")
        else:
            print("\n✗ System encountered errors!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nSystem interrupted by user")
    except Exception as e:
        print(f"\n✗ System failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
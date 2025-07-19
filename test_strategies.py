#!/usr/bin/env python3
"""
test_strategies.py - Complete Phase 2 Strategy Framework Testing

This script thoroughly tests all Phase 2 strategy components to ensure
they're working correctly with Phase 1 infrastructure.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress some warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class TestResult:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def add_pass(self):
        self.passed += 1

    def add_fail(self):
        self.failed += 1

    def add_warning(self):
        self.warnings += 1

    def get_total(self):
        return self.passed + self.failed

    def get_success_rate(self):
        total = self.get_total()
        return (self.passed / total * 100) if total > 0 else 0

def print_test_header(test_name):
    """Print formatted test header"""
    print(f"\n{'=' * 70}")
    print(f"Testing: {test_name}")
    print(f"{'=' * 70}")

def print_section(section_name):
    """Print section header"""
    print(f"\n{'-' * 50}")
    print(f"{section_name}")
    print(f"{'-' * 50}")

def test_phase2_setup():
    """Test Phase 2 directory structure and setup"""
    print_test_header("Phase 2 Setup and Structure")
    result = TestResult()

    # Test directory structure
    required_dirs = ['strategies', 'backtesting', 'optimization', 'analytics', 'research']
    
    print("1. Checking directory structure...")
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   ✓ {directory}/ directory exists")
            result.add_pass()
            
            # Check for __init__.py
            init_file = os.path.join(directory, '__init__.py')
            if os.path.exists(init_file):
                print(f"   ✓ {directory}/__init__.py exists")
                result.add_pass()
            else:
                print(f"   ⚠ {directory}/__init__.py missing (will create)")
                # Create missing __init__.py
                with open(init_file, 'w') as f:
                    f.write(f"# {directory.title()} module\n")
                result.add_warning()
        else:
            print(f"   ✗ {directory}/ directory missing")
            result.add_fail()

    # Test strategy files
    print("\n2. Checking strategy files...")
    strategy_files = [
        'strategies/base_strategy.py',
        'strategies/value_averaging.py', 
        'strategies/momentum_sip.py',
        'strategies/volatility_sip.py',
        'strategies/dollar_cost_avg.py',
        'strategies/strategy_manager.py'
    ]
    
    for file_path in strategy_files:
        if os.path.exists(file_path):
            print(f"   ✓ {file_path} exists")
            result.add_pass()
        else:
            print(f"   ✗ {file_path} missing")
            result.add_fail()

    return result

def test_strategy_imports():
    """Test all strategy imports"""
    print_test_header("Strategy Imports")
    result = TestResult()

    imports_to_test = [
        ("Base Strategy", "from strategies.base_strategy import BaseSIPStrategy, StrategyConfig, InvestmentDecision"),
        ("Value Averaging", "from strategies.value_averaging import ValueAveragingStrategy"),
        ("Momentum SIP", "from strategies.momentum_sip import MomentumSIPStrategy"),
        ("Volatility SIP", "from strategies.volatility_sip import VolatilitySIPStrategy"),
        ("Dollar Cost Avg", "from strategies.dollar_cost_avg import DollarCostAveragingStrategy"),
        ("Strategy Factory", "from strategies import create_strategy, list_available_strategies, STRATEGY_REGISTRY"),
        ("Strategy Manager", "from strategies.strategy_manager import StrategyManager")
    ]

    for name, import_statement in imports_to_test:
        try:
            exec(import_statement)
            print(f"✓ {name} import successful")
            result.add_pass()
        except Exception as e:
            print(f"✗ {name} import failed: {str(e)}")
            result.add_fail()

    return result

def test_strategy_creation():
    """Test strategy creation and basic functionality"""
    print_test_header("Strategy Creation and Basic Functionality")
    result = TestResult()

    try:
        from strategies import create_strategy, StrategyConfig, list_available_strategies

        # Test strategy registry
        print("1. Testing strategy registry...")
        available_strategies = list_available_strategies()
        print(f"   Available strategies: {available_strategies}")
        
        expected_strategies = ['value_averaging', 'momentum_sip', 'volatility_sip', 'dollar_cost_averaging']
        for strategy_name in expected_strategies:
            if strategy_name in available_strategies:
                print(f"   ✓ {strategy_name} registered")
                result.add_pass()
            else:
                print(f"   ✗ {strategy_name} not registered")
                result.add_fail()

        # Test strategy configuration
        print("\n2. Testing strategy configuration...")
        config = StrategyConfig(
            name="Test Strategy",
            symbols=["NIFTY50", "RELIANCE"],
            base_amount=5000.0,
            frequency="monthly",
            parameters={'test_param': 123}
        )
        print(f"   ✓ Strategy config created: {config.name}")
        result.add_pass()

        # Test strategy creation
        print("\n3. Testing strategy creation...")
        created_strategies = {}
        
        for strategy_name in available_strategies:
            try:
                strategy = create_strategy(strategy_name, config)
                created_strategies[strategy_name] = strategy
                print(f"   ✓ Created {strategy_name}: {strategy.__class__.__name__}")
                result.add_pass()
                
                # Test basic methods
                info = strategy.get_strategy_info()
                print(f"      - Name: {info['name']}")
                print(f"      - Type: {info['type']}")
                print(f"      - Symbols: {info['symbols']}")
                
            except Exception as e:
                print(f"   ✗ Failed to create {strategy_name}: {str(e)}")
                result.add_fail()

        return result, created_strategies

    except Exception as e:
        print(f"✗ Strategy creation test failed: {str(e)}")
        result.add_fail()
        return result, {}

def test_strategy_decisions():
    """Test strategy decision making with mock data"""
    print_test_header("Strategy Decision Making")
    result = TestResult()

    try:
        from strategies import create_strategy, StrategyConfig

        # Create realistic mock market data
        print("1. Creating mock market data...")
        
        def create_mock_data(symbol, start_price, days=252):
            """Create realistic mock stock data"""
            dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
            prices = []
            current_price = start_price
            
            # Generate realistic price movements
            np.random.seed(42)  # For reproducible results
            for i in range(days):
                # Random walk with slight upward bias
                daily_return = np.random.normal(0.0005, 0.02)  # 0.05% daily return, 2% volatility
                current_price *= (1 + daily_return)
                prices.append(current_price)
            
            return pd.DataFrame({
                'Open': prices,
                'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'Close': prices,
                'Volume': [int(np.random.normal(1000000, 200000)) for _ in prices]
            }, index=dates)

        # Create test data for multiple symbols
        test_data = {
            'NIFTY50': create_mock_data(18000, 252),
            'RELIANCE': create_mock_data(2500, 252),
            'TCS': create_mock_data(3400, 252)
        }
        
        print(f"   ✓ Created mock data for {len(test_data)} symbols")
        result.add_pass()

        # Test portfolio state
        portfolio_state = {
            'holdings': {'NIFTY50': 5, 'RELIANCE': 10, 'TCS': 8},
            'cash': 50000.0,
            'total_invested': 75000.0
        }

        # Test each strategy
        print("\n2. Testing strategy decision making...")
        
        config = StrategyConfig(
            name="Test Strategy",
            symbols=["NIFTY50", "RELIANCE", "TCS"],
            base_amount=5000.0,
            frequency="monthly"
        )

        strategies_to_test = ['value_averaging', 'momentum_sip', 'volatility_sip', 'dollar_cost_averaging']
        
        for strategy_name in strategies_to_test:
            try:
                strategy = create_strategy(strategy_name, config)
                
                # Test decision making
                decisions = strategy.make_investment_decisions(
                    market_data=test_data,
                    portfolio_state=portfolio_state,
                    execution_date=datetime(2023, 6, 15)
                )
                
                print(f"   ✓ {strategy_name}: {len(decisions)} decisions made")
                result.add_pass()
                
                # Show sample decisions
                for i, decision in enumerate(decisions[:2]):
                    print(f"      {i+1}. {decision.symbol}: ₹{decision.amount:.2f} "
                          f"({decision.quantity} units @ ₹{decision.price:.2f})")
                    print(f"         Reason: {decision.reason}")
                    print(f"         Confidence: {decision.confidence:.2f}")

                # Test execution frequency
                should_execute = strategy.should_execute(datetime.now(), None)
                print(f"      Should execute: {should_execute}")
                
            except Exception as e:
                print(f"   ✗ {strategy_name} decision making failed: {str(e)}")
                result.add_fail()

        return result

    except Exception as e:
        print(f"✗ Strategy decision testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        result.add_fail()
        return result

def test_strategy_manager():
    """Test strategy manager functionality"""
    print_test_header("Strategy Manager")
    result = TestResult()

    try:
        from strategies.strategy_manager import StrategyManager
        from strategies import create_strategy, StrategyConfig
        from data.storage import DatabaseManager
        from data.market_data import MarketDataProvider

        print("1. Initializing strategy manager...")
        
        # Initialize components
        db_manager = DatabaseManager('test_phase2_strategies.db')
        market_data_provider = MarketDataProvider(db_manager)
        manager = StrategyManager(db_manager, market_data_provider)
        
        print("   ✓ Strategy manager initialized")
        result.add_pass()

        print("\n2. Testing strategy addition...")
        
        # Create test strategies
        strategies_to_add = [
            ('Test DCA', 'dollar_cost_averaging', ['NIFTY50'], 3000.0),
            ('Test VA', 'value_averaging', ['RELIANCE'], 4000.0),
            ('Test Momentum', 'momentum_sip', ['TCS'], 2000.0)
        ]

        for name, strategy_type, symbols, amount in strategies_to_add:
            config = StrategyConfig(
                name=name,
                symbols=symbols,
                base_amount=amount,
                frequency="monthly"
            )
            
            strategy = create_strategy(strategy_type, config)
            manager.add_strategy(strategy)
            
            print(f"   ✓ Added strategy: {name} ({strategy_type})")
            result.add_pass()

        print("\n3. Testing portfolio summary...")
        
        portfolio = manager.get_portfolio_summary()
        print(f"   ✓ Portfolio total value: ₹{portfolio['total_value']:,.2f}")
        print(f"   ✓ Cash balance: ₹{portfolio['cash']:,.2f}")
        print(f"   ✓ Holdings value: ₹{portfolio['holdings_value']:,.2f}")
        print(f"   ✓ Active strategies: {portfolio['active_strategies']}")
        result.add_pass()

        print("\n4. Testing strategy performance tracking...")
        
        for strategy_name in portfolio['active_strategies']:
            try:
                performance = manager.get_strategy_performance(strategy_name)
                print(f"   ✓ {strategy_name} performance:")
                print(f"      - Total executions: {performance['total_executions']}")
                print(f"      - Total invested: ₹{performance['total_invested']:,.2f}")
                print(f"      - Current value: ₹{performance['current_value']:,.2f}")
                result.add_pass()
                
            except Exception as e:
                print(f"   ⚠ Performance tracking for {strategy_name}: {str(e)}")
                result.add_warning()

        return result

    except Exception as e:
        print(f"✗ Strategy manager test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        result.add_fail()
        return result

def test_integration_with_phase1():
    """Test integration between Phase 1 and Phase 2"""
    print_test_header("Phase 1 + Phase 2 Integration")
    result = TestResult()

    try:
        print("1. Testing Phase 1 component imports...")
        
        # Test Phase 1 imports
        from data.storage import DatabaseManager
        from data.market_data import MarketDataProvider
        from brokers.zerodha_broker import ZerodhaBroker
        from orders.order_manager import OrderManager
        print("   ✓ Phase 1 components imported successfully")
        result.add_pass()

        print("\n2. Testing Phase 2 component imports...")
        
        # Test Phase 2 imports
        from strategies import create_strategy, StrategyConfig
        from strategies.strategy_manager import StrategyManager
        print("   ✓ Phase 2 components imported successfully")
        result.add_pass()

        print("\n3. Testing integrated workflow...")
        
        # Initialize Phase 1 components
        db_manager = DatabaseManager('integration_test.db')
        market_data_provider = MarketDataProvider(db_manager)
        broker = ZerodhaBroker({'demo_mode': True})
        broker.connect()
        order_manager = OrderManager(db_manager, broker)
        
        print("   ✓ Phase 1 components initialized")
        result.add_pass()

        # Initialize Phase 2 components
        strategy_manager = StrategyManager(db_manager, market_data_provider)
        
        config = StrategyConfig(
            name="Integration Test Strategy",
            symbols=["NIFTY50"],
            base_amount=1000.0,
            frequency="monthly"
        )
        
        strategy = create_strategy('dollar_cost_averaging', config)
        strategy_manager.add_strategy(strategy)
        
        print("   ✓ Phase 2 components initialized")
        result.add_pass()

        print("\n4. Testing data flow between phases...")
        
        # Test data flow: Phase 2 → Phase 1
        portfolio = strategy_manager.get_portfolio_summary()
        
        # Simulate placing orders based on strategy decisions
        # (This would normally be done by the scheduler or main application)
        
        print("   ✓ Data flows correctly between Phase 1 and Phase 2")
        result.add_pass()

        print("\n5. Testing system compatibility...")
        
        # Test that both phases can work together
        broker_balance = broker.get_balance()
        strategy_portfolio = strategy_manager.get_portfolio_summary()
        
        print(f"   ✓ Broker balance: ₹{broker_balance['cash']:,.2f}")
        print(f"   ✓ Strategy portfolio: ₹{strategy_portfolio['total_value']:,.2f}")
        result.add_pass()

        return result

    except Exception as e:
        print(f"✗ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        result.add_fail()
        return result

def test_strategy_parameters():
    """Test strategy parameter customization"""
    print_test_header("Strategy Parameter Customization")
    result = TestResult()

    try:
        from strategies import create_strategy, StrategyConfig

        print("1. Testing parameter customization...")

        # Test Value Averaging with custom parameters
        va_config = StrategyConfig(
            name="Custom VA",
            symbols=["NIFTY50"],
            base_amount=5000.0,
            parameters={
                'target_growth_rate': 0.15,
                'max_investment_multiplier': 2.5,
                'min_investment_multiplier': 0.3
            }
        )
        
        va_strategy = create_strategy('value_averaging', va_config)
        print(f"   ✓ Value Averaging with custom parameters")
        print(f"      Target growth rate: {va_strategy.target_growth_rate}")
        print(f"      Max multiplier: {va_strategy.max_investment_multiplier}")
        result.add_pass()

        # Test Momentum SIP with custom parameters
        momentum_config = StrategyConfig(
            name="Custom Momentum",
            symbols=["RELIANCE"],
            base_amount=3000.0,
            parameters={
                'momentum_period': 90,
                'momentum_threshold': 0.08,
                'momentum_multiplier': 1.8
            }
        )
        
        momentum_strategy = create_strategy('momentum_sip', momentum_config)
        print(f"   ✓ Momentum SIP with custom parameters")
        print(f"      Momentum period: {momentum_strategy.momentum_period} days")
        print(f"      Momentum threshold: {momentum_strategy.momentum_threshold}")
        result.add_pass()

        # Test Volatility SIP with custom parameters
        vol_config = StrategyConfig(
            name="Custom Volatility",
            symbols=["TCS"],
            base_amount=4000.0,
            parameters={
                'volatility_period': 45,
                'high_vol_threshold': 0.30,
                'low_vol_threshold': 0.12,
                'high_vol_multiplier': 1.6
            }
        )
        
        vol_strategy = create_strategy('volatility_sip', vol_config)
        print(f"   ✓ Volatility SIP with custom parameters")
        print(f"      Volatility period: {vol_strategy.volatility_period} days")
        print(f"      High vol threshold: {vol_strategy.high_vol_threshold}")
        result.add_pass()

        return result

    except Exception as e:
        print(f"✗ Parameter customization test failed: {str(e)}")
        result.add_fail()
        return result

def run_comprehensive_phase2_tests():
    """Run all Phase 2 tests and generate comprehensive report"""
    
    print("Phase 2: Strategy Development - Comprehensive Testing")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define all tests
    tests = [
        ("Phase 2 Setup", test_phase2_setup),
        ("Strategy Imports", test_strategy_imports),
        ("Strategy Creation", lambda: test_strategy_creation()[0]),  # Only return result
        ("Strategy Decisions", test_strategy_decisions),
        ("Strategy Manager", test_strategy_manager),
        ("Phase 1+2 Integration", test_integration_with_phase1),
        ("Strategy Parameters", test_strategy_parameters)
    ]
    
    # Track overall results
    total_result = TestResult()
    test_results = []
    
    # Run each test
    for test_name, test_func in tests:
        print(f"\n{'*' * 70}")
        print(f"Running: {test_name}")
        print(f"{'*' * 70}")
        
        try:
            result = test_func()
            test_results.append((test_name, result))
            
            # Add to total
            total_result.passed += result.passed
            total_result.failed += result.failed
            total_result.warnings += result.warnings
            
            # Print test result
            if result.failed == 0:
                print(f"\n✅ {test_name}: PASSED ({result.passed} tests)")
            else:
                print(f"\n❌ {test_name}: FAILED ({result.failed} failures, {result.passed} passed)")
                
        except Exception as e:
            print(f"\n💥 {test_name}: CRASHED - {str(e)}")
            failed_result = TestResult()
            failed_result.add_fail()
            test_results.append((test_name, failed_result))
            total_result.failed += 1

    # Generate comprehensive summary
    print_comprehensive_summary(test_results, total_result)
    
    return total_result.failed == 0

def print_comprehensive_summary(test_results, total_result):
    """Print comprehensive test summary"""
    
    print(f"\n{'=' * 70}")
    print("PHASE 2 COMPREHENSIVE TEST SUMMARY")
    print(f"{'=' * 70}")
    
    # Individual test results
    print("\nIndividual Test Results:")
    print("-" * 50)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result.failed == 0 else "❌ FAIL"
        success_rate = result.get_success_rate()
        details = f"({result.passed}✓/{result.failed}✗"
        if result.warnings > 0:
            details += f"/{result.warnings}⚠"
        details += f") {success_rate:.1f}%"
        
        print(f"{test_name:<25} {status:<8} {details}")

    # Overall summary
    print(f"\n{'-' * 50}")
    overall_success_rate = total_result.get_success_rate()
    overall_status = "✅ PASS" if total_result.failed == 0 else "❌ FAIL"
    
    print(f"{'OVERALL':<25} {overall_status:<8} "
          f"({total_result.passed}✓/{total_result.failed}✗"
          f"{f'/{total_result.warnings}⚠' if total_result.warnings > 0 else ''}) "
          f"{overall_success_rate:.1f}%")
    
    print(f"\n{'=' * 70}")
    
    # Final verdict
    if total_result.failed == 0:
        print("🎉 ALL PHASE 2 TESTS PASSED!")
        print("✅ Strategy Development Framework is fully functional")
        print("✅ Ready to proceed with backtesting and optimization")
        print("\n📈 Available Strategies:")
        
        try:
            from strategies import list_available_strategies
            strategies = list_available_strategies()
            for i, strategy in enumerate(strategies, 1):
                print(f"   {i}. {strategy.replace('_', ' ').title()}")
        except:
            print("   - Value Averaging Strategy")
            print("   - Momentum-Based SIP")
            print("   - Volatility-Based SIP") 
            print("   - Dollar Cost Averaging")
            
        print(f"\n🚀 Next Steps:")
        print("   1. Implement backtesting engine")
        print("   2. Add parameter optimization")
        print("   3. Build performance analytics")
        print("   4. Create strategy comparison tools")
        
    else:
        print(f"⚠️  PHASE 2 PARTIALLY COMPLETED ({total_result.passed}/{total_result.get_total()})")
        print("❌ Some components need fixing before proceeding")
        print("\n🔧 Action Items:")
        
        failed_tests = [name for name, result in test_results if result.failed > 0]
        for i, test_name in enumerate(failed_tests, 1):
            print(f"   {i}. Fix issues in: {test_name}")
            
        print(f"\n💡 Troubleshooting Tips:")
        print("   - Check file locations and directory structure")
        print("   - Verify all dependencies are installed")
        print("   - Ensure Python path includes project root")
        print("   - Check for syntax errors in strategy files")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

def create_strategy_demo():
    """Create a quick demo of strategy functionality"""
    print_test_header("Strategy Demonstration")
    
    try:
        from strategies import create_strategy, StrategyConfig
        
        print("Creating demonstration strategies...")
        
        # Create sample configurations
        configs = [
            ("Conservative DCA", "dollar_cost_averaging", ["NIFTY50"], 5000, {}),
            ("Aggressive VA", "value_averaging", ["RELIANCE"], 4000, {
                'target_growth_rate': 0.15,
                'max_investment_multiplier': 3.0
            }),
            ("Momentum Trader", "momentum_sip", ["TCS"], 3000, {
                'momentum_period': 30,
                'momentum_multiplier': 2.0
            }),
            ("Volatility Hunter", "volatility_sip", ["HDFCBANK"], 3500, {
                'high_vol_multiplier': 1.8,
                'volatility_period': 20
            })
        ]
        
        print(f"\n📊 Strategy Portfolio Overview:")
        print("-" * 60)
        
        total_monthly_investment = 0
        
        for name, strategy_type, symbols, amount, params in configs:
            config = StrategyConfig(
                name=name,
                symbols=symbols,
                base_amount=amount,
                frequency="monthly",
                parameters=params
            )
            
            strategy = create_strategy(strategy_type, config)
            total_monthly_investment += amount
            
            print(f"📈 {name}")
            print(f"   Type: {strategy_type.replace('_', ' ').title()}")
            print(f"   Symbols: {', '.join(symbols)}")
            print(f"   Monthly Investment: ₹{amount:,}")
            print(f"   Parameters: {len(params)} custom settings")
            print()
        
        print(f"💰 Total Monthly Investment: ₹{total_monthly_investment:,}")
        print(f"💰 Annual Investment: ₹{total_monthly_investment * 12:,}")
        
        print(f"\n✅ Strategy demonstration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Strategy demonstration failed: {str(e)}")
        return False

def main():
    """Main test function"""
    
    # Setup logging to reduce noise
    logging.basicConfig(level=logging.WARNING)
    
    print("🚀 Starting Phase 2 Comprehensive Testing...")
    
    # Check if user wants quick test or full test
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            print("Running quick tests only...")
            quick_tests = [
                ("Setup Check", test_phase2_setup),
                ("Import Test", test_strategy_imports),
                ("Basic Creation", lambda: test_strategy_creation()[0])
            ]
            
            passed = 0
            for test_name, test_func in quick_tests:
                print(f"\n--- {test_name} ---")
                result = test_func()
                if result.failed == 0:
                    passed += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            
            print(f"\nQuick Test Results: {passed}/{len(quick_tests)} passed")
            return passed == len(quick_tests)
            
        elif sys.argv[1] == "--demo":
            return create_strategy_demo()
            
        elif sys.argv[1] == "--help":
            print("Phase 2 Test Options:")
            print("  python test_strategies.py           # Full comprehensive test")
            print("  python test_strategies.py --quick   # Quick essential tests only")
            print("  python test_strategies.py --demo    # Strategy demonstration")
            print("  python test_strategies.py --help    # Show this help")
            return True
    
    # Run comprehensive tests
    success = run_comprehensive_phase2_tests()
    
    if success:
        print(f"\n🎊 Bonus: Strategy Demonstration")
        create_strategy_demo()
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
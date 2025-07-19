# Create test_emergency_controls.py

def test_emergency_controls():
    """Test all emergency control components"""
    
    print("Testing Emergency Controls System")
    print("=" * 50)
    
    # Test Circuit Breaker
    print("\n1. Testing Circuit Breaker...")
    test_circuit_breaker()
    
    # Test Emergency Stop
    print("\n2. Testing Emergency Stop...")
    test_emergency_stop()
    
    # Test Manual Override
    print("\n3. Testing Manual Override...")
    test_manual_override()
    
    print("\n✅ All emergency controls tests completed!")

def test_circuit_breaker():
    """Test circuit breaker functionality"""
    from controls.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, TriggerCondition, TriggerType
    from datetime import timedelta
    
    # Create test configuration
    config = CircuitBreakerConfig(
        name="test_breaker",
        triggers=[
            TriggerCondition(TriggerType.PORTFOLIO_LOSS, 0.10, timedelta(minutes=5)),
            TriggerCondition(TriggerType.DAILY_LOSS, 0.05, timedelta(hours=1))
        ]
    )
    
    cb = CircuitBreaker(config)
    
    # Test evaluation
    portfolio_state = {
        'total_pnl_pct': -0.12,  # 12% loss - should trigger
        'daily_pnl_pct': -0.03   # 3% daily loss - should not trigger
    }
    
    should_trigger = cb.evaluate_conditions(portfolio_state, {}, {})
    print(f"  ✓ Circuit breaker trigger test: {'TRIGGERED' if should_trigger else 'NOT TRIGGERED'}")
    
    # Test manual trigger
    cb.manual_trigger("Test trigger")
    print(f"  ✓ Manual trigger test: {cb.state.value}")
    
    # Test status
    status = cb.get_status()
    print(f"  ✓ Status check: {status['state']}")

def test_emergency_stop():
    """Test emergency stop functionality"""
    from controls.emergency_stop import EmergencyStop, StopLevel, StopReason
    
    es = EmergencyStop("TEST_SYSTEM")
    
    # Test soft stop
    stop_id = es.emergency_stop(
        StopLevel.SOFT_STOP,
        StopReason.MANUAL_STOP,
        "Test soft stop",
        "test_user"
    )
    
    print(f"  ✓ Soft stop test: {stop_id}")
    print(f"  ✓ Stop status: {'STOPPED' if es.is_stopped else 'RUNNING'}")
    
    # Test recovery
    recovery_started = es.initiate_recovery("test_user", "Test recovery")
    print(f"  ✓ Recovery initiation: {'SUCCESS' if recovery_started else 'FAILED'}")
    
    recovery_completed = es.complete_recovery("Recovery test completed", True)
    print(f"  ✓ Recovery completion: {'SUCCESS' if recovery_completed else 'FAILED'}")

def test_manual_override():
    """Test manual override functionality"""
    from controls.manual_override import ManualOverride, OverrideLevel, OverrideAction
    
    mo = ManualOverride("TEST_SYSTEM")
    
    # Add user permission
    mo.add_user_permission("test_user", "Test User", OverrideLevel.EMERGENCY_CONTROL)
    
    # Start session
    session_id = mo.start_override_session("test_user", "Test User", "Testing manual override")
    print(f"  ✓ Session started: {session_id}")
    
    # Submit request
    request_id = mo.submit_override_request(
        session_id,
        OverrideAction.PAUSE_STRATEGY,
        {"strategy_name": "test_strategy"},
        "Testing override action"
    )
    print(f"  ✓ Override request: {request_id}")
    
    # End session
    mo.end_session(session_id, "Test completed")
    print(f"  ✓ Session ended successfully")

if __name__ == "__main__":
    test_emergency_controls()
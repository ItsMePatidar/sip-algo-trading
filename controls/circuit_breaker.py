import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import json

logger = logging.getLogger(__name__)

class TriggerType(Enum):
    PORTFOLIO_LOSS = "portfolio_loss"
    POSITION_LOSS = "position_loss"
    DAILY_LOSS = "daily_loss"
    VOLATILITY_SPIKE = "volatility_spike"
    DRAWDOWN_LIMIT = "drawdown_limit"
    ORDER_FAILURE_RATE = "order_failure_rate"
    SYSTEM_ERROR_RATE = "system_error_rate"
    MARKET_VOLATILITY = "market_volatility"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CORRELATION_BREAKDOWN = "correlation_breakdown"

class CircuitBreakerState(Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    COOLING_DOWN = "cooling_down"
    DISABLED = "disabled"

@dataclass
class TriggerCondition:
    """Circuit breaker trigger condition"""
    trigger_type: TriggerType
    threshold: float
    measurement_period: timedelta
    evaluation_function: Optional[Callable] = None
    enabled: bool = True
    description: str = ""
    
    def __post_init__(self):
        if not self.description:
            self.description = f"{self.trigger_type.value} threshold: {self.threshold}"

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    name: str
    triggers: List[TriggerCondition] = field(default_factory=list)
    cooldown_period: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    auto_reset: bool = True
    max_triggers_per_day: int = 5
    notification_channels: List[str] = field(default_factory=lambda: ['email', 'log'])
    
@dataclass
class CircuitBreakerEvent:
    """Circuit breaker trigger event"""
    timestamp: datetime
    trigger_type: TriggerType
    current_value: float
    threshold: float
    breach_ratio: float
    portfolio_state: Dict[str, Any]
    market_state: Dict[str, Any]
    action_taken: str
    event_id: str

class CircuitBreaker:
    """
    Circuit Breaker Implementation
    
    Automatically suspends trading when dangerous conditions are detected.
    Provides multiple trigger conditions and automatic recovery mechanisms.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.name = config.name
        self.state = CircuitBreakerState.ACTIVE
        self.triggered_at = None
        self.cooldown_until = None
        self.trigger_count_today = 0
        self.last_reset_date = datetime.now().date()
        
        # Event tracking
        self.trigger_history = []
        self.evaluation_history = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Callbacks
        self.on_trigger_callbacks = []
        self.on_reset_callbacks = []
        
        logger.info(f"Circuit breaker '{self.name}' initialized with {len(config.triggers)} triggers")
    
    def add_trigger_callback(self, callback: Callable):
        """Add callback function for when circuit breaker triggers"""
        self.on_trigger_callbacks.append(callback)
    
    def add_reset_callback(self, callback: Callable):
        """Add callback function for when circuit breaker resets"""
        self.on_reset_callbacks.append(callback)
    
    def evaluate_conditions(self, 
                          portfolio_state: Dict[str, Any], 
                          market_state: Dict[str, Any],
                          system_state: Dict[str, Any]) -> bool:
        """
        Evaluate all trigger conditions
        
        Returns:
            True if circuit breaker should trigger, False otherwise
        """
        with self.lock:
            # Reset daily counter if new day
            self._check_daily_reset()
            
            # Don't evaluate if already triggered or disabled
            if self.state in [CircuitBreakerState.TRIGGERED, CircuitBreakerState.DISABLED]:
                return False
            
            # Check if in cooldown
            if self.state == CircuitBreakerState.COOLING_DOWN:
                if datetime.now() >= self.cooldown_until:
                    self._reset_circuit_breaker()
                return False
            
            # Check daily trigger limit
            if self.trigger_count_today >= self.config.max_triggers_per_day:
                logger.warning(f"Circuit breaker '{self.name}' hit daily trigger limit")
                return False
            
            # Evaluate each trigger condition
            for trigger in self.config.triggers:
                if not trigger.enabled:
                    continue
                
                try:
                    should_trigger = self._evaluate_trigger(
                        trigger, portfolio_state, market_state, system_state
                    )
                    
                    if should_trigger:
                        self._trigger_circuit_breaker(trigger, portfolio_state, market_state)
                        return True
                        
                except Exception as e:
                    logger.error(f"Error evaluating trigger {trigger.trigger_type}: {str(e)}")
            
            return False
    
    def _evaluate_trigger(self, 
                         trigger: TriggerCondition, 
                         portfolio_state: Dict[str, Any],
                         market_state: Dict[str, Any],
                         system_state: Dict[str, Any]) -> bool:
        """Evaluate a specific trigger condition"""
        
        current_time = datetime.now()
        
        if trigger.trigger_type == TriggerType.PORTFOLIO_LOSS:
            current_value = portfolio_state.get('total_pnl_pct', 0)
            return current_value <= -trigger.threshold
        
        elif trigger.trigger_type == TriggerType.POSITION_LOSS:
            positions = portfolio_state.get('positions', {})
            for position in positions.values():
                pnl_pct = position.get('pnl_pct', 0)
                if pnl_pct <= -trigger.threshold:
                    return True
            return False
        
        elif trigger.trigger_type == TriggerType.DAILY_LOSS:
            daily_pnl_pct = portfolio_state.get('daily_pnl_pct', 0)
            return daily_pnl_pct <= -trigger.threshold
        
        elif trigger.trigger_type == TriggerType.DRAWDOWN_LIMIT:
            max_drawdown = portfolio_state.get('max_drawdown', 0)
            return max_drawdown >= trigger.threshold
        
        elif trigger.trigger_type == TriggerType.VOLATILITY_SPIKE:
            portfolio_vol = portfolio_state.get('volatility', 0)
            return portfolio_vol >= trigger.threshold
        
        elif trigger.trigger_type == TriggerType.ORDER_FAILURE_RATE:
            failure_rate = system_state.get('order_failure_rate', 0)
            return failure_rate >= trigger.threshold
        
        elif trigger.trigger_type == TriggerType.SYSTEM_ERROR_RATE:
            error_rate = system_state.get('error_rate', 0)
            return error_rate >= trigger.threshold
        
        elif trigger.trigger_type == TriggerType.MARKET_VOLATILITY:
            market_vol = market_state.get('volatility', 0)
            return market_vol >= trigger.threshold
        
        elif trigger.trigger_type == TriggerType.LIQUIDITY_CRISIS:
            liquidity_score = market_state.get('liquidity_score', 1.0)
            return liquidity_score <= trigger.threshold
        
        elif trigger.trigger_type == TriggerType.CORRELATION_BREAKDOWN:
            correlation_stability = portfolio_state.get('correlation_stability', 1.0)
            return correlation_stability <= trigger.threshold
        
        # Custom evaluation function
        elif trigger.evaluation_function:
            return trigger.evaluation_function(portfolio_state, market_state, system_state)
        
        return False
    
    def _trigger_circuit_breaker(self, 
                                trigger: TriggerCondition,
                                portfolio_state: Dict[str, Any],
                                market_state: Dict[str, Any]):
        """Trigger the circuit breaker"""
        
        trigger_time = datetime.now()
        self.state = CircuitBreakerState.TRIGGERED
        self.triggered_at = trigger_time
        self.trigger_count_today += 1
        
        # Calculate breach ratio
        current_value = self._get_current_value_for_trigger(trigger, portfolio_state, market_state)
        breach_ratio = abs(current_value) / trigger.threshold if trigger.threshold != 0 else float('inf')
        
        # Create event record
        event = CircuitBreakerEvent(
            timestamp=trigger_time,
            trigger_type=trigger.trigger_type,
            current_value=current_value,
            threshold=trigger.threshold,
            breach_ratio=breach_ratio,
            portfolio_state=portfolio_state.copy(),
            market_state=market_state.copy(),
            action_taken="TRADING_SUSPENDED",
            event_id=f"CB_{self.name}_{trigger_time.strftime('%Y%m%d_%H%M%S')}"
        )
        
        self.trigger_history.append(event)
        
        # Log critical event
        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {self.name}")
        logger.critical(f"Trigger: {trigger.trigger_type.value}")
        logger.critical(f"Current Value: {current_value}")
        logger.critical(f"Threshold: {trigger.threshold}")
        logger.critical(f"Breach Ratio: {breach_ratio:.2f}")
        
        # Execute callbacks
        for callback in self.on_trigger_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in trigger callback: {str(e)}")
        
        # Start cooldown if auto-reset is enabled
        if self.config.auto_reset:
            self.cooldown_until = trigger_time + self.config.cooldown_period
            self.state = CircuitBreakerState.COOLING_DOWN
            logger.info(f"Circuit breaker '{self.name}' entering cooldown until {self.cooldown_until}")
    
    def _get_current_value_for_trigger(self, 
                                     trigger: TriggerCondition,
                                     portfolio_state: Dict[str, Any],
                                     market_state: Dict[str, Any]) -> float:
        """Get current value for the trigger type"""
        
        trigger_value_map = {
            TriggerType.PORTFOLIO_LOSS: lambda: portfolio_state.get('total_pnl_pct', 0),
            TriggerType.DAILY_LOSS: lambda: portfolio_state.get('daily_pnl_pct', 0),
            TriggerType.DRAWDOWN_LIMIT: lambda: portfolio_state.get('max_drawdown', 0),
            TriggerType.VOLATILITY_SPIKE: lambda: portfolio_state.get('volatility', 0),
            TriggerType.MARKET_VOLATILITY: lambda: market_state.get('volatility', 0),
        }
        
        getter = trigger_value_map.get(trigger.trigger_type)
        return getter() if getter else 0.0
    
    def manual_trigger(self, reason: str) -> bool:
        """Manually trigger the circuit breaker"""
        with self.lock:
            if self.state == CircuitBreakerState.ACTIVE:
                trigger_time = datetime.now()
                self.state = CircuitBreakerState.TRIGGERED
                self.triggered_at = trigger_time
                self.trigger_count_today += 1
                
                logger.critical(f"Circuit breaker '{self.name}' manually triggered: {reason}")
                return True
            
            return False
    
    def manual_reset(self, reason: str) -> bool:
        """Manually reset the circuit breaker"""
        with self.lock:
            if self.state in [CircuitBreakerState.TRIGGERED, CircuitBreakerState.COOLING_DOWN]:
                self._reset_circuit_breaker()
                logger.info(f"Circuit breaker '{self.name}' manually reset: {reason}")
                return True
            
            return False
    
    def _reset_circuit_breaker(self):
        """Reset the circuit breaker to active state"""
        old_state = self.state
        self.state = CircuitBreakerState.ACTIVE
        self.triggered_at = None
        self.cooldown_until = None
        
        logger.info(f"Circuit breaker '{self.name}' reset from {old_state.value} to active")
        
        # Execute reset callbacks
        for callback in self.on_reset_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in reset callback: {str(e)}")
    
    def _check_daily_reset(self):
        """Check if daily counters should be reset"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.trigger_count_today = 0
            self.last_reset_date = current_date
            logger.info(f"Circuit breaker '{self.name}' daily counters reset")
    
    def disable(self, reason: str):
        """Disable the circuit breaker"""
        with self.lock:
            self.state = CircuitBreakerState.DISABLED
            logger.warning(f"Circuit breaker '{self.name}' disabled: {reason}")
    
    def enable(self, reason: str):
        """Enable the circuit breaker"""
        with self.lock:
            if self.state == CircuitBreakerState.DISABLED:
                self.state = CircuitBreakerState.ACTIVE
                logger.info(f"Circuit breaker '{self.name}' enabled: {reason}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        with self.lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
                'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None,
                'trigger_count_today': self.trigger_count_today,
                'max_triggers_per_day': self.config.max_triggers_per_day,
                'triggers_enabled': len([t for t in self.config.triggers if t.enabled]),
                'total_triggers': len(self.config.triggers),
                'trigger_history_count': len(self.trigger_history),
                'last_trigger': self.trigger_history[-1].timestamp.isoformat() if self.trigger_history else None
            }
    
    def get_trigger_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get trigger history"""
        with self.lock:
            recent_triggers = self.trigger_history[-limit:] if self.trigger_history else []
            return [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'trigger_type': event.trigger_type.value,
                    'current_value': event.current_value,
                    'threshold': event.threshold,
                    'breach_ratio': event.breach_ratio,
                    'event_id': event.event_id
                }
                for event in recent_triggers
            ]
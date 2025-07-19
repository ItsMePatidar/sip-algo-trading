import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class RiskAction(Enum):
    ALLOW = "ALLOW"
    REDUCE_SIZE = "REDUCE_SIZE"
    BLOCK = "BLOCK"
    EMERGENCY_STOP = "EMERGENCY_STOP"

@dataclass
class RiskLimit:
    """Risk limit configuration"""
    name: str
    limit_type: str
    threshold: float
    action: RiskAction
    enabled: bool = True
    description: str = ""

@dataclass
class RiskViolation:
    """Risk violation details"""
    limit_name: str
    current_value: float
    threshold: float
    violation_ratio: float
    risk_level: RiskLevel
    action: RiskAction
    timestamp: datetime
    description: str

@dataclass
class PortfolioState:
    """Current portfolio state for risk calculations"""
    total_value: float
    cash_available: float
    positions: Dict[str, Dict[str, Any]]
    daily_pnl: float
    inception_pnl: float
    peak_value: float
    current_drawdown: float
    last_updated: datetime

class RiskEngine:
    """
    Core Risk Management Engine
    
    Provides comprehensive risk management including:
    - Pre-trade risk checks
    - Position and concentration limits
    - Drawdown monitoring
    - Volatility controls
    - Correlation limits
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.risk_limits = {}
        self.violations_history = []
        self.emergency_stop_active = False
        self.risk_metrics_cache = {}
        
        # Initialize default risk limits
        self._initialize_default_limits()
        
        logger.info("Risk Engine initialized")
    
    def _initialize_default_limits(self):
        """Initialize default risk limits"""
        default_limits = [
            # Position limits
            RiskLimit("max_single_position", "position", 0.25, RiskAction.BLOCK,
                     description="Maximum 25% in single position"),
            RiskLimit("max_sector_concentration", "concentration", 0.40, RiskAction.REDUCE_SIZE,
                     description="Maximum 40% in single sector"),
            
            # Portfolio limits
            RiskLimit("max_portfolio_drawdown", "drawdown", 0.15, RiskAction.EMERGENCY_STOP,
                     description="Maximum 15% portfolio drawdown"),
            RiskLimit("daily_loss_limit", "daily_loss", 0.05, RiskAction.BLOCK,
                     description="Maximum 5% daily loss"),
            
            # Volatility limits
            RiskLimit("max_portfolio_volatility", "volatility", 0.30, RiskAction.REDUCE_SIZE,
                     description="Maximum 30% portfolio volatility"),
            
            # Concentration limits
            RiskLimit("max_correlation_exposure", "correlation", 0.80, RiskAction.REDUCE_SIZE,
                     description="Maximum 80% correlated exposure"),
        ]
        
        for limit in default_limits:
            self.risk_limits[limit.name] = limit
    
    def validate_order(self, order: Dict[str, Any], portfolio_state: PortfolioState) -> Tuple[bool, List[RiskViolation]]:
        """
        Validate order against all risk limits
        
        Args:
            order: Order details (symbol, side, quantity, price)
            portfolio_state: Current portfolio state
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        if self.emergency_stop_active:
            violation = RiskViolation(
                limit_name="emergency_stop",
                current_value=1.0,
                threshold=0.0,
                violation_ratio=float('inf'),
                risk_level=RiskLevel.CRITICAL,
                action=RiskAction.EMERGENCY_STOP,
                timestamp=datetime.now(),
                description="Emergency stop is active"
            )
            violations.append(violation)
            return False, violations
        
        # Check all risk limits
        checks = [
            self._check_position_limits,
            self._check_concentration_limits,
            self._check_drawdown_limits,
            self._check_daily_loss_limits,
            self._check_cash_availability,
            self._check_volatility_limits
        ]
        
        for check in checks:
            try:
                violation = check(order, portfolio_state)
                if violation:
                    violations.append(violation)
            except Exception as e:
                logger.error(f"Error in risk check {check.__name__}: {str(e)}")
        
        # Determine overall validation result
        blocking_violations = [v for v in violations if v.action in [RiskAction.BLOCK, RiskAction.EMERGENCY_STOP]]
        is_valid = len(blocking_violations) == 0
        
        # Log violations
        if violations:
            self._log_violations(violations, order)
        
        return is_valid, violations
    
    def _check_position_limits(self, order: Dict[str, Any], portfolio_state: PortfolioState) -> Optional[RiskViolation]:
        """Check single position limits"""
        limit = self.risk_limits.get("max_single_position")
        if not limit or not limit.enabled:
            return None
        
        symbol = order.get('symbol')
        side = order.get('side', 'BUY')
        quantity = order.get('quantity', 0)
        price = order.get('price', 0)
        
        if side != 'BUY':
            return None  # Only check for buy orders
        
        # Calculate new position value
        current_position_value = 0
        if symbol in portfolio_state.positions:
            pos = portfolio_state.positions[symbol]
            current_position_value = pos.get('quantity', 0) * pos.get('current_price', 0)
        
        order_value = quantity * price
        new_position_value = current_position_value + order_value
        position_percentage = new_position_value / portfolio_state.total_value
        
        if position_percentage > limit.threshold:
            return RiskViolation(
                limit_name=limit.name,
                current_value=position_percentage,
                threshold=limit.threshold,
                violation_ratio=position_percentage / limit.threshold,
                risk_level=RiskLevel.HIGH,
                action=limit.action,
                timestamp=datetime.now(),
                description=f"Position in {symbol} would be {position_percentage:.2%} of portfolio"
            )
        
        return None
    
    def _check_concentration_limits(self, order: Dict[str, Any], portfolio_state: PortfolioState) -> Optional[RiskViolation]:
        """Check sector concentration limits"""
        limit = self.risk_limits.get("max_sector_concentration")
        if not limit or not limit.enabled:
            return None
        
        # This would require sector classification data
        # For now, return None (placeholder for future implementation)
        return None
    
    def _check_drawdown_limits(self, order: Dict[str, Any], portfolio_state: PortfolioState) -> Optional[RiskViolation]:
        """Check portfolio drawdown limits"""
        limit = self.risk_limits.get("max_portfolio_drawdown")
        if not limit or not limit.enabled:
            return None
        
        current_drawdown = portfolio_state.current_drawdown
        
        if current_drawdown > limit.threshold:
            return RiskViolation(
                limit_name=limit.name,
                current_value=current_drawdown,
                threshold=limit.threshold,
                violation_ratio=current_drawdown / limit.threshold,
                risk_level=RiskLevel.CRITICAL,
                action=limit.action,
                timestamp=datetime.now(),
                description=f"Portfolio drawdown is {current_drawdown:.2%}"
            )
        
        return None
    
    def _check_daily_loss_limits(self, order: Dict[str, Any], portfolio_state: PortfolioState) -> Optional[RiskViolation]:
        """Check daily loss limits"""
        limit = self.risk_limits.get("daily_loss_limit")
        if not limit or not limit.enabled:
            return None
        
        daily_loss_percentage = abs(portfolio_state.daily_pnl) / portfolio_state.total_value
        
        if portfolio_state.daily_pnl < 0 and daily_loss_percentage > limit.threshold:
            return RiskViolation(
                limit_name=limit.name,
                current_value=daily_loss_percentage,
                threshold=limit.threshold,
                violation_ratio=daily_loss_percentage / limit.threshold,
                risk_level=RiskLevel.HIGH,
                action=limit.action,
                timestamp=datetime.now(),
                description=f"Daily loss is {daily_loss_percentage:.2%}"
            )
        
        return None
    
    def _check_cash_availability(self, order: Dict[str, Any], portfolio_state: PortfolioState) -> Optional[RiskViolation]:
        """Check cash availability"""
        side = order.get('side', 'BUY')
        quantity = order.get('quantity', 0)
        price = order.get('price', 0)
        
        if side != 'BUY':
            return None
        
        order_value = quantity * price
        required_cash = order_value * 1.02  # Add 2% buffer for fees
        
        if required_cash > portfolio_state.cash_available:
            return RiskViolation(
                limit_name="insufficient_cash",
                current_value=portfolio_state.cash_available,
                threshold=required_cash,
                violation_ratio=required_cash / portfolio_state.cash_available,
                risk_level=RiskLevel.HIGH,
                action=RiskAction.BLOCK,
                timestamp=datetime.now(),
                description=f"Insufficient cash: need {required_cash:.2f}, have {portfolio_state.cash_available:.2f}"
            )
        
        return None
    
    def _check_volatility_limits(self, order: Dict[str, Any], portfolio_state: PortfolioState) -> Optional[RiskViolation]:
        """Check portfolio volatility limits"""
        # Placeholder for volatility check
        # Would require historical returns calculation
        return None
    
    def _log_violations(self, violations: List[RiskViolation], order: Dict[str, Any]):
        """Log risk violations"""
        for violation in violations:
            self.violations_history.append(violation)
            logger.warning(f"Risk violation: {violation.limit_name} - {violation.description}")
    
    def emergency_stop(self, reason: str):
        """Activate emergency stop"""
        self.emergency_stop_active = True
        logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
        
        # Record violation
        violation = RiskViolation(
            limit_name="emergency_stop",
            current_value=1.0,
            threshold=0.0,
            violation_ratio=float('inf'),
            risk_level=RiskLevel.CRITICAL,
            action=RiskAction.EMERGENCY_STOP,
            timestamp=datetime.now(),
            description=f"Emergency stop: {reason}"
        )
        self.violations_history.append(violation)
    
    def reset_emergency_stop(self, reason: str):
        """Reset emergency stop"""
        self.emergency_stop_active = False
        logger.info(f"Emergency stop reset: {reason}")
    
    def get_risk_summary(self, portfolio_state: PortfolioState) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        summary = {
            'emergency_stop_active': self.emergency_stop_active,
            'total_violations_today': len([v for v in self.violations_history 
                                         if v.timestamp.date() == datetime.now().date()]),
            'current_drawdown': portfolio_state.current_drawdown,
            'daily_pnl_percentage': portfolio_state.daily_pnl / portfolio_state.total_value,
            'cash_utilization': (portfolio_state.total_value - portfolio_state.cash_available) / portfolio_state.total_value,
            'largest_position': self._get_largest_position_percentage(portfolio_state),
            'risk_limits_status': {name: limit.enabled for name, limit in self.risk_limits.items()},
            'recent_violations': self.violations_history[-10:] if self.violations_history else []
        }
        
        return summary
    
    def _get_largest_position_percentage(self, portfolio_state: PortfolioState) -> float:
        """Get largest position as percentage of portfolio"""
        if not portfolio_state.positions:
            return 0.0
        
        largest_value = 0
        for position in portfolio_state.positions.values():
            position_value = position.get('quantity', 0) * position.get('current_price', 0)
            largest_value = max(largest_value, position_value)
        
        return largest_value / portfolio_state.total_value if portfolio_state.total_value > 0 else 0.0
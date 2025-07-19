# =============================================================================
# risk_management/drawdown_control.py - Drawdown Control System
# =============================================================================

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class DrawdownSeverity(Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class DrawdownAction(Enum):
    CONTINUE = "CONTINUE"
    REDUCE_SIZE = "REDUCE_SIZE"
    PAUSE_TRADING = "PAUSE_TRADING"
    EMERGENCY_STOP = "EMERGENCY_STOP"

@dataclass
class DrawdownEvent:
    """Represents a drawdown event"""
    start_date: datetime
    end_date: Optional[datetime]
    peak_value: float
    trough_value: float
    max_drawdown: float
    duration_days: int
    recovery_date: Optional[datetime] = None
    severity: DrawdownSeverity = DrawdownSeverity.NORMAL
    is_active: bool = True

@dataclass
class DrawdownConfig:
    """Configuration for drawdown control"""
    # Drawdown thresholds
    caution_threshold: float = 0.05    # 5% - start monitoring closely
    warning_threshold: float = 0.10    # 10% - reduce position sizes
    critical_threshold: float = 0.15   # 15% - pause new trading
    emergency_threshold: float = 0.20  # 20% - emergency stop
    
    # Action parameters
    size_reduction_factor: float = 0.5  # Reduce position sizes by 50%
    pause_duration_days: int = 7        # Pause trading for 7 days
    recovery_buffer: float = 0.02       # 2% buffer before resuming
    
    # Monitoring parameters
    lookback_days: int = 252           # 1 year lookback for analysis
    min_recovery_days: int = 5         # Minimum days for recovery confirmation
    volatility_adjustment: bool = True  # Adjust thresholds based on volatility

class DrawdownController:
    """
    Comprehensive Drawdown Control System
    
    Features:
    - Real-time drawdown monitoring
    - Dynamic threshold adjustments
    - Progressive risk reduction
    - Recovery detection
    - Historical drawdown analysis
    """
    
    def __init__(self, config: DrawdownConfig = None):
        self.config = config or DrawdownConfig()
        self.current_drawdown_event = None
        self.drawdown_history = []
        self.portfolio_value_history = []
        self.peak_value = 0.0
        self.current_value = 0.0
        self.is_trading_paused = False
        self.pause_until_date = None
        self.size_reduction_active = False
        
        logger.info("Drawdown Controller initialized")
    
    def update_portfolio_value(self, portfolio_value: float, timestamp: datetime = None) -> Dict[str, Any]:
        """
        Update portfolio value and check for drawdown conditions
        
        Args:
            portfolio_value: Current portfolio value
            timestamp: Update timestamp (defaults to now)
            
        Returns:
            Dictionary with drawdown status and recommended actions
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update value history
        self.current_value = portfolio_value
        self.portfolio_value_history.append({
            'timestamp': timestamp,
            'value': portfolio_value
        })
        
        # Keep only recent history
        cutoff_date = timestamp - timedelta(days=self.config.lookback_days)
        self.portfolio_value_history = [
            h for h in self.portfolio_value_history 
            if h['timestamp'] >= cutoff_date
        ]
        
        # Update peak value
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            
            # Check if we're recovering from a drawdown
            if self.current_drawdown_event and self.current_drawdown_event.is_active:
                self._check_recovery(timestamp)
        
        # Calculate current drawdown
        current_drawdown = self._calculate_current_drawdown()
        
        # Determine severity and actions
        severity = self._determine_severity(current_drawdown)
        actions = self._determine_actions(severity, current_drawdown, timestamp)
        
        # Update or create drawdown event
        self._update_drawdown_event(current_drawdown, timestamp, severity)
        
        # Update control states
        self._update_control_states(actions, timestamp)
        
        return {
            'current_drawdown': current_drawdown,
            'severity': severity,
            'actions': actions,
            'is_trading_paused': self.is_trading_paused,
            'size_reduction_active': self.size_reduction_active,
            'peak_value': self.peak_value,
            'current_value': self.current_value,
            'drawdown_event': self.current_drawdown_event
        }
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if self.peak_value <= 0:
            return 0.0
        
        drawdown = (self.peak_value - self.current_value) / self.peak_value
        return max(0.0, drawdown)  # Ensure non-negative
    
    def _determine_severity(self, drawdown: float) -> DrawdownSeverity:
        """Determine drawdown severity based on thresholds"""
        # Adjust thresholds based on volatility if enabled
        thresholds = self._get_adjusted_thresholds()
        
        if drawdown >= thresholds['emergency']:
            return DrawdownSeverity.EMERGENCY
        elif drawdown >= thresholds['critical']:
            return DrawdownSeverity.CRITICAL
        elif drawdown >= thresholds['warning']:
            return DrawdownSeverity.WARNING
        elif drawdown >= thresholds['caution']:
            return DrawdownSeverity.CAUTION
        else:
            return DrawdownSeverity.NORMAL
    
    def _get_adjusted_thresholds(self) -> Dict[str, float]:
        """Get volatility-adjusted thresholds"""
        base_thresholds = {
            'caution': self.config.caution_threshold,
            'warning': self.config.warning_threshold,
            'critical': self.config.critical_threshold,
            'emergency': self.config.emergency_threshold
        }
        
        if not self.config.volatility_adjustment or len(self.portfolio_value_history) < 30:
            return base_thresholds
        
        # Calculate portfolio volatility
        volatility = self._calculate_portfolio_volatility()
        
        # Adjust thresholds based on volatility
        # Higher volatility = higher thresholds (more tolerance)
        vol_adjustment = min(2.0, max(0.5, volatility / 0.20))  # Normalize around 20% volatility
        
        adjusted_thresholds = {}
        for key, threshold in base_thresholds.items():
            adjusted_thresholds[key] = threshold * vol_adjustment
        
        return adjusted_thresholds
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility from recent history"""
        if len(self.portfolio_value_history) < 2:
            return 0.20  # Default 20% volatility
        
        values = [h['value'] for h in self.portfolio_value_history[-60:]]  # Last 60 periods
        returns = pd.Series(values).pct_change().dropna()
        
        if len(returns) < 2:
            return 0.20
        
        # Annualized volatility (assuming daily data)
        volatility = returns.std() * np.sqrt(252)
        return volatility
    
    def _determine_actions(self, severity: DrawdownSeverity, drawdown: float, timestamp: datetime) -> List[DrawdownAction]:
        """Determine required actions based on severity"""
        actions = []
        
        if severity == DrawdownSeverity.EMERGENCY:
            actions.append(DrawdownAction.EMERGENCY_STOP)
        elif severity == DrawdownSeverity.CRITICAL:
            actions.append(DrawdownAction.PAUSE_TRADING)
        elif severity == DrawdownSeverity.WARNING:
            actions.append(DrawdownAction.REDUCE_SIZE)
        elif severity == DrawdownSeverity.CAUTION:
            # Monitor more closely but continue
            pass
        
        # Check if trading should remain paused
        if self.is_trading_paused and timestamp < self.pause_until_date:
            if DrawdownAction.PAUSE_TRADING not in actions:
                actions.append(DrawdownAction.PAUSE_TRADING)
        
        if not actions:
            actions.append(DrawdownAction.CONTINUE)
        
        return actions
    
    def _update_drawdown_event(self, drawdown: float, timestamp: datetime, severity: DrawdownSeverity):
        """Update current drawdown event"""
        if drawdown > 0.001:  # Significant drawdown (0.1%)
            if self.current_drawdown_event is None or not self.current_drawdown_event.is_active:
                # Start new drawdown event
                self.current_drawdown_event = DrawdownEvent(
                    start_date=timestamp,
                    end_date=None,
                    peak_value=self.peak_value,
                    trough_value=self.current_value,
                    max_drawdown=drawdown,
                    duration_days=0,
                    severity=severity,
                    is_active=True
                )
                logger.warning(f"New drawdown event started: {drawdown:.2%}")
            else:
                # Update existing event
                self.current_drawdown_event.trough_value = min(
                    self.current_drawdown_event.trough_value, 
                    self.current_value
                )
                self.current_drawdown_event.max_drawdown = max(
                    self.current_drawdown_event.max_drawdown, 
                    drawdown
                )
                self.current_drawdown_event.duration_days = (
                    timestamp - self.current_drawdown_event.start_date
                ).days
                self.current_drawdown_event.severity = max(
                    self.current_drawdown_event.severity, 
                    severity, 
                    key=lambda x: x.value
                )
    
    def _check_recovery(self, timestamp: datetime):
        """Check if portfolio has recovered from drawdown"""
        if not self.current_drawdown_event or not self.current_drawdown_event.is_active:
            return
        
        # Check if we've exceeded the previous peak by recovery buffer
        recovery_threshold = self.current_drawdown_event.peak_value * (1 + self.config.recovery_buffer)
        
        if self.current_value >= recovery_threshold:
            # Confirm recovery over minimum period
            days_since_peak = (timestamp - self.current_drawdown_event.start_date).days
            
            if days_since_peak >= self.config.min_recovery_days:
                self._end_drawdown_event(timestamp)
    
    def _end_drawdown_event(self, timestamp: datetime):
        """End current drawdown event"""
        if self.current_drawdown_event:
            self.current_drawdown_event.is_active = False
            self.current_drawdown_event.end_date = timestamp
            self.current_drawdown_event.recovery_date = timestamp
            
            # Add to history
            self.drawdown_history.append(self.current_drawdown_event)
            
            logger.info(f"Drawdown event ended. Max drawdown: {self.current_drawdown_event.max_drawdown:.2%}, "
                       f"Duration: {self.current_drawdown_event.duration_days} days")
            
            # Reset control states
            self.is_trading_paused = False
            self.pause_until_date = None
            self.size_reduction_active = False
    
    def _update_control_states(self, actions: List[DrawdownAction], timestamp: datetime):
        """Update control states based on actions"""
        if DrawdownAction.EMERGENCY_STOP in actions:
            logger.critical("EMERGENCY STOP triggered by drawdown control")
            
        elif DrawdownAction.PAUSE_TRADING in actions:
            if not self.is_trading_paused:
                self.is_trading_paused = True
                self.pause_until_date = timestamp + timedelta(days=self.config.pause_duration_days)
                logger.warning(f"Trading paused due to drawdown until {self.pause_until_date}")
            
        elif DrawdownAction.REDUCE_SIZE in actions:
            if not self.size_reduction_active:
                self.size_reduction_active = True
                logger.warning("Position size reduction activated due to drawdown")
        
        elif DrawdownAction.CONTINUE in actions:
            # Reset states if continuing normally
            if self.size_reduction_active and self._calculate_current_drawdown() < self.config.caution_threshold:
                self.size_reduction_active = False
                logger.info("Position size reduction deactivated")
    
    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on current state"""
        if self.size_reduction_active:
            return self.config.size_reduction_factor
        return 1.0
    
    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        if self.is_trading_paused:
            if self.pause_until_date and datetime.now() > self.pause_until_date:
                # Auto-resume after pause period
                self.is_trading_paused = False
                self.pause_until_date = None
                logger.info("Trading automatically resumed after pause period")
                return True
            return False
        return True
    
    def force_resume_trading(self, reason: str):
        """Manually resume trading (override pause)"""
        self.is_trading_paused = False
        self.pause_until_date = None
        self.size_reduction_active = False
        logger.info(f"Trading manually resumed: {reason}")
    
    def get_drawdown_statistics(self) -> Dict[str, Any]:
        """Get comprehensive drawdown statistics"""
        all_events = self.drawdown_history.copy()
        if self.current_drawdown_event:
            all_events.append(self.current_drawdown_event)
        
        if not all_events:
            return {
                'total_events': 0,
                'average_drawdown': 0.0,
                'max_drawdown': 0.0,
                'average_duration': 0,
                'recovery_rate': 0.0,
                'current_drawdown': self._calculate_current_drawdown()
            }
        
        completed_events = [e for e in all_events if not e.is_active]
        
        stats = {
            'total_events': len(all_events),
            'active_events': len([e for e in all_events if e.is_active]),
            'completed_events': len(completed_events),
            'current_drawdown': self._calculate_current_drawdown(),
            'max_drawdown_ever': max([e.max_drawdown for e in all_events]),
            'average_drawdown': np.mean([e.max_drawdown for e in all_events]),
            'average_duration': np.mean([e.duration_days for e in completed_events]) if completed_events else 0,
            'longest_drawdown': max([e.duration_days for e in all_events]) if all_events else 0,
            'recovery_rate': len(completed_events) / len(all_events) if all_events else 0,
            'severity_distribution': self._get_severity_distribution(all_events),
            'recent_events': all_events[-5:] if len(all_events) > 5 else all_events
        }
        
        return stats
    
    def _get_severity_distribution(self, events: List[DrawdownEvent]) -> Dict[str, int]:
        """Get distribution of drawdown severities"""
        distribution = {severity.value: 0 for severity in DrawdownSeverity}
        
        for event in events:
            distribution[event.severity.value] += 1
        
        return distribution
    
    def export_drawdown_data(self) -> pd.DataFrame:
        """Export drawdown data for analysis"""
        data = []
        
        for event in self.drawdown_history:
            data.append({
                'start_date': event.start_date,
                'end_date': event.end_date,
                'peak_value': event.peak_value,
                'trough_value': event.trough_value,
                'max_drawdown': event.max_drawdown,
                'duration_days': event.duration_days,
                'recovery_date': event.recovery_date,
                'severity': event.severity.value,
                'is_active': event.is_active
            })
        
        return pd.DataFrame(data)
    
    def simulate_drawdown_scenarios(self, scenarios: List[float]) -> Dict[str, Any]:
        """Simulate different drawdown scenarios"""
        results = {}
        
        for scenario_drawdown in scenarios:
            scenario_name = f"{scenario_drawdown:.1%}_drawdown"
            
            # Simulate portfolio value at this drawdown level
            scenario_value = self.peak_value * (1 - scenario_drawdown)
            
            # Determine what would happen
            severity = self._determine_severity(scenario_drawdown)
            actions = self._determine_actions(severity, scenario_drawdown, datetime.now())
            
            results[scenario_name] = {
                'drawdown': scenario_drawdown,
                'portfolio_value': scenario_value,
                'severity': severity.value,
                'actions': [action.value for action in actions],
                'trading_allowed': DrawdownAction.PAUSE_TRADING not in actions and DrawdownAction.EMERGENCY_STOP not in actions,
                'position_size_multiplier': self.config.size_reduction_factor if DrawdownAction.REDUCE_SIZE in actions else 1.0
            }
        
        return results

# =============================================================================
# Helper Functions for Drawdown Analysis
# =============================================================================

def calculate_maximum_drawdown(price_series: pd.Series) -> Tuple[float, datetime, datetime]:
    """
    Calculate maximum drawdown from a price series
    
    Returns:
        Tuple of (max_drawdown, start_date, end_date)
    """
    # Calculate running maximum (peak)
    running_max = price_series.expanding().max()
    
    # Calculate drawdown series
    drawdown_series = (price_series - running_max) / running_max
    
    # Find maximum drawdown
    max_drawdown = drawdown_series.min()
    max_drawdown_date = drawdown_series.idxmin()
    
    # Find the peak before maximum drawdown
    peak_before_max_dd = running_max.loc[:max_drawdown_date].idxmax()
    
    return abs(max_drawdown), peak_before_max_dd, max_drawdown_date

def analyze_drawdown_periods(price_series: pd.Series, threshold: float = 0.05) -> List[Dict[str, Any]]:
    """
    Analyze all drawdown periods above a threshold
    
    Args:
        price_series: Time series of portfolio values
        threshold: Minimum drawdown threshold to consider
        
    Returns:
        List of drawdown period dictionaries
    """
    running_max = price_series.expanding().max()
    drawdown_series = (price_series - running_max) / running_max
    
    drawdown_periods = []
    in_drawdown = False
    current_period = None
    
    for date, drawdown in drawdown_series.items():
        if abs(drawdown) >= threshold and not in_drawdown:
            # Start of drawdown period
            in_drawdown = True
            current_period = {
                'start_date': date,
                'peak_value': running_max[date],
                'max_drawdown': abs(drawdown),
                'trough_value': price_series[date],
                'trough_date': date
            }
        
        elif abs(drawdown) >= threshold and in_drawdown:
            # Continue drawdown period
            if abs(drawdown) > current_period['max_drawdown']:
                current_period['max_drawdown'] = abs(drawdown)
                current_period['trough_value'] = price_series[date]
                current_period['trough_date'] = date
        
        elif abs(drawdown) < threshold and in_drawdown:
            # End of drawdown period
            in_drawdown = False
            current_period['end_date'] = date
            current_period['recovery_value'] = price_series[date]
            current_period['duration_days'] = (current_period['end_date'] - current_period['start_date']).days
            
            drawdown_periods.append(current_period)
            current_period = None
    
    # Handle ongoing drawdown
    if in_drawdown and current_period:
        current_period['end_date'] = None
        current_period['recovery_value'] = None
        current_period['duration_days'] = (price_series.index[-1] - current_period['start_date']).days
        drawdown_periods.append(current_period)
    
    return drawdown_periods
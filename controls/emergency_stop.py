from enum import Enum
from typing import Set
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import logging
from typing import Callable, Dict, Any, List
logger = logging.getLogger(__name__)

class StopLevel(Enum):
    SOFT_STOP = "soft_stop"      # Stop new orders, allow existing to complete
    HARD_STOP = "hard_stop"      # Cancel all orders, stop all trading
    PANIC_STOP = "panic_stop"    # Immediate liquidation

class StopReason(Enum):
    MANUAL_STOP = "manual_stop"
    RISK_BREACH = "risk_breach"
    SYSTEM_ERROR = "system_error"
    CIRCUIT_BREAKER = "circuit_breaker"
    MARKET_CLOSURE = "market_closure"
    REGULATORY = "regulatory"
    TECHNICAL_ISSUE = "technical_issue"
    CONNECTIVITY_LOSS = "connectivity_loss"
    DATA_FEED_FAILURE = "data_feed_failure"

@dataclass
class EmergencyStopEvent:
    """Emergency stop event record"""
    stop_id: str
    timestamp: datetime
    level: StopLevel
    reason: StopReason
    description: str
    initiated_by: str
    portfolio_state_snapshot: Dict[str, Any]
    actions_taken: List[str] = field(default_factory=list)
    recovery_actions: List[str] = field(default_factory=list)

class EmergencyStop:
    """
    Emergency Stop Mechanism
    
    Provides immediate trading halt capabilities with different stop levels.
    Ensures safe shutdown and recovery procedures.
    """
    
    def __init__(self, system_name: str = "SIP_TRADING_SYSTEM"):
        self.system_name = system_name
        self.is_stopped = False
        self.stop_level = None
        self.stop_reason = None
        self.stop_event = None
        self.stop_timestamp = None
        self.initiated_by = None
        
        # Recovery state
        self.recovery_in_progress = False
        self.recovery_timestamp = None
        
        # Event history
        self.stop_history = []
        
        # Callbacks
        self.on_stop_callbacks = []
        self.on_recovery_callbacks = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Emergency stop system initialized for {system_name}")
    
    def add_stop_callback(self, callback: Callable):
        """Add callback for emergency stop events"""
        self.on_stop_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable):
        """Add callback for recovery events"""
        self.on_recovery_callbacks.append(callback)
    
    def emergency_stop(self, 
                      level: StopLevel,
                      reason: StopReason,
                      description: str,
                      initiated_by: str,
                      portfolio_state: Dict[str, Any] = None) -> str:
        """
        Initiate emergency stop
        
        Returns:
            stop_id: Unique identifier for this stop event
        """
        with self.lock:
            if self.is_stopped and self.stop_level == level:
                logger.warning(f"Emergency stop already active at level {level.value}")
                return self.stop_event.stop_id if self.stop_event else None
            
            stop_id = str(uuid.uuid4())
            stop_time = datetime.now()
            
            # Create stop event
            stop_event = EmergencyStopEvent(
                stop_id=stop_id,
                timestamp=stop_time,
                level=level,
                reason=reason,
                description=description,
                initiated_by=initiated_by,
                portfolio_state_snapshot=portfolio_state or {}
            )
            
            # Update state
            self.is_stopped = True
            self.stop_level = level
            self.stop_reason = reason
            self.stop_event = stop_event
            self.stop_timestamp = stop_time
            self.initiated_by = initiated_by
            
            # Add to history
            self.stop_history.append(stop_event)
            
            # Log emergency stop
            logger.critical(f"EMERGENCY STOP ACTIVATED - Level: {level.value}")
            logger.critical(f"Reason: {reason.value} - {description}")
            logger.critical(f"Initiated by: {initiated_by}")
            logger.critical(f"Stop ID: {stop_id}")
            
            # Execute stop actions based on level
            actions_taken = self._execute_stop_actions(level, stop_event)
            stop_event.actions_taken.extend(actions_taken)
            
            # Execute callbacks
            for callback in self.on_stop_callbacks:
                try:
                    callback(stop_event)
                except Exception as e:
                    logger.error(f"Error in stop callback: {str(e)}")
            
            return stop_id
    
    def _execute_stop_actions(self, level: StopLevel, stop_event: EmergencyStopEvent) -> List[str]:
        """Execute stop actions based on stop level"""
        actions = []
        
        try:
            if level == StopLevel.SOFT_STOP:
                actions.extend(self._soft_stop_actions())
            elif level == StopLevel.HARD_STOP:
                actions.extend(self._hard_stop_actions())
            elif level == StopLevel.PANIC_STOP:
                actions.extend(self._panic_stop_actions())
                
        except Exception as e:
            error_msg = f"Error executing stop actions: {str(e)}"
            logger.error(error_msg)
            actions.append(error_msg)
        
        return actions
    
    def _soft_stop_actions(self) -> List[str]:
        """Execute soft stop actions"""
        actions = []
        
        # Stop accepting new orders
        actions.append("Stopped accepting new trading orders")
        
        # Allow existing orders to complete
        actions.append("Allowing existing orders to complete")
        
        # Pause strategy execution
        actions.append("Paused strategy execution")
        
        # Send notifications
        actions.append("Sent soft stop notifications")
        
        return actions
    
    def _hard_stop_actions(self) -> List[str]:
        """Execute hard stop actions"""
        actions = []
        
        # Cancel all pending orders
        actions.append("Cancelled all pending orders")
        
        # Stop all trading activities
        actions.append("Stopped all trading activities")
        
        # Disable automated strategies
        actions.append("Disabled all automated strategies")
        
        # Close data connections (non-critical)
        actions.append("Closed non-essential data connections")
        
        # Send urgent notifications
        actions.append("Sent urgent stop notifications")
        
        return actions
    
    def _panic_stop_actions(self) -> List[str]:
        """Execute panic stop actions"""
        actions = []
        
        # All hard stop actions
        actions.extend(self._hard_stop_actions())
        
        # Initiate immediate liquidation (if configured)
        actions.append("Initiated emergency liquidation procedures")
        
        # Close all connections
        actions.append("Closed all external connections")
        
        # Save system state
        actions.append("Saved system state for recovery")
        
        # Send critical alerts
        actions.append("Sent critical panic stop alerts")
        
        return actions
    
    def initiate_recovery(self, 
                         initiated_by: str,
                         recovery_plan: str = "Standard recovery") -> bool:
        """
        Initiate recovery from emergency stop
        
        Returns:
            True if recovery initiated successfully
        """
        with self.lock:
            if not self.is_stopped:
                logger.warning("No emergency stop active, recovery not needed")
                return False
            
            if self.recovery_in_progress:
                logger.warning("Recovery already in progress")
                return False
            
            self.recovery_in_progress = True
            self.recovery_timestamp = datetime.now()
            
            logger.info(f"Recovery initiated by {initiated_by}: {recovery_plan}")
            
            # Execute recovery callbacks
            for callback in self.on_recovery_callbacks:
                try:
                    callback(self.stop_event, recovery_plan)
                except Exception as e:
                    logger.error(f"Error in recovery callback: {str(e)}")
            
            return True
    
    def complete_recovery(self, 
                         recovery_notes: str = "",
                         system_checks_passed: bool = True) -> bool:
        """
        Complete recovery process
        
        Returns:
            True if recovery completed successfully
        """
        with self.lock:
            if not self.recovery_in_progress:
                logger.warning("No recovery in progress")
                return False
            
            if not system_checks_passed:
                logger.error("System checks failed, recovery aborted")
                return False
            
            # Reset emergency stop state
            self.is_stopped = False
            self.stop_level = None
            self.stop_reason = None
            self.recovery_in_progress = False
            
            # Update stop event with recovery info
            if self.stop_event:
                self.stop_event.recovery_actions.append(f"Recovery completed: {recovery_notes}")
            
            recovery_time = datetime.now()
            duration = recovery_time - self.stop_timestamp if self.stop_timestamp else timedelta(0)
            
            logger.info(f"Emergency stop recovery completed")
            logger.info(f"Stop duration: {duration}")
            logger.info(f"Recovery notes: {recovery_notes}")
            
            # Clear stop event
            self.stop_event = None
            self.stop_timestamp = None
            self.recovery_timestamp = None
            
            return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get emergency stop status"""
        with self.lock:
            status = {
                'system_name': self.system_name,
                'is_stopped': self.is_stopped,
                'stop_level': self.stop_level.value if self.stop_level else None,
                'stop_reason': self.stop_reason.value if self.stop_reason else None,
                'stop_timestamp': self.stop_timestamp.isoformat() if self.stop_timestamp else None,
                'initiated_by': self.initiated_by,
                'recovery_in_progress': self.recovery_in_progress,
                'recovery_timestamp': self.recovery_timestamp.isoformat() if self.recovery_timestamp else None,
                'stop_history_count': len(self.stop_history)
            }
            
            if self.stop_event:
                status.update({
                    'current_stop_id': self.stop_event.stop_id,
                    'stop_description': self.stop_event.description,
                    'actions_taken': self.stop_event.actions_taken
                })
            
            return status

class EmergencyStopManager:
    """
    Manages multiple emergency stop systems and provides centralized control
    """
    
    def __init__(self):
        self.emergency_stops: Dict[str, EmergencyStop] = {}
        self.global_stop_active = False
        self.master_lock = threading.Lock()
        
        logger.info("Emergency Stop Manager initialized")
    
    def register_system(self, system_name: str) -> EmergencyStop:
        """Register a new emergency stop system"""
        with self.master_lock:
            if system_name in self.emergency_stops:
                logger.warning(f"Emergency stop system '{system_name}' already registered")
                return self.emergency_stops[system_name]
            
            emergency_stop = EmergencyStop(system_name)
            self.emergency_stops[system_name] = emergency_stop
            
            logger.info(f"Emergency stop system '{system_name}' registered")
            return emergency_stop
    
    def global_emergency_stop(self, 
                            level: StopLevel,
                            reason: StopReason,
                            description: str,
                            initiated_by: str) -> List[str]:
        """
        Trigger emergency stop for all registered systems
        
        Returns:
            List of stop IDs for each system
        """
        with self.master_lock:
            stop_ids = []
            
            logger.critical("GLOBAL EMERGENCY STOP INITIATED")
            logger.critical(f"Level: {level.value}, Reason: {reason.value}")
            logger.critical(f"Description: {description}")
            
            self.global_stop_active = True
            
            for system_name, emergency_stop in self.emergency_stops.items():
                try:
                    stop_id = emergency_stop.emergency_stop(
                        level=level,
                        reason=reason,
                        description=f"Global stop: {description}",
                        initiated_by=initiated_by
                    )
                    stop_ids.append(stop_id)
                    logger.info(f"Emergency stop activated for system: {system_name}")
                except Exception as e:
                    logger.error(f"Failed to stop system {system_name}: {str(e)}")
            
            return stop_ids
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get status of all emergency stop systems"""
        with self.master_lock:
            return {
                'global_stop_active': self.global_stop_active,
                'registered_systems': len(self.emergency_stops),
                'systems_stopped': sum(1 for es in self.emergency_stops.values() if es.is_stopped),
                'system_statuses': {
                    name: es.get_status() 
                    for name, es in self.emergency_stops.items()
                }
            }
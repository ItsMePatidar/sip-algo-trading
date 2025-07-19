"""
Emergency Controls Module

This module provides critical safety mechanisms for the SIP trading system:
- Circuit breakers for automatic trading suspension
- Emergency stop mechanisms for immediate halt
- Manual override capabilities for human intervention
"""

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, TriggerCondition
from .emergency_stop import EmergencyStop, EmergencyStopManager, StopReason
from .manual_override import ManualOverride, OverrideSession, OverrideAction

__all__ = [
    'CircuitBreaker',
    'CircuitBreakerConfig', 
    'TriggerCondition',
    'EmergencyStop',
    'EmergencyStopManager',
    'StopReason',
    'ManualOverride',
    'OverrideSession',
    'OverrideAction'
]

__version__ = '1.0.0'
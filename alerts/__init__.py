"""
Alert System for SIP Algorithmic Trading

This package provides comprehensive alerting capabilities including:
- Alert engine for managing alerts
- Multiple notification channels (email, SMS, Slack, etc.)
- Configurable alert rules
- Alert escalation logic
- Alert history and analytics
"""

from .alert_engine import AlertEngine, Alert, AlertLevel, AlertCategory
from .notification import NotificationManager, EmailNotifier, SMSNotifier, SlackNotifier
from .alert_rules import AlertRuleEngine, AlertRule, RuleCondition
from .escalation import EscalationManager, EscalationLevel, EscalationRule

__all__ = [
    'AlertEngine',
    'Alert',
    'AlertLevel',
    'AlertCategory',
    'NotificationManager',
    'EmailNotifier',
    'SMSNotifier',
    'SlackNotifier',
    'AlertRuleEngine',
    'AlertRule',
    'RuleCondition',
    'EscalationManager',
    'EscalationLevel',
    'EscalationRule'
]

__version__ = '1.0.0'
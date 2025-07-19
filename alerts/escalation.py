# =============================================================================
# alerts/escalation.py - Alert Escalation Logic
# =============================================================================

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from queue import Queue, Empty
import uuid

# Import from other alert modules
from .alert_engine import Alert, AlertSeverity, AlertStatus, AlertCategory
from .notification import NotificationManager

logger = logging.getLogger(__name__)

class EscalationAction(Enum):
    """Types of escalation actions"""
    NOTIFY = "NOTIFY"
    CHANGE_SEVERITY = "CHANGE_SEVERITY"
    ADD_RECIPIENTS = "ADD_RECIPIENTS"
    CREATE_TICKET = "CREATE_TICKET"
    CALL_WEBHOOK = "CALL_WEBHOOK"
    EXECUTE_SCRIPT = "EXECUTE_SCRIPT"
    EMERGENCY_STOP = "EMERGENCY_STOP"

class EscalationTrigger(Enum):
    """Escalation triggers"""
    TIME_BASED = "TIME_BASED"
    NO_ACKNOWLEDGMENT = "NO_ACKNOWLEDGMENT"
    NO_RESOLUTION = "NO_RESOLUTION"
    REPEATED_OCCURRENCE = "REPEATED_OCCURRENCE"
    SEVERITY_THRESHOLD = "SEVERITY_THRESHOLD"

@dataclass
class EscalationStep:
    """Single escalation step"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    trigger: EscalationTrigger = EscalationTrigger.TIME_BASED
    delay_minutes: int = 15
    action: EscalationAction = EscalationAction.NOTIFY
    action_config: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[str] = None  # Python expression to evaluate
    enabled: bool = True
    
    def should_execute(self, alert: Alert, escalation_start_time: datetime) -> bool:
        """Check if this escalation step should execute"""
        if not self.enabled:
            return False
        
        now = datetime.now()
        
        # Check trigger conditions
        if self.trigger == EscalationTrigger.TIME_BASED:
            return now >= escalation_start_time + timedelta(minutes=self.delay_minutes)
        
        elif self.trigger == EscalationTrigger.NO_ACKNOWLEDGMENT:
            return (alert.status != AlertStatus.ACKNOWLEDGED and 
                   now >= escalation_start_time + timedelta(minutes=self.delay_minutes))
        
        elif self.trigger == EscalationTrigger.NO_RESOLUTION:
            return (alert.status != AlertStatus.RESOLVED and 
                   now >= escalation_start_time + timedelta(minutes=self.delay_minutes))
        
        elif self.trigger == EscalationTrigger.SEVERITY_THRESHOLD:
            required_severity = self.action_config.get('required_severity', 'HIGH')
            return alert.severity.value == required_severity
        
        # Custom condition evaluation
        if self.condition:
            try:
                # Create evaluation context
                context = {
                    'alert': alert,
                    'now': now,
                    'escalation_start': escalation_start_time,
                    'minutes_elapsed': (now - escalation_start_time).total_seconds() / 60
                }
                return eval(self.condition, {"__builtins__": {}}, context)
            except Exception as e:
                logger.error(f"Error evaluating escalation condition: {str(e)}")
                return False
        
        return False

@dataclass
class EscalationPolicy:
    """Escalation policy definition"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    steps: List[EscalationStep] = field(default_factory=list)
    enabled: bool = True
    
    # Policy matching criteria
    severity_filter: Optional[List[AlertSeverity]] = None
    category_filter: Optional[List[AlertCategory]] = None
    tag_filter: Optional[List[str]] = None
    source_filter: Optional[str] = None  # Regex pattern
    
    # Policy settings
    max_escalations: int = 5
    reset_on_acknowledgment: bool = True
    reset_on_resolution: bool = True
    
    def matches_alert(self, alert: Alert) -> bool:
        """Check if this policy matches the alert"""
        if not self.enabled:
            return False
        
        # Check severity filter
        if self.severity_filter and alert.severity not in self.severity_filter:
            return False
        
        # Check category filter
        if self.category_filter and alert.category not in self.category_filter:
            return False
        
        # Check tag filter (alert must have at least one matching tag)
        if self.tag_filter and not any(tag in alert.tags for tag in self.tag_filter):
            return False
        
        # Check source filter (regex)
        if self.source_filter:
            import re
            if not re.search(self.source_filter, alert.source):
                return False
        
        return True

@dataclass
class EscalationInstance:
    """Running escalation instance"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    alert_id: str = ""
    policy_id: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    executed_steps: List[str] = field(default_factory=list)
    is_active: bool = True
    last_execution_time: Optional[datetime] = None
    execution_count: int = 0

class EscalationManager:
    """
    Alert Escalation Manager
    
    Manages escalation policies and executes escalation actions
    """
    
    def __init__(self, notification_manager: NotificationManager, config: Dict[str, Any] = None):
        self.config = config or {}
        self.notification_manager = notification_manager
        
        # Escalation data
        self.policies: Dict[str, EscalationPolicy] = {}
        self.active_escalations: Dict[str, EscalationInstance] = {}
        self.escalation_history: List[Dict[str, Any]] = []
        
        # Processing
        self.escalation_queue = Queue()
        self.running = False
        self.worker_thread = None
        
        # Statistics
        self.stats = {
            'total_escalations': 0,
            'escalations_by_policy': {},
            'escalations_by_severity': {severity.value: 0 for severity in AlertSeverity},
            'successful_escalations': 0,
            'failed_escalations': 0
        }
        
        # Initialize default policies
        self._initialize_default_policies()
        
        logger.info("Escalation Manager initialized")
    
    def _initialize_default_policies(self):
        """Initialize default escalation policies"""
        
        # Critical alert escalation policy
        critical_policy = EscalationPolicy(
            id="critical_alerts",
            name="Critical Alert Escalation",
            description="Escalation for critical alerts",
            severity_filter=[AlertSeverity.CRITICAL],
            steps=[
                EscalationStep(
                    name="Immediate notification",
                    delay_minutes=0,
                    action=EscalationAction.NOTIFY,
                    action_config={'channels': ['email', 'slack', 'sms']}
                ),
                EscalationStep(
                    name="Management notification",
                    delay_minutes=15,
                    trigger=EscalationTrigger.NO_ACKNOWLEDGMENT,
                    action=EscalationAction.ADD_RECIPIENTS,
                    action_config={'additional_emails': ['manager@company.com']}
                ),
                EscalationStep(
                    name="Emergency escalation",
                    delay_minutes=30,
                    trigger=EscalationTrigger.NO_RESOLUTION,
                    action=EscalationAction.CALL_WEBHOOK,
                    action_config={'webhook_url': 'https://emergency.company.com/alert'}
                )
            ]
        )
        
        # Risk alert escalation policy
        risk_policy = EscalationPolicy(
            id="risk_alerts",
            name="Risk Alert Escalation",
            description="Escalation for risk-related alerts",
            category_filter=[AlertCategory.RISK],
            severity_filter=[AlertSeverity.HIGH, AlertSeverity.CRITICAL],
            steps=[
                EscalationStep(
                    name="Risk team notification",
                    delay_minutes=5,
                    action=EscalationAction.NOTIFY,
                    action_config={'channels': ['email', 'slack']}
                ),
                EscalationStep(
                    name="Increase severity",
                    delay_minutes=20,
                    trigger=EscalationTrigger.NO_ACKNOWLEDGMENT,
                    action=EscalationAction.CHANGE_SEVERITY,
                    action_config={'new_severity': 'CRITICAL'}
                )
            ]
        )
        
        # System alert escalation policy
        system_policy = EscalationPolicy(
            id="system_alerts",
            name="System Alert Escalation",
            description="Escalation for system alerts",
            category_filter=[AlertCategory.SYSTEM],
            steps=[
                EscalationStep(
                    name="DevOps notification",
                    delay_minutes=10,
                    action=EscalationAction.NOTIFY,
                    action_config={'channels': ['slack']}
                ),
                EscalationStep(
                    name="Emergency stop check",
                    delay_minutes=60,
                    trigger=EscalationTrigger.NO_RESOLUTION,
                    action=EscalationAction.EMERGENCY_STOP,
                    condition="alert.severity.value == 'CRITICAL'"
                )
            ]
        )
        
        self.policies['critical_alerts'] = critical_policy
        self.policies['risk_alerts'] = risk_policy
        self.policies['system_alerts'] = system_policy
    
    def start(self):
        """Start escalation processing"""
        if self.running:
            logger.warning("Escalation manager is already running")
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_escalations, daemon=True)
        self.worker_thread.start()
        logger.info("Escalation manager started")
    
    def stop(self):
        """Stop escalation processing"""
        if not self.running:
            return
        
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Escalation manager stopped")
    
    def handle_alert(self, alert: Alert):
        """Handle new alert for escalation"""
        # Find matching policies
        matching_policies = [policy for policy in self.policies.values() 
                           if policy.matches_alert(alert)]
        
        if not matching_policies:
            logger.debug(f"No escalation policies match alert: {alert.id}")
            return
        
        # Create escalation instances
        for policy in matching_policies:
            escalation = EscalationInstance(
                alert_id=alert.id,
                policy_id=policy.id
            )
            
            self.active_escalations[escalation.id] = escalation
            self.escalation_queue.put(escalation.id)
            
            logger.info(f"Escalation started for alert {alert.id} with policy {policy.name}")
    
    def handle_alert_update(self, alert: Alert):
        """Handle alert status updates"""
        # Find active escalations for this alert
        alert_escalations = [esc for esc in self.active_escalations.values() 
                           if esc.alert_id == alert.id and esc.is_active]
        
        for escalation in alert_escalations:
            policy = self.policies.get(escalation.policy_id)
            if not policy:
                continue
            
            # Check if escalation should be reset
            should_reset = False
            
            if policy.reset_on_acknowledgment and alert.status == AlertStatus.ACKNOWLEDGED:
                should_reset = True
                logger.info(f"Escalation reset due to acknowledgment: {escalation.id}")
            
            if policy.reset_on_resolution and alert.status == AlertStatus.RESOLVED:
                should_reset = True
                logger.info(f"Escalation reset due to resolution: {escalation.id}")
            
            if should_reset:
                escalation.is_active = False
                self._record_escalation_completion(escalation, "reset")
    
    def _process_escalations(self):
        """Process escalation queue"""
        while self.running:
            try:
                # Process pending escalations
                self._check_pending_escalations()
                
                # Sleep for a short interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in escalation processing: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def _check_pending_escalations(self):
        """Check and execute pending escalations"""
        for escalation in list(self.active_escalations.values()):
            if not escalation.is_active:
                continue
            
            try:
                self._process_single_escalation(escalation)
            except Exception as e:
                logger.error(f"Error processing escalation {escalation.id}: {str(e)}")
    
    def _process_single_escalation(self, escalation: EscalationInstance):
        """Process a single escalation instance"""
        policy = self.policies.get(escalation.policy_id)
        if not policy:
            escalation.is_active = False
            return
        
        # Get the alert (you'd need to fetch this from your alert engine)
        # For now, we'll create a placeholder
        alert = self._get_alert(escalation.alert_id)
        if not alert:
            escalation.is_active = False
            return
        
        # Check if max escalations reached
        if escalation.execution_count >= policy.max_escalations:
            escalation.is_active = False
            self._record_escalation_completion(escalation, "max_reached")
            return
        
        # Check each step for execution
        for step in policy.steps:
            if step.id in escalation.executed_steps:
                continue  # Already executed
            
            if step.should_execute(alert, escalation.start_time):
                success = self._execute_escalation_step(step, alert, escalation)
                
                if success:
                    escalation.executed_steps.append(step.id)
                    escalation.last_execution_time = datetime.now()
                    escalation.execution_count += 1
                    
                    logger.info(f"Escalation step executed: {step.name} for alert {alert.id}")
                else:
                    logger.error(f"Escalation step failed: {step.name} for alert {alert.id}")
    
    def _execute_escalation_step(self, step: EscalationStep, alert: Alert, escalation: EscalationInstance) -> bool:
        """Execute a single escalation step"""
        try:
            if step.action == EscalationAction.NOTIFY:
                return self._execute_notify_action(step, alert)
            
            elif step.action == EscalationAction.CHANGE_SEVERITY:
                return self._execute_change_severity_action(step, alert)
            
            elif step.action == EscalationAction.ADD_RECIPIENTS:
                return self._execute_add_recipients_action(step, alert)
            
            elif step.action == EscalationAction.CREATE_TICKET:
                return self._execute_create_ticket_action(step, alert)
            
            elif step.action == EscalationAction.CALL_WEBHOOK:
                return self._execute_webhook_action(step, alert)
            
            elif step.action == EscalationAction.EXECUTE_SCRIPT:
                return self._execute_script_action(step, alert)
            
            elif step.action == EscalationAction.EMERGENCY_STOP:
                return self._execute_emergency_stop_action(step, alert)
            
            else:
                logger.warning(f"Unknown escalation action: {step.action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing escalation step: {str(e)}")
            return False
    
    def _execute_notify_action(self, step: EscalationStep, alert: Alert) -> bool:
        """Execute notification action"""
        channels = step.action_config.get('channels', ['email'])
        
        # Create escalation alert
        escalation_alert = Alert(
            title=f"[ESCALATED] {alert.title}",
            message=f"ESCALATION: {step.name}\n\nOriginal Alert:\n{alert.message}",
            severity=alert.severity,
            category=alert.category,
            source=f"escalation:{step.id}",
            metadata={
                'original_alert_id': alert.id,
                'escalation_step': step.name,
                'escalation_action': step.action.value
            }
        )
        
        # Send through specified channels
        results = {}
        for channel in channels:
            if channel in self.notification_manager.channels:
                results[channel] = self.notification_manager.channels[channel].send_notification(escalation_alert)
        
        return any(results.values())
    
    def _execute_change_severity_action(self, step: EscalationStep, alert: Alert) -> bool:
        """Execute change severity action"""
        new_severity_str = step.action_config.get('new_severity', 'HIGH')
        
        try:
            new_severity = AlertSeverity(new_severity_str)
            old_severity = alert.severity
            alert.severity = new_severity
            
            logger.info(f"Alert severity changed from {old_severity.value} to {new_severity.value}")
            return True
            
        except ValueError:
            logger.error(f"Invalid severity value: {new_severity_str}")
            return False
    
    def _execute_add_recipients_action(self, step: EscalationStep, alert: Alert) -> bool:
        """Execute add recipients action"""
        additional_emails = step.action_config.get('additional_emails', [])
        
        # This would modify the notification manager's email configuration
        # For now, just log the action
        logger.info(f"Would add recipients: {additional_emails}")
        return True
    
    def _execute_create_ticket_action(self, step: EscalationStep, alert: Alert) -> bool:
        """Execute create ticket action"""
        ticket_system = step.action_config.get('ticket_system', 'jira')
        
        # This would integrate with external ticket systems
        logger.info(f"Would create ticket in {ticket_system} for alert {alert.id}")
        return True
    
    def _execute_webhook_action(self, step: EscalationStep, alert: Alert) -> bool:
        """Execute webhook action"""
        webhook_url = step.action_config.get('webhook_url')
        
        if not webhook_url:
            return False
        
        try:
            import requests
            
            payload = {
                'alert': alert.to_dict(),
                'escalation_step': step.name,
                'escalation_action': step.action.value,
                'timestamp': datetime.now().isoformat()
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Webhook called successfully: {webhook_url}")
            return True
            
        except Exception as e:
            logger.error(f"Webhook call failed: {str(e)}")
            return False
    
    def _execute_script_action(self, step: EscalationStep, alert: Alert) -> bool:
        """Execute script action"""
        script_path = step.action_config.get('script_path')
        
        if not script_path:
            return False
        
        try:
            import subprocess
            
            # Pass alert data as environment variables
            env = {
                'ALERT_ID': alert.id,
                'ALERT_TITLE': alert.title,
                'ALERT_SEVERITY': alert.severity.value,
                'ALERT_CATEGORY': alert.category.value
            }
            
            result = subprocess.run([script_path], env=env, timeout=60, capture_output=True)
            
            if result.returncode == 0:
                logger.info(f"Script executed successfully: {script_path}")
                return True
            else:
                logger.error(f"Script execution failed: {script_path}")
                return False
                
        except Exception as e:
            logger.error(f"Script execution error: {str(e)}")
            return False
    
    def _execute_emergency_stop_action(self, step: EscalationStep, alert: Alert) -> bool:
        """Execute emergency stop action"""
        stop_reason = f"Emergency stop triggered by escalation: {step.name}"
        
        # This would integrate with your trading system's emergency stop
        logger.critical(f"EMERGENCY STOP TRIGGERED: {stop_reason}")
        
        # You would call your risk engine's emergency stop here
        # self.risk_engine.emergency_stop(stop_reason)
        
        return True
    
    def _get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID (placeholder - integrate with your alert engine)"""
        # This would integrate with your AlertEngine
        # For now, return None
        return None
    
    def _record_escalation_completion(self, escalation: EscalationInstance, reason: str):
        """Record escalation completion"""
        completion_record = {
            'escalation_id': escalation.id,
            'alert_id': escalation.alert_id,
            'policy_id': escalation.policy_id,
            'start_time': escalation.start_time.isoformat(),
            'completion_time': datetime.now().isoformat(),
            'reason': reason,
            'steps_executed': len(escalation.executed_steps),
            'execution_count': escalation.execution_count
        }
        
        self.escalation_history.append(completion_record)
        
        # Keep only last 1000 records
        if len(self.escalation_history) > 1000:
            self.escalation_history = self.escalation_history[-1000:]
        
        logger.info(f"Escalation completed: {escalation.id} - {reason}")
    
    def add_policy(self, policy: EscalationPolicy):
        """Add escalation policy"""
        self.policies[policy.id] = policy
        logger.info(f"Escalation policy added: {policy.name}")
    
    def remove_policy(self, policy_id: str):
        """Remove escalation policy"""
        if policy_id in self.policies:
            del self.policies[policy_id]
            logger.info(f"Escalation policy removed: {policy_id}")
    
    def get_active_escalations(self) -> List[EscalationInstance]:
        """Get active escalations"""
        return [esc for esc in self.active_escalations.values() if esc.is_active]
    
    def get_escalation_stats(self) -> Dict[str, Any]:
        """Get escalation statistics"""
        active_count = len(self.get_active_escalations())
        
        return {
            **self.stats,
            'active_escalations': active_count,
            'total_policies': len(self.policies),
            'escalation_history_count': len(self.escalation_history)
        }
    
    def test_policy(self, policy_id: str, alert: Alert) -> Dict[str, Any]:
        """Test escalation policy with a sample alert"""
        policy = self.policies.get(policy_id)
        if not policy:
            return {'error': 'Policy not found'}
        
        # Check if policy matches alert
        matches = policy.matches_alert(alert)
        
        # Simulate step execution
        steps_that_would_execute = []
        for step in policy.steps:
            if step.should_execute(alert, datetime.now()):
                steps_that_would_execute.append({
                    'step_name': step.name,
                    'action': step.action.value,
                    'delay_minutes': step.delay_minutes
                })
        
        return {
            'policy_matches': matches,
            'steps_to_execute': steps_that_would_execute,
            'policy_name': policy.name
        }

# =============================================================================
# Escalation Configuration Examples
# =============================================================================

def create_sample_escalation_policies() -> List[EscalationPolicy]:
    """Create sample escalation policies for reference"""
    
    policies = []
    
    # Portfolio risk escalation
    portfolio_risk = EscalationPolicy(
        id="portfolio_risk",
        name="Portfolio Risk Escalation",
        description="Escalation for portfolio risk alerts",
        category_filter=[AlertCategory.PORTFOLIO, AlertCategory.RISK],
        severity_filter=[AlertSeverity.HIGH, AlertSeverity.CRITICAL],
        steps=[
            EscalationStep(
                name="Immediate risk team notification",
                delay_minutes=0,
                action=EscalationAction.NOTIFY,
                action_config={'channels': ['email', 'slack']}
            ),
            EscalationStep(
                name="Portfolio manager notification",
                delay_minutes=10,
                trigger=EscalationTrigger.NO_ACKNOWLEDGMENT,
                action=EscalationAction.ADD_RECIPIENTS,
                action_config={'additional_emails': ['portfolio.manager@company.com']}
            ),
            EscalationStep(
                name="Risk committee escalation",
                delay_minutes=30,
                trigger=EscalationTrigger.NO_RESOLUTION,
                action=EscalationAction.CALL_WEBHOOK,
                action_config={'webhook_url': 'https://risk.company.com/escalation'}
            )
        ]
    )
    
    # System failure escalation
    system_failure = EscalationPolicy(
        id="system_failure",
        name="System Failure Escalation",
        description="Escalation for critical system failures",
        category_filter=[AlertCategory.SYSTEM],
        severity_filter=[AlertSeverity.CRITICAL],
        steps=[
            EscalationStep(
                name="DevOps team immediate notification",
                delay_minutes=0,
                action=EscalationAction.NOTIFY,
                action_config={'channels': ['email', 'slack', 'sms']}
            ),
            EscalationStep(
                name="On-call engineer notification",
                delay_minutes=5,
                trigger=EscalationTrigger.NO_ACKNOWLEDGMENT,
                action=EscalationAction.CALL_WEBHOOK,
                action_config={'webhook_url': 'https://oncall.company.com/alert'}
            ),
            EscalationStep(
                name="Emergency trading halt",
                delay_minutes=15,
                trigger=EscalationTrigger.NO_RESOLUTION,
                action=EscalationAction.EMERGENCY_STOP,
                condition="minutes_elapsed >= 15"
            )
        ]
    )
    
    # Compliance escalation
    compliance = EscalationPolicy(
        id="compliance",
        name="Compliance Escalation",
        description="Escalation for compliance violations",
        category_filter=[AlertCategory.COMPLIANCE],
        steps=[
            EscalationStep(
                name="Compliance team notification",
                delay_minutes=0,
                action=EscalationAction.NOTIFY,
                action_config={'channels': ['email']}
            ),
            EscalationStep(
                name="Legal team notification",
                delay_minutes=60,
                trigger=EscalationTrigger.NO_ACKNOWLEDGMENT,
                action=EscalationAction.ADD_RECIPIENTS,
                action_config={'additional_emails': ['legal@company.com']}
            ),
            EscalationStep(
                name="Create compliance ticket",
                delay_minutes=120,
                trigger=EscalationTrigger.NO_RESOLUTION,
                action=EscalationAction.CREATE_TICKET,
                action_config={'ticket_system': 'compliance_tracker'}
            )
        ]
    )
    
    policies.extend([portfolio_risk, system_failure, compliance])
    return policies

# =============================================================================
# Escalation Testing and Utilities
# =============================================================================

class EscalationTester:
    """Test escalation functionality"""
    
    def __init__(self, escalation_manager: EscalationManager):
        self.escalation_manager = escalation_manager
    
    def test_all_policies(self) -> Dict[str, Any]:
        """Test all escalation policies"""
        results = {}
        
        # Create test alert
        test_alert = Alert(
            title="Test Alert for Escalation",
            message="This is a test alert to verify escalation functionality",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.SYSTEM,
            source="escalation_test"
        )
        
        for policy_id, policy in self.escalation_manager.policies.items():
            results[policy_id] = self.escalation_manager.test_policy(policy_id, test_alert)
        
        return results
    
    def simulate_escalation(self, policy_id: str, alert: Alert) -> List[str]:
        """Simulate escalation execution"""
        policy = self.escalation_manager.policies.get(policy_id)
        if not policy:
            return ["Policy not found"]
        
        simulation_log = []
        start_time = datetime.now()
        
        for step in policy.steps:
            if step.should_execute(alert, start_time):
                simulation_log.append(f"Would execute: {step.name} ({step.action.value})")
            else:
                simulation_log.append(f"Would skip: {step.name} (conditions not met)")
        
        return simulation_log

def main():
    """Test escalation system"""
    from .notification import NotificationManager
    
    # Create notification manager
    notification_config = {
        'channels': {
            'email': {'enabled': False},  # Disable for testing
            'slack': {'enabled': False}
        }
    }
    notification_manager = NotificationManager(notification_config)
    
    # Create escalation manager
    escalation_manager = EscalationManager(notification_manager)
    escalation_manager.start()
    
    # Test escalation
    test_alert = Alert(
        title="Test Critical Alert",
        message="This is a test critical alert",
        severity=AlertSeverity.CRITICAL,
        category=AlertCategory.SYSTEM,
        source="test"
    )
    
    escalation_manager.handle_alert(test_alert)
    
    # Wait a bit and check status
    import time
    time.sleep(2)
    
    active_escalations = escalation_manager.get_active_escalations()
    print(f"Active escalations: {len(active_escalations)}")
    
    stats = escalation_manager.get_escalation_stats()
    print(f"Escalation stats: {stats}")
    
    escalation_manager.stop()

if __name__ == "__main__":
    main()
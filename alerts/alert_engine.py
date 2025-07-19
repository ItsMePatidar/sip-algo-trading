import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import threading
from queue import Queue, Empty

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    SUPPRESSED = "SUPPRESSED"

class AlertCategory(Enum):
    """Alert categories"""
    PORTFOLIO = "PORTFOLIO"
    RISK = "RISK"
    MARKET = "MARKET"
    SYSTEM = "SYSTEM"
    STRATEGY = "STRATEGY"
    COMPLIANCE = "COMPLIANCE"

@dataclass
class Alert:
    """Alert data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    message: str = ""
    severity: AlertSeverity = AlertSeverity.INFO
    category: AlertCategory = AlertCategory.SYSTEM
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def acknowledge(self, user: str = "system"):
        """Acknowledge the alert"""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now()
        self.acknowledged_by = user
        self.updated_at = datetime.now()
    
    def resolve(self, user: str = "system"):
        """Resolve the alert"""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now()
        self.updated_at = datetime.now()
    
    def suppress(self):
        """Suppress the alert"""
        self.status = AlertStatus.SUPPRESSED
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by,
            'source': self.source,
            'metadata': self.metadata,
            'tags': self.tags
        }

class AlertEngine:
    """
    Core Alert Management Engine
    
    Manages alert lifecycle, storage, and processing
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_queue = Queue()
        self.alert_handlers: List[Callable] = []
        self.running = False
        self.worker_thread = None
        
        # Alert statistics
        self.stats = {
            'total_alerts': 0,
            'alerts_by_severity': {severity.value: 0 for severity in AlertSeverity},
            'alerts_by_category': {category.value: 0 for category in AlertCategory},
            'alerts_by_status': {status.value: 0 for status in AlertStatus}
        }
        
        # Alert rate limiting
        self.rate_limits = self.config.get('rate_limits', {})
        self.alert_counts = {}
        
        logger.info("Alert Engine initialized")
    
    def start(self):
        """Start alert processing"""
        if self.running:
            logger.warning("Alert engine is already running")
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self.worker_thread.start()
        logger.info("Alert engine started")
    
    def stop(self):
        """Stop alert processing"""
        if not self.running:
            return
        
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Alert engine stopped")
    
    def create_alert(self, 
                    title: str,
                    message: str,
                    severity: AlertSeverity = AlertSeverity.INFO,
                    category: AlertCategory = AlertCategory.SYSTEM,
                    source: str = "",
                    metadata: Dict[str, Any] = None,
                    tags: List[str] = None) -> Alert:
        """Create a new alert"""
        
        # Check rate limiting
        if not self._check_rate_limit(title, severity):
            logger.debug(f"Alert rate limited: {title}")
            return None
        
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            category=category,
            source=source,
            metadata=metadata or {},
            tags=tags or []
        )
        
        # Store alert
        self.alerts[alert.id] = alert
        
        # Queue for processing
        self.alert_queue.put(alert)
        
        # Update statistics
        self._update_stats(alert)
        
        logger.info(f"Alert created: {alert.title} [{alert.severity.value}]")
        return alert
    
    def _check_rate_limit(self, title: str, severity: AlertSeverity) -> bool:
        """Check if alert is rate limited"""
        now = datetime.now()
        
        # Get rate limit for this severity
        limit_key = f"{severity.value.lower()}_alerts"
        rate_limit = self.rate_limits.get(limit_key, {
            'count': 100,
            'window': 3600  # 1 hour
        })
        
        # Clean old entries
        cutoff_time = now - timedelta(seconds=rate_limit['window'])
        
        if title not in self.alert_counts:
            self.alert_counts[title] = []
        
        # Remove old timestamps
        self.alert_counts[title] = [
            timestamp for timestamp in self.alert_counts[title]
            if timestamp > cutoff_time
        ]
        
        # Check if under limit
        if len(self.alert_counts[title]) >= rate_limit['count']:
            return False
        
        # Add current timestamp
        self.alert_counts[title].append(now)
        return True
    
    def _process_alerts(self):
        """Process alerts from queue"""
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1)
                self._handle_alert(alert)
                self.alert_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {str(e)}")
    
    def _handle_alert(self, alert: Alert):
        """Handle a single alert"""
        try:
            # Call all registered handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler error: {str(e)}")
        except Exception as e:
            logger.error(f"Error handling alert {alert.id}: {str(e)}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler"""
        self.alert_handlers.append(handler)
        logger.info(f"Alert handler added: {handler.__name__}")
    
    def remove_alert_handler(self, handler: Callable):
        """Remove alert handler"""
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)
            logger.info(f"Alert handler removed: {handler.__name__}")
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID"""
        return self.alerts.get(alert_id)
    
    def get_alerts(self, 
                  status: Optional[AlertStatus] = None,
                  severity: Optional[AlertSeverity] = None,
                  category: Optional[AlertCategory] = None,
                  limit: int = 100) -> List[Alert]:
        """Get alerts with filters"""
        alerts = list(self.alerts.values())
        
        # Apply filters
        if status:
            alerts = [a for a in alerts if a.status == status]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        # Sort by creation time (newest first)
        alerts.sort(key=lambda a: a.created_at, reverse=True)
        
        return alerts[:limit]
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert"""
        alert = self.get_alert(alert_id)
        if alert:
            alert.acknowledge(user)
            logger.info(f"Alert acknowledged: {alert_id} by {user}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an alert"""
        alert = self.get_alert(alert_id)
        if alert:
            alert.resolve(user)
            logger.info(f"Alert resolved: {alert_id} by {user}")
            return True
        return False
    
    def suppress_alert(self, alert_id: str) -> bool:
        """Suppress an alert"""
        alert = self.get_alert(alert_id)
        if alert:
            alert.suppress()
            logger.info(f"Alert suppressed: {alert_id}")
            return True
        return False
    
    def _update_stats(self, alert: Alert):
        """Update alert statistics"""
        self.stats['total_alerts'] += 1
        self.stats['alerts_by_severity'][alert.severity.value] += 1
        self.stats['alerts_by_category'][alert.category.value] += 1
        self.stats['alerts_by_status'][alert.status.value] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        active_alerts = len([a for a in self.alerts.values() if a.status == AlertStatus.ACTIVE])
        critical_alerts = len([a for a in self.alerts.values() 
                              if a.severity == AlertSeverity.CRITICAL and a.status == AlertStatus.ACTIVE])
        
        return {
            **self.stats,
            'active_alerts': active_alerts,
            'critical_alerts': critical_alerts,
            'total_stored_alerts': len(self.alerts)
        }
    
    def cleanup_old_alerts(self, days: int = 30):
        """Clean up old resolved alerts"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        alerts_to_remove = [
            alert_id for alert_id, alert in self.alerts.items()
            if alert.status == AlertStatus.RESOLVED and alert.resolved_at and alert.resolved_at < cutoff_date
        ]
        
        for alert_id in alerts_to_remove:
            del self.alerts[alert_id]
        
        logger.info(f"Cleaned up {len(alerts_to_remove)} old alerts")
import logging
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from .alert_engine import Alert, AlertSeverity, AlertCategory
logger = logging.getLogger(__name__)

class NotificationChannel(ABC):
    """Abstract base class for notification channels"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.name = config.get('name', self.__class__.__name__)
    
    @abstractmethod
    def send_notification(self, alert: Alert) -> bool:
        """Send notification for alert"""
        pass
    
    def format_message(self, alert: Alert) -> str:
        """Format alert message for this channel"""
        return f"🚨 {alert.title}\n\n{alert.message}\n\nSeverity: {alert.severity.value}\nTime: {alert.created_at}"

class EmailNotification(NotificationChannel):
    """Email notification channel"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username', '')
        self.password = config.get('password', '')
        self.from_email = config.get('from_email', self.username)
        self.to_emails = config.get('to_emails', [])
    
    def send_notification(self, alert: Alert) -> bool:
        """Send email notification"""
        if not self.enabled or not self.to_emails:
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value}] {alert.title}"
            
            # Create HTML body
            html_body = self._create_html_body(alert)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")
            return False
    
    def _create_html_body(self, alert: Alert) -> str:
        """Create HTML email body"""
        severity_colors = {
            'CRITICAL': '#dc3545',
            'HIGH': '#fd7e14',
            'MEDIUM': '#ffc107',
            'LOW': '#28a745',
            'INFO': '#17a2b8'
        }
        
        color = severity_colors.get(alert.severity.value, '#6c757d')
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <div style="border-left: 4px solid {color}; padding-left: 20px;">
                <h2 style="color: {color}; margin-top: 0;">{alert.title}</h2>
                <p><strong>Severity:</strong> <span style="color: {color};">{alert.severity.value}</span></p>
                <p><strong>Category:</strong> {alert.category.value}</p>
                <p><strong>Time:</strong> {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Source:</strong> {alert.source}</p>
                <hr>
                <p>{alert.message}</p>
                {self._format_metadata(alert.metadata)}
            </div>
        </body>
        </html>
        """
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for display"""
        if not metadata:
            return ""
        
        html = "<h4>Additional Details:</h4><ul>"
        for key, value in metadata.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul>"
        return html

class SlackNotification(NotificationChannel):
    """Slack notification channel"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get('webhook_url', '')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'SIP Trading Bot')
    
    def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification"""
        if not self.enabled or not self.webhook_url:
            return False
        
        try:
            # Create Slack message
            payload = {
                'channel': self.channel,
                'username': self.username,
                'attachments': [self._create_slack_attachment(alert)]
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack notification sent for alert: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {str(e)}")
            return False
    
    def _create_slack_attachment(self, alert: Alert) -> Dict[str, Any]:
        """Create Slack attachment"""
        severity_colors = {
            'CRITICAL': 'danger',
            'HIGH': 'warning',
            'MEDIUM': 'warning',
            'LOW': 'good',
            'INFO': '#17a2b8'
        }
        
        color = severity_colors.get(alert.severity.value, 'good')
        
        return {
            'color': color,
            'title': alert.title,
            'text': alert.message,
            'fields': [
                {'title': 'Severity', 'value': alert.severity.value, 'short': True},
                {'title': 'Category', 'value': alert.category.value, 'short': True},
                {'title': 'Source', 'value': alert.source, 'short': True},
                {'title': 'Time', 'value': alert.created_at.strftime('%Y-%m-%d %H:%M:%S'), 'short': True}
            ],
            'footer': 'SIP Trading System',
            'ts': int(alert.created_at.timestamp())
        }

class SMSNotification(NotificationChannel):
    """SMS notification channel (using Twilio or similar service)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.account_sid = config.get('account_sid', '')
        self.auth_token = config.get('auth_token', '')
        self.from_number = config.get('from_number', '')
        self.to_numbers = config.get('to_numbers', [])
    
    def send_notification(self, alert: Alert) -> bool:
        """Send SMS notification"""
        if not self.enabled or not self.to_numbers:
            return False
        
        # Only send SMS for critical alerts to avoid spam
        if alert.severity not in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            return True
        
        try:
            # This would integrate with Twilio or similar service
            # For now, just log the message
            message = f"{alert.severity.value}: {alert.title} - {alert.message[:100]}"
            logger.info(f"SMS would be sent: {message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SMS notification: {str(e)}")
            return False

class WebhookNotification(NotificationChannel):
    """Webhook notification channel"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get('webhook_url', '')
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.timeout = config.get('timeout', 10)
    
    def send_notification(self, alert: Alert) -> bool:
        """Send webhook notification"""
        if not self.enabled or not self.webhook_url:
            return False
        
        try:
            payload = {
                'alert': alert.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'source': 'sip_trading_system'
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent for alert: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {str(e)}")
            return False

class NotificationManager:
    """
    Manages multiple notification channels
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.channels: Dict[str, NotificationChannel] = {}
        self.channel_rules: Dict[str, Dict[str, Any]] = {}
        
        # Initialize channels from config
        self._initialize_channels()
        
        logger.info("Notification Manager initialized")
    
    def _initialize_channels(self):
        """Initialize notification channels from config"""
        channels_config = self.config.get('channels', {})
        
        # Email channel
        if 'email' in channels_config:
            self.channels['email'] = EmailNotification(channels_config['email'])
        
        # Slack channel
        if 'slack' in channels_config:
            self.channels['slack'] = SlackNotification(channels_config['slack'])
        
        # SMS channel
        if 'sms' in channels_config:
            self.channels['sms'] = SMSNotification(channels_config['sms'])
        
        # Webhook channel
        if 'webhook' in channels_config:
            self.channels['webhook'] = WebhookNotification(channels_config['webhook'])
        
        # Channel rules
        self.channel_rules = self.config.get('rules', {
            'critical': ['email', 'slack', 'sms'],
            'high': ['email', 'slack'],
            'medium': ['email'],
            'low': ['email'],
            'info': []
        })
    
    def send_notification(self, alert: Alert) -> Dict[str, bool]:
        """Send notification through appropriate channels"""
        results = {}
        
        # Determine which channels to use
        severity_key = alert.severity.value.lower()
        channels_to_use = self.channel_rules.get(severity_key, [])
        
        # Send through each channel
        for channel_name in channels_to_use:
            if channel_name in self.channels:
                try:
                    result = self.channels[channel_name].send_notification(alert)
                    results[channel_name] = result
                except Exception as e:
                    logger.error(f"Error sending notification via {channel_name}: {str(e)}")
                    results[channel_name] = False
            else:
                logger.warning(f"Channel {channel_name} not configured")
                results[channel_name] = False
        
        return results
    
    def add_channel(self, name: str, channel: NotificationChannel):
        """Add notification channel"""
        self.channels[name] = channel
        logger.info(f"Notification channel added: {name}")
    
    def remove_channel(self, name: str):
        """Remove notification channel"""
        if name in self.channels:
            del self.channels[name]
            logger.info(f"Notification channel removed: {name}")
    
    def test_channels(self) -> Dict[str, bool]:
        """Test all notification channels"""
        test_alert = Alert(
            title="Test Alert",
            message="This is a test notification from SIP Trading System",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
            source="notification_test"
        )
        
        results = {}
        for name, channel in self.channels.items():
            try:
                result = channel.send_notification(test_alert)
                results[name] = result
                logger.info(f"Channel {name} test: {'PASS' if result else 'FAIL'}")
            except Exception as e:
                results[name] = False
                logger.error(f"Channel {name} test failed: {str(e)}")
        
        return results
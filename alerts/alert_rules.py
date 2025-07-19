# =============================================================================
# alerts/alert_rules.py - Alert Rule Definitions and Engine
# =============================================================================

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import operator
import re
import json

from .alert_engine import Alert, AlertSeverity, AlertCategory

logger = logging.getLogger(__name__)

class RuleOperator(Enum):
    """Rule operators"""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    CONTAINS = "contains"
    REGEX = "regex"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    CHANGE_PERCENT = "change_percent"

@dataclass
class RuleCondition:
    """Rule condition definition"""
    field: str
    operator: RuleOperator
    value: Any
    description: str = ""
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate condition against data"""
        field_value = self._get_field_value(data, self.field)
        
        if field_value is None:
            return False
        
        try:
            if self.operator == RuleOperator.EQUALS:
                return field_value == self.value
            elif self.operator == RuleOperator.NOT_EQUALS:
                return field_value != self.value
            elif self.operator == RuleOperator.GREATER_THAN:
                return float(field_value) > float(self.value)
            elif self.operator == RuleOperator.GREATER_EQUAL:
                return float(field_value) >= float(self.value)
            elif self.operator == RuleOperator.LESS_THAN:
                return float(field_value) < float(self.value)
            elif self.operator == RuleOperator.LESS_EQUAL:
                return float(field_value) <= float(self.value)
            elif self.operator == RuleOperator.CONTAINS:
                return str(self.value) in str(field_value)
            elif self.operator == RuleOperator.REGEX:
                return bool(re.search(str(self.value), str(field_value)))
            elif self.operator == RuleOperator.IN:
                return field_value in self.value
            elif self.operator == RuleOperator.NOT_IN:
                return field_value not in self.value
            elif self.operator == RuleOperator.BETWEEN:
                if isinstance(self.value, (list, tuple)) and len(self.value) == 2:
                    return self.value[0] <= float(field_value) <= self.value[1]
            elif self.operator == RuleOperator.CHANGE_PERCENT:
                # For percentage change calculations
                previous_value = self._get_field_value(data, f"{self.field}_previous")
                if previous_value is not None and previous_value != 0:
                    change_percent = abs((float(field_value) - float(previous_value)) / float(previous_value)) * 100
                    return change_percent >= float(self.value)
            
        except Exception as e:
            logger.error(f"Error evaluating condition {self.field} {self.operator.value} {self.value}: {str(e)}")
            return False
        
        return False
    
    def _get_field_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get field value using dot notation"""
        try:
            value = data
            for key in field_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(key)
                elif hasattr(value, key):
                    value = getattr(value, key)
                else:
                    return None
            return value
        except Exception:
            return None

@dataclass
class AlertRule:
    """Alert rule definition"""
    id: str
    name: str
    description: str
    conditions: List[RuleCondition]
    alert_title: str
    alert_message: str
    severity: AlertSeverity
    category: AlertCategory
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes default
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate all conditions (AND logic by default)"""
        if not self.enabled or not self.conditions:
            return False
        
        return all(condition.evaluate(data) for condition in self.conditions)
    
    def create_alert(self, data: Dict[str, Any]) -> Alert:
        """Create alert from rule"""
        # Format message with data
        formatted_message = self._format_message(self.alert_message, data)
        formatted_title = self._format_message(self.alert_title, data)
        
        return Alert(
            title=formatted_title,
            message=formatted_message,
            severity=self.severity,
            category=self.category,
            source=f"rule:{self.id}",
            metadata={
                'rule_id': self.id, 
                'rule_name': self.name, 
                'trigger_data': data,
                **self.metadata
            },
            tags=self.tags
        )
    
    def _format_message(self, template: str, data: Dict[str, Any]) -> str:
        """Format message template with data"""
        try:
            flattened_data = self._flatten_dict(data)
            return template.format(**flattened_data)
        except Exception as e:
            logger.warning(f"Error formatting message template: {str(e)}")
            return template
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary for template formatting"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

class AlertRuleEngine:
    """
    Alert Rule Engine
    
    Manages and evaluates alert rules against portfolio and market data
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rules: Dict[str, AlertRule] = {}
        self.rule_last_triggered: Dict[str, datetime] = {}
        
        # Initialize default rules
        self._initialize_default_rules()
        
        logger.info("Alert Rule Engine initialized")
    
    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        
        # Portfolio Risk Rules
        self.add_rule(AlertRule(
            id="portfolio_drawdown_high",
            name="High Portfolio Drawdown",
            description="Alert when portfolio drawdown exceeds 10%",
            conditions=[
                RuleCondition("portfolio.current_drawdown", RuleOperator.GREATER_THAN, 0.10)
            ],
            alert_title="High Portfolio Drawdown Alert",
            alert_message="Portfolio drawdown is {portfolio_current_drawdown:.2%}, exceeding the 10% threshold",
            severity=AlertSeverity.HIGH,
            category=AlertCategory.RISK,
            tags=["portfolio", "drawdown", "risk"]
        ))
        
        self.add_rule(AlertRule(
            id="portfolio_drawdown_critical",
            name="Critical Portfolio Drawdown",
            description="Alert when portfolio drawdown exceeds 15%",
            conditions=[
                RuleCondition("portfolio.current_drawdown", RuleOperator.GREATER_THAN, 0.15)
            ],
            alert_title="CRITICAL: Portfolio Drawdown",
            alert_message="Portfolio drawdown is {portfolio_current_drawdown:.2%}, exceeding the critical 15% threshold. Immediate action required!",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            tags=["portfolio", "drawdown", "critical", "risk"]
        ))
        
        # Position Concentration Rules
        self.add_rule(AlertRule(
            id="position_concentration_warning",
            name="Position Concentration Warning",
            description="Alert when single position exceeds 20% of portfolio",
            conditions=[
                RuleCondition("position.concentration", RuleOperator.GREATER_THAN, 0.20)
            ],
            alert_title="Position Concentration Warning",
            alert_message="Position {position_symbol} represents {position_concentration:.2%} of portfolio, exceeding 20% limit",
            severity=AlertSeverity.MEDIUM,
            category=AlertCategory.RISK,
            tags=["position", "concentration", "risk"]
        ))
        
        self.add_rule(AlertRule(
            id="position_concentration_critical",
            name="Critical Position Concentration",
            description="Alert when single position exceeds 30% of portfolio",
            conditions=[
                RuleCondition("position.concentration", RuleOperator.GREATER_THAN, 0.30)
            ],
            alert_title="CRITICAL: Position Concentration",
            alert_message="Position {position_symbol} represents {position_concentration:.2%} of portfolio, exceeding critical 30% limit",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            tags=["position", "concentration", "critical", "risk"]
        ))
        
        # Daily Loss Rules
        self.add_rule(AlertRule(
            id="daily_loss_warning",
            name="Daily Loss Warning",
            description="Alert when daily loss exceeds 3%",
            conditions=[
                RuleCondition("portfolio.daily_pnl_percent", RuleOperator.LESS_THAN, -0.03)
            ],
            alert_title="Daily Loss Warning",
            alert_message="Daily portfolio loss is {portfolio_daily_pnl_percent:.2%}, exceeding 3% threshold",
            severity=AlertSeverity.MEDIUM,
            category=AlertCategory.PORTFOLIO,
            tags=["daily_loss", "portfolio", "risk"]
        ))
        
        self.add_rule(AlertRule(
            id="daily_loss_critical",
            name="Critical Daily Loss",
            description="Alert when daily loss exceeds 5%",
            conditions=[
                RuleCondition("portfolio.daily_pnl_percent", RuleOperator.LESS_THAN, -0.05)
            ],
            alert_title="CRITICAL: Daily Loss",
            alert_message="Daily portfolio loss is {portfolio_daily_pnl_percent:.2%}, exceeding critical 5% threshold",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.PORTFOLIO,
            tags=["daily_loss", "portfolio", "critical", "risk"]
        ))
        
        # Market Volatility Rules
        self.add_rule(AlertRule(
            id="market_volatility_high",
            name="High Market Volatility",
            description="Alert when market volatility exceeds 30%",
            conditions=[
                RuleCondition("market.volatility", RuleOperator.GREATER_THAN, 0.30)
            ],
            alert_title="High Market Volatility Alert",
            alert_message="Market volatility is {market_volatility:.2%}, indicating increased market risk",
            severity=AlertSeverity.MEDIUM,
            category=AlertCategory.MARKET,
            tags=["market", "volatility", "risk"]
        ))
        
        # Cash Availability Rules
        self.add_rule(AlertRule(
            id="low_cash_warning",
            name="Low Cash Warning",
            description="Alert when available cash is below 5% of portfolio",
            conditions=[
                RuleCondition("portfolio.cash_percentage", RuleOperator.LESS_THAN, 0.05)
            ],
            alert_title="Low Cash Warning",
            alert_message="Available cash is {portfolio_cash_percentage:.2%} of portfolio, below 5% threshold",
            severity=AlertSeverity.MEDIUM,
            category=AlertCategory.PORTFOLIO,
            tags=["cash", "liquidity", "portfolio"]
        ))
        
        # System Health Rules
        self.add_rule(AlertRule(
            id="order_failure_rate_high",
            name="High Order Failure Rate",
            description="Alert when order failure rate exceeds 10%",
            conditions=[
                RuleCondition("system.order_failure_rate", RuleOperator.GREATER_THAN, 0.10)
            ],
            alert_title="High Order Failure Rate",
            alert_message="Order failure rate is {system_order_failure_rate:.2%}, indicating system issues",
            severity=AlertSeverity.HIGH,
            category=AlertCategory.SYSTEM,
            tags=["system", "orders", "failure"]
        ))
        
        # Strategy Performance Rules
        self.add_rule(AlertRule(
            id="strategy_underperformance",
            name="Strategy Underperformance",
            description="Alert when strategy underperforms benchmark by more than 5%",
            conditions=[
                RuleCondition("strategy.relative_performance", RuleOperator.LESS_THAN, -0.05)
            ],
            alert_title="Strategy Underperformance Alert",
            alert_message="Strategy {strategy_name} is underperforming benchmark by {strategy_relative_performance:.2%}",
            severity=AlertSeverity.MEDIUM,
            category=AlertCategory.STRATEGY,
            tags=["strategy", "performance", "underperformance"]
        ))
        
        # Large Position Movement Rules
        self.add_rule(AlertRule(
            id="position_large_gain",
            name="Large Position Gain",
            description="Alert when position gains exceed 20%",
            conditions=[
                RuleCondition("position.pnl_percent", RuleOperator.GREATER_THAN, 0.20)
            ],
            alert_title="Large Position Gain",
            alert_message="Position {position_symbol} has gained {position_pnl_percent:.2%}. Consider profit taking.",
            severity=AlertSeverity.INFO,
            category=AlertCategory.PORTFOLIO,
            tags=["position", "gain", "profit"],
            cooldown_seconds=3600  # 1 hour cooldown
        ))
        
        self.add_rule(AlertRule(
            id="position_large_loss",
            name="Large Position Loss",
            description="Alert when position loss exceeds 15%",
            conditions=[
                RuleCondition("position.pnl_percent", RuleOperator.LESS_THAN, -0.15)
            ],
            alert_title="Large Position Loss",
            alert_message="Position {position_symbol} has lost {position_pnl_percent:.2%}. Consider stop loss.",
            severity=AlertSeverity.HIGH,
            category=AlertCategory.PORTFOLIO,
            tags=["position", "loss", "stop_loss"]
        ))
        
        # Market Hours Rules
        self.add_rule(AlertRule(
            id="after_hours_trading",
            name="After Hours Trading Activity",
            description="Alert for significant price movements after market hours",
            conditions=[
                RuleCondition("market.is_open", RuleOperator.EQUALS, False),
                RuleCondition("price.change_percent", RuleOperator.GREATER_THAN, 5.0)
            ],
            alert_title="After Hours Price Movement",
            alert_message="Significant price movement in {symbol}: {price_change_percent:.2%} after market hours",
            severity=AlertSeverity.INFO,
            category=AlertCategory.MARKET,
            tags=["after_hours", "price_movement", "market"]
        ))
        
        # Correlation Rules
        self.add_rule(AlertRule(
            id="high_correlation_risk",
            name="High Portfolio Correlation",
            description="Alert when portfolio correlation exceeds 0.8",
            conditions=[
                RuleCondition("portfolio.correlation", RuleOperator.GREATER_THAN, 0.8)
            ],
            alert_title="High Portfolio Correlation Risk",
            alert_message="Portfolio correlation is {portfolio_correlation:.2f}, indicating concentration risk",
            severity=AlertSeverity.MEDIUM,
            category=AlertCategory.RISK,
            tags=["correlation", "risk", "diversification"]
        ))
        
        logger.info(f"Initialized {len(self.rules)} default alert rules")
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.rules[rule.id] = rule
        logger.debug(f"Alert rule added: {rule.name}")
    
    def remove_rule(self, rule_id: str):
        """Remove alert rule"""
        if rule_id in self.rules:
            rule_name = self.rules[rule_id].name
            del self.rules[rule_id]
            logger.info(f"Alert rule removed: {rule_name}")
    
    def enable_rule(self, rule_id: str):
        """Enable alert rule"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            logger.info(f"Alert rule enabled: {rule_id}")
    
    def disable_rule(self, rule_id: str):
        """Disable alert rule"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            logger.info(f"Alert rule disabled: {rule_id}")
    
    def evaluate_rules(self, data: Dict[str, Any]) -> List[Alert]:
        """Evaluate all rules against data and return triggered alerts"""
        triggered_alerts = []
        current_time = datetime.now()
        
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if self._is_in_cooldown(rule_id, current_time):
                continue
            
            try:
                if rule.evaluate(data):
                    alert = rule.create_alert(data)
                    triggered_alerts.append(alert)
                    
                    # Update last triggered time
                    self.rule_last_triggered[rule_id] = current_time
                    
                    logger.info(f"Alert rule triggered: {rule.name}")
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_id}: {str(e)}")
        
        return triggered_alerts
    
    def _is_in_cooldown(self, rule_id: str, current_time: datetime) -> bool:
        """Check if rule is in cooldown period"""
        if rule_id not in self.rule_last_triggered:
            return False
        
        rule = self.rules[rule_id]
        last_triggered = self.rule_last_triggered[rule_id]
        cooldown_end = last_triggered + timedelta(seconds=rule.cooldown_seconds)
        
        return current_time < cooldown_end
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get rule by ID"""
        return self.rules.get(rule_id)
    
    def get_rules(self, 
                  category: Optional[AlertCategory] = None,
                  severity: Optional[AlertSeverity] = None,
                  enabled_only: bool = False) -> List[AlertRule]:
        """Get rules with filters"""
        rules = list(self.rules.values())
        
        if category:
            rules = [r for r in rules if r.category == category]
        if severity:
            rules = [r for r in rules if r.severity == severity]
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        
        return rules
    
    def get_rule_status(self) -> Dict[str, Any]:
        """Get rule engine status"""
        total_rules = len(self.rules)
        enabled_rules = len([r for r in self.rules.values() if r.enabled])
        rules_in_cooldown = len([
            rule_id for rule_id in self.rule_last_triggered
            if self._is_in_cooldown(rule_id, datetime.now())
        ])
        
        return {
            'total_rules': total_rules,
            'enabled_rules': enabled_rules,
            'disabled_rules': total_rules - enabled_rules,
            'rules_in_cooldown': rules_in_cooldown,
            'rules_by_category': self._count_rules_by_category(),
            'rules_by_severity': self._count_rules_by_severity()
        }
    
    def _count_rules_by_category(self) -> Dict[str, int]:
        """Count rules by category"""
        counts = {}
        for rule in self.rules.values():
            category = rule.category.value
            counts[category] = counts.get(category, 0) + 1
        return counts
    
    def _count_rules_by_severity(self) -> Dict[str, int]:
        """Count rules by severity"""
        counts = {}
        for rule in self.rules.values():
            severity = rule.severity.value
            counts[severity] = counts.get(severity, 0) + 1
        return counts
    
    def export_rules(self) -> Dict[str, Any]:
        """Export rules configuration"""
        exported_rules = {}
        for rule_id, rule in self.rules.items():
            exported_rules[rule_id] = {
                'name': rule.name,
                'description': rule.description,
                'conditions': [
                    {
                        'field': c.field,
                        'operator': c.operator.value,
                        'value': c.value,
                        'description': c.description
                    }
                    for c in rule.conditions
                ],
                'alert_title': rule.alert_title,
                'alert_message': rule.alert_message,
                'severity': rule.severity.value,
                'category': rule.category.value,
                'enabled': rule.enabled,
                'cooldown_seconds': rule.cooldown_seconds,
                'tags': rule.tags,
                'metadata': rule.metadata
            }
        return exported_rules
    
    def import_rules(self, rules_config: Dict[str, Any]):
        """Import rules from configuration"""
        imported_count = 0
        
        for rule_id, rule_data in rules_config.items():
            try:
                # Parse conditions
                conditions = []
                for cond_data in rule_data.get('conditions', []):
                    condition = RuleCondition(
                        field=cond_data['field'],
                        operator=RuleOperator(cond_data['operator']),
                        value=cond_data['value'],
                        description=cond_data.get('description', '')
                    )
                    conditions.append(condition)
                
                # Create rule
                rule = AlertRule(
                    id=rule_id,
                    name=rule_data['name'],
                    description=rule_data['description'],
                    conditions=conditions,
                    alert_title=rule_data['alert_title'],
                    alert_message=rule_data['alert_message'],
                    severity=AlertSeverity(rule_data['severity']),
                    category=AlertCategory(rule_data['category']),
                    enabled=rule_data.get('enabled', True),
                    cooldown_seconds=rule_data.get('cooldown_seconds', 300),
                    tags=rule_data.get('tags', []),
                    metadata=rule_data.get('metadata', {})
                )
                
                self.add_rule(rule)
                imported_count += 1
                
            except Exception as e:
                logger.error(f"Error importing rule {rule_id}: {str(e)}")
        
        logger.info(f"Imported {imported_count} alert rules")
        return imported_count
    
    def test_rule(self, rule_id: str, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test a specific rule against test data"""
        rule = self.get_rule(rule_id)
        if not rule:
            return {'error': f'Rule {rule_id} not found'}
        
        try:
            # Temporarily enable rule for testing
            original_enabled = rule.enabled
            rule.enabled = True
            
            # Evaluate rule
            triggered = rule.evaluate(test_data)
            
            result = {
                'rule_id': rule_id,
                'rule_name': rule.name,
                'triggered': triggered,
                'condition_results': []
            }
            
            # Test individual conditions
            for i, condition in enumerate(rule.conditions):
                condition_result = condition.evaluate(test_data)
                result['condition_results'].append({
                    'condition_index': i,
                    'field': condition.field,
                    'operator': condition.operator.value,
                    'value': condition.value,
                    'result': condition_result,
                    'description': condition.description
                })
            
            # If triggered, create alert
            if triggered:
                alert = rule.create_alert(test_data)
                result['alert'] = alert.to_dict()
            
            # Restore original enabled state
            rule.enabled = original_enabled
            
            return result
            
        except Exception as e:
            return {'error': f'Error testing rule: {str(e)}'}

# =============================================================================
# Rule Templates and Builders
# =============================================================================

class RuleBuilder:
    """Helper class to build alert rules"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset builder state"""
        self._id = ""
        self._name = ""
        self._description = ""
        self._conditions = []
        self._alert_title = ""
        self._alert_message = ""
        self._severity = AlertSeverity.INFO
        self._category = AlertCategory.SYSTEM
        self._enabled = True
        self._cooldown_seconds = 300
        self._tags = []
        self._metadata = {}
        return self
    
    def id(self, rule_id: str):
        """Set rule ID"""
        self._id = rule_id
        return self
    
    def name(self, name: str):
        """Set rule name"""
        self._name = name
        return self
    
    def description(self, description: str):
        """Set rule description"""
        self._description = description
        return self
    
    def condition(self, field: str, operator: RuleOperator, value: Any, description: str = ""):
        """Add condition"""
        self._conditions.append(RuleCondition(field, operator, value, description))
        return self
    
    def alert_title(self, title: str):
        """Set alert title"""
        self._alert_title = title
        return self
    
    def alert_message(self, message: str):
        """Set alert message"""
        self._alert_message = message
        return self
    
    def severity(self, severity: AlertSeverity):
        """Set severity"""
        self._severity = severity
        return self
    
    def category(self, category: AlertCategory):
        """Set category"""
        self._category = category
        return self
    
    def enabled(self, enabled: bool = True):
        """Set enabled state"""
        self._enabled = enabled
        return self
    
    def cooldown(self, seconds: int):
        """Set cooldown period"""
        self._cooldown_seconds = seconds
        return self
    
    def tags(self, *tags):
        """Set tags"""
        self._tags = list(tags)
        return self
    
    def metadata(self, **metadata):
        """Set metadata"""
        self._metadata = metadata
        return self
    
    def build(self) -> AlertRule:
        """Build the rule"""
        return AlertRule(
            id=self._id,
            name=self._name,
            description=self._description,
            conditions=self._conditions,
            alert_title=self._alert_title,
            alert_message=self._alert_message,
            severity=self._severity,
            category=self._category,
            enabled=self._enabled,
            cooldown_seconds=self._cooldown_seconds,
            tags=self._tags,
            metadata=self._metadata
        )
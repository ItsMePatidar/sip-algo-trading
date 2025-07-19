# =============================================================================
# risk_management/compliance.py - Regulatory Compliance Module
# =============================================================================

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, time
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import calendar

logger = logging.getLogger(__name__)

class ComplianceStatus(Enum):
    COMPLIANT = "COMPLIANT"
    WARNING = "WARNING"
    VIOLATION = "VIOLATION"
    CRITICAL = "CRITICAL"

class RegulationType(Enum):
    SEBI = "SEBI"
    RBI = "RBI"
    INTERNAL = "INTERNAL"
    TAXATION = "TAXATION"

@dataclass
class ComplianceRule:
    """Regulatory compliance rule definition"""
    rule_id: str
    name: str
    regulation_type: RegulationType
    description: str
    threshold: float
    action_required: str
    enabled: bool = True
    last_checked: Optional[datetime] = None

@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    rule_id: str
    rule_name: str
    violation_type: str
    current_value: float
    threshold: float
    severity: ComplianceStatus
    timestamp: datetime
    description: str
    remedial_action: str
    resolved: bool = False
    resolution_date: Optional[datetime] = None

@dataclass
class TradingSession:
    """Trading session information"""
    date: datetime
    market_open: time
    market_close: time
    pre_market_start: time
    post_market_end: time
    is_trading_day: bool
    session_type: str  # "NORMAL", "MUHURAT", "HOLIDAY"

class SEBIComplianceChecker:
    """
    SEBI (Securities and Exchange Board of India) compliance checker
    
    Implements key SEBI regulations for algorithmic trading:
    - Algo trading registration requirements
    - Risk management requirements
    - Order-to-trade ratio limits
    - Position limits
    - Disclosure requirements
    """
    
    def __init__(self):
        self.rules = self._initialize_sebi_rules()
        self.order_count = 0
        self.trade_count = 0
        self.daily_turnover = 0.0
        self.positions = {}
        
    def _initialize_sebi_rules(self) -> Dict[str, ComplianceRule]:
        """Initialize SEBI compliance rules"""
        rules = {
            "algo_registration": ComplianceRule(
                rule_id="SEBI_AR_001",
                name="Algorithmic Trading Registration",
                regulation_type=RegulationType.SEBI,
                description="Algorithmic trading systems must be registered with exchanges",
                threshold=1.0,
                action_required="Ensure proper registration before live trading"
            ),
            
            "order_trade_ratio": ComplianceRule(
                rule_id="SEBI_OTR_001",
                name="Order-to-Trade Ratio",
                regulation_type=RegulationType.SEBI,
                description="Maximum order-to-trade ratio of 20:1",
                threshold=20.0,
                action_required="Reduce order frequency or improve fill rates"
            ),
            
            "position_limit_individual": ComplianceRule(
                rule_id="SEBI_PL_001",
                name="Individual Position Limit",
                regulation_type=RegulationType.SEBI,
                description="Maximum 15% of free float or 5% of total equity",
                threshold=0.15,
                action_required="Reduce position size to comply with limits"
            ),
            
            "risk_management_system": ComplianceRule(
                rule_id="SEBI_RMS_001",
                name="Risk Management System",
                regulation_type=RegulationType.SEBI,
                description="Adequate risk management systems required",
                threshold=1.0,
                action_required="Implement comprehensive risk management"
            ),
            
            "algo_audit_trail": ComplianceRule(
                rule_id="SEBI_AAT_001",
                name="Algorithm Audit Trail",
                regulation_type=RegulationType.SEBI,
                description="Maintain complete audit trail of algorithm decisions",
                threshold=1.0,
                action_required="Ensure all trading decisions are logged"
            )
        }
        
        return rules
    
    def check_order_trade_ratio(self) -> Optional[ComplianceViolation]:
        """Check order-to-trade ratio compliance"""
        rule = self.rules["order_trade_ratio"]
        
        if self.trade_count == 0:
            ratio = float('inf') if self.order_count > 0 else 0
        else:
            ratio = self.order_count / self.trade_count
        
        if ratio > rule.threshold:
            return ComplianceViolation(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                violation_type="ORDER_TRADE_RATIO",
                current_value=ratio,
                threshold=rule.threshold,
                severity=ComplianceStatus.VIOLATION,
                timestamp=datetime.now(),
                description=f"Order-to-trade ratio {ratio:.1f} exceeds limit of {rule.threshold}",
                remedial_action="Optimize algorithms to improve fill rates"
            )
        
        return None
    
    def check_position_limits(self, symbol: str, position_value: float, market_cap: float) -> Optional[ComplianceViolation]:
        """Check position limit compliance for a specific stock"""
        rule = self.rules["position_limit_individual"]
        
        # Calculate position as percentage of market cap (simplified)
        position_percentage = position_value / market_cap if market_cap > 0 else 0
        
        if position_percentage > rule.threshold:
            return ComplianceViolation(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                violation_type="POSITION_LIMIT",
                current_value=position_percentage,
                threshold=rule.threshold,
                severity=ComplianceStatus.VIOLATION,
                timestamp=datetime.now(),
                description=f"Position in {symbol} is {position_percentage:.2%} of market cap",
                remedial_action=f"Reduce position in {symbol} to comply with limits"
            )
        
        return None

class TaxationComplianceChecker:
    """
    Indian taxation compliance for trading activities
    
    Covers:
    - STT (Securities Transaction Tax)
    - Capital gains classification (STCG/LTCG)
    - TDS requirements
    - Turnover calculations for business income
    """
    
    def __init__(self):
        self.rules = self._initialize_tax_rules()
        self.transactions = []
        self.annual_turnover = 0.0
        
    def _initialize_tax_rules(self) -> Dict[str, ComplianceRule]:
        """Initialize taxation compliance rules"""
        rules = {
            "stt_compliance": ComplianceRule(
                rule_id="TAX_STT_001",
                name="Securities Transaction Tax",
                regulation_type=RegulationType.TAXATION,
                description="STT must be paid on all equity transactions",
                threshold=1.0,
                action_required="Ensure STT is calculated and reported"
            ),
            
            "turnover_threshold": ComplianceRule(
                rule_id="TAX_TO_001",
                name="Business Income Turnover Threshold",
                regulation_type=RegulationType.TAXATION,
                description="Turnover > ₹10 lakh may classify as business income",
                threshold=1000000.0,  # ₹10 lakh
                action_required="Consider tax implications of business vs investment income"
            ),
            
            "audit_requirement": ComplianceRule(
                rule_id="TAX_AUDIT_001",
                name="Tax Audit Requirement",
                regulation_type=RegulationType.TAXATION,
                description="Tax audit required if turnover > ₹1 crore",
                threshold=10000000.0,  # ₹1 crore
                action_required="Mandatory tax audit required"
            )
        }
        
        return rules
    
    def check_turnover_thresholds(self, annual_turnover: float) -> List[ComplianceViolation]:
        """Check turnover-based tax compliance"""
        violations = []
        
        # Check business income threshold
        business_rule = self.rules["turnover_threshold"]
        if annual_turnover > business_rule.threshold:
            violations.append(ComplianceViolation(
                rule_id=business_rule.rule_id,
                rule_name=business_rule.name,
                violation_type="TURNOVER_THRESHOLD",
                current_value=annual_turnover,
                threshold=business_rule.threshold,
                severity=ComplianceStatus.WARNING,
                timestamp=datetime.now(),
                description=f"Annual turnover ₹{annual_turnover:,.0f} exceeds business income threshold",
                remedial_action="Consult tax advisor for business vs investment classification"
            ))
        
        # Check audit requirement
        audit_rule = self.rules["audit_requirement"]
        if annual_turnover > audit_rule.threshold:
            violations.append(ComplianceViolation(
                rule_id=audit_rule.rule_id,
                rule_name=audit_rule.name,
                violation_type="AUDIT_REQUIREMENT",
                current_value=annual_turnover,
                threshold=audit_rule.threshold,
                severity=ComplianceStatus.CRITICAL,
                timestamp=datetime.now(),
                description=f"Annual turnover ₹{annual_turnover:,.0f} requires mandatory tax audit",
                remedial_action="Arrange for mandatory tax audit by qualified CA"
            ))
        
        return violations
    
    def calculate_capital_gains(self, buy_date: datetime, sell_date: datetime, 
                              buy_price: float, sell_price: float, quantity: int) -> Dict[str, Any]:
        """Calculate capital gains and classify as STCG/LTCG"""
        holding_period = (sell_date - buy_date).days
        
        # For equity shares: LTCG if held > 12 months
        is_long_term = holding_period > 365
        
        total_buy_value = buy_price * quantity
        total_sell_value = sell_price * quantity
        
        # STT calculation (simplified)
        stt_rate = 0.001 if not is_long_term else 0.001  # 0.1% for equity delivery
        stt_amount = total_sell_value * stt_rate
        
        capital_gain = total_sell_value - total_buy_value - stt_amount
        
        # Tax calculation (simplified)
        if is_long_term:
            # LTCG: 10% on gains > ₹1 lakh
            tax_rate = 0.10 if capital_gain > 100000 else 0.0
            exemption_limit = 100000
        else:
            # STCG: 15% flat rate
            tax_rate = 0.15
            exemption_limit = 0
        
        taxable_gain = max(0, capital_gain - exemption_limit)
        tax_liability = taxable_gain * tax_rate
        
        return {
            'holding_period_days': holding_period,
            'is_long_term': is_long_term,
            'capital_gain': capital_gain,
            'taxable_gain': taxable_gain,
            'tax_rate': tax_rate,
            'tax_liability': tax_liability,
            'stt_amount': stt_amount,
            'exemption_used': min(capital_gain, exemption_limit)
        }

class TradingHoursComplianceChecker:
    """
    Trading hours and market session compliance
    
    Ensures trading activities comply with:
    - NSE/BSE trading hours
    - Pre-market and after-market sessions
    - Holiday calendar compliance
    - Circuit breaker timings
    """
    
    def __init__(self):
        self.rules = self._initialize_trading_hours_rules()
        self.market_holidays = self._load_market_holidays()
        
    def _initialize_trading_hours_rules(self) -> Dict[str, ComplianceRule]:
        """Initialize trading hours compliance rules"""
        rules = {
            "normal_trading_hours": ComplianceRule(
                rule_id="TH_NTH_001",
                name="Normal Trading Hours",
                regulation_type=RegulationType.SEBI,
                description="Trading allowed only during market hours (9:15 AM - 3:30 PM)",
                threshold=1.0,
                action_required="Restrict trading to market hours only"
            ),
            
            "pre_market_compliance": ComplianceRule(
                rule_id="TH_PM_001",
                name="Pre-market Session",
                regulation_type=RegulationType.SEBI,
                description="Pre-market trading (9:00 AM - 9:15 AM) has specific rules",
                threshold=1.0,
                action_required="Ensure pre-market trading compliance"
            ),
            
            "holiday_trading": ComplianceRule(
                rule_id="TH_HT_001",
                name="Holiday Trading Restriction",
                regulation_type=RegulationType.SEBI,
                description="No trading on market holidays",
                threshold=0.0,
                action_required="Disable trading on market holidays"
            )
        }
        
        return rules
    
    def _load_market_holidays(self) -> List[datetime]:
        """Load market holidays for the year"""
        # This would typically load from a file or API
        # For now, return some common holidays
        year = datetime.now().year
        holidays = [
            datetime(year, 1, 26),   # Republic Day
            datetime(year, 8, 15),   # Independence Day
            datetime(year, 10, 2),   # Gandhi Jayanti
            # Add more holidays as needed
        ]
        return holidays
    
    def is_market_open(self, check_time: datetime = None) -> Tuple[bool, str]:
        """Check if market is open at given time"""
        if check_time is None:
            check_time = datetime.now()
        
        # Check if it's a market holiday
        if check_time.date() in [h.date() for h in self.market_holidays]:
            return False, "Market Holiday"
        
        # Check if it's a weekend
        if check_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False, "Weekend"
        
        # Check trading hours
        current_time = check_time.time()
        
        # Normal trading hours: 9:15 AM - 3:30 PM
        market_open = time(9, 15)
        market_close = time(15, 30)
        
        if market_open <= current_time <= market_close:
            return True, "Normal Trading Hours"
        
        # Pre-market: 9:00 AM - 9:15 AM
        pre_market_start = time(9, 0)
        if pre_market_start <= current_time < market_open:
            return True, "Pre-market Session"
        
        # After-market: 3:40 PM - 4:00 PM
        after_market_start = time(15, 40)
        after_market_end = time(16, 0)
        if after_market_start <= current_time <= after_market_end:
            return True, "After-market Session"
        
        return False, "Market Closed"
    
    def check_trading_time_compliance(self, order_time: datetime) -> Optional[ComplianceViolation]:
        """Check if order was placed during valid trading hours"""
        is_open, session_type = self.is_market_open(order_time)
        
        if not is_open:
            rule = self.rules["normal_trading_hours"]
            return ComplianceViolation(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                violation_type="TRADING_HOURS",
                current_value=0.0,
                threshold=1.0,
                severity=ComplianceStatus.VIOLATION,
                timestamp=datetime.now(),
                description=f"Order placed during {session_type}",
                remedial_action="Ensure orders are placed only during market hours"
            )
        
        return None

class InternalComplianceChecker:
    """
    Internal compliance rules specific to the organization
    
    Covers:
    - Internal risk limits
    - Trading authorization levels
    - Documentation requirements
    - Approval workflows
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rules = self._initialize_internal_rules()
        self.approval_logs = []
        
    def _initialize_internal_rules(self) -> Dict[str, ComplianceRule]:
        """Initialize internal compliance rules"""
        rules = {
            "daily_loss_limit": ComplianceRule(
                rule_id="INT_DLL_001",
                name="Daily Loss Limit",
                regulation_type=RegulationType.INTERNAL,
                description="Maximum daily loss limit of 5%",
                threshold=0.05,
                action_required="Stop trading for the day"
            ),
            
            "strategy_approval": ComplianceRule(
                rule_id="INT_SA_001",
                name="Strategy Approval",
                regulation_type=RegulationType.INTERNAL,
                description="All strategies must be approved before deployment",
                threshold=1.0,
                action_required="Obtain strategy approval before live trading"
            ),
            
            "position_concentration": ComplianceRule(
                rule_id="INT_PC_001",
                name="Position Concentration",
                regulation_type=RegulationType.INTERNAL,
                description="Maximum 20% in single position",
                threshold=0.20,
                action_required="Reduce position concentration"
            ),
            
            "documentation_requirement": ComplianceRule(
                rule_id="INT_DR_001",
                name="Documentation Requirement",
                regulation_type=RegulationType.INTERNAL,
                description="All trading decisions must be documented",
                threshold=1.0,
                action_required="Ensure proper documentation of all decisions"
            )
        }
        
        return rules
    
    def check_strategy_approval(self, strategy_name: str) -> Optional[ComplianceViolation]:
        """Check if strategy has required approvals"""
        # This would check against an approval database
        # For now, assume all strategies need approval
        
        approved_strategies = self.config.get('approved_strategies', [])
        
        if strategy_name not in approved_strategies:
            rule = self.rules["strategy_approval"]
            return ComplianceViolation(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                violation_type="STRATEGY_APPROVAL",
                current_value=0.0,
                threshold=1.0,
                severity=ComplianceStatus.CRITICAL,
                timestamp=datetime.now(),
                description=f"Strategy '{strategy_name}' is not approved for live trading",
                remedial_action="Obtain approval before deploying strategy"
            )
        
        return None

class ComplianceManager:
    """
    Main compliance management system
    
    Coordinates all compliance checkers and provides unified interface
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize compliance checkers
        self.sebi_checker = SEBIComplianceChecker()
        self.tax_checker = TaxationComplianceChecker()
        self.trading_hours_checker = TradingHoursComplianceChecker()
        self.internal_checker = InternalComplianceChecker(config)
        
        # Violation tracking
        self.all_violations = []
        self.active_violations = []
        
        logger.info("Compliance Manager initialized")
    
    def run_comprehensive_check(self, portfolio_state: Dict[str, Any], 
                               trading_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive compliance check"""
        violations = []
        
        # SEBI compliance checks
        sebi_violations = self._run_sebi_checks(portfolio_state, trading_data)
        violations.extend(sebi_violations)
        
        # Tax compliance checks
        tax_violations = self._run_tax_checks(trading_data)
        violations.extend(tax_violations)
        
        # Trading hours checks
        hours_violations = self._run_trading_hours_checks(trading_data)
        violations.extend(hours_violations)
        
        # Internal compliance checks
        internal_violations = self._run_internal_checks(portfolio_state, trading_data)
        violations.extend(internal_violations)
        
        # Update violation tracking
        self.all_violations.extend(violations)
        self.active_violations = [v for v in violations if not v.resolved]
        
        # Generate compliance report
        report = self._generate_compliance_report(violations)
        
        return report
    
    def _run_sebi_checks(self, portfolio_state: Dict[str, Any], 
                        trading_data: Dict[str, Any]) -> List[ComplianceViolation]:
        """Run SEBI compliance checks"""
        violations = []
        
        # Update order/trade counts
        self.sebi_checker.order_count = trading_data.get('daily_order_count', 0)
        self.sebi_checker.trade_count = trading_data.get('daily_trade_count', 0)
        
        # Check order-to-trade ratio
        otr_violation = self.sebi_checker.check_order_trade_ratio()
        if otr_violation:
            violations.append(otr_violation)
        
        # Check position limits for each position
        positions = portfolio_state.get('positions', {})
        for symbol, position in positions.items():
            position_value = position.get('value', 0)
            market_cap = position.get('market_cap', position_value * 100)  # Simplified
            
            pl_violation = self.sebi_checker.check_position_limits(symbol, position_value, market_cap)
            if pl_violation:
                violations.append(pl_violation)
        
        return violations
    
    def _run_tax_checks(self, trading_data: Dict[str, Any]) -> List[ComplianceViolation]:
        """Run taxation compliance checks"""
        violations = []
        
        # Check turnover thresholds
        annual_turnover = trading_data.get('annual_turnover', 0)
        turnover_violations = self.tax_checker.check_turnover_thresholds(annual_turnover)
        violations.extend(turnover_violations)
        
        return violations
    
    def _run_trading_hours_checks(self, trading_data: Dict[str, Any]) -> List[ComplianceViolation]:
        """Run trading hours compliance checks"""
        violations = []
        
        # Check recent orders for timing compliance
        recent_orders = trading_data.get('recent_orders', [])
        for order in recent_orders:
            order_time = order.get('timestamp')
            if order_time:
                timing_violation = self.trading_hours_checker.check_trading_time_compliance(order_time)
                if timing_violation:
                    violations.append(timing_violation)
        
        return violations
    
    def _run_internal_checks(self, portfolio_state: Dict[str, Any], 
                           trading_data: Dict[str, Any]) -> List[ComplianceViolation]:
        """Run internal compliance checks"""
        violations = []
        
        # Check strategy approvals
        active_strategies = trading_data.get('active_strategies', [])
        for strategy in active_strategies:
            approval_violation = self.internal_checker.check_strategy_approval(strategy)
            if approval_violation:
                violations.append(approval_violation)
        
        return violations
    
    def _generate_compliance_report(self, violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        critical_violations = [v for v in violations if v.severity == ComplianceStatus.CRITICAL]
        warning_violations = [v for v in violations if v.severity == ComplianceStatus.WARNING]
        
        report = {
            'timestamp': datetime.now(),
            'overall_status': ComplianceStatus.CRITICAL if critical_violations 
                             else ComplianceStatus.WARNING if warning_violations
                             else ComplianceStatus.COMPLIANT,
            'total_violations': len(violations),
            'critical_violations': len(critical_violations),
            'warning_violations': len(warning_violations),
            'violations_by_type': self._group_violations_by_type(violations),
            'violations_by_regulation': self._group_violations_by_regulation(violations),
            'immediate_actions_required': [v.remedial_action for v in critical_violations],
            'recommendations': [v.remedial_action for v in warning_violations],
            'detailed_violations': violations
        }
        
        return report
    
    def _group_violations_by_type(self, violations: List[ComplianceViolation]) -> Dict[str, int]:
        """Group violations by type"""
        type_counts = {}
        for violation in violations:
            violation_type = violation.violation_type
            type_counts[violation_type] = type_counts.get(violation_type, 0) + 1
        return type_counts
    
    def _group_violations_by_regulation(self, violations: List[ComplianceViolation]) -> Dict[str, int]:
        """Group violations by regulation type"""
        regulation_counts = {}
        for violation in violations:
            # Extract regulation type from rule_id
            if violation.rule_id.startswith('SEBI'):
                reg_type = 'SEBI'
            elif violation.rule_id.startswith('TAX'):
                reg_type = 'TAXATION'
            elif violation.rule_id.startswith('TH'):
                reg_type = 'TRADING_HOURS'
            elif violation.rule_id.startswith('INT'):
                reg_type = 'INTERNAL'
            else:
                reg_type = 'OTHER'
            
            regulation_counts[reg_type] = regulation_counts.get(reg_type, 0) + 1
        
        return regulation_counts
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get current compliance status"""
        active_critical = [v for v in self.active_violations if v.severity == ComplianceStatus.CRITICAL]
        active_warnings = [v for v in self.active_violations if v.severity == ComplianceStatus.WARNING]
        
        return {
            'overall_status': ComplianceStatus.CRITICAL if active_critical 
                             else ComplianceStatus.WARNING if active_warnings
                             else ComplianceStatus.COMPLIANT,
            'active_violations': len(self.active_violations),
            'critical_count': len(active_critical),
            'warning_count': len(active_warnings),
            'total_violations_today': len([v for v in self.all_violations 
                                         if v.timestamp.date() == datetime.now().date()]),
            'last_check': datetime.now(),
            'trading_allowed': len(active_critical) == 0
        }
    
    def resolve_violation(self, violation_id: str, resolution_notes: str):
        """Mark a violation as resolved"""
        for violation in self.active_violations:
            if violation.rule_id == violation_id:
                violation.resolved = True
                violation.resolution_date = datetime.now()
                logger.info(f"Violation {violation_id} resolved: {resolution_notes}")
                break
        
        # Update active violations list
        self.active_violations = [v for v in self.active_violations if not v.resolved]
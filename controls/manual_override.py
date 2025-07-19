# =============================================================================
# controls/manual_override.py - Fixed Manual Control Interface
# =============================================================================

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import threading
import time
import uuid
import hashlib

logger = logging.getLogger(__name__)

# Fixed: Use IntEnum for proper comparison
class OverrideLevel(IntEnum):
    READ_ONLY = 1           # View-only access
    PARAMETER_ADJUST = 2    # Adjust parameters
    ORDER_CONTROL = 3       # Manual order control
    STRATEGY_CONTROL = 4    # Enable/disable strategies
    SYSTEM_CONTROL = 5      # Full system control
    EMERGENCY_CONTROL = 6   # Emergency controls

class OverrideAction(Enum):
    # Parameter actions
    ADJUST_POSITION_SIZE = "adjust_position_size"
    MODIFY_RISK_LIMITS = "modify_risk_limits"
    CHANGE_SIP_AMOUNT = "change_sip_amount"
    
    # Order actions
    PLACE_MANUAL_ORDER = "place_manual_order"
    CANCEL_ORDER = "cancel_order"
    MODIFY_ORDER = "modify_order"
    
    # Strategy actions
    PAUSE_STRATEGY = "pause_strategy"
    RESUME_STRATEGY = "resume_strategy"
    FORCE_STRATEGY_EXECUTION = "force_strategy_execution"
    
    # System actions
    OVERRIDE_CIRCUIT_BREAKER = "override_circuit_breaker"
    RESET_EMERGENCY_STOP = "reset_emergency_stop"
    FORCE_SYSTEM_SHUTDOWN = "force_system_shutdown"
    MODIFY_SYSTEM_CONFIG = "modify_system_config"

@dataclass
class OverrideSession:
    """Manual override session"""
    session_id: str
    user_id: str
    user_name: str
    override_level: OverrideLevel
    start_time: datetime
    end_time: Optional[datetime] = None
    max_duration: timedelta = field(default_factory=lambda: timedelta(hours=4))
    actions_performed: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    reason: str = ""
    approval_required: bool = False
    approved_by: Optional[str] = None

@dataclass
class OverrideRequest:
    """Override action request"""
    request_id: str
    session_id: str
    action: OverrideAction
    parameters: Dict[str, Any]
    timestamp: datetime
    user_id: str
    reason: str
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    approved: bool = False
    executed: bool = False
    result: Optional[Dict[str, Any]] = None

class ManualOverride:
    """
    Manual Override System
    
    Provides controlled manual intervention capabilities with:
    - Multi-level access control
    - Session management
    - Action logging and audit trail
    - Risk assessment for manual actions
    - Approval workflows for high-risk actions
    """
    
    def __init__(self, system_name: str = "SIP_TRADING_SYSTEM"):
        self.system_name = system_name
        self.active_sessions: Dict[str, OverrideSession] = {}
        self.session_history: List[OverrideSession] = []
        self.pending_requests: Dict[str, OverrideRequest] = {}
        self.executed_requests: List[OverrideRequest] = []
        
        # Access control
        self.user_permissions: Dict[str, OverrideLevel] = {}
        self.approval_required_actions: Set[OverrideAction] = {
            OverrideAction.OVERRIDE_CIRCUIT_BREAKER,
            OverrideAction.RESET_EMERGENCY_STOP,
            OverrideAction.FORCE_SYSTEM_SHUTDOWN,
            OverrideAction.MODIFY_SYSTEM_CONFIG
        }
        
        # Safety limits
        self.max_concurrent_sessions = 3
        self.max_session_duration = timedelta(hours=8)
        self.cooldown_period = timedelta(minutes=30)
        
        # Callbacks
        self.on_session_start_callbacks = []
        self.on_action_executed_callbacks = []
        self.on_high_risk_action_callbacks = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Manual override system initialized for {system_name}")
    
    def add_user_permission(self, user_id: str, user_name: str, override_level: OverrideLevel):
        """Add user permissions"""
        with self.lock:
            self.user_permissions[user_id] = override_level
            logger.info(f"User {user_name} ({user_id}) granted {override_level.name} access")
    
    def start_override_session(self, 
                             user_id: str,
                             user_name: str,
                             reason: str,
                             requested_level: OverrideLevel = OverrideLevel.PARAMETER_ADJUST,
                             max_duration: timedelta = None) -> Optional[str]:
        """
        Start a manual override session
        
        Returns:
            session_id if successful, None if failed
        """
        with self.lock:
            # Check user permissions
            if user_id not in self.user_permissions:
                logger.warning(f"User {user_id} not authorized for manual override")
                return None
            
            user_level = self.user_permissions[user_id]
            
            # Fixed: Proper enum comparison
            if user_level < requested_level:
                logger.warning(f"User {user_id} requested {requested_level.name} but only has {user_level.name}")
                return None
            
            # Check session limits
            active_count = len(self.active_sessions)
            if active_count >= self.max_concurrent_sessions:
                logger.warning(f"Maximum concurrent sessions ({self.max_concurrent_sessions}) reached")
                return None
            
            # Create session - Fixed: Use proper min comparison
            effective_level = user_level if user_level <= requested_level else requested_level
            
            session_id = self._generate_session_id(user_id)
            session = OverrideSession(
                session_id=session_id,
                user_id=user_id,
                user_name=user_name,
                override_level=effective_level,
                start_time=datetime.now(),
                max_duration=max_duration or self.max_session_duration,
                reason=reason
            )
            
            self.active_sessions[session_id] = session
            
            logger.info(f"Override session started: {session_id}")
            logger.info(f"User: {user_name} ({user_id})")
            logger.info(f"Level: {session.override_level.name}")
            logger.info(f"Reason: {reason}")
            
            # Execute callbacks
            for callback in self.on_session_start_callbacks:
                try:
                    callback(session)
                except Exception as e:
                    logger.error(f"Error in session start callback: {str(e)}")
            
            return session_id
    
    def submit_override_request(self, 
                              session_id: str,
                              action: OverrideAction,
                              parameters: Dict[str, Any],
                              reason: str) -> Optional[str]:
        """
        Submit an override action request
        
        Returns:
            request_id if successful, None if failed
        """
        with self.lock:
            # Validate session
            session = self._validate_session(session_id)
            if not session:
                return None
            
            # Check action permissions
            if not self._check_action_permission(session, action):
                logger.warning(f"Action {action.value} not permitted for session {session_id}")
                return None
            
            # Create request
            request_id = self._generate_request_id(session_id, action)
            request = OverrideRequest(
                request_id=request_id,
                session_id=session_id,
                action=action,
                parameters=parameters,
                timestamp=datetime.now(),
                user_id=session.user_id,
                reason=reason
            )
            
            # Risk assessment
            request.risk_assessment = self._assess_action_risk(action, parameters)
            
            # Check if approval required
            if action in self.approval_required_actions or request.risk_assessment.get('high_risk', False):
                request.approval_required = True
                self.pending_requests[request_id] = request
                logger.warning(f"Override request {request_id} requires approval")
                
                # Notify for approval
                for callback in self.on_high_risk_action_callbacks:
                    try:
                        callback(request)
                    except Exception as e:
                        logger.error(f"Error in high risk action callback: {str(e)}")
                
                return request_id
            else:
                # Execute immediately
                return self._execute_override_request(request) is not None
    
    def approve_request(self, request_id: str, approver_id: str, approver_name: str) -> bool:
        """Approve a pending override request"""
        with self.lock:
            if request_id not in self.pending_requests:
                logger.warning(f"Request {request_id} not found in pending requests")
                return False
            
            request = self.pending_requests[request_id]
            request.approved = True
            request.approved_by = f"{approver_name} ({approver_id})"
            
            logger.info(f"Override request {request_id} approved by {approver_name}")
            
            # Execute the approved request
            return self._execute_override_request(request) is not None
    
    def _execute_override_request(self, request: OverrideRequest) -> Optional[str]:
        """Execute an override request"""
        try:
            # Validate session again
            session = self._validate_session(request.session_id)
            if not session:
                return None
            
            # Execute action
            result = self._execute_action(request.action, request.parameters, session)
            
            # Update request
            request.executed = True
            request.result = result
            
            # Add to session history
            session.actions_performed.append({
                'request_id': request.request_id,
                'action': request.action.value,
                'parameters': request.parameters,
                'timestamp': request.timestamp.isoformat(),
                'reason': request.reason,
                'result': result
            })
            
            # Move to executed requests
            if request.request_id in self.pending_requests:
                del self.pending_requests[request.request_id]
            self.executed_requests.append(request)
            
            logger.info(f"Override action executed: {request.action.value}")
            logger.info(f"Request ID: {request.request_id}")
            logger.info(f"Result: {result}")
            
            # Execute callbacks
            for callback in self.on_action_executed_callbacks:
                try:
                    callback(request, result)
                except Exception as e:
                    logger.error(f"Error in action executed callback: {str(e)}")
            
            return request.request_id
            
        except Exception as e:
            error_msg = f"Error executing override request: {str(e)}"
            logger.error(error_msg)
            request.result = {'error': error_msg}
            return None
    
    def _execute_action(self, action: OverrideAction, parameters: Dict[str, Any], session: OverrideSession) -> Dict[str, Any]:
        """Execute a specific override action"""
        
        if action == OverrideAction.ADJUST_POSITION_SIZE:
            return self._adjust_position_size(parameters)
        
        elif action == OverrideAction.MODIFY_RISK_LIMITS:
            return self._modify_risk_limits(parameters)
        
        elif action == OverrideAction.CHANGE_SIP_AMOUNT:
            return self._change_sip_amount(parameters)
        
        elif action == OverrideAction.PLACE_MANUAL_ORDER:
            return self._place_manual_order(parameters)
        
        elif action == OverrideAction.CANCEL_ORDER:
            return self._cancel_order(parameters)
        
        elif action == OverrideAction.PAUSE_STRATEGY:
            return self._pause_strategy(parameters)
        
        elif action == OverrideAction.RESUME_STRATEGY:
            return self._resume_strategy(parameters)
        
        elif action == OverrideAction.OVERRIDE_CIRCUIT_BREAKER:
            return self._override_circuit_breaker(parameters)
        
        elif action == OverrideAction.RESET_EMERGENCY_STOP:
            return self._reset_emergency_stop(parameters)
        
        elif action == OverrideAction.FORCE_SYSTEM_SHUTDOWN:
            return self._force_system_shutdown(parameters)
        
        else:
            return {'error': f'Unknown action: {action.value}'}
    
    def _adjust_position_size(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust position size"""
        symbol = parameters.get('symbol')
        new_size = parameters.get('new_size')
        
        # Implementation would adjust actual position size
        logger.info(f"Adjusting position size for {symbol} to {new_size}")
        
        return {
            'success': True,
            'action': 'Position size adjusted',
            'symbol': symbol,
            'new_size': new_size
        }
    
    def _modify_risk_limits(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Modify risk limits"""
        limit_type = parameters.get('limit_type')
        new_value = parameters.get('new_value')
        
        # Implementation would modify actual risk limits
        logger.info(f"Modifying risk limit {limit_type} to {new_value}")
        
        return {
            'success': True,
            'action': 'Risk limit modified',
            'limit_type': limit_type,
            'new_value': new_value
        }
    
    def _change_sip_amount(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Change SIP amount"""
        strategy = parameters.get('strategy')
        new_amount = parameters.get('new_amount')
        
        # Implementation would modify SIP configuration
        logger.info(f"Changing SIP amount for {strategy} to {new_amount}")
        
        return {
            'success': True,
            'action': 'SIP amount changed',
            'strategy': strategy,
            'new_amount': new_amount
        }
    
    def _place_manual_order(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Place manual order"""
        symbol = parameters.get('symbol')
        side = parameters.get('side')
        quantity = parameters.get('quantity')
        order_type = parameters.get('order_type', 'MARKET')
        
        # Implementation would place actual order
        logger.info(f"Placing manual order: {side} {quantity} {symbol} ({order_type})")
        
        return {
            'success': True,
            'action': 'Manual order placed',
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'order_type': order_type,
            'order_id': f"MANUAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    
    def _cancel_order(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel order"""
        order_id = parameters.get('order_id')
        
        # Implementation would cancel actual order
        logger.info(f"Cancelling order {order_id}")
        
        return {
            'success': True,
            'action': 'Order cancelled',
            'order_id': order_id
        }
    
    def _pause_strategy(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Pause strategy"""
        strategy_name = parameters.get('strategy_name')
        
        # Implementation would pause actual strategy
        logger.info(f"Pausing strategy {strategy_name}")
        
        return {
            'success': True,
            'action': 'Strategy paused',
            'strategy_name': strategy_name
        }
    
    def _resume_strategy(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Resume strategy"""
        strategy_name = parameters.get('strategy_name')
        
        # Implementation would resume actual strategy
        logger.info(f"Resuming strategy {strategy_name}")
        
        return {
            'success': True,
            'action': 'Strategy resumed',
            'strategy_name': strategy_name
        }
    
    def _override_circuit_breaker(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Override circuit breaker"""
        breaker_name = parameters.get('breaker_name')
        override_duration = parameters.get('override_duration_minutes', 30)
        
        # Implementation would override actual circuit breaker
        logger.critical(f"OVERRIDING CIRCUIT BREAKER: {breaker_name} for {override_duration} minutes")
        
        return {
            'success': True,
            'action': 'Circuit breaker overridden',
            'breaker_name': breaker_name,
            'override_duration': override_duration
        }
    
    def _reset_emergency_stop(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Reset emergency stop"""
        system_name = parameters.get('system_name')
        
        # Implementation would reset actual emergency stop
        logger.critical(f"RESETTING EMERGENCY STOP: {system_name}")
        
        return {
            'success': True,
            'action': 'Emergency stop reset',
            'system_name': system_name
        }
    
    def _force_system_shutdown(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Force system shutdown"""
        shutdown_type = parameters.get('shutdown_type', 'graceful')
        
        # Implementation would shutdown actual system
        logger.critical(f"FORCING SYSTEM SHUTDOWN: {shutdown_type}")
        
        return {
            'success': True,
            'action': 'System shutdown initiated',
            'shutdown_type': shutdown_type
        }
    
    def _validate_session(self, session_id: str) -> Optional[OverrideSession]:
        """Validate session"""
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found")
            return None
        
        session = self.active_sessions[session_id]
        
        # Check if session expired
        if datetime.now() > session.start_time + session.max_duration:
            logger.warning(f"Session {session_id} expired")
            self.end_session(session_id, "Session expired")
            return None
        
        return session
    
    def _check_action_permission(self, session: OverrideSession, action: OverrideAction) -> bool:
        """Check if action is permitted for session level"""
        
        # Define what actions are allowed at each permission level
        level_permissions = {
            OverrideLevel.READ_ONLY: [],
            
            OverrideLevel.PARAMETER_ADJUST: [
                OverrideAction.ADJUST_POSITION_SIZE,
                OverrideAction.CHANGE_SIP_AMOUNT
            ],
            
            OverrideLevel.ORDER_CONTROL: [
                OverrideAction.PLACE_MANUAL_ORDER,
                OverrideAction.CANCEL_ORDER,
                OverrideAction.MODIFY_ORDER
            ],
            
            OverrideLevel.STRATEGY_CONTROL: [
                OverrideAction.PAUSE_STRATEGY,
                OverrideAction.RESUME_STRATEGY,
                OverrideAction.FORCE_STRATEGY_EXECUTION
            ],
            
            OverrideLevel.SYSTEM_CONTROL: [
                OverrideAction.MODIFY_RISK_LIMITS,
                OverrideAction.MODIFY_SYSTEM_CONFIG
            ],
            
            OverrideLevel.EMERGENCY_CONTROL: [
                OverrideAction.OVERRIDE_CIRCUIT_BREAKER,
                OverrideAction.RESET_EMERGENCY_STOP,
                OverrideAction.FORCE_SYSTEM_SHUTDOWN
            ]
        }
        
        # Get all allowed actions for this session's level and all lower levels
        allowed_actions = []
        
        # Convert enum to integer for comparison
        session_level_value = list(OverrideLevel).index(session.override_level)
        
        for level in OverrideLevel:
            level_value = list(OverrideLevel).index(level)
            
            # If this level is at or below the session's level, include its actions
            if level_value <= session_level_value:
                allowed_actions.extend(level_permissions.get(level, []))
        
        # Debug logging
        logger.info(f"Session {session.session_id} level: {session.override_level.value}")
        logger.info(f"Requested action: {action.value}")
        logger.info(f"Allowed actions: {[a.value for a in allowed_actions]}")
        
        return action in allowed_actions
    
    def _assess_action_risk(self, action: OverrideAction, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk level of an action"""
        high_risk_actions = {
            OverrideAction.OVERRIDE_CIRCUIT_BREAKER,
            OverrideAction.RESET_EMERGENCY_STOP,
            OverrideAction.FORCE_SYSTEM_SHUTDOWN
        }
        
        medium_risk_actions = {
            OverrideAction.MODIFY_RISK_LIMITS,
            OverrideAction.PLACE_MANUAL_ORDER,
            OverrideAction.MODIFY_SYSTEM_CONFIG
        }
        
        if action in high_risk_actions:
            risk_level = "HIGH"
        elif action in medium_risk_actions:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Additional risk assessment based on parameters
        risk_factors = []
        if action == OverrideAction.PLACE_MANUAL_ORDER:
            order_value = parameters.get('quantity', 0) * parameters.get('price', 0)
            if order_value > 100000:  # Large order
                risk_factors.append("Large order value")
        
        return {
            'risk_level': risk_level,
            'high_risk': risk_level == "HIGH",
            'risk_factors': risk_factors,
            'requires_approval': action in self.approval_required_actions
        }
    
    def _generate_session_id(self, user_id: str) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hash_input = f"{user_id}_{timestamp}_{self.system_name}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"OVERRIDE_{timestamp}_{hash_value}"
    
    def _generate_request_id(self, session_id: str, action: OverrideAction) -> str:
        """Generate unique request ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        return f"REQ_{session_id}_{action.value}_{timestamp}"
    
    def end_session(self, session_id: str, reason: str = "Manual termination") -> bool:
        """End an override session"""
        with self.lock:
            if session_id not in self.active_sessions:
                logger.warning(f"Session {session_id} not found")
                return False
            
            session = self.active_sessions[session_id]
            session.end_time = datetime.now()
            session.is_active = False
            
            # Move to history
            self.session_history.append(session)
            del self.active_sessions[session_id]
            
            duration = session.end_time - session.start_time
            
            logger.info(f"Override session ended: {session_id}")
            logger.info(f"Duration: {duration}")
            logger.info(f"Actions performed: {len(session.actions_performed)}")
            logger.info(f"Reason: {reason}")
            
            return True
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active sessions"""
        with self.lock:
            return [
                {
                    'session_id': session.session_id,
                    'user_name': session.user_name,
                    'override_level': session.override_level.name,  # Use .name instead of .value
                    'start_time': session.start_time.isoformat(),
                    'actions_count': len(session.actions_performed),
                    'reason': session.reason
                }
                for session in self.active_sessions.values()
            ]
    
    def get_session_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed session information"""
        with self.lock:
            session = self.active_sessions.get(session_id)
            if not session:
                return None
            
            return {
                'session_id': session.session_id,
                'user_id': session.user_id,
                'user_name': session.user_name,
                'override_level': session.override_level.name,  # Use .name instead of .value
                'start_time': session.start_time.isoformat(),
                'max_duration': str(session.max_duration),
                'actions_performed': session.actions_performed,
                'is_active': session.is_active,
                'reason': session.reason
            }
    
    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests"""
        with self.lock:
            return [
                {
                    'request_id': request.request_id,
                    'session_id': request.session_id,
                    'action': request.action.value,
                    'user_id': request.user_id,
                    'timestamp': request.timestamp.isoformat(),
                    'reason': request.reason,
                    'risk_assessment': request.risk_assessment
                }
                for request in self.pending_requests.values()
            ]
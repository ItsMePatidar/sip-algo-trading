# =============================================================================
# risk_management/position_limits.py - Position Limit Management
# =============================================================================

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from enum import Enum
import json

logger = logging.getLogger(__name__)

@dataclass
class PositionLimit:
    """Position limit configuration"""
    symbol: str
    limit_type: str  # 'absolute', 'percentage', 'volatility_adjusted'
    max_value: float
    max_percentage: float
    enabled: bool = True
    reason: str = ""
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class SectorLimit:
    """Sector exposure limit"""
    sector: str
    max_percentage: float
    current_exposure: float = 0.0
    positions: List[str] = None
    enabled: bool = True
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = []

@dataclass
class CorrelationLimit:
    """Correlation-based position limit"""
    symbols: List[str]
    max_correlation: float
    max_combined_exposure: float
    current_correlation: float = 0.0
    current_exposure: float = 0.0
    enabled: bool = True

class PositionLimitManager:
    """
    Manages position-specific risk limits and controls
    
    Features:
    - Individual position limits
    - Sector concentration limits
    - Correlation-based limits
    - Dynamic position sizing
    - Volatility-adjusted limits
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.position_limits = {}
        self.sector_limits = {}
        self.correlation_limits = []
        self.correlation_matrix = None
        self.sector_mappings = {}
        
        # Default limits
        self.default_limits = {
            'max_single_position': 0.25,      # 25% max in single position
            'max_sector_exposure': 0.40,      # 40% max in single sector
            'max_correlated_exposure': 0.60,  # 60% max in highly correlated positions
            'min_position_size': 0.01,        # 1% minimum position
            'volatility_multiplier': 2.0      # Volatility adjustment factor
        }
        
        # Initialize sector mappings for Indian stocks
        self._initialize_sector_mappings()
        self._initialize_default_sector_limits()
        
        logger.info("Position Limit Manager initialized")
    
    def _initialize_sector_mappings(self):
        """Initialize sector mappings for common Indian stocks"""
        self.sector_mappings = {
            # Banking & Financial Services
            'HDFCBANK': 'Banking',
            'ICICIBANK': 'Banking',
            'SBIN': 'Banking',
            'AXISBANK': 'Banking',
            'KOTAKBANK': 'Banking',
            'BAJFINANCE': 'Financial Services',
            'BAJAJFINSV': 'Financial Services',
            'HDFCLIFE': 'Insurance',
            'SBILIFE': 'Insurance',
            
            # Information Technology
            'TCS': 'Information Technology',
            'INFY': 'Information Technology',
            'WIPRO': 'Information Technology',
            'HCLTECH': 'Information Technology',
            'TECHM': 'Information Technology',
            'LTI': 'Information Technology',
            
            # Oil & Gas
            'RELIANCE': 'Oil & Gas',
            'ONGC': 'Oil & Gas',
            'IOC': 'Oil & Gas',
            'BPCL': 'Oil & Gas',
            'GAIL': 'Oil & Gas',
            
            # FMCG
            'HINDUNILVR': 'FMCG',
            'ITC': 'FMCG',
            'NESTLEIND': 'FMCG',
            'BRITANNIA': 'FMCG',
            'DABUR': 'FMCG',
            
            # Pharmaceuticals
            'SUNPHARMA': 'Pharmaceuticals',
            'DRREDDY': 'Pharmaceuticals',
            'CIPLA': 'Pharmaceuticals',
            'LUPIN': 'Pharmaceuticals',
            'BIOCON': 'Pharmaceuticals',
            
            # Automobiles
            'MARUTI': 'Automobiles',
            'TATAMOTORS': 'Automobiles',
            'M&M': 'Automobiles',
            'BAJAJ-AUTO': 'Automobiles',
            'HEROMOTOCO': 'Automobiles',
            
            # Metals & Mining
            'TATASTEEL': 'Metals',
            'HINDALCO': 'Metals',
            'VEDL': 'Metals',
            'JSWSTEEL': 'Metals',
            'COALINDIA': 'Mining',
            
            # Telecom
            'BHARTIARTL': 'Telecom',
            'JIOFINANCE': 'Telecom',
            
            # Indices
            'NIFTY50': 'Index',
            'SENSEX': 'Index',
            'NIFTYBEES': 'Index ETF',
            'SENSEXETF': 'Index ETF'
        }
    
    def _initialize_default_sector_limits(self):
        """Initialize default sector limits"""
        sectors = [
            'Banking', 'Information Technology', 'Oil & Gas', 'FMCG',
            'Pharmaceuticals', 'Automobiles', 'Metals', 'Mining',
            'Telecom', 'Insurance', 'Financial Services', 'Index', 'Index ETF'
        ]
        
        for sector in sectors:
            # Higher limits for diversified sectors like Index ETFs
            if sector in ['Index', 'Index ETF']:
                max_percentage = 0.60  # 60% for index funds
            elif sector in ['Banking', 'Information Technology']:
                max_percentage = 0.35  # 35% for major sectors
            else:
                max_percentage = 0.25  # 25% for other sectors
            
            self.sector_limits[sector] = SectorLimit(
                sector=sector,
                max_percentage=max_percentage,
                enabled=True
            )
    
    def set_position_limit(self, symbol: str, limit_type: str, max_value: float, max_percentage: float, reason: str = ""):
        """Set position limit for a specific symbol"""
        self.position_limits[symbol] = PositionLimit(
            symbol=symbol,
            limit_type=limit_type,
            max_value=max_value,
            max_percentage=max_percentage,
            reason=reason
        )
        logger.info(f"Position limit set for {symbol}: {max_percentage:.2%} / ₹{max_value:,.2f}")
    
    def set_sector_limit(self, sector: str, max_percentage: float):
        """Set sector exposure limit"""
        if sector in self.sector_limits:
            self.sector_limits[sector].max_percentage = max_percentage
        else:
            self.sector_limits[sector] = SectorLimit(
                sector=sector,
                max_percentage=max_percentage
            )
        logger.info(f"Sector limit set for {sector}: {max_percentage:.2%}")
    
    def add_correlation_limit(self, symbols: List[str], max_correlation: float, max_combined_exposure: float):
        """Add correlation-based limit"""
        correlation_limit = CorrelationLimit(
            symbols=symbols,
            max_correlation=max_correlation,
            max_combined_exposure=max_combined_exposure
        )
        self.correlation_limits.append(correlation_limit)
        logger.info(f"Correlation limit added for {symbols}: max {max_combined_exposure:.2%} combined exposure")
    
    def check_position_limit(self, symbol: str, new_quantity: int, current_price: float, 
                           portfolio_value: float, current_positions: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if new position would violate position limits
        
        Args:
            symbol: Stock symbol
            new_quantity: Proposed quantity to buy
            current_price: Current stock price
            portfolio_value: Total portfolio value
            current_positions: Current position holdings
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Calculate new position value
        current_quantity = current_positions.get(symbol, {}).get('quantity', 0)
        total_quantity = current_quantity + new_quantity
        total_position_value = total_quantity * current_price
        position_percentage = total_position_value / portfolio_value
        
        # Check specific position limit
        if symbol in self.position_limits:
            limit = self.position_limits[symbol]
            if not limit.enabled:
                return True, "Position limit disabled"
            
            if position_percentage > limit.max_percentage:
                return False, f"Position limit exceeded: {position_percentage:.2%} > {limit.max_percentage:.2%}"
            
            if total_position_value > limit.max_value:
                return False, f"Position value limit exceeded: ₹{total_position_value:,.2f} > ₹{limit.max_value:,.2f}"
        
        # Check default position limit
        max_single_position = self.default_limits['max_single_position']
        if position_percentage > max_single_position:
            return False, f"Default position limit exceeded: {position_percentage:.2%} > {max_single_position:.2%}"
        
        # Check minimum position size
        min_position_size = self.default_limits['min_position_size']
        if total_quantity > 0 and position_percentage < min_position_size:
            return False, f"Position too small: {position_percentage:.2%} < {min_position_size:.2%}"
        
        return True, "Position limit check passed"
    
    def check_sector_limit(self, symbol: str, new_quantity: int, current_price: float,
                          portfolio_value: float, current_positions: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if new position would violate sector limits"""
        sector = self.get_symbol_sector(symbol)
        if not sector:
            return True, "No sector classification available"
        
        if sector not in self.sector_limits:
            return True, "No sector limit configured"
        
        sector_limit = self.sector_limits[sector]
        if not sector_limit.enabled:
            return True, "Sector limit disabled"
        
        # Calculate current sector exposure
        current_sector_exposure = self.calculate_sector_exposure(sector, current_positions)
        
        # Calculate new sector exposure
        new_position_value = new_quantity * current_price
        new_sector_exposure = (current_sector_exposure * portfolio_value + new_position_value) / portfolio_value
        
        if new_sector_exposure > sector_limit.max_percentage:
            return False, f"Sector limit exceeded for {sector}: {new_sector_exposure:.2%} > {sector_limit.max_percentage:.2%}"
        
        return True, "Sector limit check passed"
    
    def check_correlation_limits(self, symbol: str, new_quantity: int, current_price: float,
                               portfolio_value: float, current_positions: Dict[str, Any]) -> Tuple[bool, str]:
        """Check correlation-based limits"""
        for correlation_limit in self.correlation_limits:
            if symbol not in correlation_limit.symbols:
                continue
            
            if not correlation_limit.enabled:
                continue
            
            # Calculate current combined exposure for correlated symbols
            combined_exposure = 0.0
            for corr_symbol in correlation_limit.symbols:
                if corr_symbol in current_positions:
                    pos = current_positions[corr_symbol]
                    position_value = pos.get('quantity', 0) * pos.get('current_price', current_price)
                    combined_exposure += position_value / portfolio_value
            
            # Add new position exposure
            new_position_value = new_quantity * current_price
            new_combined_exposure = combined_exposure + (new_position_value / portfolio_value)
            
            if new_combined_exposure > correlation_limit.max_combined_exposure:
                return False, f"Correlation limit exceeded for {correlation_limit.symbols}: {new_combined_exposure:.2%} > {correlation_limit.max_combined_exposure:.2%}"
        
        return True, "Correlation limit check passed"
    
    def validate_position(self, symbol: str, new_quantity: int, current_price: float,
                         portfolio_value: float, current_positions: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Comprehensive position validation
        
        Returns:
            Tuple of (is_valid, list_of_reasons)
        """
        reasons = []
        is_valid = True
        
        # Check position limit
        valid, reason = self.check_position_limit(symbol, new_quantity, current_price, portfolio_value, current_positions)
        if not valid:
            is_valid = False
            reasons.append(reason)
        
        # Check sector limit
        valid, reason = self.check_sector_limit(symbol, new_quantity, current_price, portfolio_value, current_positions)
        if not valid:
            is_valid = False
            reasons.append(reason)
        
        # Check correlation limits
        valid, reason = self.check_correlation_limits(symbol, new_quantity, current_price, portfolio_value, current_positions)
        if not valid:
            is_valid = False
            reasons.append(reason)
        
        if is_valid:
            reasons.append("All position limit checks passed")
        
        return is_valid, reasons
    
    def calculate_optimal_position_size(self, symbol: str, target_amount: float, current_price: float,
                                      portfolio_value: float, current_positions: Dict[str, Any],
                                      volatility: float = None) -> Tuple[int, float, str]:
        """
        Calculate optimal position size considering all limits
        
        Args:
            symbol: Stock symbol
            target_amount: Target investment amount
            current_price: Current stock price
            portfolio_value: Total portfolio value
            current_positions: Current positions
            volatility: Asset volatility (optional)
            
        Returns:
            Tuple of (optimal_quantity, actual_amount, reason)
        """
        # Start with target quantity
        target_quantity = int(target_amount / current_price)
        
        # Apply volatility adjustment if available
        if volatility is not None:
            vol_multiplier = self.default_limits['volatility_multiplier']
            vol_adjustment = max(0.5, min(2.0, vol_multiplier / max(volatility, 0.1)))
            target_quantity = int(target_quantity * vol_adjustment)
        
        # Check limits and adjust if necessary
        max_attempts = 10
        attempts = 0
        
        while attempts < max_attempts:
            is_valid, reasons = self.validate_position(symbol, target_quantity, current_price, portfolio_value, current_positions)
            
            if is_valid:
                actual_amount = target_quantity * current_price
                return target_quantity, actual_amount, "Optimal size calculated successfully"
            
            # Reduce quantity by 10% and try again
            target_quantity = int(target_quantity * 0.9)
            attempts += 1
            
            if target_quantity <= 0:
                return 0, 0.0, "Cannot determine valid position size"
        
        # Final attempt with minimal quantity
        if target_quantity > 0:
            actual_amount = target_quantity * current_price
            return target_quantity, actual_amount, f"Reduced size due to limits (attempts: {attempts})"
        
        return 0, 0.0, "Position size reduced to zero due to risk limits"
    
    def get_symbol_sector(self, symbol: str) -> Optional[str]:
        """Get sector for a symbol"""
        return self.sector_mappings.get(symbol)
    
    def calculate_sector_exposure(self, sector: str, current_positions: Dict[str, Any]) -> float:
        """Calculate current sector exposure as percentage"""
        sector_value = 0.0
        total_value = 0.0
        
        for symbol, position in current_positions.items():
            position_value = position.get('quantity', 0) * position.get('current_price', 0)
            total_value += position_value
            
            if self.get_symbol_sector(symbol) == sector:
                sector_value += position_value
        
        return sector_value / total_value if total_value > 0 else 0.0
    
    def get_sector_allocation(self, current_positions: Dict[str, Any]) -> Dict[str, float]:
        """Get current sector allocation breakdown"""
        sector_allocation = {}
        total_value = sum(pos.get('quantity', 0) * pos.get('current_price', 0) 
                         for pos in current_positions.values())
        
        if total_value == 0:
            return sector_allocation
        
        for symbol, position in current_positions.items():
            sector = self.get_symbol_sector(symbol)
            if sector:
                position_value = position.get('quantity', 0) * position.get('current_price', 0)
                sector_percentage = position_value / total_value
                
                if sector in sector_allocation:
                    sector_allocation[sector] += sector_percentage
                else:
                    sector_allocation[sector] = sector_percentage
        
        return sector_allocation
    
    def get_position_limits_summary(self) -> Dict[str, Any]:
        """Get summary of all position limits"""
        return {
            'default_limits': self.default_limits,
            'position_limits': {symbol: {
                'max_percentage': limit.max_percentage,
                'max_value': limit.max_value,
                'enabled': limit.enabled,
                'reason': limit.reason
            } for symbol, limit in self.position_limits.items()},
            'sector_limits': {sector: {
                'max_percentage': limit.max_percentage,
                'current_exposure': limit.current_exposure,
                'enabled': limit.enabled
            } for sector, limit in self.sector_limits.items()},
            'correlation_limits': [{
                'symbols': limit.symbols,
                'max_combined_exposure': limit.max_combined_exposure,
                'current_exposure': limit.current_exposure,
                'enabled': limit.enabled
            } for limit in self.correlation_limits]
        }
    
    def update_correlation_matrix(self, correlation_matrix: pd.DataFrame):
        """Update correlation matrix for correlation-based limits"""
        self.correlation_matrix = correlation_matrix
        
        # Update correlation limits with current correlations
        for correlation_limit in self.correlation_limits:
            if len(correlation_limit.symbols) >= 2:
                symbols = correlation_limit.symbols
                if all(symbol in correlation_matrix.index for symbol in symbols):
                    # Calculate average correlation among the symbols
                    correlations = []
                    for i, symbol1 in enumerate(symbols):
                        for symbol2 in symbols[i+1:]:
                            corr = correlation_matrix.loc[symbol1, symbol2]
                            correlations.append(abs(corr))
                    
                    if correlations:
                        correlation_limit.current_correlation = np.mean(correlations)
        
        logger.info("Correlation matrix updated for position limits")
    
    def save_limits_to_file(self, filename: str):
        """Save position limits to JSON file"""
        limits_data = {
            'default_limits': self.default_limits,
            'position_limits': {symbol: {
                'symbol': limit.symbol,
                'limit_type': limit.limit_type,
                'max_value': limit.max_value,
                'max_percentage': limit.max_percentage,
                'enabled': limit.enabled,
                'reason': limit.reason,
                'created_at': limit.created_at.isoformat()
            } for symbol, limit in self.position_limits.items()},
            'sector_limits': {sector: {
                'sector': limit.sector,
                'max_percentage': limit.max_percentage,
                'enabled': limit.enabled
            } for sector, limit in self.sector_limits.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(limits_data, f, indent=2)
        
        logger.info(f"Position limits saved to {filename}")
    
    def load_limits_from_file(self, filename: str):
        """Load position limits from JSON file"""
        try:
            with open(filename, 'r') as f:
                limits_data = json.load(f)
            
            # Load default limits
            if 'default_limits' in limits_data:
                self.default_limits.update(limits_data['default_limits'])
            
            # Load position limits
            if 'position_limits' in limits_data:
                for symbol, limit_data in limits_data['position_limits'].items():
                    self.position_limits[symbol] = PositionLimit(
                        symbol=limit_data['symbol'],
                        limit_type=limit_data['limit_type'],
                        max_value=limit_data['max_value'],
                        max_percentage=limit_data['max_percentage'],
                        enabled=limit_data.get('enabled', True),
                        reason=limit_data.get('reason', ''),
                        created_at=datetime.fromisoformat(limit_data.get('created_at', datetime.now().isoformat()))
                    )
            
            # Load sector limits
            if 'sector_limits' in limits_data:
                for sector, limit_data in limits_data['sector_limits'].items():
                    self.sector_limits[sector] = SectorLimit(
                        sector=limit_data['sector'],
                        max_percentage=limit_data['max_percentage'],
                        enabled=limit_data.get('enabled', True)
                    )
            
            logger.info(f"Position limits loaded from {filename}")
            
        except Exception as e:
            logger.error(f"Error loading position limits from {filename}: {str(e)}")
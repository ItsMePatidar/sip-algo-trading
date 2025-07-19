# =============================================================================
# risk_management/correlation_monitor.py - Portfolio Correlation Monitoring
# =============================================================================

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CorrelationLevel(Enum):
    VERY_LOW = "VERY_LOW"      # < 0.3
    LOW = "LOW"                # 0.3 - 0.5
    MODERATE = "MODERATE"      # 0.5 - 0.7
    HIGH = "HIGH"              # 0.7 - 0.85
    VERY_HIGH = "VERY_HIGH"    # > 0.85

@dataclass
class CorrelationAlert:
    """Correlation-based risk alert"""
    alert_type: str
    symbols: List[str]
    correlation: float
    risk_level: CorrelationLevel
    portfolio_exposure: float
    threshold: float
    timestamp: datetime
    description: str
    recommended_action: str

@dataclass
class CorrelationMetrics:
    """Portfolio correlation metrics"""
    average_correlation: float
    max_correlation: float
    min_correlation: float
    correlation_matrix: pd.DataFrame
    cluster_analysis: Dict[str, List[str]]
    diversification_ratio: float
    effective_positions: float
    concentration_risk: float

class CorrelationMonitor:
    """
    Monitors portfolio correlation risk and provides diversification insights
    
    Key Features:
    - Real-time correlation calculation
    - Correlation-based risk alerts
    - Portfolio diversification analysis
    - Cluster analysis for similar assets
    - Dynamic correlation tracking
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Configuration parameters
        self.lookback_period = self.config.get('lookback_period', 252)  # 1 year
        self.min_periods = self.config.get('min_periods', 60)  # Minimum periods for correlation
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.high_correlation_threshold = self.config.get('high_correlation_threshold', 0.85)
        self.cluster_threshold = self.config.get('cluster_threshold', 0.6)
        
        # Risk thresholds
        self.max_correlated_exposure = self.config.get('max_correlated_exposure', 0.6)  # 60%
        self.max_cluster_exposure = self.config.get('max_cluster_exposure', 0.4)  # 40%
        
        # Data storage
        self.price_data = {}
        self.returns_data = {}
        self.correlation_matrix = None
        self.correlation_history = []
        self.alerts_history = []
        
        # Cache for performance
        self.last_update = None
        self.cache_duration = timedelta(hours=1)
        
        logger.info("Correlation Monitor initialized")
    
    def update_price_data(self, symbol: str, price_data: pd.DataFrame):
        """
        Update price data for a symbol
        
        Args:
            symbol: Stock symbol
            price_data: DataFrame with price data (must have 'Close' column)
        """
        if 'Close' not in price_data.columns:
            logger.error(f"Price data for {symbol} missing 'Close' column")
            return
        
        # Store price data
        self.price_data[symbol] = price_data.copy()
        
        # Calculate returns
        returns = price_data['Close'].pct_change().dropna()
        if len(returns) >= self.min_periods:
            self.returns_data[symbol] = returns
        
        logger.debug(f"Updated price data for {symbol}: {len(price_data)} records")
    
    def calculate_correlation_matrix(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix for given symbols
        
        Args:
            symbols: List of symbols to include (None for all available)
            
        Returns:
            Correlation matrix DataFrame
        """
        if symbols is None:
            symbols = list(self.returns_data.keys())
        
        # Filter symbols with sufficient data
        valid_symbols = []
        for symbol in symbols:
            if symbol in self.returns_data and len(self.returns_data[symbol]) >= self.min_periods:
                valid_symbols.append(symbol)
        
        if len(valid_symbols) < 2:
            logger.warning("Insufficient data for correlation calculation")
            return pd.DataFrame()
        
        # Create returns matrix
        returns_matrix = pd.DataFrame()
        for symbol in valid_symbols:
            returns_matrix[symbol] = self.returns_data[symbol]
        
        # Align data (use common date range)
        returns_matrix = returns_matrix.dropna()
        
        if len(returns_matrix) < self.min_periods:
            logger.warning(f"Insufficient aligned data: {len(returns_matrix)} periods")
            return pd.DataFrame()
        
        # Calculate correlation matrix
        try:
            correlation_matrix = returns_matrix.corr()
            self.correlation_matrix = correlation_matrix
            
            logger.info(f"Calculated correlation matrix for {len(valid_symbols)} symbols")
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            return pd.DataFrame()
    
    def analyze_portfolio_correlation(self, 
                                    portfolio_weights: Dict[str, float]) -> CorrelationMetrics:
        """
        Analyze correlation risk for given portfolio weights
        
        Args:
            portfolio_weights: Dictionary of symbol weights
            
        Returns:
            CorrelationMetrics object with analysis results
        """
        symbols = list(portfolio_weights.keys())
        
        # Ensure correlation matrix is up to date
        correlation_matrix = self.calculate_correlation_matrix(symbols)
        
        if correlation_matrix.empty:
            logger.warning("Cannot analyze portfolio correlation: no correlation matrix")
            return self._empty_correlation_metrics()
        
        # Filter correlation matrix to portfolio symbols
        portfolio_symbols = [s for s in symbols if s in correlation_matrix.index]
        if len(portfolio_symbols) < 2:
            return self._empty_correlation_metrics()
        
        corr_subset = correlation_matrix.loc[portfolio_symbols, portfolio_symbols]
        
        # Calculate metrics
        metrics = CorrelationMetrics(
            average_correlation=self._calculate_average_correlation(corr_subset),
            max_correlation=self._calculate_max_correlation(corr_subset),
            min_correlation=self._calculate_min_correlation(corr_subset),
            correlation_matrix=corr_subset,
            cluster_analysis=self._perform_cluster_analysis(corr_subset),
            diversification_ratio=self._calculate_diversification_ratio(corr_subset, portfolio_weights),
            effective_positions=self._calculate_effective_positions(corr_subset, portfolio_weights),
            concentration_risk=self._calculate_concentration_risk(corr_subset, portfolio_weights)
        )
        
        return metrics
    
    def check_correlation_risks(self, 
                              portfolio_weights: Dict[str, float]) -> List[CorrelationAlert]:
        """
        Check for correlation-based risks in portfolio
        
        Args:
            portfolio_weights: Dictionary of symbol weights
            
        Returns:
            List of correlation alerts
        """
        alerts = []
        
        # Analyze portfolio correlation
        metrics = self.analyze_portfolio_correlation(portfolio_weights)
        
        if metrics.correlation_matrix.empty:
            return alerts
        
        # Check high correlation pairs
        alerts.extend(self._check_high_correlation_pairs(metrics, portfolio_weights))
        
        # Check cluster concentration
        alerts.extend(self._check_cluster_concentration(metrics, portfolio_weights))
        
        # Check overall diversification
        alerts.extend(self._check_diversification_level(metrics))
        
        # Store alerts
        self.alerts_history.extend(alerts)
        
        # Keep only recent alerts
        cutoff_date = datetime.now() - timedelta(days=30)
        self.alerts_history = [a for a in self.alerts_history if a.timestamp > cutoff_date]
        
        return alerts
    
    def _check_high_correlation_pairs(self, 
                                    metrics: CorrelationMetrics, 
                                    portfolio_weights: Dict[str, float]) -> List[CorrelationAlert]:
        """Check for highly correlated position pairs"""
        alerts = []
        corr_matrix = metrics.correlation_matrix
        
        # Find high correlation pairs
        for i in range(len(corr_matrix.index)):
            for j in range(i + 1, len(corr_matrix.columns)):
                symbol1 = corr_matrix.index[i]
                symbol2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                
                if abs(correlation) > self.correlation_threshold:
                    # Calculate combined exposure
                    exposure = portfolio_weights.get(symbol1, 0) + portfolio_weights.get(symbol2, 0)
                    
                    if exposure > self.max_correlated_exposure:
                        alert = CorrelationAlert(
                            alert_type="HIGH_CORRELATION_EXPOSURE",
                            symbols=[symbol1, symbol2],
                            correlation=correlation,
                            risk_level=self._get_correlation_level(abs(correlation)),
                            portfolio_exposure=exposure,
                            threshold=self.max_correlated_exposure,
                            timestamp=datetime.now(),
                            description=f"High correlation ({correlation:.3f}) between {symbol1} and {symbol2} with {exposure:.1%} combined exposure",
                            recommended_action="Consider reducing position in one of the assets"
                        )
                        alerts.append(alert)
        
        return alerts
    
    def _check_cluster_concentration(self, 
                                   metrics: CorrelationMetrics, 
                                   portfolio_weights: Dict[str, float]) -> List[CorrelationAlert]:
        """Check for concentration in correlated clusters"""
        alerts = []
        
        for cluster_name, cluster_symbols in metrics.cluster_analysis.items():
            if len(cluster_symbols) < 2:
                continue
            
            # Calculate cluster exposure
            cluster_exposure = sum(portfolio_weights.get(symbol, 0) for symbol in cluster_symbols)
            
            if cluster_exposure > self.max_cluster_exposure:
                # Calculate average correlation within cluster
                cluster_corr = self._calculate_cluster_correlation(metrics.correlation_matrix, cluster_symbols)
                
                alert = CorrelationAlert(
                    alert_type="CLUSTER_CONCENTRATION",
                    symbols=cluster_symbols,
                    correlation=cluster_corr,
                    risk_level=self._get_correlation_level(cluster_corr),
                    portfolio_exposure=cluster_exposure,
                    threshold=self.max_cluster_exposure,
                    timestamp=datetime.now(),
                    description=f"High concentration ({cluster_exposure:.1%}) in correlated cluster: {', '.join(cluster_symbols)}",
                    recommended_action="Diversify across different asset clusters"
                )
                alerts.append(alert)
        
        return alerts
    
    def _check_diversification_level(self, metrics: CorrelationMetrics) -> List[CorrelationAlert]:
        """Check overall portfolio diversification"""
        alerts = []
        
        # Check if effective positions are too low
        if metrics.effective_positions < 3:
            alert = CorrelationAlert(
                alert_type="LOW_DIVERSIFICATION",
                symbols=list(metrics.correlation_matrix.index),
                correlation=metrics.average_correlation,
                risk_level=CorrelationLevel.HIGH,
                portfolio_exposure=1.0,
                threshold=3.0,
                timestamp=datetime.now(),
                description=f"Low diversification: only {metrics.effective_positions:.1f} effective positions",
                recommended_action="Increase diversification by adding uncorrelated assets"
            )
            alerts.append(alert)
        
        # Check if average correlation is too high
        if metrics.average_correlation > 0.6:
            alert = CorrelationAlert(
                alert_type="HIGH_AVERAGE_CORRELATION",
                symbols=list(metrics.correlation_matrix.index),
                correlation=metrics.average_correlation,
                risk_level=self._get_correlation_level(metrics.average_correlation),
                portfolio_exposure=1.0,
                threshold=0.6,
                timestamp=datetime.now(),
                description=f"High average correlation: {metrics.average_correlation:.3f}",
                recommended_action="Add assets with lower correlation to existing holdings"
            )
            alerts.append(alert)
        
        return alerts
    
    def _calculate_average_correlation(self, correlation_matrix: pd.DataFrame) -> float:
        """Calculate average correlation excluding diagonal"""
        if correlation_matrix.empty:
            return 0.0
        
        # Get upper triangle excluding diagonal
        upper_triangle = np.triu(correlation_matrix.values, k=1)
        non_zero_elements = upper_triangle[upper_triangle != 0]
        
        if len(non_zero_elements) == 0:
            return 0.0
        
        return np.mean(np.abs(non_zero_elements))
    
    def _calculate_max_correlation(self, correlation_matrix: pd.DataFrame) -> float:
        """Calculate maximum correlation excluding diagonal"""
        if correlation_matrix.empty:
            return 0.0
        
        upper_triangle = np.triu(correlation_matrix.values, k=1)
        return np.max(np.abs(upper_triangle))
    
    def _calculate_min_correlation(self, correlation_matrix: pd.DataFrame) -> float:
        """Calculate minimum correlation excluding diagonal"""
        if correlation_matrix.empty:
            return 0.0
        
        upper_triangle = np.triu(correlation_matrix.values, k=1)
        non_zero_elements = upper_triangle[upper_triangle != 0]
        
        if len(non_zero_elements) == 0:
            return 0.0
        
        return np.min(np.abs(non_zero_elements))
    
    def _perform_cluster_analysis(self, correlation_matrix: pd.DataFrame) -> Dict[str, List[str]]:
        """Perform clustering analysis on correlation matrix"""
        if correlation_matrix.empty:
            return {}
        
        clusters = {}
        processed_symbols = set()
        cluster_id = 0
        
        for symbol in correlation_matrix.index:
            if symbol in processed_symbols:
                continue
            
            # Find highly correlated symbols
            cluster_symbols = [symbol]
            correlations = correlation_matrix.loc[symbol]
            
            for other_symbol in correlation_matrix.index:
                if (other_symbol != symbol and 
                    other_symbol not in processed_symbols and
                    abs(correlations[other_symbol]) > self.cluster_threshold):
                    cluster_symbols.append(other_symbol)
            
            if len(cluster_symbols) > 1:
                clusters[f"cluster_{cluster_id}"] = cluster_symbols
                processed_symbols.update(cluster_symbols)
                cluster_id += 1
            else:
                processed_symbols.add(symbol)
        
        return clusters
    
    def _calculate_diversification_ratio(self, 
                                       correlation_matrix: pd.DataFrame, 
                                       portfolio_weights: Dict[str, float]) -> float:
        """Calculate portfolio diversification ratio"""
        if correlation_matrix.empty:
            return 1.0
        
        try:
            # Filter weights to match correlation matrix
            symbols = correlation_matrix.index.tolist()
            weights = np.array([portfolio_weights.get(symbol, 0) for symbol in symbols])
            
            # Normalize weights
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                return 1.0
            
            # Calculate portfolio variance
            portfolio_variance = np.dot(weights, np.dot(correlation_matrix.values, weights))
            
            # Calculate average individual variance (assuming equal variances)
            avg_individual_variance = 1.0
            
            # Diversification ratio
            diversification_ratio = np.sqrt(avg_individual_variance / portfolio_variance)
            
            return min(diversification_ratio, len(symbols))  # Cap at number of assets
            
        except Exception as e:
            logger.error(f"Error calculating diversification ratio: {str(e)}")
            return 1.0
    
    def _calculate_effective_positions(self, 
                                     correlation_matrix: pd.DataFrame, 
                                     portfolio_weights: Dict[str, float]) -> float:
        """Calculate effective number of positions (Herfindahl-based)"""
        if correlation_matrix.empty:
            return 0.0
        
        try:
            # Filter weights to match correlation matrix
            symbols = correlation_matrix.index.tolist()
            weights = np.array([portfolio_weights.get(symbol, 0) for symbol in symbols])
            
            # Normalize weights
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                return 0.0
            
            # Calculate Herfindahl index
            herfindahl_index = np.sum(weights ** 2)
            
            # Effective positions
            effective_positions = 1.0 / herfindahl_index
            
            return effective_positions
            
        except Exception as e:
            logger.error(f"Error calculating effective positions: {str(e)}")
            return 0.0
    
    def _calculate_concentration_risk(self, 
                                    correlation_matrix: pd.DataFrame, 
                                    portfolio_weights: Dict[str, float]) -> float:
        """Calculate concentration risk score"""
        if correlation_matrix.empty:
            return 0.0
        
        # Simple concentration risk based on largest position
        max_weight = max(portfolio_weights.values()) if portfolio_weights else 0.0
        
        # Adjust for correlation
        avg_correlation = self._calculate_average_correlation(correlation_matrix)
        concentration_risk = max_weight * (1 + avg_correlation)
        
        return min(concentration_risk, 1.0)  # Cap at 100%
    
    def _calculate_cluster_correlation(self, 
                                     correlation_matrix: pd.DataFrame, 
                                     cluster_symbols: List[str]) -> float:
        """Calculate average correlation within a cluster"""
        if len(cluster_symbols) < 2:
            return 0.0
        
        correlations = []
        for i in range(len(cluster_symbols)):
            for j in range(i + 1, len(cluster_symbols)):
                symbol1, symbol2 = cluster_symbols[i], cluster_symbols[j]
                if symbol1 in correlation_matrix.index and symbol2 in correlation_matrix.columns:
                    corr = correlation_matrix.loc[symbol1, symbol2]
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _get_correlation_level(self, correlation: float) -> CorrelationLevel:
        """Get correlation level enum based on correlation value"""
        abs_corr = abs(correlation)
        
        if abs_corr < 0.3:
            return CorrelationLevel.VERY_LOW
        elif abs_corr < 0.5:
            return CorrelationLevel.LOW
        elif abs_corr < 0.7:
            return CorrelationLevel.MODERATE
        elif abs_corr < 0.85:
            return CorrelationLevel.HIGH
        else:
            return CorrelationLevel.VERY_HIGH
    
    def _empty_correlation_metrics(self) -> CorrelationMetrics:
        """Return empty correlation metrics"""
        return CorrelationMetrics(
            average_correlation=0.0,
            max_correlation=0.0,
            min_correlation=0.0,
            correlation_matrix=pd.DataFrame(),
            cluster_analysis={},
            diversification_ratio=1.0,
            effective_positions=0.0,
            concentration_risk=0.0
        )
    
    def get_correlation_report(self, portfolio_weights: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive correlation report"""
        metrics = self.analyze_portfolio_correlation(portfolio_weights)
        alerts = self.check_correlation_risks(portfolio_weights)
        
        report = {
            'timestamp': datetime.now(),
            'portfolio_symbols': list(portfolio_weights.keys()),
            'metrics': {
                'average_correlation': metrics.average_correlation,
                'max_correlation': metrics.max_correlation,
                'min_correlation': metrics.min_correlation,
                'diversification_ratio': metrics.diversification_ratio,
                'effective_positions': metrics.effective_positions,
                'concentration_risk': metrics.concentration_risk
            },
            'clusters': metrics.cluster_analysis,
            'alerts': [
                {
                    'type': alert.alert_type,
                    'symbols': alert.symbols,
                    'correlation': alert.correlation,
                    'risk_level': alert.risk_level.value,
                    'exposure': alert.portfolio_exposure,
                    'description': alert.description,
                    'action': alert.recommended_action
                } for alert in alerts
            ],
            'correlation_matrix': metrics.correlation_matrix.to_dict() if not metrics.correlation_matrix.empty else {},
            'risk_summary': {
                'overall_risk': self._assess_overall_correlation_risk(metrics, alerts),
                'high_risk_pairs': len([a for a in alerts if a.alert_type == "HIGH_CORRELATION_EXPOSURE"]),
                'cluster_risks': len([a for a in alerts if a.alert_type == "CLUSTER_CONCENTRATION"]),
                'diversification_issues': len([a for a in alerts if a.alert_type in ["LOW_DIVERSIFICATION", "HIGH_AVERAGE_CORRELATION"]])
            }
        }
        
        return report
    
    def _assess_overall_correlation_risk(self, 
                                       metrics: CorrelationMetrics, 
                                       alerts: List[CorrelationAlert]) -> str:
        """Assess overall correlation risk level"""
        if not alerts:
            return "LOW"
        
        critical_alerts = len([a for a in alerts if a.risk_level in [CorrelationLevel.HIGH, CorrelationLevel.VERY_HIGH]])
        
        if critical_alerts > 2:
            return "HIGH"
        elif critical_alerts > 0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def update_correlation_history(self, portfolio_weights: Dict[str, float]):
        """Update correlation tracking history"""
        metrics = self.analyze_portfolio_correlation(portfolio_weights)
        
        history_entry = {
            'timestamp': datetime.now(),
            'average_correlation': metrics.average_correlation,
            'max_correlation': metrics.max_correlation,
            'effective_positions': metrics.effective_positions,
            'diversification_ratio': metrics.diversification_ratio
        }
        
        self.correlation_history.append(history_entry)
        
        # Keep only last 100 entries
        if len(self.correlation_history) > 100:
            self.correlation_history = self.correlation_history[-100:]
    
    def get_correlation_trend(self, days: int = 30) -> Dict[str, Any]:
        """Get correlation trend over specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_history = [h for h in self.correlation_history if h['timestamp'] > cutoff_date]
        
        if len(recent_history) < 2:
            return {'trend': 'insufficient_data', 'data_points': len(recent_history)}
        
        # Calculate trends
        avg_correlations = [h['average_correlation'] for h in recent_history]
        effective_positions = [h['effective_positions'] for h in recent_history]
        
        return {
            'trend': {
                'correlation_direction': 'increasing' if avg_correlations[-1] > avg_correlations[0] else 'decreasing',
                'correlation_change': avg_correlations[-1] - avg_correlations[0],
                'diversification_direction': 'improving' if effective_positions[-1] > effective_positions[0] else 'deteriorating',
                'diversification_change': effective_positions[-1] - effective_positions[0]
            },
            'current_values': {
                'average_correlation': avg_correlations[-1],
                'effective_positions': effective_positions[-1]
            },
            'data_points': len(recent_history),
            'period_days': days
        }
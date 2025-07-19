# =============================================================================
# monitoring/market_monitor.py - Market Condition Monitoring
# =============================================================================

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import threading
import time

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_MARKET = "BULL_MARKET"
    BEAR_MARKET = "BEAR_MARKET"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    CRISIS = "CRISIS"

class MarketCondition(Enum):
    """Current market condition"""
    NORMAL = "NORMAL"
    STRESSED = "STRESSED"
    VOLATILE = "VOLATILE"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGE_BOUND = "RANGE_BOUND"

@dataclass
class MarketMetrics:
    """Market metrics snapshot"""
    timestamp: datetime
    indices: Dict[str, float]  # Index levels
    volatilities: Dict[str, float]  # VIX-like metrics
    correlations: Dict[str, float]  # Cross-asset correlations
    momentum: Dict[str, float]  # Momentum indicators
    breadth: Dict[str, float]  # Market breadth indicators
    sentiment: Dict[str, Any]  # Sentiment indicators

@dataclass
class MarketAlert:
    """Market condition alert"""
    alert_id: str
    timestamp: datetime
    alert_type: str
    severity: str  # 'INFO', 'WARNING', 'CRITICAL'
    market_metric: str
    current_value: float
    threshold: float
    description: str
    recommendation: str

class MarketMonitor:
    """
    Real-time Market Condition Monitor
    
    Monitors market conditions and generates alerts for:
    - Market regime changes
    - Volatility spikes
    - Correlation breakdowns
    - Unusual market movements
    - Sentiment shifts
    """
    
    def __init__(self, market_data_provider, config: Dict[str, Any] = None):
        self.market_data_provider = market_data_provider
        self.config = config or {}
        
        # Monitoring settings
        self.update_interval = self.config.get('update_interval', 60)  # seconds
        self.lookback_periods = {
            'short': 20,   # 20 days
            'medium': 60,  # 60 days
            'long': 252    # 1 year
        }
        
        # Market symbols to monitor
        self.benchmark_symbols = self.config.get('benchmark_symbols', [
            'NIFTY50', 'SENSEX', 'BANKNIFTY', 'NIFTYJR', 'NIFTYMID'
        ])
        
        self.sector_symbols = self.config.get('sector_symbols', [
            'CNXIT', 'CNXFMCG', 'CNXPHARMA', 'CNXAUTO', 'CNXBANK'
        ])
        
        # Alert thresholds
        self.alert_thresholds = {
            'volatility_spike': 2.0,        # 2x normal volatility
            'correlation_breakdown': 0.3,    # Correlation drops below 30%
            'momentum_reversal': 0.15,       # 15% momentum change
            'market_decline': 0.05,          # 5% daily decline
            'volume_surge': 3.0,             # 3x average volume
            'breadth_deterioration': 0.3     # Breadth below 30%
        }
        
        # State tracking
        self.current_metrics = None
        self.historical_metrics = []
        self.current_regime = MarketRegime.SIDEWAYS
        self.current_condition = MarketCondition.NORMAL
        self.alerts_generated = []
        
        # Monitoring control
        self.is_monitoring = False
        self.monitor_thread = None
        
        logger.info("Market Monitor initialized")
    
    def start_monitoring(self):
        """Start real-time market monitoring"""
        if self.is_monitoring:
            logger.warning("Market monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Market monitoring started")
    
    def stop_monitoring(self):
        """Stop market monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Market monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Update market metrics
                self.update_market_metrics()
                
                # Analyze market conditions
                self.analyze_market_conditions()
                
                # Check for alerts
                self.check_alert_conditions()
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in market monitoring loop: {str(e)}")
                time.sleep(self.update_interval)
    
    def update_market_metrics(self):
        """Update current market metrics"""
        try:
            timestamp = datetime.now()
            
            # Get current market data
            indices = self._get_index_levels()
            volatilities = self._calculate_volatilities()
            correlations = self._calculate_correlations()
            momentum = self._calculate_momentum_indicators()
            breadth = self._calculate_market_breadth()
            sentiment = self._get_sentiment_indicators()
            
            # Create metrics snapshot
            self.current_metrics = MarketMetrics(
                timestamp=timestamp,
                indices=indices,
                volatilities=volatilities,
                correlations=correlations,
                momentum=momentum,
                breadth=breadth,
                sentiment=sentiment
            )
            
            # Add to historical record
            self.historical_metrics.append(self.current_metrics)
            
            # Keep only recent history (last 30 days)
            if len(self.historical_metrics) > 30 * 24 * 60 // self.update_interval:
                self.historical_metrics = self.historical_metrics[-1000:]
            
            logger.debug("Market metrics updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating market metrics: {str(e)}")
    
    def _get_index_levels(self) -> Dict[str, float]:
        """Get current index levels"""
        indices = {}
        
        for symbol in self.benchmark_symbols:
            try:
                price = self.market_data_provider.get_current_price(symbol)
                if price:
                    indices[symbol] = price
            except Exception as e:
                logger.warning(f"Could not get price for {symbol}: {str(e)}")
        
        return indices
    
    def _calculate_volatilities(self) -> Dict[str, float]:
        """Calculate current volatilities"""
        volatilities = {}
        
        for symbol in self.benchmark_symbols:
            try:
                # Get historical data
                data = self.market_data_provider.fetch_historical_data(symbol, period='3mo')
                if data.empty:
                    continue
                
                # Calculate rolling volatility
                returns = data['Close'].pct_change().dropna()
                
                if len(returns) >= 20:
                    # 20-day volatility (annualized)
                    volatility = returns.tail(20).std() * np.sqrt(252)
                    volatilities[symbol] = volatility
                
            except Exception as e:
                logger.warning(f"Could not calculate volatility for {symbol}: {str(e)}")
        
        return volatilities
    
    def _calculate_correlations(self) -> Dict[str, float]:
        """Calculate cross-asset correlations"""
        correlations = {}
        
        try:
            # Get data for all benchmark symbols
            price_data = {}
            for symbol in self.benchmark_symbols[:3]:  # Limit to top 3 for performance
                data = self.market_data_provider.fetch_historical_data(symbol, period='3mo')
                if not data.empty:
                    price_data[symbol] = data['Close']
            
            if len(price_data) >= 2:
                # Create DataFrame
                df = pd.DataFrame(price_data)
                df = df.dropna()
                
                if len(df) >= 20:
                    # Calculate returns
                    returns = df.pct_change().dropna()
                    
                    # Calculate correlation matrix
                    corr_matrix = returns.corr()
                    
                    # Average correlation (excluding diagonal)
                    n = len(corr_matrix)
                    if n > 1:
                        total_corr = corr_matrix.sum().sum() - n  # Exclude diagonal
                        avg_correlation = total_corr / (n * (n - 1))
                        correlations['average_correlation'] = avg_correlation
                    
                    # Specific pairs
                    if 'NIFTY50' in returns.columns and 'SENSEX' in returns.columns:
                        correlations['nifty_sensex'] = returns['NIFTY50'].corr(returns['SENSEX'])
                
        except Exception as e:
            logger.warning(f"Could not calculate correlations: {str(e)}")
        
        return correlations
    
    def _calculate_momentum_indicators(self) -> Dict[str, float]:
        """Calculate momentum indicators"""
        momentum = {}
        
        for symbol in self.benchmark_symbols:
            try:
                # Get historical data
                data = self.market_data_provider.fetch_historical_data(symbol, period='6mo')
                if data.empty:
                    continue
                
                current_price = data['Close'].iloc[-1]
                
                # Short-term momentum (20 days)
                if len(data) >= 20:
                    price_20d = data['Close'].iloc[-20]
                    momentum_20d = (current_price - price_20d) / price_20d
                    momentum[f'{symbol}_20d'] = momentum_20d
                
                # Medium-term momentum (60 days)
                if len(data) >= 60:
                    price_60d = data['Close'].iloc[-60]
                    momentum_60d = (current_price - price_60d) / price_60d
                    momentum[f'{symbol}_60d'] = momentum_60d
                
                # RSI calculation
                if len(data) >= 14:
                    rsi = self._calculate_rsi(data['Close'], 14)
                    momentum[f'{symbol}_rsi'] = rsi
                
            except Exception as e:
                logger.warning(f"Could not calculate momentum for {symbol}: {str(e)}")
        
        return momentum
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
            
        except Exception:
            return 50.0  # Neutral RSI
    
    def _calculate_market_breadth(self) -> Dict[str, float]:
        """Calculate market breadth indicators"""
        breadth = {}
        
        try:
            # For demonstration, using sector performance as breadth proxy
            sector_performances = []
            
            for symbol in self.sector_symbols:
                try:
                    data = self.market_data_provider.fetch_historical_data(symbol, period='1mo')
                    if not data.empty and len(data) >= 2:
                        daily_return = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
                        sector_performances.append(daily_return)
                except Exception:
                    continue
            
            if sector_performances:
                # Percentage of sectors with positive performance
                positive_sectors = sum(1 for perf in sector_performances if perf > 0)
                breadth_ratio = positive_sectors / len(sector_performances)
                breadth['positive_sectors_ratio'] = breadth_ratio
                
                # Average sector performance
                breadth['average_sector_performance'] = np.mean(sector_performances)
                
        except Exception as e:
            logger.warning(f"Could not calculate market breadth: {str(e)}")
        
        return breadth
    
    def _get_sentiment_indicators(self) -> Dict[str, Any]:
        """Get market sentiment indicators"""
        sentiment = {}
        
        try:
            # VIX-like calculation (simplified)
            if 'NIFTY50' in self.current_metrics.volatilities if self.current_metrics else {}:
                nifty_vol = self.current_metrics.volatilities['NIFTY50']
                sentiment['volatility_regime'] = 'HIGH' if nifty_vol > 0.25 else 'LOW'
            
            # Fear & Greed proxy using momentum and volatility
            if self.current_metrics:
                momentum_scores = [v for k, v in self.current_metrics.momentum.items() if '_20d' in k]
                volatility_scores = list(self.current_metrics.volatilities.values())
                
                if momentum_scores and volatility_scores:
                    avg_momentum = np.mean(momentum_scores)
                    avg_volatility = np.mean(volatility_scores)
                    
                    # Simple fear/greed score (0-100)
                    # High momentum + low volatility = Greed
                    # Low momentum + high volatility = Fear
                    fear_greed_score = 50 + (avg_momentum * 100) - (avg_volatility * 50)
                    fear_greed_score = max(0, min(100, fear_greed_score))
                    
                    sentiment['fear_greed_index'] = fear_greed_score
                    
                    if fear_greed_score > 70:
                        sentiment['sentiment_label'] = 'GREED'
                    elif fear_greed_score < 30:
                        sentiment['sentiment_label'] = 'FEAR'
                    else:
                        sentiment['sentiment_label'] = 'NEUTRAL'
            
        except Exception as e:
            logger.warning(f"Could not calculate sentiment indicators: {str(e)}")
        
        return sentiment
    
    def analyze_market_conditions(self):
        """Analyze current market conditions and regime"""
        if not self.current_metrics:
            return
        
        try:
            # Determine market regime
            self.current_regime = self._determine_market_regime()
            
            # Determine market condition
            self.current_condition = self._determine_market_condition()
            
            logger.debug(f"Market regime: {self.current_regime.value}, Condition: {self.current_condition.value}")
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {str(e)}")
    
    def _determine_market_regime(self) -> MarketRegime:
        """Determine current market regime"""
        try:
            # Use momentum and volatility to determine regime
            momentum_values = [v for k, v in self.current_metrics.momentum.items() if '_60d' in k]
            volatility_values = list(self.current_metrics.volatilities.values())
            
            if momentum_values and volatility_values:
                avg_momentum = np.mean(momentum_values)
                avg_volatility = np.mean(volatility_values)
                
                # Crisis regime (high volatility + negative momentum)
                if avg_volatility > 0.35 and avg_momentum < -0.15:
                    return MarketRegime.CRISIS
                
                # High volatility regime
                elif avg_volatility > 0.30:
                    return MarketRegime.HIGH_VOLATILITY
                
                # Low volatility regime
                elif avg_volatility < 0.15:
                    return MarketRegime.LOW_VOLATILITY
                
                # Bull market (positive momentum)
                elif avg_momentum > 0.05:
                    return MarketRegime.BULL_MARKET
                
                # Bear market (negative momentum)
                elif avg_momentum < -0.05:
                    return MarketRegime.BEAR_MARKET
                
                # Sideways market
                else:
                    return MarketRegime.SIDEWAYS
            
            return MarketRegime.SIDEWAYS
            
        except Exception:
            return MarketRegime.SIDEWAYS
    
    def _determine_market_condition(self) -> MarketCondition:
        """Determine current market condition"""
        try:
            if not self.current_metrics:
                return MarketCondition.NORMAL
            
            # Check for stressed conditions
            volatility_values = list(self.current_metrics.volatilities.values())
            momentum_values = [v for k, v in self.current_metrics.momentum.items() if '_20d' in k]
            
            if volatility_values:
                avg_volatility = np.mean(volatility_values)
                
                # Volatile condition
                if avg_volatility > 0.30:
                    return MarketCondition.VOLATILE
                
                # Stressed condition
                elif avg_volatility > 0.25:
                    return MarketCondition.STRESSED
            
            if momentum_values:
                avg_momentum = np.mean(momentum_values)
                
                # Trending conditions
                if avg_momentum > 0.03:
                    return MarketCondition.TRENDING_UP
                elif avg_momentum < -0.03:
                    return MarketCondition.TRENDING_DOWN
                else:
                    return MarketCondition.RANGE_BOUND
            
            return MarketCondition.NORMAL
            
        except Exception:
            return MarketCondition.NORMAL
    
    def check_alert_conditions(self):
        """Check for alert conditions"""
        if not self.current_metrics:
            return
        
        try:
            alerts = []
            
            # Check volatility spike
            alerts.extend(self._check_volatility_alerts())
            
            # Check correlation breakdown
            alerts.extend(self._check_correlation_alerts())
            
            # Check momentum reversal
            alerts.extend(self._check_momentum_alerts())
            
            # Check market decline
            alerts.extend(self._check_decline_alerts())
            
            # Check breadth deterioration
            alerts.extend(self._check_breadth_alerts())
            
            # Add new alerts
            for alert in alerts:
                self.alerts_generated.append(alert)
                logger.warning(f"Market Alert: {alert.alert_type} - {alert.description}")
            
            # Keep only recent alerts (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.alerts_generated = [a for a in self.alerts_generated if a.timestamp > cutoff_time]
            
        except Exception as e:
            logger.error(f"Error checking alert conditions: {str(e)}")
    
    def _check_volatility_alerts(self) -> List[MarketAlert]:
        """Check for volatility spike alerts"""
        alerts = []
        
        try:
            threshold = self.alert_thresholds['volatility_spike']
            
            for symbol, volatility in self.current_metrics.volatilities.items():
                # Compare with historical average (if available)
                historical_vols = []
                for metrics in self.historical_metrics[-20:]:  # Last 20 observations
                    if symbol in metrics.volatilities:
                        historical_vols.append(metrics.volatilities[symbol])
                
                if historical_vols:
                    avg_vol = np.mean(historical_vols)
                    vol_ratio = volatility / avg_vol
                    
                    if vol_ratio > threshold:
                        alert = MarketAlert(
                            alert_id=f"vol_spike_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            timestamp=datetime.now(),
                            alert_type="VOLATILITY_SPIKE",
                            severity="WARNING",
                            market_metric=f"{symbol}_volatility",
                            current_value=volatility,
                            threshold=avg_vol * threshold,
                            description=f"{symbol} volatility spiked to {volatility:.2%} ({vol_ratio:.1f}x normal)",
                            recommendation="Consider reducing position sizes and increasing monitoring"
                        )
                        alerts.append(alert)
        
        except Exception as e:
            logger.warning(f"Error checking volatility alerts: {str(e)}")
        
        return alerts
    
    def _check_correlation_alerts(self) -> List[MarketAlert]:
        """Check for correlation breakdown alerts"""
        alerts = []
        
        try:
            threshold = self.alert_thresholds['correlation_breakdown']
            
            avg_corr = self.current_metrics.correlations.get('average_correlation')
            if avg_corr is not None and avg_corr < threshold:
                alert = MarketAlert(
                    alert_id=f"corr_breakdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    alert_type="CORRELATION_BREAKDOWN",
                    severity="WARNING",
                    market_metric="average_correlation",
                    current_value=avg_corr,
                    threshold=threshold,
                    description=f"Market correlation dropped to {avg_corr:.2f}",
                    recommendation="Diversification benefits may be reduced"
                )
                alerts.append(alert)
        
        except Exception as e:
            logger.warning(f"Error checking correlation alerts: {str(e)}")
        
        return alerts
    
    def _check_momentum_alerts(self) -> List[MarketAlert]:
        """Check for momentum reversal alerts"""
        alerts = []
        
        try:
            threshold = self.alert_thresholds['momentum_reversal']
            
            for key, momentum in self.current_metrics.momentum.items():
                if '_20d' in key:  # Focus on short-term momentum
                    if abs(momentum) > threshold:
                        severity = "CRITICAL" if abs(momentum) > 0.20 else "WARNING"
                        direction = "positive" if momentum > 0 else "negative"
                        
                        alert = MarketAlert(
                            alert_id=f"momentum_{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            timestamp=datetime.now(),
                            alert_type="MOMENTUM_CHANGE",
                            severity=severity,
                            market_metric=key,
                            current_value=momentum,
                            threshold=threshold,
                            description=f"Strong {direction} momentum in {key}: {momentum:.2%}",
                            recommendation=f"Monitor for trend continuation or reversal"
                        )
                        alerts.append(alert)
        
        except Exception as e:
            logger.warning(f"Error checking momentum alerts: {str(e)}")
        
        return alerts
    
    def _check_decline_alerts(self) -> List[MarketAlert]:
        """Check for market decline alerts"""
        alerts = []
        
        try:
            threshold = self.alert_thresholds['market_decline']
            
            for key, momentum in self.current_metrics.momentum.items():
                if '_20d' in key and momentum < -threshold:
                    symbol = key.replace('_20d', '')
                    
                    alert = MarketAlert(
                        alert_id=f"decline_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        timestamp=datetime.now(),
                        alert_type="MARKET_DECLINE",
                        severity="WARNING",
                        market_metric=key,
                        current_value=momentum,
                        threshold=-threshold,
                        description=f"{symbol} declined {momentum:.2%} over 20 days",
                        recommendation="Consider defensive positioning"
                    )
                    alerts.append(alert)
        
        except Exception as e:
            logger.warning(f"Error checking decline alerts: {str(e)}")
        
        return alerts
    
    def _check_breadth_alerts(self) -> List[MarketAlert]:
        """Check for market breadth alerts"""
        alerts = []
        
        try:
            threshold = self.alert_thresholds['breadth_deterioration']
            
            breadth_ratio = self.current_metrics.breadth.get('positive_sectors_ratio')
            if breadth_ratio is not None and breadth_ratio < threshold:
                alert = MarketAlert(
                    alert_id=f"breadth_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    alert_type="BREADTH_DETERIORATION",
                    severity="WARNING",
                    market_metric="positive_sectors_ratio",
                    current_value=breadth_ratio,
                    threshold=threshold,
                    description=f"Market breadth deteriorated: only {breadth_ratio:.1%} sectors positive",
                    recommendation="Broad market weakness detected"
                )
                alerts.append(alert)
        
        except Exception as e:
            logger.warning(f"Error checking breadth alerts: {str(e)}")
        
        return alerts
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive market summary"""
        if not self.current_metrics:
            return {"status": "No data available"}
        
        summary = {
            "timestamp": self.current_metrics.timestamp.isoformat(),
            "market_regime": self.current_regime.value,
            "market_condition": self.current_condition.value,
            "indices": self.current_metrics.indices,
            "volatilities": {k: f"{v:.2%}" for k, v in self.current_metrics.volatilities.items()},
            "key_metrics": {
                "average_volatility": f"{np.mean(list(self.current_metrics.volatilities.values())):.2%}" if self.current_metrics.volatilities else "N/A",
                "average_correlation": f"{self.current_metrics.correlations.get('average_correlation', 0):.2f}",
                "market_sentiment": self.current_metrics.sentiment.get('sentiment_label', 'UNKNOWN'),
                "fear_greed_index": self.current_metrics.sentiment.get('fear_greed_index', 50)
            },
            "recent_alerts": len([a for a in self.alerts_generated if a.timestamp > datetime.now() - timedelta(hours=1)]),
            "monitoring_status": "ACTIVE" if self.is_monitoring else "INACTIVE"
        }
        
        return summary
    
    def get_recent_alerts(self, hours: int = 24) -> List[MarketAlert]:
        """Get recent alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alerts_generated if a.timestamp > cutoff_time]
    
    def get_alert_summary(self) -> Dict[str, int]:
        """Get alert summary by type and severity"""
        recent_alerts = self.get_recent_alerts()
        
        summary = {
            "total_alerts": len(recent_alerts),
            "by_severity": {},
            "by_type": {}
        }
        
        for alert in recent_alerts:
            # Count by severity
            summary["by_severity"][alert.severity] = summary["by_severity"].get(alert.severity, 0) + 1
            
            # Count by type
            summary["by_type"][alert.alert_type] = summary["by_type"].get(alert.alert_type, 0) + 1
        
        return summary
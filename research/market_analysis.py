import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_id: int
    name: str
    start_date: datetime
    end_date: datetime
    characteristics: Dict[str, float]
    avg_return: float
    volatility: float
    sharpe_ratio: float

@dataclass
class MarketResearchReport:
    """Market analysis research report"""
    title: str
    analysis_period: Tuple[datetime, datetime]
    market_regimes: List[MarketRegime]
    correlation_analysis: Dict[str, Any]
    volatility_analysis: Dict[str, Any]
    technical_indicators: Dict[str, pd.DataFrame]
    key_insights: List[str]
    recommendations: List[str]
    charts: Dict[str, Any]

class TechnicalIndicators:
    """Technical analysis indicators"""
    
    @staticmethod
    def simple_moving_average(prices: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def exponential_moving_average(prices: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=window).mean()
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper_band = sma + (num_std * std)
        lower_band = sma - (num_std * std)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def relative_strength_index(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast_period).mean()
        ema_slow = prices.ewm(span=slow_period).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def volatility(prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling volatility"""
        returns = prices.pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252)

class CorrelationAnalysis:
    """Correlation analysis tools"""
    
    def __init__(self, market_data: Dict[str, pd.DataFrame]):
        self.market_data = market_data
        self.returns = self._calculate_returns()
    
    def _calculate_returns(self) -> pd.DataFrame:
        """Calculate returns for all assets"""
        returns_dict = {}
        
        for symbol, data in self.market_data.items():
            if 'Close' in data.columns:
                returns_dict[symbol] = data['Close'].pct_change().dropna()
        
        return pd.DataFrame(returns_dict)
    
    def static_correlation(self) -> pd.DataFrame:
        """Calculate static correlation matrix"""
        return self.returns.corr()
    
    def rolling_correlation(self, window: int = 60) -> Dict[str, pd.Series]:
        """Calculate rolling correlation between all pairs"""
        rolling_corrs = {}
        symbols = list(self.returns.columns)
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                key = f"{symbol1}_vs_{symbol2}"
                rolling_corrs[key] = self.returns[symbol1].rolling(window).corr(self.returns[symbol2])
        
        return rolling_corrs
    
    def correlation_clusters(self, method: str = 'ward') -> Dict[str, Any]:
        """Perform hierarchical clustering based on correlation"""
        from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
        from scipy.spatial.distance import squareform
        
        corr_matrix = self.static_correlation()
        
        # Convert correlation to distance
        distance_matrix = 1 - corr_matrix.abs()
        
        # Perform hierarchical clustering
        condensed_distances = squareform(distance_matrix.values)
        linkage_matrix = linkage(condensed_distances, method=method)
        
        # Get clusters
        clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')
        
        return {
            'linkage_matrix': linkage_matrix,
            'clusters': dict(zip(corr_matrix.index, clusters)),
            'distance_matrix': distance_matrix
        }

class VolatilityAnalysis:
    """Volatility analysis and modeling"""
    
    def __init__(self, market_data: Dict[str, pd.DataFrame]):
        self.market_data = market_data
        self.returns = self._calculate_returns()
    
    def _calculate_returns(self) -> pd.DataFrame:
        """Calculate returns for all assets"""
        returns_dict = {}
        
        for symbol, data in self.market_data.items():
            if 'Close' in data.columns:
                returns_dict[symbol] = data['Close'].pct_change().dropna()
        
        return pd.DataFrame(returns_dict)
    
    def realized_volatility(self, window: int = 20) -> pd.DataFrame:
        """Calculate realized volatility"""
        return self.returns.rolling(window=window).std() * np.sqrt(252) * 100
    
    def volatility_regimes(self, symbol: str, n_regimes: int = 3) -> Dict[str, Any]:
        """Identify volatility regimes using clustering"""
        if symbol not in self.returns.columns:
            raise ValueError(f"Symbol {symbol} not found in data")
        
        # Calculate rolling volatility
        vol = self.realized_volatility(window=20)[symbol].dropna()
        
        # Prepare data for clustering
        vol_data = vol.values.reshape(-1, 1)
        scaler = StandardScaler()
        vol_scaled = scaler.fit_transform(vol_data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regimes = kmeans.fit_predict(vol_scaled)
        
        # Create regime series
        regime_series = pd.Series(regimes, index=vol.index)
        
        # Calculate regime statistics
        regime_stats = {}
        for regime_id in range(n_regimes):
            mask = regimes == regime_id
            regime_vol = vol[regime_series == regime_id]
            
            regime_stats[regime_id] = {
                'avg_volatility': regime_vol.mean(),
                'min_volatility': regime_vol.min(),
                'max_volatility': regime_vol.max(),
                'periods': mask.sum(),
                'percentage': mask.sum() / len(regimes) * 100
            }
        
        return {
            'regimes': regime_series,
            'regime_stats': regime_stats,
            'cluster_centers': scaler.inverse_transform(kmeans.cluster_centers_).flatten()
        }
    
    def volatility_spillover(self) -> pd.DataFrame:
        """Calculate volatility spillover between assets"""
        # Calculate squared returns as proxy for volatility
        squared_returns = self.returns ** 2
        
        # Calculate correlations of squared returns
        spillover_matrix = squared_returns.corr()
        
        return spillover_matrix

class MarketAnalyzer:
    """
    Comprehensive market analysis toolkit
    
    Features:
    - Market regime identification
    - Correlation analysis
    - Volatility modeling
    - Technical analysis
    - Sector analysis
    - Risk factor analysis
    """
    
    def __init__(self, market_data: Dict[str, pd.DataFrame]):
        self.market_data = market_data
        self.technical_indicators = TechnicalIndicators()
        self.correlation_analysis = CorrelationAnalysis(market_data)
        self.volatility_analysis = VolatilityAnalysis(market_data)
        
        # Calculate combined market data
        self.combined_data = self._combine_market_data()
    
    def _combine_market_data(self) -> pd.DataFrame:
        """Combine market data into single DataFrame"""
        combined = {}
        
        for symbol, data in self.market_data.items():
            if 'Close' in data.columns:
                combined[f'{symbol}_Close'] = data['Close']
                combined[f'{symbol}_Volume'] = data.get('Volume', pd.Series(index=data.index))
        
        return pd.DataFrame(combined)
    
    def identify_market_regimes(self, 
                               symbol: str = None,
                               regime_indicators: List[str] = None,
                               n_regimes: int = 3) -> List[MarketRegime]:
        """
        Identify market regimes using multiple indicators
        
        Args:
            symbol: Primary symbol for regime identification
            regime_indicators: List of indicators to use
            n_regimes: Number of regimes to identify
            
        Returns:
            List of MarketRegime objects
        """
        logger.info(f"Identifying {n_regimes} market regimes")
        
        if symbol is None:
            symbol = list(self.market_data.keys())[0]
        
        if regime_indicators is None:
            regime_indicators = ['returns', 'volatility', 'trend']
        
        # Prepare regime identification data
        data = self.market_data[symbol].copy()
        returns = data['Close'].pct_change().dropna()
        
        # Calculate indicators
        indicators_data = []
        
        if 'returns' in regime_indicators:
            indicators_data.append(returns.rolling(20).mean())
        
        if 'volatility' in regime_indicators:
            vol = returns.rolling(20).std()
            indicators_data.append(vol)
        
        if 'trend' in regime_indicators:
            sma_short = self.technical_indicators.simple_moving_average(data['Close'], 20)
            sma_long = self.technical_indicators.simple_moving_average(data['Close'], 50)
            trend = (sma_short - sma_long) / sma_long
            indicators_data.append(trend)
        
        # Combine indicators
        regime_df = pd.concat(indicators_data, axis=1).dropna()
        regime_df.columns = regime_indicators
        
        # Standardize data
        scaler = StandardScaler()
        regime_scaled = scaler.fit_transform(regime_df)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regime_labels = kmeans.fit_predict(regime_scaled)
        
        # Create regime periods
        regime_series = pd.Series(regime_labels, index=regime_df.index)
        regimes = []
        
        current_regime = regime_labels[0]
        start_date = regime_df.index[0]
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] != current_regime:
                # Regime change detected
                end_date = regime_df.index[i-1]
                
                # Calculate regime characteristics
                regime_data = returns[start_date:end_date]
                characteristics = self._calculate_regime_characteristics(regime_data, regime_df.loc[start_date:end_date])
                
                regimes.append(MarketRegime(
                    regime_id=current_regime,
                    name=f"Regime_{current_regime}",
                    start_date=start_date,
                    end_date=end_date,
                    characteristics=characteristics,
                    avg_return=regime_data.mean() * 252,  # Annualized
                    volatility=regime_data.std() * np.sqrt(252),  # Annualized
                    sharpe_ratio=regime_data.mean() / regime_data.std() * np.sqrt(252) if regime_data.std() > 0 else 0
                ))
                
                current_regime = regime_labels[i]
                start_date = regime_df.index[i]
        
        # Add final regime
        end_date = regime_df.index[-1]
        regime_data = returns[start_date:end_date]
        characteristics = self._calculate_regime_characteristics(regime_data, regime_df.loc[start_date:end_date])
        
        regimes.append(MarketRegime(
            regime_id=current_regime,
            name=f"Regime_{current_regime}",
            start_date=start_date,
            end_date=end_date,
            characteristics=characteristics,
            avg_return=regime_data.mean() * 252,
            volatility=regime_data.std() * np.sqrt(252),
            sharpe_ratio=regime_data.mean() / regime_data.std() * np.sqrt(252) if regime_data.std() > 0 else 0
        ))
        
        return regimes
    
    def _calculate_regime_characteristics(self, 
                                       returns: pd.Series, 
                                       indicators: pd.DataFrame) -> Dict[str, float]:
        """Calculate characteristics for a market regime"""
        characteristics = {}
        
        for col in indicators.columns:
            characteristics[f'avg_{col}'] = indicators[col].mean()
            characteristics[f'std_{col}'] = indicators[col].std()
        
        characteristics['skewness'] = stats.skew(returns.dropna())
        characteristics['kurtosis'] = stats.kurtosis(returns.dropna())
        characteristics['var_95'] = np.percentile(returns.dropna(), 5)
        
        return characteristics
    
    def analyze_correlations(self) -> Dict[str, Any]:
        """Comprehensive correlation analysis"""
        logger.info("Performing correlation analysis")
        
        analysis = {
            'static_correlation': self.correlation_analysis.static_correlation(),
            'rolling_correlation': self.correlation_analysis.rolling_correlation(),
            'correlation_clusters': self.correlation_analysis.correlation_clusters()
        }
        
        # Average correlation over time
        rolling_corrs = analysis['rolling_correlation']
        avg_correlations = {}
        
        for pair, series in rolling_corrs.items():
            avg_correlations[pair] = {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max()
            }
        
        analysis['average_correlations'] = avg_correlations
        
        return analysis
    
    def analyze_volatility(self) -> Dict[str, Any]:
        """Comprehensive volatility analysis"""
        logger.info("Performing volatility analysis")
        
        analysis = {
            'realized_volatility': self.volatility_analysis.realized_volatility(),
            'volatility_spillover': self.volatility_analysis.volatility_spillover()
        }
        
        # Volatility regimes for each symbol
        vol_regimes = {}
        for symbol in self.market_data.keys():
            try:
                vol_regimes[symbol] = self.volatility_analysis.volatility_regimes(symbol)
            except Exception as e:
                logger.warning(f"Could not calculate volatility regimes for {symbol}: {str(e)}")
        
        analysis['volatility_regimes'] = vol_regimes
        
        return analysis
    
    def calculate_technical_indicators(self) -> Dict[str, pd.DataFrame]:
        """Calculate technical indicators for all symbols"""
        logger.info("Calculating technical indicators")
        
        indicators = {}
        
        for symbol, data in self.market_data.items():
            if 'Close' not in data.columns:
                continue
            
            prices = data['Close']
            symbol_indicators = pd.DataFrame(index=data.index)
            
            # Moving averages
            symbol_indicators['SMA_20'] = self.technical_indicators.simple_moving_average(prices, 20)
            symbol_indicators['SMA_50'] = self.technical_indicators.simple_moving_average(prices, 50)
            symbol_indicators['EMA_12'] = self.technical_indicators.exponential_moving_average(prices, 12)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.technical_indicators.bollinger_bands(prices)
            symbol_indicators['BB_Upper'] = bb_upper
            symbol_indicators['BB_Middle'] = bb_middle
            symbol_indicators['BB_Lower'] = bb_lower
            symbol_indicators['BB_Width'] = (bb_upper - bb_lower) / bb_middle
            
            # RSI
            symbol_indicators['RSI'] = self.technical_indicators.relative_strength_index(prices)
            
            # MACD
            macd, signal, histogram = self.technical_indicators.macd(prices)
            symbol_indicators['MACD'] = macd
            symbol_indicators['MACD_Signal'] = signal
            symbol_indicators['MACD_Histogram'] = histogram
            
            # Volatility
            symbol_indicators['Volatility'] = self.technical_indicators.volatility(prices)
            
            indicators[symbol] = symbol_indicators
        
        return indicators
    
    def generate_market_research_report(self,
                                      analysis_start: datetime,
                                      analysis_end: datetime,
                                      primary_symbol: str = None) -> MarketResearchReport:
        """
        Generate comprehensive market research report
        
        Args:
            analysis_start: Start date for analysis
            analysis_end: End date for analysis
            primary_symbol: Primary symbol for regime analysis
            
        Returns:
            MarketResearchReport object
        """
        logger.info(f"Generating market research report for {analysis_start} to {analysis_end}")
        
        if primary_symbol is None:
            primary_symbol = list(self.market_data.keys())[0]
        
        # Perform analyses
        market_regimes = self.identify_market_regimes(primary_symbol)
        correlation_analysis = self.analyze_correlations()
        volatility_analysis = self.analyze_volatility()
        technical_indicators = self.calculate_technical_indicators()
        
        # Generate insights
        key_insights = self._generate_market_insights(
            market_regimes, correlation_analysis, volatility_analysis
        )
        
        # Generate recommendations
        recommendations = self._generate_market_recommendations(
            market_regimes, correlation_analysis, volatility_analysis
        )
        
        # Create charts
        charts = self._create_market_charts(
            market_regimes, correlation_analysis, volatility_analysis, technical_indicators
        )
        
        return MarketResearchReport(
            title=f"Market Analysis Report: {analysis_start.strftime('%Y-%m-%d')} to {analysis_end.strftime('%Y-%m-%d')}",
            analysis_period=(analysis_start, analysis_end),
            market_regimes=market_regimes,
            correlation_analysis=correlation_analysis,
            volatility_analysis=volatility_analysis,
            technical_indicators=technical_indicators,
            key_insights=key_insights,
            recommendations=recommendations,
            charts=charts
        )
    
    def _generate_market_insights(self,
                                market_regimes: List[MarketRegime],
                                correlation_analysis: Dict[str, Any],
                                volatility_analysis: Dict[str, Any]) -> List[str]:
        """Generate key market insights"""
        insights = []
        
        # Market regime insights
        if market_regimes:
            high_vol_regimes = [r for r in market_regimes if r.volatility > 0.25]
            if high_vol_regimes:
                insights.append(f"Identified {len(high_vol_regimes)} high volatility periods with average volatility of {np.mean([r.volatility for r in high_vol_regimes]):.1f}%")
            
            best_regime = max(market_regimes, key=lambda x: x.sharpe_ratio)
            insights.append(f"Best performing regime had {best_regime.avg_return:.1f}% annualized return with {best_regime.volatility:.1f}% volatility")
        
        # Correlation insights
        if 'static_correlation' in correlation_analysis:
            corr_matrix = correlation_analysis['static_correlation']
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            insights.append(f"Average correlation between assets is {avg_correlation:.2f}")
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.8:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                insights.append(f"Found {len(high_corr_pairs)} highly correlated asset pairs (>0.8 correlation)")
        
        # Volatility insights
        if 'realized_volatility' in volatility_analysis:
            vol_data = volatility_analysis['realized_volatility']
            current_vol = vol_data.iloc[-1].mean()
            historical_avg = vol_data.mean().mean()
            
            if current_vol > historical_avg * 1.2:
                insights.append("Current market volatility is elevated compared to historical average")
            elif current_vol < historical_avg * 0.8:
                insights.append("Current market volatility is below historical average")
        
        return insights
    
    def _generate_market_recommendations(self,
                                       market_regimes: List[MarketRegime],
                                       correlation_analysis: Dict[str, Any],
                                       volatility_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable market recommendations"""
        recommendations = []
        
        # Regime-based recommendations
        if market_regimes:
            current_regime = market_regimes[-1] if market_regimes else None
            if current_regime and current_regime.volatility > 0.25:
                recommendations.append("Consider increasing SIP amounts during high volatility periods for better long-term returns")
                recommendations.append("Implement volatility-based position sizing to manage risk")
        
        # Correlation-based recommendations
        if 'static_correlation' in correlation_analysis:
            corr_matrix = correlation_analysis['static_correlation']
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            
            if avg_correlation > 0.7:
                recommendations.append("High correlation between assets suggests need for broader diversification")
            
            # Find least correlated assets for diversification
            min_corr_pair = None
            min_corr_value = 1.0
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] < min_corr_value:
                        min_corr_value = corr_matrix.iloc[i, j]
                        min_corr_pair = (corr_matrix.columns[i], corr_matrix.columns[j])
            
            if min_corr_pair:
                recommendations.append(f"Consider pairing {min_corr_pair[0]} and {min_corr_pair[1]} for better diversification")
        
        # Volatility-based recommendations
        if 'volatility_regimes' in volatility_analysis:
            recommendations.append("Use volatility regime detection to adjust investment timing")
            recommendations.append("Increase SIP frequency during low volatility periods")
        
        recommendations.append("Monitor correlation changes for early warning of market regime shifts")
        recommendations.append("Implement dynamic allocation based on current market regime")
        
        return recommendations
    
    def _create_market_charts(self,
                            market_regimes: List[MarketRegime],
                            correlation_analysis: Dict[str, Any],
                            volatility_analysis: Dict[str, Any],
                            technical_indicators: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create market analysis charts"""
        charts = {}
        
        # Market regime chart
        if market_regimes and len(self.market_data) > 0:
            primary_symbol = list(self.market_data.keys())[0]
            price_data = self.market_data[primary_symbol]['Close']
            
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(
                go.Scatter(x=price_data.index, y=price_data.values,
                          mode='lines', name='Price', line=dict(color='blue', width=1))
            )
            
            # Add regime backgrounds
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
            for i, regime in enumerate(market_regimes):
                fig.add_vrect(
                    x0=regime.start_date, x1=regime.end_date,
                    fillcolor=colors[regime.regime_id % len(colors)],
                    opacity=0.3,
                    line_width=0,
                    annotation_text=f"Regime {regime.regime_id}",
                    annotation_position="top left"
                )
            
            fig.update_layout(
                title='Market Regimes Analysis',
                xaxis_title='Date',
                yaxis_title='Price',
                height=500
            )
            
            charts['market_regimes'] = fig
        
        # Correlation heatmap
        if 'static_correlation' in correlation_analysis:
            corr_matrix = correlation_analysis['static_correlation']
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Asset Correlation Matrix',
                height=500
            )
            
            charts['correlation_heatmap'] = fig
        
        # Volatility analysis
        if 'realized_volatility' in volatility_analysis:
            vol_data = volatility_analysis['realized_volatility']
            
            fig = go.Figure()
            
            for symbol in vol_data.columns:
                fig.add_trace(
                    go.Scatter(x=vol_data.index, y=vol_data[symbol],
                              mode='lines', name=f'{symbol} Volatility',
                              line=dict(width=2))
                )
            
            fig.update_layout(
                title='Realized Volatility Over Time',
                xaxis_title='Date',
                yaxis_title='Volatility (%)',
                height=400
            )
            
            charts['volatility_analysis'] = fig
        
        # Technical indicators chart (for primary symbol)
        if technical_indicators and len(technical_indicators) > 0:
            primary_symbol = list(technical_indicators.keys())[0]
            indicators = technical_indicators[primary_symbol]
            price_data = self.market_data[primary_symbol]['Close']
            
            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                subplot_titles=('Price & Moving Averages', 'RSI', 'MACD'),
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Price and moving averages
            fig.add_trace(
                go.Scatter(x=price_data.index, y=price_data.values,
                          mode='lines', name='Price', line=dict(color='black', width=2)),
                row=1, col=1
            )
            
            if 'SMA_20' in indicators.columns:
                fig.add_trace(
                    go.Scatter(x=indicators.index, y=indicators['SMA_20'],
                              mode='lines', name='SMA 20', line=dict(color='blue', width=1)),
                    row=1, col=1
                )
            
            if 'SMA_50' in indicators.columns:
                fig.add_trace(
                    go.Scatter(x=indicators.index, y=indicators['SMA_50'],
                              mode='lines', name='SMA 50', line=dict(color='red', width=1)),
                    row=1, col=1
                )
            
            # RSI
            if 'RSI' in indicators.columns:
                fig.add_trace(
                    go.Scatter(x=indicators.index, y=indicators['RSI'],
                              mode='lines', name='RSI', line=dict(color='purple', width=2)),
                    row=2, col=1
                )
                
                # Add RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # MACD
            if 'MACD' in indicators.columns and 'MACD_Signal' in indicators.columns:
                fig.add_trace(
                    go.Scatter(x=indicators.index, y=indicators['MACD'],
                              mode='lines', name='MACD', line=dict(color='blue', width=2)),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=indicators.index, y=indicators['MACD_Signal'],
                              mode='lines', name='Signal', line=dict(color='red', width=2)),
                    row=3, col=1
                )
            
            fig.update_layout(height=800, showlegend=True)
            charts['technical_indicators'] = fig
        
        return charts
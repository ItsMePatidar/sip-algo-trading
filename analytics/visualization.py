import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging


logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PerformanceVisualizer:
    """
    Comprehensive visualization tools for performance analysis
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def create_comprehensive_report(self, 
                                  portfolio_data: Dict[str, pd.Series],
                                  benchmark_data: Optional[pd.Series] = None,
                                  save_path: Optional[str] = None) -> None:
        """
        Create comprehensive performance report
        
        Args:
            portfolio_data: Dictionary of portfolio time series {name: values}
            benchmark_data: Benchmark time series (optional)
            save_path: Path to save the report (optional)
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Cumulative Returns
        ax1 = axes[0, 0]
        self._plot_cumulative_returns_subplot(portfolio_data, benchmark_data, ax1)
        
        # Plot 2: Rolling Sharpe Ratio
        ax2 = axes[0, 1]
        self._plot_rolling_sharpe_subplot(portfolio_data, ax2)
        
        # Plot 3: Drawdown
        ax3 = axes[0, 2]
        self._plot_drawdown_subplot(portfolio_data, ax3)
        
        # Plot 4: Returns Distribution
        ax4 = axes[1, 0]
        self._plot_returns_distribution_subplot(portfolio_data, ax4)
        
        # Plot 5: Risk-Return Scatter
        ax5 = axes[1, 1]
        self._plot_risk_return_scatter_subplot(portfolio_data, benchmark_data, ax5)
        
        # Plot 6: Monthly Returns Heatmap
        ax6 = axes[1, 2]
        self._plot_monthly_returns_heatmap_subplot(portfolio_data, ax6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance report saved to {save_path}")
        
        plt.show()
    
    def _plot_cumulative_returns_subplot(self, 
                                       portfolio_data: Dict[str, pd.Series],
                                       benchmark_data: Optional[pd.Series],
                                       ax: plt.Axes) -> None:
        """Plot cumulative returns subplot"""
        for i, (name, values) in enumerate(portfolio_data.items()):
            returns = calculate_returns(values)
            cumulative = (1 + returns).cumprod()
            ax.plot(cumulative.index, cumulative.values, 
                   label=name, color=self.colors[i % len(self.colors)], linewidth=2)
        
        if benchmark_data is not None:
            bench_returns = calculate_returns(benchmark_data)
            bench_cumulative = (1 + bench_returns).cumprod()
            ax.plot(bench_cumulative.index, bench_cumulative.values, 
                   label='Benchmark', color='black', linewidth=2, linestyle='--')
        
        ax.set_title('Cumulative Returns', fontweight='bold')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_rolling_sharpe_subplot(self, portfolio_data: Dict[str, pd.Series], ax: plt.Axes) -> None:
        """Plot rolling Sharpe ratio subplot"""
        for i, (name, values) in enumerate(portfolio_data.items()):
            returns = calculate_returns(values)
            rolling_sharpe = calculate_rolling_sharpe(returns, window=60)
            ax.plot(rolling_sharpe.index, rolling_sharpe.values, 
                   label=name, color=self.colors[i % len(self.colors)], alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title('Rolling Sharpe Ratio (60-day)', fontweight='bold')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_drawdown_subplot(self, portfolio_data: Dict[str, pd.Series], ax: plt.Axes) -> None:
        """Plot drawdown subplot"""
        for i, (name, values) in enumerate(portfolio_data.items()):
            returns = calculate_returns(values)
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            ax.fill_between(drawdown.index, drawdown.values, 0, 
                          alpha=0.3, color=self.colors[i % len(self.colors)], label=name)
        
        ax.set_title('Drawdown Analysis', fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_returns_distribution_subplot(self, portfolio_data: Dict[str, pd.Series], ax: plt.Axes) -> None:
        """Plot returns distribution subplot"""
        for i, (name, values) in enumerate(portfolio_data.items()):
            returns = calculate_returns(values)
            ax.hist(returns, bins=50, alpha=0.6, 
                   color=self.colors[i % len(self.colors)], label=name, density=True)
        
        ax.set_title('Returns Distribution', fontweight='bold')
        ax.set_xlabel('Daily Returns')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_risk_return_scatter_subplot(self, 
                                        portfolio_data: Dict[str, pd.Series],
                                        benchmark_data: Optional[pd.Series],
                                        ax: plt.Axes) -> None:
        """Plot risk-return scatter subplot"""
        returns_list = []
        volatility_list = []
        names = []
        
        for name, values in portfolio_data.items():
            returns = calculate_returns(values)
            annual_return = returns.mean() * 252
            annual_vol = calculate_annualized_volatility(returns)
            
            returns_list.append(annual_return)
            volatility_list.append(annual_vol)
            names.append(name)
        
        # Add benchmark if provided
        if benchmark_data is not None:
            bench_returns = calculate_returns(benchmark_data)
            bench_annual_return = bench_returns.mean() * 252
            bench_annual_vol = calculate_annualized_volatility(bench_returns)
            
            returns_list.append(bench_annual_return)
            volatility_list.append(bench_annual_vol)
            names.append('Benchmark')
        
        scatter = ax.scatter(volatility_list, returns_list, 
                           c=range(len(names)), cmap='viridis', s=100, alpha=0.7)
        
        # Add labels
        for i, name in enumerate(names):
            ax.annotate(name, (volatility_list[i], returns_list[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_title('Risk vs Return', fontweight='bold')
        ax.set_xlabel('Annualized Volatility')
        ax.set_ylabel('Annualized Return')
        ax.grid(True, alpha=0.3)
    
    def _plot_monthly_returns_heatmap_subplot(self, portfolio_data: Dict[str, pd.Series], ax: plt.Axes) -> None:
        """Plot monthly returns heatmap for first portfolio"""
        if not portfolio_data:
            return
        
        # Use first portfolio for heatmap
        name, values = next(iter(portfolio_data.items()))
        returns = calculate_returns(values)
        
        # Group by year and month
        monthly_returns = returns.groupby([returns.index.year, returns.index.month]).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Create pivot table
        monthly_pivot = monthly_returns.unstack(level=1, fill_value=0)
        
        if monthly_pivot.empty:
            ax.text(0.5, 0.5, 'Insufficient data for heatmap', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create heatmap
        sns.heatmap(monthly_pivot, annot=True, fmt='.1%', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': 'Monthly Return'})
        
        ax.set_title(f'Monthly Returns Heatmap - {name}', fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')

# Standalone plotting functions
def plot_cumulative_returns(portfolio_data: Dict[str, pd.Series],
                          benchmark_data: Optional[pd.Series] = None,
                          title: str = "Cumulative Returns",
                          figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot cumulative returns for multiple portfolios
    
    Args:
        portfolio_data: Dictionary of portfolio time series {name: values}
        benchmark_data: Benchmark time series (optional)
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, values) in enumerate(portfolio_data.items()):
        returns = calculate_returns(values)
        cumulative = (1 + returns).cumprod()
        ax.plot(cumulative.index, cumulative.values, 
               label=name, color=colors[i % len(colors)], linewidth=2)
    
    if benchmark_data is not None:
        bench_returns = calculate_returns(benchmark_data)
        bench_cumulative = (1 + bench_returns).cumprod()
        ax.plot(bench_cumulative.index, bench_cumulative.values, 
               label='Benchmark', color='black', linewidth=2, linestyle='--')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_rolling_sharpe(portfolio_data: Dict[str, pd.Series],
                       window: int = 60,
                       title: str = "Rolling Sharpe Ratio",
                       figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot rolling Sharpe ratio
    
    Args:
        portfolio_data: Dictionary of portfolio time series {name: values}
        window: Rolling window size
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, values) in enumerate(portfolio_data.items()):
        returns = calculate_returns(values)
        rolling_sharpe = calculate_rolling_sharpe(returns, window=window)
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, 
               label=name, color=colors[i % len(colors)], alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
    
    ax.set_title(f'{title} ({window}-day)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_drawdown(portfolio_data: Dict[str, pd.Series],
                 title: str = "Drawdown Analysis",
                 figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot drawdown analysis
    
    Args:
        portfolio_data: Dictionary of portfolio time series {name: values}
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, values) in enumerate(portfolio_data.items()):
        returns = calculate_returns(values)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        ax.fill_between(drawdown.index, drawdown.values, 0, 
                      alpha=0.3, color=colors[i % len(colors)], label=name)
        ax.plot(drawdown.index, drawdown.values, 
               color=colors[i % len(colors)], linewidth=1)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    plt.tight_layout()
    plt.show()

def plot_returns_distribution(portfolio_data: Dict[str, pd.Series],
                            title: str = "Returns Distribution",
                            figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot returns distribution
    
    Args:
        portfolio_data: Dictionary of portfolio time series {name: values}
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, values) in enumerate(portfolio_data.items()):
        returns = calculate_returns(values)
        ax.hist(returns, bins=50, alpha=0.6, 
               color=colors[i % len(colors)], label=name, density=True)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Daily Returns', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_risk_return_scatter(portfolio_data: Dict[str, pd.Series],
                           benchmark_data: Optional[pd.Series] = None,
                           title: str = "Risk vs Return",
                           figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot risk-return scatter plot
    
    Args:
        portfolio_data: Dictionary of portfolio time series {name: values}
        benchmark_data: Benchmark time series (optional)
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    returns_list = []
    volatility_list = []
    names = []
    
    for name, values in portfolio_data.items():
        returns = calculate_returns(values)
        annual_return = returns.mean() * 252
        annual_vol = calculate_annualized_volatility(returns)
        
        returns_list.append(annual_return)
        volatility_list.append(annual_vol)
        names.append(name)
    
    # Add benchmark if provided
    if benchmark_data is not None:
        bench_returns = calculate_returns(benchmark_data)
        bench_annual_return = bench_returns.mean() * 252
        bench_annual_vol = calculate_annualized_volatility(bench_returns)
        
        returns_list.append(bench_annual_return)
        volatility_list.append(bench_annual_vol)
        names.append('Benchmark')
    
    scatter = ax.scatter(volatility_list, returns_list, 
                       c=range(len(names)), cmap='viridis', s=150, alpha=0.7)
    
    # Add labels
    for i, name in enumerate(names):
        ax.annotate(name, (volatility_list[i], returns_list[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Annualized Volatility', fontsize=12)
    ax.set_ylabel('Annualized Return', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format axes as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(portfolio_data: Dict[str, pd.Series],
                           title: str = "Correlation Matrix",
                           figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot correlation heatmap
    
    Args:
        portfolio_data: Dictionary of portfolio time series {name: values}
        title: Plot title
        figsize: Figure size
    """
    # Calculate returns for all portfolios
    returns_df = pd.DataFrame()
    for name, values in portfolio_data.items():
        returns_df[name] = calculate_returns(values)
    
    # Calculate correlation matrix
    correlation_matrix = returns_df.corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
               square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def create_performance_dashboard(portfolio_data: Dict[str, pd.Series],
                               benchmark_data: Optional[pd.Series] = None,
                               interactive: bool = True) -> None:
    """
    Create interactive performance dashboard using Plotly
    
    Args:
        portfolio_data: Dictionary of portfolio time series {name: values}
        benchmark_data: Benchmark time series (optional)
        interactive: Whether to create interactive plots
    """
    if not interactive:
        # Use matplotlib version
        visualizer = PerformanceVisualizer()
        visualizer.create_comprehensive_report(portfolio_data, benchmark_data)
        return
    
    # Create interactive Plotly dashboard
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Cumulative Returns', 'Rolling Sharpe Ratio', 'Drawdown',
                       'Returns Distribution', 'Risk vs Return', 'Monthly Returns'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = px.colors.qualitative.Set1
    
    # Plot 1: Cumulative Returns
    for i, (name, values) in enumerate(portfolio_data.items()):
        returns = calculate_returns(values)
        cumulative = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(x=cumulative.index, y=cumulative.values, 
                      name=name, line=dict(color=colors[i % len(colors)])),
            row=1, col=1
        )
    
    if benchmark_data is not None:
        bench_returns = calculate_returns(benchmark_data)
        bench_cumulative = (1 + bench_returns).cumprod()
        fig.add_trace(
            go.Scatter(x=bench_cumulative.index, y=bench_cumulative.values,
                      name='Benchmark', line=dict(color='black', dash='dash')),
            row=1, col=1
        )
    
    # Plot 2: Rolling Sharpe Ratio
    for i, (name, values) in enumerate(portfolio_data.items()):
        returns = calculate_returns(values)
        rolling_sharpe = calculate_rolling_sharpe(returns, window=60)
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                      name=f'{name} Sharpe', line=dict(color=colors[i % len(colors)]),
                      showlegend=False),
            row=1, col=2
        )
    
    # Plot 3: Drawdown
    for i, (name, values) in enumerate(portfolio_data.items()):
        returns = calculate_returns(values)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values,
                      name=f'{name} DD', fill='tonexty' if i == 0 else 'tozeroy',
                      line=dict(color=colors[i % len(colors)]),
                      showlegend=False),
            row=1, col=3
        )
    
    # Update layout
    fig.update_layout(
        title_text="Performance Dashboard",
        showlegend=True,
        height=800,
        template="plotly_white"
    )
    
    fig.show()

def plot_strategy_comparison(strategy_results: Dict[str, Dict[str, float]],
                           metrics: List[str] = ['sharpe_ratio', 'max_drawdown', 'cagr'],
                           title: str = "Strategy Comparison",
                           figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot strategy comparison across multiple metrics
    
    Args:
        strategy_results: Dictionary of strategy results {strategy_name: {metric: value}}
        metrics: List of metrics to compare
        title: Plot title
        figsize: Figure size
    """
    # Prepare data
    strategies = list(strategy_results.keys())
    metric_data = {metric: [strategy_results[strategy].get(metric, 0) for strategy in strategies] 
                  for metric in metrics}
    
    # Create subplots
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(strategies, metric_data[metric], 
                     color=[colors[j % len(colors)] for j in range(len(strategies))],
                     alpha=0.7)
        
        ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title())
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_data[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom')
        
        # Rotate x-axis labels if needed
        if len(max(strategies, key=len)) > 10:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Test function for the analytics module
def test_analytics_module():
    """Test the analytics module with sample data"""
    print("Testing Analytics Module...")
    print("=" * 50)
    
    # Generate sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Sample portfolio data
    portfolio_data = {}
    
    # Strategy 1: Steady growth with low volatility
    returns1 = np.random.normal(0.0008, 0.015, len(dates))  # 0.08% daily, 1.5% vol
    values1 = pd.Series((1 + pd.Series(returns1)).cumprod() * 100000, index=dates)
    portfolio_data['Conservative Strategy'] = values1
    
    # Strategy 2: Higher growth with higher volatility
    returns2 = np.random.normal(0.0012, 0.025, len(dates))  # 0.12% daily, 2.5% vol
    values2 = pd.Series((1 + pd.Series(returns2)).cumprod() * 100000, index=dates)
    portfolio_data['Aggressive Strategy'] = values2
    
    # Benchmark data
    bench_returns = np.random.normal(0.0010, 0.020, len(dates))  # 0.10% daily, 2.0% vol
    benchmark_data = pd.Series((1 + pd.Series(bench_returns)).cumprod() * 100000, index=dates)
    
    print("✓ Sample data generated")
    
    # Test performance analyzer
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze_strategy_performance(values1, benchmark_data)
    print(f"✓ Performance analysis completed. Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
    
    # Test risk analyzer
    risk_analyzer = RiskAnalyzer()
    risk_metrics = risk_analyzer.analyze_portfolio_risk(calculate_returns(values1))
    print(f"✓ Risk analysis completed. Max drawdown: {risk_metrics.get('max_drawdown', 0):.3f}")
    
    # Test visualization
    print("✓ Creating performance plots...")
    plot_cumulative_returns(portfolio_data, benchmark_data)
    plot_risk_return_scatter(portfolio_data, benchmark_data)
    
    print("✓ Analytics module test completed successfully!")

if __name__ == "__main__":
    test_analytics_module()
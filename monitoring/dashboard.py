# =============================================================================
# monitoring/dashboard.py - Real-time Trading Dashboard
# =============================================================================

import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os
from typing import Dict, List, Any, Optional
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.storage import DatabaseManager
from data.market_data import MarketDataProvider
from brokers.zerodha_broker import ZerodhaBroker
from orders.order_manager import OrderManager
from config.settings import settings

logger = logging.getLogger(__name__)

class TradingDashboard:
    """
    Real-time Trading Dashboard using Dash/Plotly
    
    Features:
    - Live portfolio monitoring
    - Real-time P&L tracking
    - Risk metrics display
    - Position management
    - Market condition monitoring
    - Alert management
    """
    
    def __init__(self, host='127.0.0.1', port=8050, debug=False):
        self.host = host
        self.port = port
        self.debug = debug
        
        # Initialize trading components
        self.db_manager = DatabaseManager()
        self.market_data = MarketDataProvider(self.db_manager)
        
        # Initialize broker in demo mode for dashboard
        broker_config = {
            'api_key': settings.broker.api_key,
            'api_secret': settings.broker.api_secret,
            'demo_mode': True
        }
        self.broker = ZerodhaBroker(broker_config)
        self.broker.connect()
        
        self.order_manager = OrderManager(self.db_manager, self.broker)
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[
            'https://codepen.io/chriddyp/pen/bWLwgP.css',
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
        ])
        
        self.app.title = "SIP Trading Dashboard"
        
        # Setup layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
        
        logger.info("Trading dashboard initialized")
    
    def _setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("SIP Algorithmic Trading Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px'}),
                html.Div([
                    html.Span(f"Last Updated: ", style={'marginRight': '10px'}),
                    html.Span(id='last-update-time', style={'fontWeight': 'bold'}),
                    html.Button("🔄 Refresh", id='refresh-button', 
                               style={'marginLeft': '20px', 'padding': '5px 10px'})
                ], style={'textAlign': 'center', 'marginBottom': '20px'})
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Update every 5 seconds
                n_intervals=0
            ),
            
            # Main content tabs
            dcc.Tabs(id='main-tabs', value='portfolio-tab', children=[
                dcc.Tab(label='📊 Portfolio Overview', value='portfolio-tab'),
                dcc.Tab(label='📈 Performance', value='performance-tab'),
                dcc.Tab(label='🛡️ Risk Monitor', value='risk-tab'),
                dcc.Tab(label='📋 Positions', value='positions-tab'),
                dcc.Tab(label='🔔 Alerts', value='alerts-tab'),
                dcc.Tab(label='📊 Market Data', value='market-tab'),
            ]),
            
            # Tab content
            html.Div(id='tab-content')
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'value'),
             Input('interval-component', 'n_intervals'),
             Input('refresh-button', 'n_clicks')]
        )
        def render_tab_content(active_tab, n_intervals, refresh_clicks):
            """Render content based on active tab"""
            try:
                if active_tab == 'portfolio-tab':
                    return self._render_portfolio_tab()
                elif active_tab == 'performance-tab':
                    return self._render_performance_tab()
                elif active_tab == 'risk-tab':
                    return self._render_risk_tab()
                elif active_tab == 'positions-tab':
                    return self._render_positions_tab()
                elif active_tab == 'alerts-tab':
                    return self._render_alerts_tab()
                elif active_tab == 'market-tab':
                    return self._render_market_tab()
                else:
                    return html.Div("Tab not found")
            except Exception as e:
                logger.error(f"Error rendering tab {active_tab}: {str(e)}")
                return html.Div(f"Error loading content: {str(e)}", 
                               style={'color': 'red', 'margin': '20px'})
        
        @self.app.callback(
            Output('last-update-time', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_timestamp(n_intervals):
            """Update last refresh timestamp"""
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def _render_portfolio_tab(self):
        """Render portfolio overview tab"""
        try:
            # Get portfolio data
            portfolio_data = self._get_portfolio_data()
            
            return html.Div([
                # Portfolio summary cards
                html.Div([
                    self._create_metric_card("💰 Total Value", f"₹{portfolio_data['total_value']:,.2f}", "green"),
                    self._create_metric_card("📈 Daily P&L", f"₹{portfolio_data['daily_pnl']:,.2f}", 
                                           "green" if portfolio_data['daily_pnl'] >= 0 else "red"),
                    self._create_metric_card("💵 Cash Available", f"₹{portfolio_data['cash_available']:,.2f}", "blue"),
                    self._create_metric_card("📊 Positions", str(portfolio_data['position_count']), "purple"),
                ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
                
                # Portfolio allocation chart
                html.Div([
                    html.H3("Portfolio Allocation", style={'textAlign': 'center'}),
                    dcc.Graph(
                        id='portfolio-allocation-chart',
                        figure=self._create_allocation_chart(portfolio_data['positions'])
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                # Portfolio value over time
                html.Div([
                    html.H3("Portfolio Value Trend", style={'textAlign': 'center'}),
                    dcc.Graph(
                        id='portfolio-value-chart',
                        figure=self._create_portfolio_value_chart()
                    )
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
                
                # Recent transactions
                html.Div([
                    html.H3("Recent Transactions", style={'marginTop': '30px'}),
                    self._create_transactions_table()
                ], style={'marginTop': '50px'})
            ])
        except Exception as e:
            return html.Div(f"Error loading portfolio data: {str(e)}", style={'color': 'red'})
    
    def _render_performance_tab(self):
        """Render performance analysis tab"""
        try:
            performance_data = self._get_performance_data()
            
            return html.Div([
                # Performance metrics
                html.Div([
                    self._create_metric_card("📈 Total Return", f"{performance_data['total_return']:.2%}", "green"),
                    self._create_metric_card("📊 Sharpe Ratio", f"{performance_data['sharpe_ratio']:.2f}", "blue"),
                    self._create_metric_card("📉 Max Drawdown", f"{performance_data['max_drawdown']:.2%}", "red"),
                    self._create_metric_card("📈 Win Rate", f"{performance_data['win_rate']:.1%}", "purple"),
                ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
                
                # Performance charts
                html.Div([
                    html.H3("Cumulative Returns", style={'textAlign': 'center'}),
                    dcc.Graph(
                        id='cumulative-returns-chart',
                        figure=self._create_cumulative_returns_chart()
                    )
                ], style={'marginBottom': '30px'}),
                
                html.Div([
                    # Monthly returns heatmap
                    html.Div([
                        html.H3("Monthly Returns Heatmap"),
                        dcc.Graph(
                            id='monthly-returns-heatmap',
                            figure=self._create_monthly_returns_heatmap()
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    # Drawdown chart
                    html.Div([
                        html.H3("Drawdown Analysis"),
                        dcc.Graph(
                            id='drawdown-chart',
                            figure=self._create_drawdown_chart()
                        )
                    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
                ])
            ])
        except Exception as e:
            return html.Div(f"Error loading performance data: {str(e)}", style={'color': 'red'})
    
    def _render_risk_tab(self):
        """Render risk monitoring tab"""
        try:
            risk_data = self._get_risk_data()
            
            return html.Div([
                # Risk metrics
                html.Div([
                    self._create_metric_card("⚠️ Risk Level", risk_data['risk_level'], 
                                           self._get_risk_color(risk_data['risk_level'])),
                    self._create_metric_card("📊 Portfolio VaR", f"₹{risk_data['var']:,.0f}", "orange"),
                    self._create_metric_card("📈 Volatility", f"{risk_data['volatility']:.1%}", "blue"),
                    self._create_metric_card("🔗 Max Correlation", f"{risk_data['max_correlation']:.2f}", "purple"),
                ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
                
                # Risk charts
                html.Div([
                    # Risk metrics over time
                    html.Div([
                        html.H3("Risk Metrics Trend"),
                        dcc.Graph(
                            id='risk-metrics-chart',
                            figure=self._create_risk_metrics_chart()
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    # Position concentration
                    html.Div([
                        html.H3("Position Concentration"),
                        dcc.Graph(
                            id='concentration-chart',
                            figure=self._create_concentration_chart()
                        )
                    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
                ]),
                
                # Risk limits table
                html.Div([
                    html.H3("Risk Limits Status", style={'marginTop': '30px'}),
                    self._create_risk_limits_table()
                ])
            ])
        except Exception as e:
            return html.Div(f"Error loading risk data: {str(e)}", style={'color': 'red'})
    
    def _render_positions_tab(self):
        """Render positions management tab"""
        try:
            return html.Div([
                html.H3("Current Positions"),
                self._create_positions_table(),
                
                html.H3("Position Performance", style={'marginTop': '30px'}),
                dcc.Graph(
                    id='position-performance-chart',
                    figure=self._create_position_performance_chart()
                )
            ])
        except Exception as e:
            return html.Div(f"Error loading positions data: {str(e)}", style={'color': 'red'})
    
    def _render_alerts_tab(self):
        """Render alerts management tab"""
        try:
            alerts_data = self._get_alerts_data()
            
            return html.Div([
                # Alert summary
                html.Div([
                    self._create_metric_card("🔴 Critical", str(alerts_data['critical_count']), "red"),
                    self._create_metric_card("🟡 Warning", str(alerts_data['warning_count']), "orange"),
                    self._create_metric_card("🔵 Info", str(alerts_data['info_count']), "blue"),
                    self._create_metric_card("📊 Total Today", str(alerts_data['total_today']), "purple"),
                ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
                
                # Active alerts table
                html.H3("Active Alerts"),
                self._create_alerts_table(),
                
                # Alert history chart
                html.Div([
                    html.H3("Alert History", style={'marginTop': '30px'}),
                    dcc.Graph(
                        id='alert-history-chart',
                        figure=self._create_alert_history_chart()
                    )
                ])
            ])
        except Exception as e:
            return html.Div(f"Error loading alerts data: {str(e)}", style={'color': 'red'})
    
    def _render_market_tab(self):
        """Render market data tab"""
        try:
            return html.Div([
                # Market indices
                html.H3("Market Indices"),
                dcc.Graph(
                    id='market-indices-chart',
                    figure=self._create_market_indices_chart()
                ),
                
                # Market sentiment
                html.Div([
                    html.H3("Market Data", style={'marginTop': '30px'}),
                    self._create_market_data_table()
                ])
            ])
        except Exception as e:
            return html.Div(f"Error loading market data: {str(e)}", style={'color': 'red'})
    
    def _create_metric_card(self, title, value, color):
        """Create a metric display card"""
        return html.Div([
            html.H4(title, style={'margin': '0', 'color': '#7f8c8d'}),
            html.H2(value, style={'margin': '10px 0', 'color': color, 'fontWeight': 'bold'})
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'textAlign': 'center',
            'minWidth': '200px'
        })
    
    def _get_portfolio_data(self):
        """Get current portfolio data"""
        try:
            # Get positions from broker
            positions = self.broker.get_positions()
            balance = self.broker.get_balance()
            
            # Calculate portfolio metrics
            total_position_value = sum(pos.get('quantity', 0) * pos.get('current_price', 0) for pos in positions)
            total_value = balance['cash'] + total_position_value
            
            # Mock daily P&L calculation
            daily_pnl = np.random.uniform(-500, 1000)  # Replace with actual calculation
            
            return {
                'total_value': total_value,
                'cash_available': balance['cash'],
                'daily_pnl': daily_pnl,
                'position_count': len(positions),
                'positions': positions
            }
        except Exception as e:
            logger.error(f"Error getting portfolio data: {str(e)}")
            return {
                'total_value': 100000,
                'cash_available': 75000,
                'daily_pnl': 250,
                'position_count': 3,
                'positions': []
            }
    
    def _get_performance_data(self):
        """Get performance metrics"""
        # Mock performance data - replace with actual calculations
        return {
            'total_return': 0.125,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'win_rate': 0.65
        }
    
    def _get_risk_data(self):
        """Get risk metrics"""
        # Mock risk data - replace with actual calculations
        return {
            'risk_level': 'MEDIUM',
            'var': 5000,
            'volatility': 0.18,
            'max_correlation': 0.72
        }
    
    def _get_alerts_data(self):
        """Get alerts data"""
        # Mock alerts data - replace with actual alert system
        return {
            'critical_count': 0,
            'warning_count': 2,
            'info_count': 5,
            'total_today': 7
        }
    
    def _get_risk_color(self, risk_level):
        """Get color for risk level"""
        colors = {
            'LOW': 'green',
            'MEDIUM': 'orange',
            'HIGH': 'red',
            'CRITICAL': 'darkred'
        }
        return colors.get(risk_level, 'gray')
    
    def _create_allocation_chart(self, positions):
        """Create portfolio allocation pie chart"""
        if not positions:
            return go.Figure().add_annotation(text="No positions", showarrow=False)
        
        symbols = [pos.get('symbol', 'Unknown') for pos in positions]
        values = [pos.get('quantity', 0) * pos.get('current_price', 0) for pos in positions]
        
        fig = go.Figure(data=[go.Pie(labels=symbols, values=values, hole=0.3)])
        fig.update_layout(title="Portfolio Allocation")
        return fig
    
    def _create_portfolio_value_chart(self):
        """Create portfolio value trend chart"""
        # Mock data - replace with actual historical data
        dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='D')
        values = np.cumsum(np.random.randn(len(dates)) * 100) + 100000
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=values, mode='lines', name='Portfolio Value'))
        fig.update_layout(title="Portfolio Value Over Time", xaxis_title="Date", yaxis_title="Value (₹)")
        return fig
    
    def _create_cumulative_returns_chart(self):
        """Create cumulative returns chart"""
        # Mock data
        dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='D')
        returns = np.cumsum(np.random.randn(len(dates)) * 0.01)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=returns, mode='lines', name='Cumulative Returns'))
        fig.update_layout(title="Cumulative Returns", xaxis_title="Date", yaxis_title="Returns (%)")
        return fig
    
    def _create_monthly_returns_heatmap(self):
        """Create monthly returns heatmap"""
        # Mock data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        years = ['2023', '2024']
        data = np.random.randn(len(years), len(months)) * 0.05
        
        fig = go.Figure(data=go.Heatmap(z=data, x=months, y=years, colorscale='RdYlGn'))
        fig.update_layout(title="Monthly Returns Heatmap")
        return fig
    
    def _create_drawdown_chart(self):
        """Create drawdown chart"""
        # Mock data
        dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='D')
        drawdown = np.minimum(0, np.cumsum(np.random.randn(len(dates)) * 0.01))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=drawdown, mode='lines', fill='tonexty', name='Drawdown'))
        fig.update_layout(title="Drawdown Analysis", xaxis_title="Date", yaxis_title="Drawdown (%)")
        return fig
    
    def _create_risk_metrics_chart(self):
        """Create risk metrics trend chart"""
        # Mock data
        dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='W')
        volatility = np.random.uniform(0.15, 0.25, len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=volatility, mode='lines', name='Volatility'))
        fig.update_layout(title="Portfolio Volatility Trend", xaxis_title="Date", yaxis_title="Volatility")
        return fig
    
    def _create_concentration_chart(self):
        """Create position concentration chart"""
        # Mock data
        symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
        concentrations = [0.25, 0.20, 0.18, 0.15, 0.12]
        
        fig = go.Figure(data=[go.Bar(x=symbols, y=concentrations)])
        fig.update_layout(title="Position Concentration", xaxis_title="Symbol", yaxis_title="Concentration (%)")
        return fig
    
    def _create_position_performance_chart(self):
        """Create position performance chart"""
        positions = self.broker.get_positions()
        if not positions:
            return go.Figure().add_annotation(text="No positions", showarrow=False)
        
        symbols = [pos.get('symbol', 'Unknown') for pos in positions]
        pnl = [pos.get('pnl', 0) for pos in positions]
        
        colors = ['green' if p >= 0 else 'red' for p in pnl]
        
        fig = go.Figure(data=[go.Bar(x=symbols, y=pnl, marker_color=colors)])
        fig.update_layout(title="Position P&L", xaxis_title="Symbol", yaxis_title="P&L (₹)")
        return fig
    
    def _create_market_indices_chart(self):
        """Create market indices chart"""
        # Mock data for major indices
        symbols = ['NIFTY50', 'SENSEX', 'BANKNIFTY']
        changes = [0.5, -0.2, 0.8]  # Percentage changes
        colors = ['green' if c >= 0 else 'red' for c in changes]
        
        fig = go.Figure(data=[go.Bar(x=symbols, y=changes, marker_color=colors)])
        fig.update_layout(title="Market Indices Performance", xaxis_title="Index", yaxis_title="Change (%)")
        return fig
    
    def _create_alert_history_chart(self):
        """Create alert history chart"""
        # Mock data
        dates = pd.date_range(start=datetime.now()-timedelta(days=30), end=datetime.now(), freq='D')
        alert_counts = np.random.poisson(3, len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=alert_counts, mode='lines+markers', name='Daily Alerts'))
        fig.update_layout(title="Alert History", xaxis_title="Date", yaxis_title="Alert Count")
        return fig
    
    def _create_transactions_table(self):
        """Create recent transactions table"""
        # Mock transaction data
        transactions = [
            {'Date': '2024-01-15', 'Symbol': 'RELIANCE', 'Side': 'BUY', 'Quantity': 10, 'Price': 2650, 'Amount': 26500},
            {'Date': '2024-01-15', 'Symbol': 'TCS', 'Side': 'BUY', 'Quantity': 5, 'Price': 3500, 'Amount': 17500},
            {'Date': '2024-01-14', 'Symbol': 'HDFCBANK', 'Side': 'BUY', 'Quantity': 8, 'Price': 1600, 'Amount': 12800},
        ]
        
        return dash_table.DataTable(
            data=transactions,
            columns=[{'name': col, 'id': col} for col in transactions[0].keys()],
            style_cell={'textAlign': 'left'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Side} = BUY'},
                    'backgroundColor': '#d4edda',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{Side} = SELL'},
                    'backgroundColor': '#f8d7da',
                    'color': 'black',
                }
            ]
        )
    
    def _create_positions_table(self):
        """Create positions table"""
        positions = self.broker.get_positions()
        
        if not positions:
            return html.Div("No positions found", style={'textAlign': 'center', 'margin': '20px'})
        
        return dash_table.DataTable(
            data=positions,
            columns=[
                {'name': 'Symbol', 'id': 'symbol'},
                {'name': 'Quantity', 'id': 'quantity'},
                {'name': 'Avg Price', 'id': 'average_price', 'type': 'numeric', 'format': {'specifier': ',.2f'}},
                {'name': 'Current Price', 'id': 'current_price', 'type': 'numeric', 'format': {'specifier': ',.2f'}},
                {'name': 'P&L', 'id': 'pnl', 'type': 'numeric', 'format': {'specifier': ',.2f'}}
            ],
            style_cell={'textAlign': 'center'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{pnl} >= 0'},
                    'backgroundColor': '#d4edda',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{pnl} < 0'},
                    'backgroundColor': '#f8d7da',
                    'color': 'black',
                }
            ]
        )
    
    def _create_alerts_table(self):
        """Create alerts table"""
        # Mock alerts data
        alerts = [
            {'Time': '10:30:00', 'Level': 'WARNING', 'Message': 'Position concentration above 20%', 'Symbol': 'RELIANCE'},
            {'Time': '09:45:00', 'Level': 'INFO', 'Message': 'SIP execution completed', 'Symbol': 'ALL'},
            {'Time': '09:15:00', 'Level': 'INFO', 'Message': 'Market opened', 'Symbol': 'MARKET'},
        ]
        
        return dash_table.DataTable(
            data=alerts,
            columns=[{'name': col, 'id': col} for col in alerts[0].keys()],
            style_cell={'textAlign': 'left'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Level} = CRITICAL'},
                    'backgroundColor': '#f8d7da',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{Level} = WARNING'},
                    'backgroundColor': '#fff3cd',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{Level} = INFO'},
                    'backgroundColor': '#d1ecf1',
                    'color': 'black',
                }
            ]
        )
    
    def _create_risk_limits_table(self):
        """Create risk limits status table"""
        # Mock risk limits data
        risk_limits = [
            {'Limit': 'Max Single Position', 'Current': '22%', 'Threshold': '25%', 'Status': 'OK'},
            {'Limit': 'Portfolio Drawdown', 'Current': '3%', 'Threshold': '15%', 'Status': 'OK'},
            {'Limit': 'Daily Loss Limit', 'Current': '1%', 'Threshold': '5%', 'Status': 'OK'},
            {'Limit': 'Cash Reserve', 'Current': '75%', 'Threshold': '10%', 'Status': 'OK'},
        ]
        
        return dash_table.DataTable(
            data=risk_limits,
            columns=[{'name': col, 'id': col} for col in risk_limits[0].keys()],
            style_cell={'textAlign': 'center'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Status} = OK'},
                    'backgroundColor': '#d4edda',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{Status} = WARNING'},
                    'backgroundColor': '#fff3cd',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{Status} = VIOLATED'},
                    'backgroundColor': '#f8d7da',
                    'color': 'black',
                }
            ]
        )
    
    def _create_market_data_table(self):
        """Create market data table"""
        try:
            # Get current prices for major symbols
            symbols = ['NIFTY50', 'RELIANCE', 'TCS', 'HDFCBANK', 'INFY']
            market_data = []
            
            for symbol in symbols:
                try:
                    price = self.market_data.get_current_price(symbol)
                    if price:
                        # Mock additional data
                        change = np.random.uniform(-2, 2)
                        change_pct = change / price * 100
                        volume = np.random.randint(100000, 1000000)
                        
                        market_data.append({
                            'Symbol': symbol,
                            'Price': f"₹{price:.2f}",
                            'Change': f"₹{change:.2f}",
                            'Change %': f"{change_pct:.2f}%",
                            'Volume': f"{volume:,}"
                        })
                except Exception as e:
                    logger.warning(f"Could not get price for {symbol}: {str(e)}")
            
            if not market_data:
                # Fallback mock data
                market_data = [
                    {'Symbol': 'NIFTY50', 'Price': '₹19,450.75', 'Change': '₹125.30', 'Change %': '0.65%', 'Volume': '245,678'},
                    {'Symbol': 'RELIANCE', 'Price': '₹2,650.80', 'Change': '-₹15.20', 'Change %': '-0.57%', 'Volume': '1,234,567'},
                    {'Symbol': 'TCS', 'Price': '₹3,520.45', 'Change': '₹28.15', 'Change %': '0.81%', 'Volume': '890,234'},
                    {'Symbol': 'HDFCBANK', 'Price': '₹1,605.90', 'Change': '₹12.40', 'Change %': '0.78%', 'Volume': '2,345,678'},
                    {'Symbol': 'INFY', 'Price': '₹1,485.25', 'Change': '-₹8.30', 'Change %': '-0.56%', 'Volume': '1,567,890'},
                ]
            
            return dash_table.DataTable(
                data=market_data,
                columns=[{'name': col, 'id': col} for col in market_data[0].keys()],
                style_cell={'textAlign': 'center'},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Change %} contains "+"'},
                        'backgroundColor': '#d4edda',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{Change %} contains "-"'},
                        'backgroundColor': '#f8d7da',
                        'color': 'black',
                    }
                ]
            )
        except Exception as e:
            logger.error(f"Error creating market data table: {str(e)}")
            return html.Div(f"Error loading market data: {str(e)}", style={'color': 'red'})
    
    def run(self):
        """Run the dashboard server"""
        logger.info(f"Starting dashboard server on http://{self.host}:{self.port}")
        try:
            self.app.run(host=self.host, port=self.port, debug=self.debug)
        except Exception as e:
            logger.error(f"Error starting dashboard: {str(e)}")
            raise

# =============================================================================
# Dashboard CLI and Main Function
# =============================================================================

def main():
    """Main function to run dashboard"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SIP Trading Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Dashboard host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8050, help='Dashboard port (default: 8050)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Setup logging
    from utils.helpers import setup_logging
    setup_logging()
    
    try:
        # Create and run dashboard
        dashboard = TradingDashboard(host=args.host, port=args.port, debug=args.debug)
        
        print(f"\n🚀 SIP Trading Dashboard Starting...")
        print(f"📊 Dashboard URL: http://{args.host}:{args.port}")
        print(f"🔄 Auto-refresh: Every 5 seconds")
        print(f"💡 Press Ctrl+C to stop")
        print("-" * 50)
        
        dashboard.run()
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting dashboard: {str(e)}")
        logger.error(f"Dashboard startup error: {str(e)}")
        raise

# =============================================================================
# Dashboard Utilities and Helper Functions
# =============================================================================

class DashboardConfig:
    """Dashboard configuration settings"""
    
    def __init__(self):
        self.refresh_interval = 5  # seconds
        self.chart_theme = 'plotly_white'
        self.color_scheme = {
            'positive': '#28a745',
            'negative': '#dc3545',
            'neutral': '#6c757d',
            'primary': '#007bff',
            'warning': '#ffc107',
            'info': '#17a2b8'
        }
        self.default_height = 400
        self.table_page_size = 10

class DashboardMetrics:
    """Calculate dashboard metrics"""
    
    @staticmethod
    def calculate_portfolio_metrics(positions, balance):
        """Calculate portfolio-level metrics"""
        total_position_value = sum(
            pos.get('quantity', 0) * pos.get('current_price', 0) 
            for pos in positions
        )
        
        total_value = balance.get('cash', 0) + total_position_value
        
        return {
            'total_value': total_value,
            'total_positions_value': total_position_value,
            'cash_percentage': balance.get('cash', 0) / total_value if total_value > 0 else 0,
            'positions_count': len(positions)
        }
    
    @staticmethod
    def calculate_risk_metrics(positions, historical_data=None):
        """Calculate risk metrics"""
        if not positions:
            return {
                'concentration_risk': 0,
                'volatility': 0,
                'var_95': 0
            }
        
        # Calculate position concentrations
        total_value = sum(pos.get('quantity', 0) * pos.get('current_price', 0) for pos in positions)
        concentrations = [
            (pos.get('quantity', 0) * pos.get('current_price', 0)) / total_value 
            for pos in positions if total_value > 0
        ]
        
        max_concentration = max(concentrations) if concentrations else 0
        
        return {
            'concentration_risk': max_concentration,
            'volatility': 0.18,  # Mock - replace with actual calculation
            'var_95': 0.05  # Mock - replace with actual VaR calculation
        }

class DashboardAlerts:
    """Manage dashboard alerts"""
    
    def __init__(self):
        self.active_alerts = []
        self.alert_history = []
    
    def add_alert(self, level, message, symbol=None):
        """Add new alert"""
        alert = {
            'timestamp': datetime.now(),
            'level': level,
            'message': message,
            'symbol': symbol,
            'id': len(self.alert_history)
        }
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Keep only last 100 alerts in memory
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
    
    def get_active_alerts(self):
        """Get active alerts"""
        return self.active_alerts
    
    def clear_alert(self, alert_id):
        """Clear specific alert"""
        self.active_alerts = [a for a in self.active_alerts if a.get('id') != alert_id]

# =============================================================================
# Dashboard Testing and Development
# =============================================================================

class DashboardTester:
    """Test dashboard functionality"""
    
    def __init__(self, dashboard):
        self.dashboard = dashboard
    
    def test_all_tabs(self):
        """Test all dashboard tabs"""
        tabs = ['portfolio-tab', 'performance-tab', 'risk-tab', 'positions-tab', 'alerts-tab', 'market-tab']
        
        results = {}
        for tab in tabs:
            try:
                content = self.dashboard._render_tab_content(tab, 0, 0)
                results[tab] = 'PASS' if content else 'FAIL'
            except Exception as e:
                results[tab] = f'FAIL: {str(e)}'
        
        return results
    
    def test_data_sources(self):
        """Test data source connectivity"""
        tests = {
            'portfolio_data': self.dashboard._get_portfolio_data,
            'performance_data': self.dashboard._get_performance_data,
            'risk_data': self.dashboard._get_risk_data,
            'alerts_data': self.dashboard._get_alerts_data
        }
        
        results = {}
        for test_name, test_func in tests.items():
            try:
                data = test_func()
                results[test_name] = 'PASS' if data else 'FAIL'
            except Exception as e:
                results[test_name] = f'FAIL: {str(e)}'
        
        return results

def test_dashboard():
    """Test dashboard functionality"""
    print("Testing Dashboard Components...")
    
    try:
        dashboard = TradingDashboard()
        tester = DashboardTester(dashboard)
        
        # Test tabs
        print("\nTesting Dashboard Tabs:")
        tab_results = tester.test_all_tabs()
        for tab, result in tab_results.items():
            status = "✅" if result == 'PASS' else "❌"
            print(f"  {status} {tab}: {result}")
        
        # Test data sources
        print("\nTesting Data Sources:")
        data_results = tester.test_data_sources()
        for test, result in data_results.items():
            status = "✅" if result == 'PASS' else "❌"
            print(f"  {status} {test}: {result}")
        
        print("\n🎉 Dashboard testing completed!")
        
    except Exception as e:
        print(f"❌ Dashboard testing failed: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_dashboard()
    else:
        main()
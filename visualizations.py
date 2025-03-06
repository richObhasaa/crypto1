import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import logging
from typing import List, Union

from data_fetcher import CryptoDataFetcher
from statistical_analysis import CryptoStatAnalyzer

class CryptoVisualizer:
    """
    Advanced visualization class for cryptocurrency data
    """
    def __init__(self):
        """
        Initialize visualizer with data sources
        """
        self.logger = logging.getLogger(__name__)
        self.data_fetcher = CryptoDataFetcher()
        self.stat_analyzer = CryptoStatAnalyzer()

    def pie_chart(
        self, 
        data: pd.DataFrame, 
        title: str = 'Market Cap Distribution'
    ) -> go.Figure:
        """
        Create a pie chart of market cap distribution
        
        Args:
            data: DataFrame with market cap data
            title: Chart title
        
        Returns:
            Plotly Figure object
        """
        fig = px.pie(
            data, 
            values='market_cap', 
            names='name', 
            title=title,
            hole=0.3,
            color_discrete_sequence=px.colors.sequential.Plasma_r
        )
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label'
        )
        
        return fig

    def line_chart(
        self, 
        token: str, 
        days: int = 30
    ) -> go.Figure:
        """
        Create line chart for token price history
        
        Args:
            token: Cryptocurrency token ID
            days: Number of historical days
        
        Returns:
            Plotly Figure object
        """
        historical_data = self.data_fetcher.get_token_historical_data(token, days)
        
        fig = px.line(
            historical_data, 
            title=f'{token.capitalize()} Price History',
            labels={'timestamp': 'Date', 'price': 'Price (USD)'}
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_dark'
        )
        
        return fig

    def comparative_line_chart(
        self, 
        tokens: List[str], 
        days: int = 30
    ) -> go.Figure:
        """
        Create comparative line chart for multiple tokens
        
        Args:
            tokens: List of cryptocurrency tokens
            days: Number of historical days
        
        Returns:
            Plotly Figure object
        """
        # Fetch historical data for each token
        token_data = {}
        for token in tokens:
            token_data[token] = self.data_fetcher.get_token_historical_data(token, days)
        
        # Combine data into a single DataFrame for plotting
        combined_data = pd.DataFrame({
            token: data['price'] for token, data in token_data.items()
        }).reset_index()
        
        # Melt the DataFrame for easier plotting
        melted_data = combined_data.melt(
            id_vars=['timestamp'], 
            var_name='Token', 
            value_name='Price'
        )
        
        # Create comparative line chart
        fig = px.line(
            melted_data, 
            x='timestamp', 
            y='Price', 
            color='Token',
            title='Comparative Cryptocurrency Price Analysis',
            labels={'timestamp': 'Date', 'Price': 'Price (USD)'}
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_white',
            legend_title_text='Cryptocurrencies'
        )
        
        return fig

    def correlation_heatmap(
        self, 
        tokens: List[str]
    ) -> go.Figure:
        """
        Create correlation heatmap for multiple tokens
        
        Args:
            tokens: List of cryptocurrency tokens
        
        Returns:
            Plotly Figure object
        """
        # Compute correlation matrix
        correlation_matrix = self.stat_analyzer.correlation_analysis(tokens)
        
        # Create heatmap
        fig = px.imshow(
            correlation_matrix, 
            text_auto=True,
            aspect='auto',
            title='Cryptocurrency Price Correlation Heatmap',
            color_continuous_scale='RdBu_r'
        )
        
        fig.update_layout(
            xaxis_title='Tokens',
            yaxis_title='Tokens',
            template='plotly_white'
        )
        
        return fig

    def volatility_boxplot(
        self, 
        tokens: List[str]
    ) -> go.Figure:
        """
        Create boxplot to compare token price volatilities
        
        Args:
            tokens: List of cryptocurrency tokens
        
        Returns:
            Plotly Figure object
        """
        # Prepare data for boxplot
        volatility_data = []
        
        for token in tokens:
            historical_data = self.data_fetcher.get_token_historical_data(token)
            returns = historical_data['price'].pct_change().dropna()
            
            volatility_data.append(
                go.Box(
                    y=returns, 
                    name=token.capitalize(),
                    boxmean=True
                )
            )
        
        # Create figure
        fig = go.Figure(data=volatility_data)
        
        fig.update_layout(
            title='Cryptocurrency Price Returns Volatility',
            yaxis_title='Daily Returns',
            template='plotly_white'
        )
        
        return fig

    def market_cap_treemap(
        self, 
        top_n: int = 20
    ) -> go.Figure:
        """
        Create treemap visualization of market capitalization
        
        Args:
            top_n: Number of top cryptocurrencies to visualize
        
        Returns:
            Plotly Figure object
        """
        # Fetch top tokens
        top_tokens = self.data_fetcher.get_top_tokens(top_n)
        
        # Create treemap
        fig = px.treemap(
            top_tokens, 
            path=['name'], 
            values='market_cap',
            title='Cryptocurrency Market Cap Distribution',
            color='market_cap',
            color_continuous_scale='Viridis'
        )
        
        fig.update_traces(
            texttemplate='%{label}<br>$%{value:,.0f}',
            textposition='middle'
        )
        
        return fig

    def anomaly_scatter(
        self, 
        token: str
    ) -> go.Figure:
        """
        Create scatter plot highlighting price anomalies
        
        Args:
            token: Cryptocurrency token
        
        Returns:
            Plotly Figure object
        """
        # Fetch historical data
        historical_data = self.data_fetcher.get_token_historical_data(token)
        
        # Detect anomalies
        anomalies = self.stat_analyzer.anomaly_detection(token)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add normal data points
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data['price'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add anomaly points
        fig.add_trace(
            go.Scatter(
                x=anomalies.index,
                y=anomalies.values,
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color='red', 
                    size=10, 
                    symbol='cross'
                )
            )
        )
        
        fig.update_layout(
            title=f'{token.capitalize()} Price Anomalies',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_white'
        )
        
        return fig

def main():
    """
    Test the CryptoVisualizer
    """
    logging.basicConfig(level=logging.INFO)
    
    visualizer = CryptoVisualizer()
    
    # Test various visualizations
    top_tokens = ['bitcoin', 'ethereum', 'binancecoin', 'cardano', 'ripple']
    
    # Market Cap Pie Chart
    top_tokens_data = visualizer.data_fetcher.get_top_tokens(5)
    pie_chart = visualizer.pie_chart(top_tokens_data)
    pie_chart.show()
    
    # Comparative Line Chart
    line_chart = visualizer.comparative_line_chart(top_tokens)
    line_chart.show()
    
    # Correlation Heatmap
    correlation_heatmap = visualizer.correlation_heatmap(top_tokens)
    correlation_heatmap.show()
    
    # Volatility Boxplot
    volatility_plot = visualizer.volatility_boxplot(top_tokens)
    volatility_plot.show()
    
    # Market Cap Treemap
    market_cap_treemap = visualizer.market_cap_treemap()
    market_cap_treemap.show()
    
    # Anomaly Scatter for Bitcoin
    bitcoin_anomalies = visualizer.anomaly_scatter('bitcoin')
    bitcoin_anomalies.show()

if __name__ == "__main__":
    main()
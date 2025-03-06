import streamlit as st
import pandas as pd

# Import custom modules
from data_fetcher import CryptoDataFetcher
from statistical_analysis import CryptoStatAnalyzer
from visualizations import CryptoVisualizer
from price_predictor import CryptoPricePredictor
from ai_analyzer import CryptoAIAnalyzer
from crypto_trends import CryptoTrendAnalyzer

class CryptoMarketCapApp:
    def __init__(self):
        """
        Initialize the Crypto Market Cap Analysis Streamlit Application
        """
        st.set_page_config(
            page_title="Crypto Market Cap Analysis",
            page_icon="📊",
            layout="wide"
        )
        
        # Initialize components
        self.data_fetcher = CryptoDataFetcher()
        self.stat_analyzer = CryptoStatAnalyzer()
        self.visualizer = CryptoVisualizer()
        self.price_predictor = CryptoPricePredictor()
        self.ai_analyzer = CryptoAIAnalyzer()
        self.trend_analyzer = CryptoTrendAnalyzer()

    def run(self):
        """
        Main application runner
        """
        st.title("🚀 Crypto Market Cap Comparison & Prediction")
        
        # Sidebar navigation
        menu = st.sidebar.radio(
            "Navigation", 
            [
                "Dashboard Overview", 
                "Market Statistics", 
                "Price Predictions", 
                "Project Analysis", 
                "Crypto Trends"
            ]
        )

        # Routing based on menu selection
        if menu == "Dashboard Overview":
            self.dashboard_overview()
        elif menu == "Market Statistics":
            self.market_statistics()
        elif menu == "Price Predictions":
            self.price_predictions()
        elif menu == "Project Analysis":
            self.project_analysis()
        elif menu == "Crypto Trends":
            self.crypto_trends()

    def dashboard_overview(self):
        """
        Main dashboard with key market insights
        """
        st.header("Market Overview")
        
        # Fetch latest market data
        market_data = self.data_fetcher.get_global_market_data()
        
        # Create columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Market Cap", f"${market_data['total_market_cap']:,.0f}")
        
        with col2:
            st.metric("24h Volume", f"${market_data['total_24h_volume']:,.0f}")
        
        with col3:
            st.metric("Cryptocurrencies", market_data['active_cryptocurrencies'])
        
        # Visualize market distribution
        st.subheader("Market Cap Distribution")
        top_tokens = self.data_fetcher.get_top_tokens(limit=10)
        self.visualizer.pie_chart(top_tokens, 'Market Cap Distribution')

    def market_statistics(self):
        """
        Detailed market statistics and comparisons
        """
        st.header("Cryptocurrency Market Statistics")
        
        # Token selection
        selected_tokens = st.multiselect(
            "Select Cryptocurrencies", 
            self.data_fetcher.get_all_token_names()
        )
        
        if selected_tokens:
            # Statistical analysis
            stats_df = self.stat_analyzer.compare_tokens(selected_tokens)
            st.dataframe(stats_df)
            
            # Comparative visualizations
            self.visualizer.comparative_line_chart(selected_tokens)

    def price_predictions(self):
        """
        Machine learning based price predictions
        """
        st.header("Cryptocurrency Price Predictions")
        
        token = st.selectbox(
            "Select Cryptocurrency", 
            self.data_fetcher.get_all_token_names()
        )
        
        prediction_horizon = st.slider(
            "Prediction Horizon (Days)", 
            min_value=7, 
            max_value=365, 
            value=30
        )
        
        if st.button("Predict Price"):
            predictions = self.price_predictor.predict_price(
                token, 
                horizon=prediction_horizon
            )
            st.line_chart(predictions)

    def project_analysis(self):
        """
        AI-powered cryptocurrency project analysis
        """
        st.header("Cryptocurrency Project Insights")
        
        token = st.selectbox(
            "Select Cryptocurrency Project", 
            self.data_fetcher.get_all_token_names()
        )
        
        if st.button("Analyze Project"):
            project_report = self.ai_analyzer.analyze_project(token)
            st.write(project_report)

    def crypto_trends(self):
        """
        Latest cryptocurrency trends
        """
        st.header("Crypto Market Trends")
        
        trend_type = st.radio(
            "Select Trend Source", 
            ["Social Media", "News", "Forums"]
        )
        
        trends = self.trend_analyzer.get_trends(trend_type)
        st.dataframe(trends)

def main():
    """
    Entry point for the Streamlit application
    """
    app = CryptoMarketCapApp()
    app.run()

if __name__ == "__main__":
    main()
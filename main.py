import streamlit as st
import pandas as pd
import sys
import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Relative imports with error handling
try:
    from data_fetcher import CryptoDataFetcher
    from data_processor import CryptoDataProcessor
    from statistical_analysis import CryptoStatAnalyzer
    from visualizations import CryptoVisualizer
    from price_predictor import CryptoPricePredictor
    from ai_analyzer import CryptoAIAnalyzer
    from crypto_trends import CryptoTrendAnalyzer
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please check your project structure and ensure all modules are present.")
    raise

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
        
        # Initialize components with error handling
        try:
            self.data_fetcher = CryptoDataFetcher()
            self.data_processor = CryptoDataProcessor()
            self.stat_analyzer = CryptoStatAnalyzer()
            self.visualizer = CryptoVisualizer()
            self.price_predictor = CryptoPricePredictor()
            self.ai_analyzer = CryptoAIAnalyzer()
            self.trend_analyzer = CryptoTrendAnalyzer()
        except Exception as e:
            st.error(f"Initialization Error: {e}")
            raise

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
        try:
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
        except Exception as e:
            st.error(f"Navigation Error: {e}")

    def dashboard_overview(self):
        """
        Main dashboard with key market insights
        """
        st.header("Market Overview")
        
        try:
            # Fetch latest market data
            market_data = self.data_fetcher.get_global_crypto_data()
            
            # Create columns for key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Market Cap", f"${market_data.get('total_market_cap', 0):,.0f}")
            
            with col2:
                st.metric("24h Volume", f"${market_data.get('total_volume', 0):,.0f}")
            
            with col3:
                st.metric("Active Cryptocurrencies", market_data.get('active_cryptocurrencies', 0))
            
            # Visualize market distribution
            st.subheader("Market Cap Distribution")
            top_tokens = self.data_fetcher.get_top_cryptocurrencies(limit=10)
            
            # Use processor to clean and prepare data
            cleaned_tokens = self.data_processor.clean_market_data(top_tokens)
            
            # Create pie chart
            pie_chart = self.visualizer.pie_chart(cleaned_tokens, 'Market Cap Distribution')
            st.plotly_chart(pie_chart)
        
        except Exception as e:
            st.error(f"Dashboard Overview Error: {e}")

    # Other methods remain the same...

def main():
    """
    Entry point for the Streamlit application
    """
    try:
        app = CryptoMarketCapApp()
        app.run()
    except Exception as e:
        st.error(f"Application Initialization Error: {e}")

if __name__ == "__main__":
    main()

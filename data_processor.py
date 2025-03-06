import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple

from data_fetcher import CryptoDataFetcher

class CryptoDataProcessor:
    """
    Advanced data processing module for cryptocurrency datasets
    """
    def __init__(self):
        """
        Initialize data processor with logging and data fetcher
        """
        self.logger = logging.getLogger(__name__)
        self.data_fetcher = CryptoDataFetcher()

    def clean_market_data(
        self, 
        data: pd.DataFrame, 
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Clean and preprocess market data
        
        Args:
            data: Input DataFrame
            columns: Columns to process (default all)
        
        Returns:
            Cleaned DataFrame
        """
        try:
            # Create a copy to avoid modifying original
            df = data.copy()
            
            # Default columns if not specified
            if columns is None:
                columns = df.columns
            
            # Handle missing values
            for col in columns:
                if df[col].dtype in ['float64', 'int64']:
                    # Replace infinite values
                    df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
                    
                    # Fill numeric columns
                    df[col].fillna(df[col].median(), inplace=True)
                elif df[col].dtype == 'object':
                    # Fill string columns
                    df[col].fillna('Unknown', inplace=True)
            
            # Remove duplicates
            df.drop_duplicates(inplace=True)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Market data cleaning error: {e}")
            return pd.DataFrame()

    def normalize_market_data(
        self, 
        data: pd.DataFrame, 
        method: str = 'minmax',
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Normalize market data using different methods
        
        Args:
            data: Input DataFrame
            method: Normalization method (minmax, zscore, robust)
            columns: Columns to normalize
        
        Returns:
            Normalized DataFrame
        """
        try:
            # Create a copy
            df = data.copy()
            
            # Default to numeric columns if not specified
            if columns is None:
                columns = df.select_dtypes(include=['float64', 'int64']).columns
            
            # Normalization methods
            if method == 'minmax':
                # Min-Max scaling (0-1 range)
                for col in columns:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df[col] = (df[col] - min_val) / (max_val - min_val)
            
            elif method == 'zscore':
                # Z-score normalization (mean=0, std=1)
                for col in columns:
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
            
            elif method == 'robust':
                # Robust scaling (using median and IQR)
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                df[columns] = scaler.fit_transform(df[columns])
            
            else:
                raise ValueError(f"Unsupported normalization method: {method}")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Data normalization error: {e}")
            return pd.DataFrame()

    def compute_technical_indicators(
        self, 
        price_data: pd.DataFrame, 
        indicators: List[str] = None
    ) -> pd.DataFrame:
        """
        Compute technical indicators for price data
        
        Args:
            price_data: Historical price DataFrame
            indicators: List of indicators to compute
        
        Returns:
            DataFrame with additional technical indicators
        """
        try:
            # Create a copy of price data
            df = price_data.copy()
            
            # Default indicators if not specified
            if indicators is None:
                indicators = [
                    'moving_average', 
                    'exponential_moving_average', 
                    'rsi', 
                    'macd'
                ]
            
            # Compute indicators
            if 'moving_average' in indicators:
                # Simple Moving Average (SMA)
                for window in [7, 14, 30]:
                    df[f'sma_{window}'] = df['price'].rolling(window=window).mean()
            
            if 'exponential_moving_average' in indicators:
                # Exponential Moving Average (EMA)
                for window in [7, 14, 30]:
                    df[f'ema_{window}'] = df['price'].ewm(span=window, adjust=False).mean()
            
            if 'rsi' in indicators:
                # Relative Strength Index (RSI)
                delta = df['price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
            
            if 'macd' in indicators:
                # Moving Average Convergence Divergence (MACD)
                exp1 = df['price'].ewm(span=12, adjust=False).mean()
                exp2 = df['price'].ewm(span=26, adjust=False).mean()
                df['macd'] = exp1 - exp2
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            return df
        
        except Exception as e:
            self.logger.error(f"Technical indicators computation error: {e}")
            return pd.DataFrame()

    def detect_market_anomalies(
        self, 
        data: pd.DataFrame, 
        column: str = 'price', 
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        Detect anomalies in market data
        
        Args:
            data: Input DataFrame
            column: Column to check for anomalies
            method: Anomaly detection method
        
        Returns:
            DataFrame with anomaly flags
        """
        try:
            # Create a copy
            df = data.copy()
            
            if method == 'zscore':
                # Z-score method: identify points more than 3 std from mean
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                df['is_anomaly'] = z_scores > 3
            
            elif method == 'iqr':
                # Interquartile Range method
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df['is_anomaly'] = (df[column] < lower_bound) | (df[column] > upper_bound)
            
            else:
                raise ValueError(f"Unsupported anomaly detection method: {method}")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Anomaly detection error: {e}")
            return pd.DataFrame()

    def merge_multiple_sources(
        self, 
        sources: List[Dict[str, pd.DataFrame]], 
        merge_key: str = 'token'
    ) -> pd.DataFrame:
        """
        Merge data from multiple cryptocurrency sources
        
        Args:
            sources: List of dictionaries with source name and DataFrame
            merge_key: Key to use for merging
        
        Returns:
            Merged DataFrame
        """
        try:
            # Start with the first source
            merged_df = sources[0][list(sources[0].keys())[0]]
            
            # Merge subsequent sources
            for source in sources[1:]:
                source_name = list(source.keys())[0]
                source_df = source[source_name]
                
                # Merge dataframes
                merged_df = pd.merge(
                    merged_df, 
                    source_df, 
                    on=merge_key, 
                    how='outer'
                )
            
            return merged_df
        
        except Exception as e:
            self.logger.error(f"Multi-source merge error: {e}")
            return pd.DataFrame()

    def categorize_market_cap(
        self, 
        market_cap: float
    ) -> str:
        """
        Categorize cryptocurrency by market capitalization
        
        Args:
            market_cap: Market capitalization value
        
        Returns:
            Market cap category
        """
        if market_cap >= 100_000_000_000:
            return 'Mega Cap'
        elif market_cap >= 10_000_000_000:
            return 'Large Cap'
        elif market_cap >= 1_000_000_000:
            return 'Mid Cap'
        elif market_cap >= 100_000_000:
            return 'Small Cap'
        else:
            return 'Micro Cap'

def main():
    """
    Test the CryptoDataProcessor
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize data fetcher and processor
    data_fetcher = CryptoDataFetcher()
    processor = CryptoDataProcessor()
    
    # Test with real cryptocurrency data
    print("Testing Data Processing:")
    
    # Fetch top cryptocurrencies
    top_coins = data_fetcher.get_top_cryptocurrencies(limit=20)
    
    # Test cleaning
    print("\n1. Data Cleaning:")
    cleaned_data = processor.clean_market_data(top_coins)
    print(cleaned_data.head())
    
    # Test normalization
    print("\n2. Data Normalization:")
    normalized_data = processor.normalize_market_data(
        cleaned_data, 
        method='minmax', 
        columns=['market_cap', 'total_volume']
    )
    print(normalized_data.head())
    
    # Test historical price processing
    print("\n3. Technical Indicators:")
    bitcoin_prices = data_fetcher.get_historical_price_data('bitcoin', days=30)
    indicators = processor.compute_technical_indicators(bitcoin_prices)
    print(indicators.tail())
    
    # Test anomaly detection
    print("\n4. Anomaly Detection:")
    anomalies = processor.detect_market_anomalies(bitcoin_prices)
    print(anomalies[anomalies['is_anomaly']].head())

if __name__ == "__main__":
    main()
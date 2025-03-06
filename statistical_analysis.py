import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import List, Dict
import logging

from data_fetcher import CryptoDataFetcher

class CryptoStatAnalyzer:
    """
    Provides comprehensive statistical analysis for cryptocurrencies
    """
    def __init__(self):
        """
        Initialize the statistical analyzer
        """
        self.logger = logging.getLogger(__name__)
        self.data_fetcher = CryptoDataFetcher()

    def compare_tokens(self, tokens: List[str]) -> pd.DataFrame:
        """
        Generate comparative statistical analysis for multiple tokens
        
        Args:
            tokens: List of cryptocurrency names/IDs
        
        Returns:
            DataFrame with comprehensive statistical metrics
        """
        analysis_results = []
        
        for token in tokens:
            try:
                # Fetch historical price data
                historical_data = self.data_fetcher.get_token_historical_data(token)
                
                # Compute statistical metrics
                token_stats = self._compute_token_statistics(historical_data['price'])
                token_stats['token'] = token
                analysis_results.append(token_stats)
            
            except Exception as e:
                self.logger.error(f"Error analyzing {token}: {e}")
        
        return pd.DataFrame(analysis_results).set_index('token')

    def _compute_token_statistics(self, price_series: pd.Series) -> Dict:
        """
        Compute detailed statistical metrics for a price series
        
        Args:
            price_series: Pandas Series of cryptocurrency prices
        
        Returns:
            Dictionary of statistical metrics
        """
        # Basic descriptive statistics
        descriptive_stats = {
            'mean_price': price_series.mean(),
            'median_price': price_series.median(),
            'std_dev': price_series.std(),
            'min_price': price_series.min(),
            'max_price': price_series.max(),
        }
        
        # Advanced statistical tests
        try:
            # Normality test
            _, normality_p_value = stats.normaltest(price_series)
            descriptive_stats['is_normally_distributed'] = normality_p_value > 0.05
            
            # Skewness and Kurtosis
            descriptive_stats['skewness'] = stats.skew(price_series)
            descriptive_stats['kurtosis'] = stats.kurtosis(price_series)
        except Exception as e:
            self.logger.warning(f"Statistical test error: {e}")
        
        # Volatility measures
        returns = price_series.pct_change().dropna()
        descriptive_stats.update({
            'volatility': returns.std(),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(price_series)
        })
        
        return descriptive_stats

    def _calculate_sharpe_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: float = 0.01
    ) -> float:
        """
        Calculate Sharpe Ratio for investment returns
        
        Args:
            returns: Series of percentage returns
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Sharpe Ratio
        """
        # Annualize the sharpe ratio
        annual_returns = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        try:
            sharpe_ratio = (annual_returns - risk_free_rate) / annual_volatility
            return sharpe_ratio
        except ZeroDivisionError:
            return 0.0

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """
        Calculate maximum drawdown for a price series
        
        Args:
            prices: Series of prices
        
        Returns:
            Maximum drawdown percentage
        """
        cumulative_max = prices.cummax()
        drawdown = (prices - cumulative_max) / cumulative_max
        return drawdown.min()

    def correlation_analysis(self, tokens: List[str]) -> pd.DataFrame:
        """
        Compute correlation matrix between multiple tokens
        
        Args:
            tokens: List of cryptocurrency names/IDs
        
        Returns:
            Correlation matrix DataFrame
        """
        # Fetch historical data for all tokens
        historical_data = {}
        for token in tokens:
            historical_data[token] = self.data_fetcher.get_token_historical_data(token)['price']
        
        # Combine into a single DataFrame
        combined_data = pd.DataFrame(historical_data)
        
        # Compute correlation matrix
        correlation_matrix = combined_data.corr()
        
        return correlation_matrix

    def anomaly_detection(
        self, 
        token: str, 
        method: str = 'z_score'
    ) -> pd.Series:
        """
        Detect price anomalies using different methods
        
        Args:
            token: Cryptocurrency token
            method: Anomaly detection method
        
        Returns:
            Series of anomalies
        """
        # Fetch historical price data
        historical_data = self.data_fetcher.get_token_historical_data(token)['price']
        
        if method == 'z_score':
            # Z-score method: identify points more than 3 standard deviations from mean
            z_scores = np.abs(stats.zscore(historical_data))
            anomalies = historical_data[z_scores > 3]
        elif method == 'iqr':
            # Interquartile Range method
            Q1 = historical_data.quantile(0.25)
            Q3 = historical_data.quantile(0.75)
            IQR = Q3 - Q1
            anomalies = historical_data[
                (historical_data < (Q1 - 1.5 * IQR)) | 
                (historical_data > (Q3 + 1.5 * IQR))
            ]
        else:
            raise ValueError("Unsupported anomaly detection method")
        
        return anomalies

def main():
    """
    Test the CryptoStatAnalyzer
    """
    logging.basicConfig(level=logging.INFO)
    
    analyzer = CryptoStatAnalyzer()
    
    # Compare top cryptocurrencies
    top_tokens = ['bitcoin', 'ethereum', 'binancecoin']
    comparison = analyzer.compare_tokens(top_tokens)
    print("Token Comparison:\n", comparison)
    
    # Correlation Analysis
    correlation = analyzer.correlation_analysis(top_tokens)
    print("\nCorrelation Matrix:\n", correlation)
    
    # Anomaly Detection
    bitcoin_anomalies = analyzer.anomaly_detection('bitcoin')
    print("\nBitcoin Price Anomalies:\n", bitcoin_anomalies)

if __name__ == "__main__":
    main()
import requests
import pandas as pd
from typing import Dict, List
from config import Config
import time
import logging

class CryptoDataFetcher:
    """
    Handles data retrieval from multiple cryptocurrency APIs
    """
    def __init__(self):
        """
        Initialize data sources and API configurations
        """
        self.logger = logging.getLogger(__name__)
        self.apis = {
            'coingecko': {
                'base_url': Config.COINGECKO_ENDPOINT,
                'api_key': Config.COINGECKO_API_KEY
            },
            'coinmarketcap': {
                'base_url': Config.COINMARKETCAP_ENDPOINT,
                'api_key': Config.COINMARKETCAP_API_KEY
            },
            'binance': {
                'base_url': Config.BINANCE_ENDPOINT,
                'api_key': Config.BINANCE_API_KEY
            }
        }
        self.session = requests.Session()

    def get_global_market_data(self) -> Dict:
        """
        Retrieve global cryptocurrency market data
        
        Returns:
            Dict with market-wide statistics
        """
        try:
            url = f"{self.apis['coingecko']['base_url']}/global"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()['data']
            
            return {
                'total_market_cap': data['total_market_cap']['usd'],
                'total_24h_volume': data['total_volume']['usd'],
                'active_cryptocurrencies': data['active_cryptocurrencies'],
                'market_cap_percentage': data['market_cap_percentage']
            }
        except requests.RequestException as e:
            self.logger.error(f"Error fetching global market data: {e}")
            return {}

    def get_top_tokens(self, limit: int = 10) -> pd.DataFrame:
        """
        Retrieve top cryptocurrencies by market cap
        
        Args:
            limit: Number of top tokens to retrieve
        
        Returns:
            DataFrame with top cryptocurrency details
        """
        try:
            url = f"{self.apis['coingecko']['base_url']}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': limit,
                'page': 1,
                'sparkline': False
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Process historical data
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            prices.set_index('timestamp', inplace=True)
            
            return prices
        except requests.RequestException as e:
            self.logger.error(f"Error fetching historical data for {token_id}: {e}")
            return pd.DataFrame()

    def get_all_token_names(self) -> List[str]:
        """
        Retrieve list of all available cryptocurrency names
        
        Returns:
            List of cryptocurrency names
        """
        try:
            url = f"{self.apis['coingecko']['base_url']}/coins/list"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return [coin['name'] for coin in data]
        except requests.RequestException as e:
            self.logger.error(f"Error fetching token names: {e}")
            return []

    def fetch_multi_source_data(
        self, 
        token_ids: List[str], 
        data_type: str = 'market_data'
    ) -> pd.DataFrame:
        """
        Fetch data from multiple sources for cross-validation
        
        Args:
            token_ids: List of cryptocurrency IDs
            data_type: Type of data to fetch
        
        Returns:
            DataFrame with multi-source data
        """
        results = {}
        
        # Fetch from CoinGecko
        try:
            url = f"{self.apis['coingecko']['base_url']}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'ids': ','.join(token_ids),
                'order': 'market_cap_desc',
                'per_page': len(token_ids),
                'sparkline': False
            }
            cg_response = self.session.get(url, params=params, timeout=10)
            cg_data = cg_response.json()
            results['coingecko'] = pd.DataFrame(cg_data)
        except Exception as e:
            self.logger.warning(f"CoinGecko data fetch failed: {e}")
        
        # Fetch from CoinMarketCap
        try:
            url = f"{self.apis['coinmarketcap']['base_url']}/cryptocurrency/quotes/latest"
            headers = {
                'X-CMC_PRO_API_KEY': self.apis['coinmarketcap']['api_key']
            }
            params = {
                'symbol': ','.join(token_ids),
                'convert': 'USD'
            }
            cmc_response = self.session.get(url, headers=headers, params=params, timeout=10)
            cmc_data = cmc_response.json()
            results['coinmarketcap'] = pd.DataFrame.from_dict(cmc_data['data'], orient='index')
        except Exception as e:
            self.logger.warning(f"CoinMarketCap data fetch failed: {e}")
        
        # Merge and reconcile data sources
        if len(results) > 1:
            return self._reconcile_data_sources(results)
        elif results:
            return list(results.values())[0]
        
        return pd.DataFrame()

    def _reconcile_data_sources(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Reconcile and validate data from multiple sources
        
        Args:
            data_sources: Dictionary of DataFrames from different sources
        
        Returns:
            Consolidated and validated DataFrame
        """
        # Implement sophisticated data reconciliation logic
        # This could include:
        # 1. Cross-validation of key metrics
        # 2. Handling discrepancies
        # 3. Weighted averaging of data points
        
        # Placeholder implementation
        reconciled_data = pd.DataFrame()
        for source, df in data_sources.items():
            if reconciled_data.empty:
                reconciled_data = df
            else:
                # Basic merging strategy
                reconciled_data = reconciled_data.combine_first(df)
        
        return reconciled_data

def main():
    """
    Test the CryptoDataFetcher
    """
    logging.basicConfig(level=logging.INFO)
    
    fetcher = CryptoDataFetcher()
    
    # Test global market data
    global_data = fetcher.get_global_market_data()
    print("Global Market Data:", global_data)
    
    # Test top tokens
    top_tokens = fetcher.get_top_tokens(5)
    print("\nTop 5 Tokens:\n", top_tokens)
    
    # Test historical data for Bitcoin
    historical_data = fetcher.get_token_historical_data('bitcoin')
    print("\nBitcoin Historical Prices:\n", historical_data.head())

if __name__ == "__main__":
    main() timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return pd.DataFrame(data)[['id', 'symbol', 'name', 'market_cap']]
        except requests.RequestException as e:
            self.logger.error(f"Error fetching top tokens: {e}")
            return pd.DataFrame()

    def get_token_historical_data(
        self, 
        token_id: str, 
        days: int = 30
    ) -> pd.DataFrame:
        """
        Retrieve historical price data for a specific token
        
        Args:
            token_id: CoinGecko token ID
            days: Number of historical days to retrieve
        
        Returns:
            DataFrame with historical price data
        """
        try:
            url = f"{self.apis['coingecko']['base_url']}/coins/{token_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days
            }
            response = self.session.get(url, params=params,
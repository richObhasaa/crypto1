import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration management for Crypto Market Cap Analysis App
    """
    # API Keys
    COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY', '')
    COINMARKETCAP_API_KEY = os.getenv('COINMARKETCAP_API_KEY', '')
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

    # Database Configuration
    DATABASE_TYPE = os.getenv('DATABASE_TYPE', 'postgresql')
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/cryptodb')

    # Machine Learning Configuration
    ML_MODEL_TYPE = os.getenv('ML_MODEL_TYPE', 'LSTM')
    PREDICTION_HORIZON = int(os.getenv('PREDICTION_HORIZON', 30))  # Days

    # API Endpoints
    COINGECKO_ENDPOINT = 'https://api.coingecko.com/api/v3'
    COINMARKETCAP_ENDPOINT = 'https://pro-api.coinmarketcap.com/v1'
    BINANCE_ENDPOINT = 'https://api.binance.com'

    # Social Media Scraping Configuration
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN', '')
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')

    @classmethod
    def validate_config(cls):
        """
        Validate critical configuration parameters
        """
        required_keys = [
            'COINGECKO_API_KEY', 
            'COINMARKETCAP_API_KEY', 
            'OPENAI_API_KEY'
        ]
        
        for key in required_keys:
            if not getattr(cls, key):
                raise ValueError(f"Missing critical configuration: {key}")

# Validate configuration on import
Config.validate_config()
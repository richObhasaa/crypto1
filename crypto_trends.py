import requests
import pandas as pd
import tweepy
import praw
import logging
from typing import List, Dict
from datetime import datetime, timedelta

from config import Config

class CryptoTrendAnalyzer:
    """
    Comprehensive cryptocurrency trends analysis from multiple sources
    """
    def __init__(self):
        """
        Initialize trend analysis components
        """
        self.logger = logging.getLogger(__name__)
        
        # Twitter API Setup
        try:
            self.twitter_client = tweepy.Client(
                bearer_token=Config.TWITTER_BEARER_TOKEN
            )
        except Exception as e:
            self.logger.error(f"Twitter API initialization failed: {e}")
            self.twitter_client = None
        
        # Reddit API Setup
        try:
            self.reddit_client = praw.Reddit(
                client_id=Config.REDDIT_CLIENT_ID,
                client_secret=Config.REDDIT_CLIENT_SECRET,
                user_agent='CryptoTrendAnalyzer/1.0'
            )
        except Exception as e:
            self.logger.error(f"Reddit API initialization failed: {e}")
            self.reddit_client = None

    def get_twitter_trends(
        self, 
        crypto_keywords: List[str] = ['bitcoin', 'ethereum', 'crypto']
    ) -> pd.DataFrame:
        """
        Fetch trending cryptocurrency topics from Twitter
        
        Args:
            crypto_keywords: List of cryptocurrency keywords to search
        
        Returns:
            DataFrame with trending tweets
        """
        if not self.twitter_client:
            self.logger.warning("Twitter client not initialized")
            return pd.DataFrame()
        
        trends_data = []
        
        try:
            # Set time range for recent tweets
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            for keyword in crypto_keywords:
                # Search recent tweets
                tweets = self.twitter_client.search_recent_tweets(
                    query=keyword,
                    max_results=100,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if tweets.data:
                    for tweet in tweets.data:
                        trends_data.append({
                            'keyword': keyword,
                            'text': tweet.text,
                            'created_at': tweet.created_at,
                            'retweet_count': tweet.public_metrics.get('retweet_count', 0),
                            'like_count': tweet.public_metrics.get('like_count', 0)
                        })
        
        except Exception as e:
            self.logger.error(f"Twitter trend fetch error: {e}")
        
        return pd.DataFrame(trends_data)

    def get_reddit_trends(
        self, 
        subreddits: List[str] = ['CryptoCurrency', 'Bitcoin', 'ethereum']
    ) -> pd.DataFrame:
        """
        Fetch trending cryptocurrency discussions from Reddit
        
        Args:
            subreddits: List of cryptocurrency-related subreddits
        
        Returns:
            DataFrame with trending Reddit posts
        """
        if not self.reddit_client:
            self.logger.warning("Reddit client not initialized")
            return pd.DataFrame()
        
        trends_data = []
        
        try:
            for subreddit_name in subreddits:
                subreddit = self.reddit_client.subreddit(subreddit_name)
                
                # Get hot posts
                for post in subreddit.hot(limit=50):
                    trends_data.append({
                        'subreddit': subreddit_name,
                        'title': post.title,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'url': post.url,
                        'created_utc': datetime.fromtimestamp(post.created_utc)
                    })
        
        except Exception as e:
            self.logger.error(f"Reddit trend fetch error: {e}")
        
        return pd.DataFrame(trends_data)

    def get_news_trends(
        self, 
        crypto_keywords: List[str] = ['bitcoin', 'ethereum', 'blockchain']
    ) -> pd.DataFrame:
        """
        Fetch cryptocurrency news from various sources
        
        Args:
            crypto_keywords: List of cryptocurrency keywords
        
        Returns:
            DataFrame with recent news articles
        """
        news_data = []
        
        try:
            # Use NewsAPI or similar service (requires API key)
            # This is a placeholder implementation
            base_url = "https://newsapi.org/v2/everything"
            
            for keyword in crypto_keywords:
                params = {
                    'apiKey': Config.NEWSAPI_KEY,
                    'q': keyword,
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'pageSize': 20
                }
                
                response = requests.get(base_url, params=params)
                
                if response.status_code == 200:
                    articles = response.json().get('articles', [])
                    
                    for article in articles:
                        news_data.append({
                            'keyword': keyword,
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                            'source': article.get('source', {}).get('name', '')
                        })
        
        except Exception as e:
            self.logger.error(f"News trend fetch error: {e}")
        
        return pd.DataFrame(news_data)

    def aggregate_trends(
        self, 
        trend_type: str = 'all'
    ) -> pd.DataFrame:
        """
        Aggregate trends from multiple sources
        
        Args:
            trend_type: Type of trends to aggregate
        
        Returns:
            Consolidated trends DataFrame
        """
        try:
            if trend_type == 'twitter':
                return self.get_twitter_trends()
            elif trend_type == 'reddit':
                return self.get_reddit_trends()
            elif trend_type == 'news':
                return self.get_news_trends()
            else:
                # Combine all trends
                twitter_trends = self.get_twitter_trends()
                reddit_trends = self.get_reddit_trends()
                news_trends = self.get_news_trends()
                
                # Consolidate trends
                combined_trends = pd.concat([
                    twitter_trends, 
                    reddit_trends, 
                    news_trends
                ], ignore_index=True)
                
                return combined_trends
        
        except Exception as e:
            self.logger.error(f"Trend aggregation error: {e}")
            return pd.DataFrame()

def main():
    """
    Test the CryptoTrendAnalyzer
    """
    logging.basicConfig(level=logging.INFO)
    
    trend_analyzer = CryptoTrendAnalyzer()
    
    # Test individual trend sources
    print("Twitter Trends:")
    twitter_trends = trend_analyzer.get_twitter_trends()
    print(twitter_trends)
    
    print("\nReddit Trends:")
    reddit_trends = trend_analyzer.get_reddit_trends()
    print(reddit_trends)
    
    print("\nNews Trends:")
    news_trends = trend_analyzer.get_news_trends()
    print(news_trends)
    
    print("\nAggregated Trends:")
    all_trends = trend_analyzer.aggregate_trends()
    print(all_trends)

if __name__ == "__main__":
    main()
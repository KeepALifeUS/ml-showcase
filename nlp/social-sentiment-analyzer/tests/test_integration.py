"""
Integration tests for Social Media Sentiment Analyzer

Testing full functionality system with enterprise patterns.
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from src.utils.config import Config
from src.analysis.realtime_analyzer import RealtimeSentimentAnalyzer
from src.connectors.twitter_connector import TwitterConnector
from src.api.rest_api import SocialSentimentAPI

@pytest.fixture
def test_config():
    """Test configuration."""
    config = Mock(spec=Config)
    config.database_url = "sqlite:///:memory:"
    config.redis_url = "redis://localhost:6379/1"
    config.ml = Mock()
    config.ml.device = "cpu"
    config.ml.batch_size = 16
    config.ml.ensemble_models = ["cardiffnlp/twitter-roberta-base-sentiment-latest"]
    config.realtime_batch_size = 16
    config.realtime_max_queue_size = 100
    config.realtime_timeout = 1.0
    config.twitter = Mock()
    config.twitter.bearer_token = "test_token"
    config.twitter.is_configured = True
    return config

@pytest.fixture
def sample_tweets():
    """Samples tweets for testing."""
    return [
        {
            "id": "123456789",
            "text": "Bitcoin is going to the moon! 🚀 HODL strong!",
            "author_username": "crypto_bull",
            "platform": "twitter",
            "crypto_symbols": ["$BTC"]
        },
        {
            "id": "123456790",
            "text": "Market is crashing, time to panic sell everything!",
            "author_username": "panic_trader",
            "platform": "twitter", 
            "crypto_symbols": ["$BTC", "$ETH"]
        },
        {
            "id": "123456791",
            "text": "Ethereum gas fees are too high, need Layer 2 solutions",
            "author_username": "eth_dev",
            "platform": "twitter",
            "crypto_symbols": ["$ETH"]
        }
    ]

class TestRealtimeAnalyzer:
    """Tests for analyzer real time."""
    
    @pytest.mark.asyncio
    async def test_analyzer_initialization(self, test_config):
        """Test initialization analyzer."""
        
        with patch('src.analysis.realtime_analyzer.EnsembleModel') as mock_ensemble:
            mock_ensemble.return_value.initialize = AsyncMock()
            
            with patch('aioredis.from_url') as mock_redis:
                mock_redis.return_value = Mock()
                
                analyzer = RealtimeSentimentAnalyzer(test_config)
                await analyzer.initialize()
                
                assert analyzer.config == test_config
                assert analyzer.batch_size == 16
                mock_ensemble.return_value.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_single_sentiment_analysis(self, test_config):
        """Test analysis one messages."""
        
        with patch('src.analysis.realtime_analyzer.EnsembleModel') as mock_ensemble:
            mock_model = Mock()
            mock_model.predict = AsyncMock(return_value={
                "positive": 0.8,
                "negative": 0.1, 
                "neutral": 0.1
            })
            mock_model.version = "test-v1.0"
            mock_ensemble.return_value = mock_model
            
            with patch('aioredis.from_url'):
                analyzer = RealtimeSentimentAnalyzer(test_config)
                analyzer.ensemble_model = mock_model
                analyzer.preprocessor = Mock()
                analyzer.preprocessor.preprocess.return_value = "bitcoin moon hodl"
                analyzer.preprocessor.extract_crypto_symbols.return_value = ["$BTC"]
                
                result = await analyzer.analyze_sentiment(
                    text="Bitcoin is going to the moon! 🚀 HODL strong!",
                    platform="twitter"
                )
                
                assert result.sentiment == "positive"
                assert result.confidence > 0.7
                assert "$BTC" in result.crypto_symbols
                assert result.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_batch_sentiment_analysis(self, test_config, sample_tweets):
        """Test batch analysis."""
        
        with patch('src.analysis.realtime_analyzer.EnsembleModel') as mock_ensemble:
            mock_model = Mock()
            mock_model.predict_batch = AsyncMock(return_value=[
                {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
                {"positive": 0.1, "negative": 0.8, "neutral": 0.1},
                {"positive": 0.2, "negative": 0.3, "neutral": 0.5}
            ])
            mock_model.version = "test-v1.0"
            mock_ensemble.return_value = mock_model
            
            with patch('aioredis.from_url'):
                analyzer = RealtimeSentimentAnalyzer(test_config)
                analyzer.ensemble_model = mock_model
                analyzer.preprocessor = Mock()
                analyzer.preprocessor.preprocess.side_effect = lambda x: x.lower()
                analyzer.preprocessor.extract_crypto_symbols.return_value = ["$BTC", "$ETH"]
                
                texts = [tweet["text"] for tweet in sample_tweets]
                results = await analyzer.analyze_batch(texts, "twitter")
                
                assert len(results) == 3
                assert results[0].sentiment == "positive"  # Moon tweet
                assert results[1].sentiment == "negative"  # Crash tweet
                assert results[2].sentiment == "neutral"   # Neutral tweet

class TestTwitterConnector:
    """Tests for Twitter connector."""
    
    @pytest.mark.asyncio
    async def test_twitter_connection(self, test_config):
        """Test connections to Twitter."""
        
        with patch('tweepy.Client') as mock_client:
            mock_tweepy = Mock()
            mock_tweepy.get_me.return_value = Mock(id="123", username="test_bot")
            mock_client.return_value = mock_tweepy
            
            connector = TwitterConnector(test_config)
            connector._client = mock_tweepy
            
            success = await connector.connect()
            
            assert success is True
            assert connector._connected is True
    
    @pytest.mark.asyncio
    async def test_search_crypto_tweets(self, test_config):
        """Test search crypto tweets."""
        
        mock_tweet = Mock()
        mock_tweet.id = "123456789"
        mock_tweet.text = "Bitcoin is pumping! 🚀"
        mock_tweet.created_at = Mock()
        mock_tweet.created_at.isoformat.return_value = "2024-01-01T00:00:00Z"
        mock_tweet.author_id = "user123"
        mock_tweet.public_metrics = {"like_count": 100, "retweet_count": 50}
        mock_tweet.lang = "en"
        mock_tweet.source = "Twitter for iPhone"
        
        mock_response = Mock()
        mock_response.data = [mock_tweet]
        mock_response.includes = {"users": [Mock(id="user123", username="crypto_trader")]}
        
        with patch('tweepy.Client') as mock_client:
            mock_tweepy = Mock()
            mock_tweepy.search_recent_tweets.return_value = mock_response
            mock_client.return_value = mock_tweepy
            
            connector = TwitterConnector(test_config)
            connector._client = mock_tweepy
            connector._connected = True
            
            tweets = await connector.search_recent_tweets(
                query="bitcoin OR ethereum",
                max_results=10,
                crypto_focus=True
            )
            
            assert len(tweets) == 1
            assert tweets[0]["id"] == "123456789"
            assert tweets[0]["text"] == "Bitcoin is pumping! 🚀"
            assert tweets[0]["platform"] == "twitter"

class TestAPIIntegration:
    """Tests for REST API."""
    
    @pytest.fixture
    def api_client(self):
        """Test API client."""
        from fastapi.testclient import TestClient
        
        api = SocialSentimentAPI()
        return TestClient(api.app)
    
    def test_health_endpoint(self, api_client):
        """Test health check endpoint."""
        
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "components" in data
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis_endpoint(self, api_client):
        """Test analysis sentiment through API."""
        
        with patch('src.api.rest_api.RealtimeSentimentAnalyzer') as mock_analyzer:
            mock_result = Mock()
            mock_result.sentiment = "positive"
            mock_result.confidence = 0.85
            mock_result.scores = {"positive": 0.85, "negative": 0.10, "neutral": 0.05}
            mock_result.crypto_symbols = ["$BTC"]
            mock_result.processing_time_ms = 25.5
            mock_result.model_version = "ensemble-v1.0"
            mock_result.timestamp = "2024-01-01T00:00:00Z"
            
            mock_analyzer.return_value.analyze_sentiment = AsyncMock(return_value=mock_result)
            
            request_data = {
                "text": "Bitcoin is going to the moon! 🚀",
                "platform": "twitter",
                "priority": 1
            }
            
            response = api_client.post("/sentiment/analyze", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["sentiment"] == "positive"
            assert data["confidence"] == 0.85
            assert "$BTC" in data["crypto_symbols"]

class TestEndToEndWorkflow:
    """End-to-end tests full workflow."""
    
    @pytest.mark.asyncio
    async def test_full_sentiment_pipeline(self, test_config, sample_tweets):
        """Test full pipeline analysis sentiments."""
        
        # Mocks for all components
        with patch('src.connectors.twitter_connector.TwitterConnector') as mock_connector:
            with patch('src.analysis.realtime_analyzer.RealtimeSentimentAnalyzer') as mock_analyzer:
                
                # Configuration mock connector
                mock_connector_instance = Mock()
                mock_connector_instance.search_recent_tweets = AsyncMock(return_value=sample_tweets)
                mock_connector.return_value = mock_connector_instance
                
                # Configuration mock analyzer
                mock_analyzer_instance = Mock()
                mock_results = [
                    Mock(sentiment="positive", confidence=0.85, crypto_symbols=["$BTC"]),
                    Mock(sentiment="negative", confidence=0.75, crypto_symbols=["$BTC", "$ETH"]),
                    Mock(sentiment="neutral", confidence=0.65, crypto_symbols=["$ETH"])
                ]
                mock_analyzer_instance.analyze_batch = AsyncMock(return_value=mock_results)
                mock_analyzer.return_value = mock_analyzer_instance
                
                # Execution workflow
                connector = mock_connector(test_config)
                analyzer = mock_analyzer(test_config)
                
                # 1. Collection tweets
                tweets = await connector.search_recent_tweets()
                assert len(tweets) == 3
                
                # 2. Analysis sentiments
                texts = [tweet["text"] for tweet in tweets]
                results = await analyzer.analyze_batch(texts)
                
                # 3. Validation results
                assert len(results) == 3
                assert results[0].sentiment == "positive"
                assert results[1].sentiment == "negative"
                assert results[2].sentiment == "neutral"
                
                # 4. Validation distribution crypto symbols
                all_symbols = []
                for result in results:
                    all_symbols.extend(result.crypto_symbols)
                
                assert "$BTC" in all_symbols
                assert "$ETH" in all_symbols

    @pytest.mark.asyncio
    async def test_error_handling_and_fallbacks(self, test_config):
        """Test processing errors and fallback mechanisms."""
        
        with patch('src.analysis.realtime_analyzer.EnsembleModel') as mock_ensemble:
            # Imitation errors model
            mock_ensemble.return_value.predict = AsyncMock(side_effect=Exception("Model error"))
            
            with patch('aioredis.from_url'):
                analyzer = RealtimeSentimentAnalyzer(test_config)
                analyzer.ensemble_model = mock_ensemble.return_value
                analyzer.preprocessor = Mock()
                analyzer.preprocessor.preprocess.return_value = "test text"
                analyzer.preprocessor.extract_crypto_symbols.return_value = []
                
                # Must return fallback result
                result = await analyzer.analyze_sentiment("Test text", "api")
                
                assert result.sentiment == "neutral"
                assert result.confidence == 0.5
                assert result.model_version == "fallback"

class TestPerformanceMetrics:
    """Tests performance and metrics."""
    
    @pytest.mark.asyncio
    async def test_throughput_measurement(self, test_config):
        """Test measurements throughput analyzer."""
        
        with patch('src.analysis.realtime_analyzer.EnsembleModel') as mock_ensemble:
            mock_model = Mock()
            mock_model.predict_batch = AsyncMock(return_value=[
                {"positive": 0.7, "negative": 0.2, "neutral": 0.1}
                for _ in range(100)
            ])
            mock_model.version = "test-v1.0"
            mock_ensemble.return_value = mock_model
            
            with patch('aioredis.from_url'):
                analyzer = RealtimeSentimentAnalyzer(test_config)
                analyzer.ensemble_model = mock_model
                analyzer.preprocessor = Mock()
                analyzer.preprocessor.preprocess.side_effect = lambda x: x
                analyzer.preprocessor.extract_crypto_symbols.return_value = []
                
                # Test batch processing
                import time
                start_time = time.time()
                
                texts = [f"Test message {i}" for i in range(100)]
                results = await analyzer.analyze_batch(texts)
                
                end_time = time.time()
                processing_time = end_time - start_time
                throughput = len(texts) / processing_time
                
                assert len(results) == 100
                assert throughput > 50  # Minimum 50 messages/second
                print(f"Throughput: {throughput:.1f} messages/second")
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, test_config):
        """Test monitoring usage memory."""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('src.analysis.realtime_analyzer.EnsembleModel') as mock_ensemble:
            mock_model = Mock()
            mock_model.predict_batch = AsyncMock(return_value=[
                {"positive": 0.5, "negative": 0.3, "neutral": 0.2}
                for _ in range(1000)
            ])
            mock_ensemble.return_value = mock_model
            
            with patch('aioredis.from_url'):
                analyzer = RealtimeSentimentAnalyzer(test_config)
                analyzer.ensemble_model = mock_model
                analyzer.preprocessor = Mock()
                analyzer.preprocessor.preprocess.side_effect = lambda x: x
                analyzer.preprocessor.extract_crypto_symbols.return_value = []
                
                # Processing large batch
                texts = [f"Large batch test message {i}" for i in range(1000)]
                await analyzer.analyze_batch(texts)
                
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory
                
                # Not must be significant increase memory
                assert memory_increase < 100  # Less 100MB increase
                print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
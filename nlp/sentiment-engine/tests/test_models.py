"""
Tests for sentiment analysis models
"""

import pytest
import asyncio
from src.models.ensemble_sentiment import EnsembleSentimentModel
from src.models.vader_sentiment import VADERSentimentAnalyzer
from src.utils.validators import SentimentScore


class TestVADERSentiment:
 """Tests for VADER sentiment analyzer"""

 @pytest.fixture
 def vader_analyzer(self):
 """Fixture for VADER analyzer"""
 return VADERSentimentAnalyzer

 @pytest.mark.asyncio
 async def test_positive_sentiment(self, vader_analyzer):
 """Test positive sentiment detection"""
 text = "Bitcoin is going to the moon! Great investment!"
 result = await vader_analyzer.predict(text)

 assert isinstance(result, SentimentScore)
 assert result.value > 0
 assert 0 <= result.confidence <= 1
 assert result.model_name == "vader"

 @pytest.mark.asyncio
 async def test_negative_sentiment(self, vader_analyzer):
 """Test negative sentiment detection"""
 text = "Crypto market crash! Selling everything!"
 result = await vader_analyzer.predict(text)

 assert isinstance(result, SentimentScore)
 assert result.value < 0
 assert result.model_name == "vader"

 @pytest.mark.asyncio
 async def test_neutral_sentiment(self, vader_analyzer):
 """Test neutral sentiment detection"""
 text = "Bitcoin price is $50000 today."
 result = await vader_analyzer.predict(text)

 assert isinstance(result, SentimentScore)
 assert -0.1 <= result.value <= 0.1

 @pytest.mark.asyncio
 async def test_crypto_lexicon(self, vader_analyzer):
 """Test crypto-specific lexicon"""
 text = "HODL diamond hands!"
 result = await vader_analyzer.predict(text)

 # Should be positive due to crypto lexicon
 assert result.value > 0

 @pytest.mark.asyncio
 async def test_batch_prediction(self, vader_analyzer):
 """Test batch prediction"""
 texts = [
 "Bitcoin to the moon!",
 "Market crash incoming",
 "Neutral market update"
 ]

 results = await vader_analyzer.predict_batch(texts)

 assert len(results) == 3
 assert all(isinstance(r, SentimentScore) for r in results)
 assert results[0].value > 0 # Positive
 assert results[1].value < 0 # Negative


@pytest.mark.slow
class TestEnsembleModel:
 """Tests for ensemble sentiment model"""

 @pytest.fixture
 async def ensemble_model(self):
 """Fixture for ensemble model"""
 model = EnsembleSentimentModel
 await model.initialize
 return model

 @pytest.mark.asyncio
 async def test_ensemble_prediction(self, ensemble_model):
 """Test ensemble prediction"""
 text = "Ethereum upgrade looks very promising for DeFi!"
 result = await ensemble_model.predict(text, source="news")

 assert isinstance(result, SentimentScore)
 assert -1 <= result.value <= 1
 assert 0 <= result.confidence <= 1
 assert "ensemble" in result.model_name

 @pytest.mark.asyncio
 async def test_source_adaptive_weighting(self, ensemble_model):
 """Test source-specific weighting"""
 text = "BTC pumping hard! 🚀🚀🚀"

 # Twitter source should use different weights
 twitter_result = await ensemble_model.predict(text, source="twitter")
 news_result = await ensemble_model.predict(text, source="news")

 # Both should be positive, but potentially different scores
 assert twitter_result.value > 0
 assert news_result.value > 0

 @pytest.mark.asyncio
 async def test_batch_ensemble_prediction(self, ensemble_model):
 """Test batch ensemble prediction"""
 texts = [
 "Bitcoin reaching new ATH!",
 "Major crypto exchange hacked",
 "Stablecoin regulation update"
 ]

 results = await ensemble_model.predict_batch(texts, source="news")

 assert len(results) == 3
 assert all(isinstance(r, SentimentScore) for r in results)
 assert results[0].value > 0 # Positive news
 assert results[1].value < 0 # Negative news

 @pytest.mark.asyncio
 async def test_model_stats(self, ensemble_model):
 """Test model statistics"""
 # Make some predictions first
 await ensemble_model.predict("Test sentiment", source="test")

 stats = ensemble_model.get_stats

 assert "model_type" in stats
 assert "predictions_made" in stats
 assert "avg_inference_time_ms" in stats
 assert stats["predictions_made"] > 0


class TestSentimentValidation:
 """Tests for sentiment validation"""

 def test_valid_sentiment_score(self):
 """Test valid sentiment score creation"""
 score = SentimentScore(
 value=0.75,
 confidence=0.85,
 model_name="test_model"
 )

 assert score.value == 0.75
 assert score.confidence == 0.85
 assert score.model_name == "test_model"

 def test_invalid_sentiment_range(self):
 """Test invalid sentiment value range"""
 with pytest.raises(ValueError):
 SentimentScore(
 value=2.0, # Out of range
 confidence=0.5,
 model_name="test"
 )

 def test_invalid_confidence_range(self):
 """Test invalid confidence range"""
 with pytest.raises(ValueError):
 SentimentScore(
 value=0.5,
 confidence=1.5, # Out of range
 model_name="test"
 )
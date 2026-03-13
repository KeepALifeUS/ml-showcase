"""
Data validation utilities for ML-Framework ML Sentiment Engine

Enterprise-grade validation with and typed validations.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator
from pydantic.validators import float_validator, int_validator


class SentimentScore(BaseModel):
 """Validated sentiment score"""

 value: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score from -1 to 1")
 confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0 to 1")
 model_name: str = Field(..., min_length=1, max_length=100, description="Model name")

 @validator("value")
 def validate_sentiment_value(cls, v):
 """Validate sentiment value"""
 if not isinstance(v, (int, float)):
 raise ValueError("Sentiment value must be numeric")
 if not -1.0 <= v <= 1.0:
 raise ValueError("Sentiment value must be between -1.0 and 1.0")
 return float(v)

 @validator("confidence")
 def validate_confidence(cls, v):
 """Validate confidence value"""
 if not isinstance(v, (int, float)):
 raise ValueError("Confidence must be numeric")
 if not 0.0 <= v <= 1.0:
 raise ValueError("Confidence must be between 0.0 and 1.0")
 return float(v)


class CryptoSymbol(BaseModel):
 """Validated cryptocurrency symbol"""

 symbol: str = Field(..., min_length=2, max_length=10, description="Cryptocurrency symbol")
 base: Optional[str] = Field(None, min_length=2, max_length=10, description="Base currency")
 quote: Optional[str] = Field(None, min_length=2, max_length=10, description="Quote currency")

 @validator("symbol", pre=True)
 def validate_symbol(cls, v):
 """Validate cryptocurrency symbol"""
 if not isinstance(v, str):
 raise ValueError("Symbol must be a string")

 # Convert to uppercase
 symbol = v.upper.strip

 # Format check
 if not re.match(r"^[A-Z0-9]{2,10}$", symbol):
 raise ValueError("Invalid crypto symbol format")

 return symbol

 @validator("base", "quote", pre=True)
 def validate_currency(cls, v):
 """Validate currency codes"""
 if v is None:
 return v

 if not isinstance(v, str):
 raise ValueError("Currency code must be a string")

 currency = v.upper.strip
 if not re.match(r"^[A-Z0-9]{2,10}$", currency):
 raise ValueError("Invalid currency code format")

 return currency


class TextContent(BaseModel):
 """Validated text content"""

 text: str = Field(..., min_length=1, max_length=10000, description="Text content")
 language: Optional[str] = Field(None, max_length=5, description="Language code")
 source: str = Field(..., min_length=1, max_length=50, description="Content source")
 created_at: Optional[datetime] = Field(None, description="Creation time")

 @validator("text", pre=True)
 def validate_text(cls, v):
 """Text validation"""
 if not isinstance(v, str):
 raise ValueError("Text must be a string")

 text = v.strip
 if not text:
 raise ValueError("Text cannot be empty")

 if len(text) > 10000:
 raise ValueError("Text too long (max 10000 characters)")

 return text

 @validator("language", pre=True)
 def validate_language(cls, v):
 """Language code validation"""
 if v is None:
 return v

 if not isinstance(v, str):
 raise ValueError("Language code must be a string")

 lang = v.lower.strip
 if not re.match(r"^[a-z]{2}(-[a-z]{2})?$", lang):
 raise ValueError("Invalid language code format (expected: 'en' or 'en-us')")

 return lang

 @validator("source", pre=True)
 def validate_source(cls, v):
 """Source validation"""
 if not isinstance(v, str):
 raise ValueError("Source must be a string")

 source = v.strip.lower
 allowed_sources = [
 "twitter", "reddit", "telegram", "discord", "news",
 "web", "blog", "forum", "chat", "social"
 ]

 if source not in allowed_sources:
 raise ValueError(f"Invalid source. Allowed: {', '.join(allowed_sources)}")

 return source


class TimeRange(BaseModel):
 """Validated time range"""

 start_time: datetime = Field(..., description="Start time")
 end_time: datetime = Field(..., description="End time")

 @validator("end_time")
 def validate_time_range(cls, v, values):
 """Time range validation"""
 if "start_time" in values and v <= values["start_time"]:
 raise ValueError("End time must be after start time")

 # Check for reasonable range (no more than 1 year)
 if "start_time" in values:
 diff = v - values["start_time"]
 if diff.days > 365:
 raise ValueError("Time range cannot exceed 1 year")

 return v


class APIRequestData(BaseModel):
 """Validated API request data"""

 method: str = Field(..., description="HTTP method")
 endpoint: str = Field(..., description="API endpoint")
 parameters: Dict[str, Any] = Field(default_factory=dict, description="Request parameters")
 headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers")

 @validator("method", pre=True)
 def validate_method(cls, v):
 """HTTP method validation"""
 if not isinstance(v, str):
 raise ValueError("Method must be a string")

 method = v.upper.strip
 allowed_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

 if method not in allowed_methods:
 raise ValueError(f"Invalid HTTP method. Allowed: {', '.join(allowed_methods)}")

 return method

 @validator("endpoint", pre=True)
 def validate_endpoint(cls, v):
 """Endpoint validation"""
 if not isinstance(v, str):
 raise ValueError("Endpoint must be a string")

 endpoint = v.strip
 if not endpoint.startswith("/"):
 endpoint = "/" + endpoint

 # Basic URL path validation
 if not re.match(r"^/[a-zA-Z0-9/_\-\.]*$", endpoint):
 raise ValueError("Invalid endpoint format")

 return endpoint


class ModelConfiguration(BaseModel):
 """Validated ML model configuration"""

 model_name: str = Field(..., min_length=1, max_length=100, description="Model name")
 model_type: str = Field(..., description="Model type")
 parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
 threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Decision threshold")

 @validator("model_type", pre=True)
 def validate_model_type(cls, v):
 """Model type validation"""
 if not isinstance(v, str):
 raise ValueError("Model type must be a string")

 model_type = v.lower.strip
 allowed_types = [
 "bert", "finbert", "vader", "textblob", "ensemble",
 "lstm", "cnn", "transformer", "custom"
 ]

 if model_type not in allowed_types:
 raise ValueError(f"Invalid model type. Allowed: {', '.join(allowed_types)}")

 return model_type


def validate_crypto_symbol(symbol: str) -> bool:
 """
 Validate cryptocurrency symbol

 Args:
 symbol: Symbol to validate

 Returns:
 bool: True if symbol is valid
 """
 try:
 CryptoSymbol(symbol=symbol)
 return True
 except Exception:
 return False


def validate_sentiment_score(score: float, confidence: float = 1.0) -> bool:
 """
 Sentiment score validation

 Args:
 score: Sentiment score
 confidence: Confidence score

 Returns:
 bool: True if score is valid
 """
 try:
 SentimentScore(value=score, confidence=confidence, model_name="validator")
 return True
 except Exception:
 return False


def validate_text_content(text: str, source: str) -> bool:
 """
 Text content validation

 Args:
 text: Text to validate
 source: Text source

 Returns:
 bool: True if content is valid
 """
 try:
 TextContent(text=text, source=source)
 return True
 except Exception:
 return False


def sanitize_text(text: str) -> str:
 """
 Clean text from potentially dangerous content

 Args:
 text: Original text

 Returns:
 str: Cleaned text
 """
 if not isinstance(text, str):
 return ""

 # Remove HTML tags
 text = re.sub(r"<[^>]+>", " ", text)

 # Remove URLs
 text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", text)

 # Remove email addresses
 text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", " ", text)

 # Remove special characters (except basic ones)
 text = re.sub(r"[^\w\s\.\,\!\?\:\;\-\(\)\'\"]+", " ", text)

 # Normalize whitespace
 text = re.sub(r"\s+", " ", text.strip)

 return text


def validate_aggregation_weights(weights: Dict[str, float]) -> bool:
 """
 Validate weights for aggregation

 Args:
 weights: Weights dictionary

 Returns:
 bool: True if weights are valid
 """
 if not isinstance(weights, dict):
 return False

 if not weights:
 return False

 # Check that all weights are numeric and positive
 for weight in weights.values:
 if not isinstance(weight, (int, float)):
 return False
 if weight < 0:
 return False

 # Check that sum of weights is approximately 1.0
 total_weight = sum(weights.values)
 if not 0.95 <= total_weight <= 1.05:
 return False

 return True


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
 """
 Normalize weights to sum 1.0

 Args:
 weights: Original weights

 Returns:
 Dict[str, float]: Normalized weights
 """
 if not weights:
 return {}

 total = sum(weights.values)
 if total == 0:
 # Equal weights if sum is 0
 equal_weight = 1.0 / len(weights)
 return {key: equal_weight for key in weights.keys}

 return {key: value / total for key, value in weights.items}


class ValidationError(Exception):
 """Custom exception for validation errors"""

 def __init__(self, message: str, field: str = None, value: Any = None):
 super.__init__(message)
 self.field = field
 self.value = value
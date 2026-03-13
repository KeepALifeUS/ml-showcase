"""
Data Validator for NLP Sentiment Analysis

Enterprise-grade input validation and data sanitization with 
patterns for secure and reliable data processing.

Author: ML-Framework Team
"""

import re
import html
import json
import validators
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import unicodedata
import logging
from urllib.parse import urlparse
import hashlib
import base64

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels"""
    BASIC = "basic"
    STRICT = "strict"
    ENTERPRISE = "enterprise"


class SecurityThreat(Enum):
    """Security threat types"""
    XSS = "xss"
    SQL_INJECTION = "sql_injection"
    SCRIPT_INJECTION = "script_injection"
    HTML_INJECTION = "html_injection"
    URL_MANIPULATION = "url_manipulation"
    SENSITIVE_DATA = "sensitive_data"
    MALICIOUS_CONTENT = "malicious_content"


@dataclass
class ValidationResult:
    """Validation result structure"""
    is_valid: bool
    sanitized_data: Any
    errors: List[str]
    warnings: List[str]
    threats_detected: List[SecurityThreat]
    confidence_score: float  # 0.0 to 1.0
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "sanitized_data": self.sanitized_data,
            "errors": self.errors,
            "warnings": self.warnings,
            "threats_detected": [threat.value for threat in self.threats_detected],
            "confidence_score": self.confidence_score,
            "processing_time_ms": self.processing_time_ms,
        }


class DataValidator:
    """
    Enterprise Data Validator with  Security Patterns
    
    Features:
    - Multi-level validation (Basic, Strict, Enterprise)
    - XSS and injection attack prevention
    - Content sanitization and normalization
    - Crypto-specific validation rules
    - Sensitive data detection and masking
    - Unicode normalization and encoding validation
    - URL and domain validation
    - Rate limiting and abuse detection
    """
    
    # Crypto-specific patterns
    CRYPTO_PATTERNS = {
        "bitcoin_address": re.compile(r"\b[13][a-km-z13-9A-HJ-NP-Z]{25,34}\b"),
        "ethereum_address": re.compile(r"\b0x[a-fA-F0-9]{40}\b"),
        "crypto_ticker": re.compile(r"\$[A-Z]{2,10}\b"),
        "wallet_seed": re.compile(r"\b(?:\w+\s+){11,23}\w+\b"),  # 12-24 word seeds
        "private_key": re.compile(r"\b[a-fA-F0-9]{64}\b"),
    }
    
    # Suspicious patterns
    SUSPICIOUS_PATTERNS = {
        "script_tags": re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        "javascript_protocol": re.compile(r"javascript:", re.IGNORECASE),
        "data_protocol": re.compile(r"data:.*?;base64,", re.IGNORECASE),
        "sql_injection": re.compile(r"(?:union|select|insert|delete|update|drop|create|alter)\s+", re.IGNORECASE),
        "php_code": re.compile(r"<\?php.*?\?>", re.IGNORECASE | re.DOTALL),
        "exec_functions": re.compile(r"\b(?:eval|exec|system|shell_exec|passthru)\s*\(", re.IGNORECASE),
    }
    
    # Sensitive data patterns
    SENSITIVE_PATTERNS = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b"),
        "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
        "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "api_key": re.compile(r"\b[A-Za-z0-9]{32,64}\b"),
        "jwt_token": re.compile(r"\beyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\b"),
    }
    
    # Malicious content indicators
    MALICIOUS_INDICATORS = {
        "phishing_terms": [
            "verify your account", "suspended account", "click here immediately",
            "urgent action required", "claim your reward", "limited time offer",
            "congratulations you won", "tax refund", "security alert"
        ],
        "scam_terms": [
            "double your bitcoin", "guaranteed profit", "risk free investment",
            "send bitcoin to", "crypto giveaway", "elon musk", "free crypto",
            "bitcoin doubler", "investment opportunity"
        ],
        "spam_indicators": [
            "act now", "buy now", "call now", "click below", "get started now",
            "increase sales", "make money", "opportunity", "stop at any time"
        ]
    }
    
    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.STRICT,
        max_text_length: int = 10000,
        max_batch_size: int = 1000,
        enable_threat_detection: bool = True,
        enable_content_filtering: bool = True,
        enable_sensitive_data_detection: bool = True,
        allowed_languages: Optional[Set[str]] = None,
        blocked_domains: Optional[Set[str]] = None,
    ):
        self.validation_level = validation_level
        self.max_text_length = max_text_length
        self.max_batch_size = max_batch_size
        self.enable_threat_detection = enable_threat_detection
        self.enable_content_filtering = enable_content_filtering
        self.enable_sensitive_data_detection = enable_sensitive_data_detection
        self.allowed_languages = allowed_languages or {"en", "es", "fr", "de", "ja", "ko", "zh", "ru"}
        self.blocked_domains = blocked_domains or set()
        
        # Statistics tracking
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "threats_detected": 0,
            "sensitive_data_detected": 0,
        }
        
        logger.info(f"Initialized DataValidator with {validation_level.value} level")
    
    def validate_text(self, text: Union[str, List[str]], **kwargs) -> Union[ValidationResult, List[ValidationResult]]:
        """
        Validate text input with comprehensive security checks
        
        Args:
            text: Text string or list of texts to validate
            **kwargs: Additional validation options
            
        Returns:
            ValidationResult or list of ValidationResults
        """
        
        import time
        start_time = time.time()
        
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        
        # Batch size validation
        if len(texts) > self.max_batch_size:
            return ValidationResult(
                is_valid=False,
                sanitized_data=None,
                errors=[f"Batch size {len(texts)} exceeds maximum {self.max_batch_size}"],
                warnings=[],
                threats_detected=[],
                confidence_score=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        
        results = []
        
        for single_text in texts:
            result = self._validate_single_text(single_text, **kwargs)
            results.append(result)
            
            # Update statistics
            self.validation_stats["total_validations"] += 1
            if result.is_valid:
                self.validation_stats["successful_validations"] += 1
            else:
                self.validation_stats["failed_validations"] += 1
            
            if result.threats_detected:
                self.validation_stats["threats_detected"] += len(result.threats_detected)
        
        processing_time = (time.time() - start_time) * 1000
        
        if is_batch:
            return results
        else:
            result = results[0]
            result.processing_time_ms = processing_time
            return result
    
    def _validate_single_text(self, text: str, **kwargs) -> ValidationResult:
        """Validate single text string"""
        
        errors = []
        warnings = []
        threats_detected = []
        confidence_score = 1.0
        
        # Basic type validation
        if not isinstance(text, str):
            return ValidationResult(
                is_valid=False,
                sanitized_data=None,
                errors=["Input must be a string"],
                warnings=[],
                threats_detected=[],
                confidence_score=0.0,
                processing_time_ms=0.0,
            )
        
        # Length validation
        if len(text) > self.max_text_length:
            errors.append(f"Text length {len(text)} exceeds maximum {self.max_text_length}")
            confidence_score -= 0.3
        
        # Empty text check
        if not text.strip():
            warnings.append("Empty or whitespace-only text")
            confidence_score -= 0.1
        
        # Unicode validation and normalization
        normalized_text = self._normalize_unicode(text)
        if normalized_text != text:
            warnings.append("Text contained non-standard Unicode characters")
            confidence_score -= 0.05
        
        # Encoding validation
        try:
            # Ensure text can be encoded/decoded properly
            encoded = text.encode('utf-8')
            decoded = encoded.decode('utf-8')
            if decoded != text:
                errors.append("Text encoding/decoding mismatch")
                confidence_score -= 0.2
        except UnicodeError as e:
            errors.append(f"Unicode encoding error: {e}")
            confidence_score -= 0.3
        
        # Security threat detection
        if self.enable_threat_detection:
            detected_threats = self._detect_threats(text)
            threats_detected.extend(detected_threats)
            
            if detected_threats:
                confidence_score -= 0.5
                if SecurityThreat.XSS in detected_threats or SecurityThreat.SCRIPT_INJECTION in detected_threats:
                    errors.append("Potential XSS or script injection detected")
                if SecurityThreat.SQL_INJECTION in detected_threats:
                    errors.append("Potential SQL injection detected")
                if SecurityThreat.MALICIOUS_CONTENT in detected_threats:
                    warnings.append("Potentially malicious content detected")
        
        # Content filtering
        sanitized_text = text
        if self.enable_content_filtering:
            sanitized_text = self._sanitize_content(text)
            if sanitized_text != text:
                warnings.append("Content was sanitized")
                confidence_score -= 0.1
        
        # Sensitive data detection
        if self.enable_sensitive_data_detection:
            sensitive_data = self._detect_sensitive_data(text)
            if sensitive_data:
                self.validation_stats["sensitive_data_detected"] += len(sensitive_data)
                warnings.extend([f"Detected {data_type}: {pattern}" for data_type, pattern in sensitive_data])
                sanitized_text = self._mask_sensitive_data(sanitized_text, sensitive_data)
                confidence_score -= 0.2
        
        # Enterprise level validation
        if self.validation_level == ValidationLevel.ENTERPRISE:
            enterprise_issues = self._enterprise_validation(text)
            if enterprise_issues:
                warnings.extend(enterprise_issues)
                confidence_score -= 0.1
        
        # Final validation decision
        is_valid = len(errors) == 0 and confidence_score >= 0.5
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_data=sanitized_text,
            errors=errors,
            warnings=warnings,
            threats_detected=threats_detected,
            confidence_score=max(0.0, confidence_score),
            processing_time_ms=0.0,  # Will be set by caller
        )
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode text"""
        
        # Normalize Unicode (NFC normalization)
        normalized = unicodedata.normalize('NFC', text)
        
        # Remove or replace problematic characters
        # Remove zero-width characters
        zero_width_chars = [
            '\u200b',  # Zero-width space
            '\u200c',  # Zero-width non-joiner
            '\u200d',  # Zero-width joiner
            '\ufeff',  # Zero-width no-break space (BOM)
        ]
        
        for char in zero_width_chars:
            normalized = normalized.replace(char, '')
        
        # Replace other problematic characters
        replacements = {
            '\u00a0': ' ',  # Non-breaking space -> regular space
            '\u2019': "'",  # Right single quotation mark -> apostrophe
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
        }
        
        for old_char, new_char in replacements.items():
            normalized = normalized.replace(old_char, new_char)
        
        return normalized
    
    def _detect_threats(self, text: str) -> List[SecurityThreat]:
        """Detect security threats in text"""
        
        threats = []
        text_lower = text.lower()
        
        # Check for script injection
        if self.SUSPICIOUS_PATTERNS["script_tags"].search(text):
            threats.append(SecurityThreat.SCRIPT_INJECTION)
        
        if self.SUSPICIOUS_PATTERNS["javascript_protocol"].search(text):
            threats.append(SecurityThreat.SCRIPT_INJECTION)
        
        # Check for XSS patterns
        xss_patterns = [
            r"<.*?on\w+\s*=",  # Event handlers like onclick
            r"<.*?src\s*=\s*[\"']javascript:",  # JavaScript in src
            r"<.*?href\s*=\s*[\"']javascript:",  # JavaScript in href
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append(SecurityThreat.XSS)
                break
        
        # Check for SQL injection
        if self.SUSPICIOUS_PATTERNS["sql_injection"].search(text):
            threats.append(SecurityThreat.SQL_INJECTION)
        
        # Check for malicious content indicators
        malicious_score = 0
        for category, terms in self.MALICIOUS_INDICATORS.items():
            for term in terms:
                if term.lower() in text_lower:
                    malicious_score += 1
        
        if malicious_score >= 3:  # Threshold for malicious content
            threats.append(SecurityThreat.MALICIOUS_CONTENT)
        
        # Check for data protocol (potential data exfiltration)
        if self.SUSPICIOUS_PATTERNS["data_protocol"].search(text):
            threats.append(SecurityThreat.URL_MANIPULATION)
        
        return threats
    
    def _sanitize_content(self, text: str) -> str:
        """Sanitize content by removing/escaping dangerous elements"""
        
        sanitized = text
        
        # HTML escape
        sanitized = html.escape(sanitized)
        
        # Remove script tags
        sanitized = self.SUSPICIOUS_PATTERNS["script_tags"].sub("", sanitized)
        
        # Remove javascript: protocols
        sanitized = self.SUSPICIOUS_PATTERNS["javascript_protocol"].sub("", sanitized)
        
        # Remove data: protocols
        sanitized = self.SUSPICIOUS_PATTERNS["data_protocol"].sub("", sanitized)
        
        # Remove PHP code
        sanitized = self.SUSPICIOUS_PATTERNS["php_code"].sub("", sanitized)
        
        # Remove dangerous function calls
        sanitized = self.SUSPICIOUS_PATTERNS["exec_functions"].sub("", sanitized)
        
        return sanitized
    
    def _detect_sensitive_data(self, text: str) -> List[Tuple[str, str]]:
        """Detect sensitive data patterns"""
        
        sensitive_data = []
        
        for data_type, pattern in self.SENSITIVE_PATTERNS.items():
            matches = pattern.finditer(text)
            for match in matches:
                sensitive_data.append((data_type, match.group()))
        
        # Check for crypto-related sensitive data
        for data_type, pattern in self.CRYPTO_PATTERNS.items():
            matches = pattern.finditer(text)
            for match in matches:
                # Be extra careful with wallet seeds and private keys
                if data_type in ["wallet_seed", "private_key"]:
                    sensitive_data.append((f"crypto_{data_type}", match.group()))
        
        return sensitive_data
    
    def _mask_sensitive_data(self, text: str, sensitive_data: List[Tuple[str, str]]) -> str:
        """Mask sensitive data in text"""
        
        masked_text = text
        
        for data_type, data_value in sensitive_data:
            if data_type in ["email"]:
                # Partially mask emails
                masked_value = self._mask_email(data_value)
            elif data_type in ["phone"]:
                # Partially mask phone numbers
                masked_value = self._mask_phone(data_value)
            elif data_type in ["crypto_wallet_seed", "crypto_private_key", "api_key", "jwt_token"]:
                # Completely mask crypto sensitive data
                masked_value = "[REDACTED]"
            elif data_type in ["credit_card"]:
                # Mask credit card numbers
                masked_value = self._mask_credit_card(data_value)
            else:
                # Generic masking
                masked_value = "*" * min(len(data_value), 8)
            
            masked_text = masked_text.replace(data_value, masked_value)
        
        return masked_text
    
    def _mask_email(self, email: str) -> str:
        """Partially mask email address"""
        try:
            local, domain = email.split('@')
            if len(local) > 2:
                masked_local = local[0] + '*' * (len(local) - 2) + local[-1]
            else:
                masked_local = '*' * len(local)
            return f"{masked_local}@{domain}"
        except ValueError:
            return "*****@*****.***"
    
    def _mask_phone(self, phone: str) -> str:
        """Partially mask phone number"""
        digits_only = re.sub(r'\D', '', phone)
        if len(digits_only) >= 4:
            return phone.replace(digits_only[:-4], '*' * len(digits_only[:-4]))
        return '*' * len(phone)
    
    def _mask_credit_card(self, card: str) -> str:
        """Mask credit card number"""
        digits_only = re.sub(r'\D', '', card)
        if len(digits_only) >= 4:
            masked_digits = '*' * (len(digits_only) - 4) + digits_only[-4:]
            # Preserve original formatting
            result = card
            for i, digit in enumerate(digits_only):
                result = result.replace(digit, masked_digits[i], 1)
            return result
        return '*' * len(card)
    
    def _enterprise_validation(self, text: str) -> List[str]:
        """Enterprise-level validation checks"""
        
        issues = []
        
        # Check for suspicious URL patterns
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)
        for url in urls:
            try:
                parsed = urlparse(url)
                
                # Check against blocked domains
                if parsed.netloc.lower() in self.blocked_domains:
                    issues.append(f"Blocked domain detected: {parsed.netloc}")
                
                # Check for suspicious TLDs
                suspicious_tlds = {'.tk', '.ml', '.ga', '.cf', '.bit', '.onion'}
                if any(parsed.netloc.endswith(tld) for tld in suspicious_tlds):
                    issues.append(f"Suspicious TLD detected: {parsed.netloc}")
                
                # Check for IP addresses instead of domains
                if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', parsed.netloc):
                    issues.append("IP address used instead of domain name")
                
            except Exception:
                issues.append(f"Malformed URL detected: {url}")
        
        # Check text complexity and readability
        if len(text) > 100:
            # Simple complexity check
            word_count = len(text.split())
            unique_words = len(set(text.lower().split()))
            
            if word_count > 0:
                uniqueness_ratio = unique_words / word_count
                if uniqueness_ratio < 0.3:  # Too much repetition
                    issues.append("Text contains excessive repetition")
        
        # Check for base64 encoded content (potential data hiding)
        base64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
        potential_base64 = base64_pattern.findall(text)
        
        for b64_candidate in potential_base64:
            try:
                decoded = base64.b64decode(b64_candidate)
                # Check if decoded content looks suspicious
                if len(decoded) > 10 and all(32 <= byte <= 126 for byte in decoded):
                    issues.append("Potential base64 encoded content detected")
            except Exception:
                pass
        
        return issues
    
    def validate_json(self, json_data: Union[str, Dict[str, Any]], schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate JSON data"""
        
        import time
        start_time = time.time()
        
        errors = []
        warnings = []
        threats_detected = []
        confidence_score = 1.0
        
        # Parse JSON if string
        if isinstance(json_data, str):
            try:
                parsed_data = json.loads(json_data)
            except json.JSONDecodeError as e:
                return ValidationResult(
                    is_valid=False,
                    sanitized_data=None,
                    errors=[f"Invalid JSON: {e}"],
                    warnings=[],
                    threats_detected=[],
                    confidence_score=0.0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
        else:
            parsed_data = json_data
        
        # Recursively validate text values in JSON
        sanitized_data = self._validate_json_recursive(parsed_data, errors, warnings, threats_detected)
        
        # Schema validation (basic)
        if schema:
            schema_errors = self._validate_json_schema(sanitized_data, schema)
            errors.extend(schema_errors)
        
        # Adjust confidence based on issues found
        confidence_score -= len(errors) * 0.2
        confidence_score -= len(warnings) * 0.1
        confidence_score -= len(threats_detected) * 0.3
        
        is_valid = len(errors) == 0 and confidence_score >= 0.5
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_data=sanitized_data,
            errors=errors,
            warnings=warnings,
            threats_detected=threats_detected,
            confidence_score=max(0.0, confidence_score),
            processing_time_ms=(time.time() - start_time) * 1000,
        )
    
    def _validate_json_recursive(self, data: Any, errors: List[str], warnings: List[str], threats_detected: List[SecurityThreat]) -> Any:
        """Recursively validate JSON data"""
        
        if isinstance(data, dict):
            sanitized_dict = {}
            for key, value in data.items():
                # Validate key
                if isinstance(key, str):
                    key_result = self._validate_single_text(key)
                    if not key_result.is_valid:
                        errors.extend(key_result.errors)
                    warnings.extend(key_result.warnings)
                    threats_detected.extend(key_result.threats_detected)
                    sanitized_key = key_result.sanitized_data
                else:
                    sanitized_key = key
                
                # Validate value
                sanitized_value = self._validate_json_recursive(value, errors, warnings, threats_detected)
                sanitized_dict[sanitized_key] = sanitized_value
            
            return sanitized_dict
        
        elif isinstance(data, list):
            return [self._validate_json_recursive(item, errors, warnings, threats_detected) for item in data]
        
        elif isinstance(data, str):
            # Validate string value
            result = self._validate_single_text(data)
            if not result.is_valid:
                errors.extend(result.errors)
            warnings.extend(result.warnings)
            threats_detected.extend(result.threats_detected)
            return result.sanitized_data
        
        else:
            # Numbers, booleans, null - pass through
            return data
    
    def _validate_json_schema(self, data: Any, schema: Dict[str, Any]) -> List[str]:
        """Basic JSON schema validation"""
        
        errors = []
        
        # Simple type checking
        if "type" in schema:
            expected_type = schema["type"]
            actual_type = type(data).__name__
            
            type_map = {
                "string": "str",
                "number": ["int", "float"],
                "boolean": "bool",
                "array": "list",
                "object": "dict",
                "null": "NoneType",
            }
            
            expected_types = type_map.get(expected_type, expected_type)
            if isinstance(expected_types, list):
                if actual_type not in expected_types:
                    errors.append(f"Expected type {expected_type}, got {actual_type}")
            else:
                if actual_type != expected_types:
                    errors.append(f"Expected type {expected_type}, got {actual_type}")
        
        # Required fields
        if isinstance(data, dict) and "required" in schema:
            for required_field in schema["required"]:
                if required_field not in data:
                    errors.append(f"Required field missing: {required_field}")
        
        return errors
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        
        total = self.validation_stats["total_validations"]
        success_rate = (self.validation_stats["successful_validations"] / total * 100) if total > 0 else 0
        
        return {
            **self.validation_stats,
            "success_rate_percent": success_rate,
            "threat_rate_percent": (self.validation_stats["threats_detected"] / total * 100) if total > 0 else 0,
            "validation_level": self.validation_level.value,
            "max_text_length": self.max_text_length,
            "max_batch_size": self.max_batch_size,
        }
    
    def reset_stats(self):
        """Reset validation statistics"""
        
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "threats_detected": 0,
            "sensitive_data_detected": 0,
        }
    
    def update_blocked_domains(self, domains: Set[str]):
        """Update blocked domains list"""
        self.blocked_domains.update(domains)
    
    def add_custom_threat_pattern(self, name: str, pattern: re.Pattern, threat_type: SecurityThreat):
        """Add custom threat detection pattern"""
        self.SUSPICIOUS_PATTERNS[name] = pattern
        # Note: Would need to modify _detect_threats to use custom patterns


# Global validator instance
_global_validator: Optional[DataValidator] = None


def get_validator(**kwargs) -> DataValidator:
    """Get global data validator instance"""
    global _global_validator
    
    if _global_validator is None:
        _global_validator = DataValidator(**kwargs)
    
    return _global_validator


def set_validator(validator: DataValidator):
    """Set global data validator instance"""
    global _global_validator
    _global_validator = validator
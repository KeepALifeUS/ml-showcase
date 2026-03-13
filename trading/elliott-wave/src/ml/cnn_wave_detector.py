"""
CNN-based Elliott Wave Pattern Detection.

Deep learning for high-accuracy wave pattern
recognition with crypto market adaptations and real-time inference.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime
from pathlib import Path

from ..utils.logger import get_logger, trading_logger, performance_monitor
from ..utils.config import config
from ..patterns.impulse_wave import ImpulseWave, WavePoint
from ..patterns.corrective_wave import CorrectiveWave

logger = get_logger(__name__)


class ModelArchitecture(str, Enum):
    """CNN architecture types."""
    BASIC_CNN = "basic_cnn"
    RESNET_WAVE = "resnet_wave"
    INCEPTION_WAVE = "inception_wave"
    EFFICIENTNET_WAVE = "efficientnet_wave"
    CUSTOM_CRYPTO = "custom_crypto"


class PatternType(str, Enum):
    """Wave pattern types for classification."""
    IMPULSE_1 = "impulse_1"
    IMPULSE_2 = "impulse_2"
    IMPULSE_3 = "impulse_3"
    IMPULSE_4 = "impulse_4"
    IMPULSE_5 = "impulse_5"
    CORRECTIVE_A = "corrective_a"
    CORRECTIVE_B = "corrective_b"
    CORRECTIVE_C = "corrective_c"
    DIAGONAL = "diagonal"
    TRIANGLE = "triangle"
    FLAT = "flat"
    ZIGZAG = "zigzag"
    NO_PATTERN = "no_pattern"


@dataclass
class PatternDetection:
    """CNN pattern detection result."""
    pattern_type: PatternType
    confidence: float
    probability_distribution: Dict[PatternType, float]
    bounding_box: Tuple[int, int, int, int]  # start_idx, end_idx, min_price, max_price
    feature_maps: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None


class WaveDataset(Dataset):
    """
    PyTorch dataset for Elliott Wave pattern training.
    
    Efficient data pipeline with crypto market preprocessing.
    """
    
    def __init__(self,
                 price_data: List[pd.DataFrame],
                 labels: List[PatternType],
                 sequence_length: int = 100,
                 transform=None):
        """
        Initialize wave dataset.
        
        Args:
            price_data: List of OHLCV DataFrames
            labels: List of corresponding pattern labels
            sequence_length: Length of price sequences
            transform: Optional data transformations
        """
        self.price_data = price_data
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Preprocess data
        self.sequences, self.processed_labels = self._preprocess_data()
        
    def _preprocess_data(self) -> Tuple[List[torch.Tensor], List[int]]:
        """Preprocess price data into CNN-ready sequences."""
        sequences = []
        processed_labels = []
        
        label_to_idx = {pattern: idx for idx, pattern in enumerate(PatternType)}
        
        for data, label in zip(self.price_data, self.labels):
            # Convert OHLCV to features
            features = self._extract_features(data)
            
            # Create sliding windows
            for i in range(len(features) - self.sequence_length + 1):
                sequence = features[i:i + self.sequence_length]
                sequences.append(torch.tensor(sequence, dtype=torch.float32))
                processed_labels.append(label_to_idx[label])
                
        return sequences, processed_labels
        
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract CNN features from OHLCV data."""
        features = []
        
        # Price features (normalized)
        high = data['high'].values
        low = data['low'].values
        open_price = data['open'].values
        close = data['close'].values
        
        # Normalize to relative values
        base_price = close[0] if len(close) > 0 else 1.0
        
        # Price channels
        features.append((high - base_price) / base_price)
        features.append((low - base_price) / base_price)
        features.append((open_price - base_price) / base_price)
        features.append((close - base_price) / base_price)
        
        # Volume (if available)
        if 'volume' in data.columns:
            volume = data['volume'].values
            volume_norm = volume / (np.mean(volume) + 1e-8)
            features.append(volume_norm)
        else:
            features.append(np.ones_like(close))
            
        # Technical indicators
        features.append(self._calculate_rsi(close))
        features.append(self._calculate_macd(close))
        features.append(self._calculate_bollinger_bands(close))
        
        return np.array(features).T  # Shape: (sequence_length, num_features)
        
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return np.full_like(prices, 0.5)
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # Pad to original length
        rsi_padded = np.full_like(prices, 0.5)
        rsi_padded[period:] = rsi / 100.0  # Normalize to 0-1
        
        return rsi_padded
        
    def _calculate_macd(self, prices: np.ndarray) -> np.ndarray:
        """Calculate MACD indicator."""
        if len(prices) < 26:
            return np.zeros_like(prices)
            
        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)
        macd = ema12 - ema26
        
        # Normalize
        macd_std = np.std(macd) + 1e-8
        return macd / macd_std
        
    def _ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
            
        return ema
        
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate Bollinger Bands position."""
        if len(prices) < period:
            return np.full_like(prices, 0.5)
            
        sma = np.convolve(prices, np.ones(period)/period, mode='valid')
        std = np.array([np.std(prices[i:i+period]) for i in range(len(prices) - period + 1)])
        
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        # Calculate position within bands
        bb_position = np.full_like(prices, 0.5)
        valid_prices = prices[period-1:]
        
        for i, (price, upper, lower) in enumerate(zip(valid_prices, upper_band, lower_band)):
            if upper - lower > 1e-8:
                bb_position[period-1+i] = (price - lower) / (upper - lower)
                
        return bb_position
        
    def __len__(self) -> int:
        return len(self.sequences)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sequence = self.sequences[idx]
        label = self.processed_labels[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
            
        return sequence, label


class WavePatternCNN(nn.Module):
    """
    Convolutional Neural Network for Elliott Wave Pattern Recognition.
    
    State-of-the-art CNN architecture with crypto market adaptations.
    """
    
    def __init__(self,
                 input_features: int = 8,
                 sequence_length: int = 100,
                 num_classes: int = 13,
                 architecture: ModelArchitecture = ModelArchitecture.CUSTOM_CRYPTO):
        """
        Initialize CNN model.
        
        Args:
            input_features: Number of input features per timestep
            sequence_length: Length of input sequences
            num_classes: Number of pattern classes
            architecture: Model architecture type
        """
        super(WavePatternCNN, self).__init__()
        
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.architecture = architecture
        
        # Build model based on architecture
        if architecture == ModelArchitecture.CUSTOM_CRYPTO:
            self._build_custom_crypto_model()
        elif architecture == ModelArchitecture.BASIC_CNN:
            self._build_basic_cnn()
        elif architecture == ModelArchitecture.RESNET_WAVE:
            self._build_resnet_wave()
        else:
            self._build_custom_crypto_model()  # Default
            
        # Initialize weights
        self.apply(self._init_weights)
        
    def _build_custom_crypto_model(self):
        """Build custom CNN optimized for crypto wave patterns."""
        # Multi-scale convolutional layers
        self.conv_layers = nn.ModuleList([
            # Short-term patterns
            nn.Sequential(
                nn.Conv1d(self.input_features, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            # Medium-term patterns
            nn.Sequential(
                nn.Conv1d(self.input_features, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            # Long-term patterns
            nn.Sequential(
                nn.Conv1d(self.input_features, 64, kernel_size=15, padding=7),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        ])
        
        # Feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Temporal processing
        self.temporal_layers = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.4)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_classes)
        )
        
    def _build_basic_cnn(self):
        """Build basic CNN model."""
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.input_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.num_classes)
        )
        
    def _build_resnet_wave(self):
        """Build ResNet-style model for wave patterns."""
        # Implementation would include residual blocks
        # For brevity, using simplified version
        self._build_basic_cnn()
        
    def _init_weights(self, m):
        """Initialize model weights."""
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input shape: (batch_size, sequence_length, features)
        # Convert to (batch_size, features, sequence_length) for Conv1d
        x = x.transpose(1, 2)
        
        if self.architecture == ModelArchitecture.CUSTOM_CRYPTO:
            # Multi-scale feature extraction
            conv_outputs = []
            for conv_layer in self.conv_layers:
                conv_out = conv_layer(x)
                conv_outputs.append(conv_out)
                
            # Concatenate multi-scale features
            x = torch.cat(conv_outputs, dim=1)
            
            # Fusion and temporal processing
            x = self.fusion_conv(x)
            x = self.temporal_layers(x)
            
            # Flatten for classification
            x = x.view(x.size(0), -1)
            
        else:
            # Basic CNN forward pass
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            
        # Classification
        output = self.classifier(x)
        
        return output
        
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract feature maps for visualization."""
        feature_maps = {}
        
        x = x.transpose(1, 2)
        
        if self.architecture == ModelArchitecture.CUSTOM_CRYPTO:
            # Extract multi-scale features
            for i, conv_layer in enumerate(self.conv_layers):
                conv_out = conv_layer(x)
                feature_maps[f'scale_{i}'] = conv_out
                
        return feature_maps


class CNNWaveDetector:
    """
    CNN-based Elliott Wave Pattern Detector.
    
    
    - High-performance inference
    - Model versioning and management
    - Real-time pattern detection
    - Crypto market optimizations
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 architecture: ModelArchitecture = ModelArchitecture.CUSTOM_CRYPTO,
                 device: str = "auto"):
        """
        Initialize CNN wave detector.
        
        Args:
            model_path: Path to pre-trained model
            architecture: Model architecture type
            device: Device for inference (auto, cpu, cuda)
        """
        self.architecture = architecture
        self.device = self._get_device(device)
        
        # Initialize model
        self.model = WavePatternCNN(architecture=architecture)
        self.model.to(self.device)
        
        # Load pre-trained weights if available
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.warning("No pre-trained model found. Using randomly initialized weights.")
            
        self.model.eval()
        
        # Performance statistics
        self.detection_stats = {
            "total_detections": 0,
            "inference_times": [],
            "confidence_scores": [],
            "pattern_counts": {pattern.value: 0 for pattern in PatternType}
        }
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
            
    @performance_monitor
    async def detect_patterns(self,
                            price_data: pd.DataFrame,
                            symbol: str,
                            timeframe: str,
                            confidence_threshold: float = 0.7) -> List[PatternDetection]:
        """
        Detect Elliott Wave patterns using CNN.
        
        High-performance async pattern detection.
        
        Args:
            price_data: OHLCV price data
            symbol: Trading symbol
            timeframe: Timeframe string
            confidence_threshold: Minimum confidence for valid detection
            
        Returns:
            List[PatternDetection]: Detected patterns
        """
        start_time = datetime.utcnow()
        
        # Validate input data
        if len(price_data) < 100:  # Minimum required for analysis
            logger.warning(f"Insufficient data for CNN analysis: {len(price_data)} bars")
            return []
            
        # Prepare data for inference
        input_tensor = await self._prepare_inference_data(price_data)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
        # Process results
        detections = await self._process_inference_results(
            probabilities, price_data, confidence_threshold
        )
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Update statistics
        self.detection_stats["total_detections"] += len(detections)
        self.detection_stats["inference_times"].append(processing_time)
        
        for detection in detections:
            self.detection_stats["confidence_scores"].append(detection.confidence)
            self.detection_stats["pattern_counts"][detection.pattern_type.value] += 1
            
        # Log detection results
        trading_logger.log_wave_detection(
            symbol=symbol,
            timeframe=timeframe,
            wave_type="cnn_patterns",
            confidence=max([d.confidence for d in detections]) if detections else 0,
            detected_count=len(detections),
            processing_time_ms=processing_time
        )
        
        return detections
        
    async def _prepare_inference_data(self, price_data: pd.DataFrame) -> torch.Tensor:
        """Prepare price data for CNN inference."""
        # Create dataset with dummy labels for inference
        dataset = WaveDataset(
            price_data=[price_data],
            labels=[PatternType.NO_PATTERN],  # Dummy label
            sequence_length=100
        )
        
        # Get the sequence (ignore label)
        if len(dataset) > 0:
            sequence, _ = dataset[0]
            return sequence.unsqueeze(0).to(self.device)  # Add batch dimension
        else:
            # Create empty tensor with correct shape
            return torch.zeros(1, 100, 8, device=self.device)
            
    async def _process_inference_results(self,
                                       probabilities: torch.Tensor,
                                       price_data: pd.DataFrame,
                                       confidence_threshold: float) -> List[PatternDetection]:
        """Process CNN inference results into pattern detections."""
        detections = []
        
        # Get predictions
        probs_numpy = probabilities.cpu().numpy()
        
        for i, prob_dist in enumerate(probs_numpy):
            # Find highest probability pattern
            max_idx = np.argmax(prob_dist)
            max_confidence = prob_dist[max_idx]
            
            if max_confidence >= confidence_threshold:
                pattern_type = list(PatternType)[max_idx]
                
                # Create probability distribution dict
                prob_dict = {
                    pattern: float(prob) 
                    for pattern, prob in zip(PatternType, prob_dist)
                }
                
                # Estimate bounding box (simplified)
                start_idx = max(0, len(price_data) - 100)
                end_idx = len(price_data) - 1
                min_price = price_data.iloc[start_idx:end_idx]['low'].min()
                max_price = price_data.iloc[start_idx:end_idx]['high'].max()
                
                detection = PatternDetection(
                    pattern_type=pattern_type,
                    confidence=max_confidence,
                    probability_distribution=prob_dict,
                    bounding_box=(start_idx, end_idx, min_price, max_price)
                )
                
                detections.append(detection)
                
        return detections
        
    def load_model(self, model_path: str) -> None:
        """Load pre-trained model weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            logger.info(f"Successfully loaded model from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            
    def save_model(self, model_path: str) -> None:
        """Save model weights."""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'architecture': self.architecture.value,
                'num_classes': self.model.num_classes,
                'input_features': self.model.input_features,
                'sequence_length': self.model.sequence_length
            }, model_path)
            
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model to {model_path}: {e}")
            
    async def train_model(self,
                        train_dataset: WaveDataset,
                        val_dataset: Optional[WaveDataset] = None,
                        num_epochs: int = 100,
                        learning_rate: float = 0.001,
                        batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the CNN model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            
        Returns:
            Dict[str, Any]: Training history and metrics
        """
        self.model.train()
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training history
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        best_val_accuracy = 0.0
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
                
            # Calculate training metrics
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100.0 * train_correct / train_total
            
            history["train_loss"].append(avg_train_loss)
            history["train_accuracy"].append(train_accuracy)
            
            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, targets in val_loader:
                        data, targets = data.to(self.device), targets.to(self.device)
                        
                        outputs = self.model(data)
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += targets.size(0)
                        val_correct += (predicted == targets).sum().item()
                        
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100.0 * val_correct / val_total
                
                history["val_loss"].append(avg_val_loss)
                history["val_accuracy"].append(val_accuracy)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Save best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    self.save_model("best_model.pth")
                    
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
                
        self.model.eval()
        return history
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "architecture": self.architecture.value,
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_features": self.model.input_features,
            "sequence_length": self.model.sequence_length,
            "num_classes": self.model.num_classes,
            "detection_stats": self.detection_stats
        }


# Export main classes
__all__ = [
    'CNNWaveDetector',
    'WavePatternCNN',
    'WaveDataset',
    'PatternDetection',
    'PatternType',
    'ModelArchitecture'
]
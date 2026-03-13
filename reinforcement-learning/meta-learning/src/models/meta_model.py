"""
Meta-Learning Model Architectures
Flexible Model Architecture

Base architectures models for meta-training with support
various types tasks and adaptive components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class MetaModelConfig:
    """Configuration for meta-model"""
    
    # Architecture
    input_dim: int = 50
    hidden_dims: List[int] = None
    output_dim: int = 3
    
    # Model
    model_type: str = "mlp"  # mlp, cnn, transformer
    activation: str = "relu"  # relu, gelu, swish
    normalization: str = "batch"  # batch, layer, none
    dropout_rate: float = 0.1
    
    # Meta-training
    meta_learnable: bool = True
    use_meta_params: bool = False
    adaptation_layers: List[str] = None  # Which layers adapt
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]
        if self.adaptation_layers is None:
            self.adaptation_layers = ["all"]


class MetaModel(nn.Module):
    """
    Base meta-model for cryptocurrency trading
    
    Adaptable Neural Architecture
    - Flexible layer configuration
    - Meta-learnable parameters
    - Task-specific adaptation
    """
    
    def __init__(self, config: MetaModelConfig):
        super().__init__()
        
        self.config = config
        
        # Create main architecture
        if config.model_type == "mlp":
            self.backbone = self._create_mlp_backbone()
        elif config.model_type == "cnn":
            self.backbone = self._create_cnn_backbone()
        elif config.model_type == "transformer":
            self.backbone = self._create_transformer_backbone()
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        # Final classification layer
        self.classifier = nn.Linear(
            self._get_backbone_output_dim(), 
            config.output_dim
        )
        
        # Meta-parameters if enabled
        if config.use_meta_params:
            self.meta_params = self._create_meta_parameters()
        
        self._initialize_weights()
    
    def _create_mlp_backbone(self) -> nn.Module:
        """Creates MLP backbone"""
        layers = []
        
        current_dim = self.config.input_dim
        
        for hidden_dim in self.config.hidden_dims:
            # Linear layer
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            # Normalization
            if self.config.normalization == "batch":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif self.config.normalization == "layer":
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            if self.config.activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif self.config.activation == "gelu":
                layers.append(nn.GELU())
            elif self.config.activation == "swish":
                layers.append(nn.SiLU())
            
            # Dropout
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
            
            current_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _create_cnn_backbone(self) -> nn.Module:
        """Creates CNN backbone for temporal series"""
        layers = []
        
        # Assume, that input has form (batch, channels, sequence_length)
        in_channels = 1
        out_channels = 32
        
        # First block
        layers.extend([
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        ])
        
        # Second block
        in_channels = out_channels
        out_channels = 64
        layers.extend([
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        ])
        
        # Third block
        in_channels = out_channels
        out_channels = 128
        layers.extend([
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        ])
        
        backbone = nn.Sequential(*layers)
        
        # Add flatten
        return nn.Sequential(
            backbone,
            nn.Flatten(),
            nn.Linear(128, self.config.hidden_dims[-1])
        )
    
    def _create_transformer_backbone(self) -> nn.Module:
        """Creates Transformer backbone"""
        # Simplified version Transformer
        embed_dim = self.config.hidden_dims[0]
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Input projection
        self.input_projection = nn.Linear(self.config.input_dim, embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=self.config.dropout_rate,
            activation=self.config.activation
        )
        
        transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=len(self.config.hidden_dims)
        )
        
        return nn.Sequential(
            self.input_projection,
            self.pos_encoding,
            transformer,
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
    
    def _get_backbone_output_dim(self) -> int:
        """Returns dimensionality output backbone"""
        if self.config.model_type == "mlp":
            return self.config.hidden_dims[-1]
        elif self.config.model_type == "cnn":
            return self.config.hidden_dims[-1]
        elif self.config.model_type == "transformer":
            return self.config.hidden_dims[0]
        else:
            return self.config.hidden_dims[-1]
    
    def _create_meta_parameters(self) -> nn.ParameterDict:
        """Creates meta-parameters for fast adaptation"""
        meta_params = nn.ParameterDict()
        
        # Meta-parameters for scaling and shifting
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Scale and bias parameters for linear layers
                meta_params[f"{name}_scale"] = nn.Parameter(
                    torch.ones(module.out_features)
                )
                meta_params[f"{name}_shift"] = nn.Parameter(
                    torch.zeros(module.out_features)
                )
        
        return meta_params
    
    def _initialize_weights(self):
        """Initialization weights model"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through model
        
        Args:
            x: Input data
            
        Returns:
            Output model
        """
        # Processing input in dependencies from type model
        if self.config.model_type == "cnn" and len(x.shape) == 2:
            # Add channel dimension for CNN
            x = x.unsqueeze(1)
        
        # Backbone
        features = self.backbone(x)
        
        # Apply meta-parameters if they exists
        if self.config.use_meta_params and hasattr(self, 'meta_params'):
            features = self._apply_meta_parameters(features, 'backbone')
        
        # Classification
        output = self.classifier(features)
        
        return output
    
    def _apply_meta_parameters(
        self, 
        features: torch.Tensor, 
        layer_name: str
    ) -> torch.Tensor:
        """Applies meta-parameters to features"""
        scale_key = f"{layer_name}_scale"
        shift_key = f"{layer_name}_shift"
        
        if scale_key in self.meta_params and shift_key in self.meta_params:
            scale = self.meta_params[scale_key]
            shift = self.meta_params[shift_key]
            features = features * scale + shift
        
        return features
    
    def get_adaptable_parameters(self) -> Dict[str, nn.Parameter]:
        """Returns parameters for adaptation"""
        if "all" in self.config.adaptation_layers:
            return dict(self.named_parameters())
        
        adaptable_params = {}
        for name, param in self.named_parameters():
            for layer_name in self.config.adaptation_layers:
                if layer_name in name:
                    adaptable_params[name] = param
                    break
        
        return adaptable_params
    
    def get_config(self) -> Dict[str, Any]:
        """Returns configuration model"""
        return {
            'input_dim': self.config.input_dim,
            'hidden_dims': self.config.hidden_dims,
            'output_dim': self.config.output_dim,
            'model_type': self.config.model_type,
            'activation': self.config.activation,
            'normalization': self.config.normalization,
            'dropout_rate': self.config.dropout_rate
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class AdaptationModule(nn.Module):
    """
    Module adaptation for fast settings to new tasks
    
    Task-Specific Adaptation
    - Parameter-efficient adaptation
    - Task-specific layers
    - Gradient-based optimization
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        adaptation_type: str = "linear",
        adaptation_dim: int = 64
    ):
        super().__init__()
        
        self.base_model = base_model
        self.adaptation_type = adaptation_type
        
        # Retrieve dimensionality features from base model
        with torch.no_grad():
            dummy_input = torch.randn(1, base_model.config.input_dim)
            dummy_features = base_model.backbone(dummy_input)
            feature_dim = dummy_features.shape[-1]
        
        # Create adaptation layers
        if adaptation_type == "linear":
            self.adaptation_layer = nn.Linear(feature_dim, adaptation_dim)
            self.task_classifier = nn.Linear(adaptation_dim, base_model.config.output_dim)
        
        elif adaptation_type == "attention":
            self.adaptation_layer = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True
            )
            self.task_classifier = nn.Linear(feature_dim, base_model.config.output_dim)
        
        elif adaptation_type == "film":
            # Feature-wise Linear Modulation
            self.gamma_layer = nn.Linear(adaptation_dim, feature_dim)
            self.beta_layer = nn.Linear(adaptation_dim, feature_dim)
            self.task_encoder = nn.Linear(base_model.config.input_dim, adaptation_dim)
            self.task_classifier = nn.Linear(feature_dim, base_model.config.output_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        task_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with adaptation
        
        Args:
            x: Input data
            task_context: Context tasks (for FiLM)
            
        Returns:
            Adapted output
        """
        # Retrieve features from base model
        features = self.base_model.backbone(x)
        
        if self.adaptation_type == "linear":
            adapted_features = self.adaptation_layer(features)
            adapted_features = F.relu(adapted_features)
            output = self.task_classifier(adapted_features)
        
        elif self.adaptation_type == "attention":
            # Self-attention adaptation
            attended_features, _ = self.adaptation_layer(
                features.unsqueeze(1), features.unsqueeze(1), features.unsqueeze(1)
            )
            attended_features = attended_features.squeeze(1)
            output = self.task_classifier(attended_features)
        
        elif self.adaptation_type == "film":
            # FiLM adaptation
            if task_context is None:
                task_context = x.mean(dim=1)  # Simple task encoding
            
            task_embedding = self.task_encoder(task_context)
            gamma = self.gamma_layer(task_embedding)
            beta = self.beta_layer(task_embedding)
            
            # Apply FiLM
            modulated_features = gamma * features + beta
            output = self.task_classifier(modulated_features)
        
        return output
    
    def adapt_to_task(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        num_steps: int = 10,
        learning_rate: float = 0.01
    ) -> None:
        """
        Adapts module to specific task
        
        Args:
            support_data: Support data
            support_labels: Support labels
            num_steps: Number steps adaptation
            learning_rate: Speed training
        """
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze adaptation layers
        for param in self.parameters():
            if param not in self.base_model.parameters():
                param.requires_grad = True
        
        # Optimizer for adaptation
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=learning_rate
        )
        
        # Adaptation
        self.train()
        for step in range(num_steps):
            optimizer.zero_grad()
            
            predictions = self(support_data)
            loss = F.cross_entropy(predictions, support_labels.long())
            
            loss.backward()
            optimizer.step()
        
        # Unfreeze base model
        for param in self.base_model.parameters():
            param.requires_grad = True


class MetaModelFactory:
    """
    Factory for creation meta-models
    
    Model Factory
    - Centralized model creation
    - Configuration-based instantiation
    - Pre-configured architectures
    """
    
    @staticmethod
    def create_model(
        model_type: str,
        input_dim: int,
        output_dim: int,
        **kwargs
    ) -> MetaModel:
        """
        Creates meta-model specified type
        
        Args:
            model_type: Type model (simple_mlp, deep_mlp, cnn, transformer)
            input_dim: Dimensionality input
            output_dim: Dimensionality output
            **kwargs: Additional parameters
            
        Returns:
            Instance meta-model
        """
        
        if model_type == "simple_mlp":
            config = MetaModelConfig(
                input_dim=input_dim,
                hidden_dims=[64, 32],
                output_dim=output_dim,
                model_type="mlp",
                activation="relu",
                dropout_rate=0.1
            )
        
        elif model_type == "deep_mlp":
            config = MetaModelConfig(
                input_dim=input_dim,
                hidden_dims=[256, 128, 64],
                output_dim=output_dim,
                model_type="mlp",
                activation="gelu",
                normalization="layer",
                dropout_rate=0.2
            )
        
        elif model_type == "cnn":
            config = MetaModelConfig(
                input_dim=input_dim,
                hidden_dims=[128, 64],
                output_dim=output_dim,
                model_type="cnn",
                activation="relu",
                dropout_rate=0.1
            )
        
        elif model_type == "transformer":
            config = MetaModelConfig(
                input_dim=input_dim,
                hidden_dims=[128],
                output_dim=output_dim,
                model_type="transformer",
                activation="gelu",
                dropout_rate=0.1
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Apply additional parameters
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return MetaModel(config)
    
    @staticmethod
    def create_crypto_trading_model(
        input_dim: int,
        task_type: str = "classification",
        complexity: str = "medium"
    ) -> MetaModel:
        """
        Creates model for cryptocurrency trading
        
        Args:
            input_dim: Dimensionality input data
            task_type: Type tasks (classification, regression)
            complexity: Complexity model (simple, medium, complex)
            
        Returns:
            Optimized model for crypto-trading
        """
        
        # Define output dimensionality
        if task_type == "classification":
            output_dim = 3  # BUY, SELL, HOLD
        elif task_type == "regression":
            output_dim = 1  # Continuous value
        elif task_type == "portfolio":
            output_dim = 10  # Portfolio weights
        else:
            output_dim = 1
        
        # Select architecture on basis complexity
        if complexity == "simple":
            model_type = "simple_mlp"
        elif complexity == "medium":
            model_type = "deep_mlp"
        elif complexity == "complex":
            model_type = "transformer"
        else:
            model_type = "deep_mlp"
        
        return MetaModelFactory.create_model(
            model_type=model_type,
            input_dim=input_dim,
            output_dim=output_dim,
            use_meta_params=True,
            meta_learnable=True
        )
"""
Model Registry for NLP Sentiment Analysis

Enterprise model versioning and registry system with enterprise patterns
for managing model lifecycle, versions, and deployment.

Author: ML-Framework Team
"""

import json
import sqlite3
import hashlib
import shutil
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import uuid
import pickle
import torch
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata structure"""
    id: str
    name: str
    version: str
    model_type: str
    framework: str  # pytorch, transformers, sklearn, etc.
    created_at: datetime
    updated_at: datetime
    
    # Model details
    architecture: str
    parameters_count: int
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    
    # Training details
    training_dataset: Optional[str] = None
    training_metrics: Optional[Dict[str, float]] = None
    validation_metrics: Optional[Dict[str, float]] = None
    
    # Deployment details
    status: str = "registered"  # registered, validated, deployed, deprecated
    environment: str = "development"  # development, staging, production
    deployment_config: Optional[Dict[str, Any]] = None
    
    # File information
    model_path: Optional[str] = None
    model_size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    
    # Metadata
    description: Optional[str] = None
    tags: List[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    
    # Performance metrics
    inference_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    accuracy: Optional[float] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary"""
        # Convert ISO strings to datetime objects
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        
        return cls(**data)


class ModelStorage:
    """Model file storage backend"""
    
    def __init__(self, storage_path: Union[str, Path]):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def store_model(self, model_id: str, model_data: Union[bytes, torch.nn.Module, Any]) -> Tuple[str, int, str]:
        """
        Store model data and return path, size, and checksum
        
        Args:
            model_id: Unique model identifier
            model_data: Model data to store
            
        Returns:
            Tuple of (file_path, file_size, checksum)
        """
        
        model_dir = self.storage_path / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(model_data, torch.nn.Module):
            # PyTorch model
            model_path = model_dir / "model.pth"
            torch.save(model_data.state_dict(), model_path)
        elif isinstance(model_data, bytes):
            # Raw bytes
            model_path = model_dir / "model.bin"
            with open(model_path, "wb") as f:
                f.write(model_data)
        else:
            # Generic Python object
            model_path = model_dir / "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)
        
        # Calculate file size and checksum
        file_size = model_path.stat().st_size
        
        # Calculate MD5 checksum
        hash_md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        checksum = hash_md5.hexdigest()
        
        return str(model_path), file_size, checksum
    
    def load_model(self, model_path: str, model_type: str = "pytorch") -> Any:
        """Load model from storage"""
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if model_type == "pytorch":
            return torch.load(model_path, map_location="cpu")
        elif model_path.suffix == ".pkl":
            with open(model_path, "rb") as f:
                return pickle.load(f)
        elif model_path.suffix == ".bin":
            with open(model_path, "rb") as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def delete_model(self, model_path: str) -> bool:
        """Delete model from storage"""
        
        model_path = Path(model_path)
        
        try:
            if model_path.is_file():
                model_path.unlink()
            elif model_path.is_dir():
                shutil.rmtree(model_path)
            return True
        except Exception as e:
            logger.error(f"Error deleting model {model_path}: {e}")
            return False
    
    def copy_model(self, source_path: str, dest_id: str) -> Tuple[str, int, str]:
        """Copy model to new location"""
        
        source_path = Path(source_path)
        dest_dir = self.storage_path / dest_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = dest_dir / source_path.name
        shutil.copy2(source_path, dest_path)
        
        # Calculate new file info
        file_size = dest_path.stat().st_size
        
        hash_md5 = hashlib.md5()
        with open(dest_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        checksum = hash_md5.hexdigest()
        
        return str(dest_path), file_size, checksum


class ModelRegistry:
    """
    Enterprise Model Registry with enterprise integration
    
    Features:
    - Model versioning and lifecycle management
    - Metadata storage and retrieval
    - Model validation and testing
    - Deployment tracking
    - Performance monitoring
    - A/B testing support
    - Model comparison and rollback
    """
    
    def __init__(
        self,
        registry_path: Union[str, Path] = "model_registry",
        storage_backend: Optional[ModelStorage] = None,
    ):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Database for metadata
        self.db_path = self.registry_path / "models.db"
        self._init_database()
        
        # Storage backend
        if storage_backend is None:
            storage_path = self.registry_path / "models"
            storage_backend = ModelStorage(storage_path)
        
        self.storage = storage_backend
        
        logger.info(f"Initialized Model Registry at {self.registry_path}")
    
    def _init_database(self):
        """Initialize SQLite database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    architecture TEXT,
                    parameters_count INTEGER,
                    input_shape TEXT,
                    output_shape TEXT,
                    training_dataset TEXT,
                    training_metrics TEXT,
                    validation_metrics TEXT,
                    status TEXT DEFAULT 'registered',
                    environment TEXT DEFAULT 'development',
                    deployment_config TEXT,
                    model_path TEXT,
                    model_size_bytes INTEGER,
                    checksum TEXT,
                    description TEXT,
                    tags TEXT,
                    author TEXT,
                    license TEXT,
                    inference_time_ms REAL,
                    memory_usage_mb REAL,
                    accuracy REAL,
                    UNIQUE(name, version)
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_name ON models(name)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON models(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_environment ON models(environment)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON models(created_at)')
    
    def register_model(
        self,
        name: str,
        model_data: Union[torch.nn.Module, bytes, Any],
        version: Optional[str] = None,
        model_type: str = "transformer",
        framework: str = "pytorch",
        architecture: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        training_metrics: Optional[Dict[str, float]] = None,
        validation_metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> ModelMetadata:
        """
        Register a new model in the registry
        
        Args:
            name: Model name
            model_data: Model object or data
            version: Model version (auto-generated if None)
            model_type: Type of model
            framework: ML framework used
            architecture: Model architecture description
            description: Model description
            tags: Model tags
            training_metrics: Training performance metrics
            validation_metrics: Validation performance metrics
            **kwargs: Additional metadata
            
        Returns:
            ModelMetadata object
        """
        
        # Generate version if not provided
        if version is None:
            existing_versions = self.list_versions(name)
            if existing_versions:
                # Increment latest version
                latest_version = max(existing_versions, key=lambda v: v.split('.'))
                major, minor, patch = map(int, latest_version.split('.'))
                version = f"{major}.{minor}.{patch + 1}"
            else:
                version = "1.0.0"
        
        # Generate unique ID
        model_id = str(uuid.uuid4())
        
        # Store model data
        model_path, file_size, checksum = self.storage.store_model(model_id, model_data)
        
        # Calculate parameters count for PyTorch models
        parameters_count = 0
        if isinstance(model_data, torch.nn.Module):
            parameters_count = sum(p.numel() for p in model_data.parameters())
        
        # Create metadata
        now = datetime.now(timezone.utc)
        metadata = ModelMetadata(
            id=model_id,
            name=name,
            version=version,
            model_type=model_type,
            framework=framework,
            created_at=now,
            updated_at=now,
            architecture=architecture,
            parameters_count=parameters_count,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            model_path=model_path,
            model_size_bytes=file_size,
            checksum=checksum,
            description=description,
            tags=tags or [],
            **kwargs
        )
        
        # Store metadata in database
        self._store_metadata(metadata)
        
        logger.info(f"Registered model {name} v{version} with ID {model_id}")
        return metadata
    
    def _store_metadata(self, metadata: ModelMetadata):
        """Store metadata in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Convert complex fields to JSON
            training_metrics = json.dumps(metadata.training_metrics) if metadata.training_metrics else None
            validation_metrics = json.dumps(metadata.validation_metrics) if metadata.validation_metrics else None
            deployment_config = json.dumps(metadata.deployment_config) if metadata.deployment_config else None
            input_shape = json.dumps(metadata.input_shape) if metadata.input_shape else None
            output_shape = json.dumps(metadata.output_shape) if metadata.output_shape else None
            tags = json.dumps(metadata.tags) if metadata.tags else None
            
            conn.execute('''
                INSERT OR REPLACE INTO models (
                    id, name, version, model_type, framework, created_at, updated_at,
                    architecture, parameters_count, input_shape, output_shape,
                    training_dataset, training_metrics, validation_metrics,
                    status, environment, deployment_config,
                    model_path, model_size_bytes, checksum,
                    description, tags, author, license,
                    inference_time_ms, memory_usage_mb, accuracy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.id, metadata.name, metadata.version, metadata.model_type,
                metadata.framework, metadata.created_at.isoformat(), metadata.updated_at.isoformat(),
                metadata.architecture, metadata.parameters_count, input_shape, output_shape,
                metadata.training_dataset, training_metrics, validation_metrics,
                metadata.status, metadata.environment, deployment_config,
                metadata.model_path, metadata.model_size_bytes, metadata.checksum,
                metadata.description, tags, metadata.author, metadata.license,
                metadata.inference_time_ms, metadata.memory_usage_mb, metadata.accuracy
            ))
    
    def _load_metadata(self, row: Tuple) -> ModelMetadata:
        """Load metadata from database row"""
        
        (id, name, version, model_type, framework, created_at, updated_at,
         architecture, parameters_count, input_shape, output_shape,
         training_dataset, training_metrics, validation_metrics,
         status, environment, deployment_config,
         model_path, model_size_bytes, checksum,
         description, tags, author, license,
         inference_time_ms, memory_usage_mb, accuracy) = row
        
        # Parse JSON fields
        training_metrics = json.loads(training_metrics) if training_metrics else None
        validation_metrics = json.loads(validation_metrics) if validation_metrics else None
        deployment_config = json.loads(deployment_config) if deployment_config else None
        input_shape = tuple(json.loads(input_shape)) if input_shape else None
        output_shape = tuple(json.loads(output_shape)) if output_shape else None
        tags = json.loads(tags) if tags else []
        
        return ModelMetadata(
            id=id,
            name=name,
            version=version,
            model_type=model_type,
            framework=framework,
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(updated_at),
            architecture=architecture,
            parameters_count=parameters_count,
            input_shape=input_shape,
            output_shape=output_shape,
            training_dataset=training_dataset,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            status=status,
            environment=environment,
            deployment_config=deployment_config,
            model_path=model_path,
            model_size_bytes=model_size_bytes,
            checksum=checksum,
            description=description,
            tags=tags,
            author=author,
            license=license,
            inference_time_ms=inference_time_ms,
            memory_usage_mb=memory_usage_mb,
            accuracy=accuracy,
        )
    
    def get_model(self, name: str, version: Optional[str] = None) -> Optional[ModelMetadata]:
        """
        Get model metadata
        
        Args:
            name: Model name
            version: Model version (latest if None)
            
        Returns:
            ModelMetadata or None if not found
        """
        
        with sqlite3.connect(self.db_path) as conn:
            if version:
                cursor = conn.execute(
                    'SELECT * FROM models WHERE name = ? AND version = ?',
                    (name, version)
                )
            else:
                # Get latest version
                cursor = conn.execute(
                    'SELECT * FROM models WHERE name = ? ORDER BY created_at DESC LIMIT 1',
                    (name,)
                )
            
            row = cursor.fetchone()
            if row:
                return self._load_metadata(row)
        
        return None
    
    def load_model(self, name: str, version: Optional[str] = None) -> Any:
        """Load model from registry"""
        
        metadata = self.get_model(name, version)
        if not metadata:
            raise ValueError(f"Model {name} not found")
        
        if not metadata.model_path:
            raise ValueError(f"Model {name} v{metadata.version} has no stored data")
        
        return self.storage.load_model(metadata.model_path, metadata.framework)
    
    def list_models(
        self,
        status: Optional[str] = None,
        environment: Optional[str] = None,
        model_type: Optional[str] = None,
        tag: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ModelMetadata]:
        """List models with optional filtering"""
        
        query = 'SELECT * FROM models WHERE 1=1'
        params = []
        
        if status:
            query += ' AND status = ?'
            params.append(status)
        
        if environment:
            query += ' AND environment = ?'
            params.append(environment)
        
        if model_type:
            query += ' AND model_type = ?'
            params.append(model_type)
        
        if tag:
            query += ' AND tags LIKE ?'
            params.append(f'%"{tag}"%')
        
        query += ' ORDER BY created_at DESC'
        
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        
        models = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                models.append(self._load_metadata(row))
        
        return models
    
    def list_versions(self, name: str) -> List[str]:
        """List all versions of a model"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT version FROM models WHERE name = ? ORDER BY created_at DESC',
                (name,)
            )
            return [row[0] for row in cursor.fetchall()]
    
    def update_model_status(self, name: str, version: str, status: str) -> bool:
        """Update model status"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'UPDATE models SET status = ?, updated_at = ? WHERE name = ? AND version = ?',
                (status, datetime.now(timezone.utc).isoformat(), name, version)
            )
            return cursor.rowcount > 0
    
    def update_model_metrics(
        self,
        name: str,
        version: str,
        inference_time_ms: Optional[float] = None,
        memory_usage_mb: Optional[float] = None,
        accuracy: Optional[float] = None,
        **metrics
    ) -> bool:
        """Update model performance metrics"""
        
        updates = []
        params = []
        
        if inference_time_ms is not None:
            updates.append('inference_time_ms = ?')
            params.append(inference_time_ms)
        
        if memory_usage_mb is not None:
            updates.append('memory_usage_mb = ?')
            params.append(memory_usage_mb)
        
        if accuracy is not None:
            updates.append('accuracy = ?')
            params.append(accuracy)
        
        if not updates:
            return False
        
        updates.append('updated_at = ?')
        params.append(datetime.now(timezone.utc).isoformat())
        params.extend([name, version])
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f'UPDATE models SET {", ".join(updates)} WHERE name = ? AND version = ?',
                params
            )
            return cursor.rowcount > 0
    
    def delete_model(self, name: str, version: str) -> bool:
        """Delete model from registry"""
        
        metadata = self.get_model(name, version)
        if not metadata:
            return False
        
        # Delete model file
        if metadata.model_path:
            self.storage.delete_model(metadata.model_path)
        
        # Delete metadata
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'DELETE FROM models WHERE name = ? AND version = ?',
                (name, version)
            )
            return cursor.rowcount > 0
    
    def copy_model(self, source_name: str, source_version: str, dest_name: str, dest_version: str) -> Optional[ModelMetadata]:
        """Copy model to new name/version"""
        
        source_metadata = self.get_model(source_name, source_version)
        if not source_metadata:
            return None
        
        # Generate new ID
        new_id = str(uuid.uuid4())
        
        # Copy model file
        new_path, file_size, checksum = self.storage.copy_model(source_metadata.model_path, new_id)
        
        # Create new metadata
        now = datetime.now(timezone.utc)
        new_metadata = ModelMetadata(
            id=new_id,
            name=dest_name,
            version=dest_version,
            model_type=source_metadata.model_type,
            framework=source_metadata.framework,
            created_at=now,
            updated_at=now,
            architecture=source_metadata.architecture,
            parameters_count=source_metadata.parameters_count,
            input_shape=source_metadata.input_shape,
            output_shape=source_metadata.output_shape,
            training_dataset=source_metadata.training_dataset,
            training_metrics=source_metadata.training_metrics,
            validation_metrics=source_metadata.validation_metrics,
            status="registered",
            environment=source_metadata.environment,
            model_path=new_path,
            model_size_bytes=file_size,
            checksum=checksum,
            description=f"Copy of {source_name} v{source_version}",
            tags=source_metadata.tags.copy(),
            author=source_metadata.author,
            license=source_metadata.license,
        )
        
        # Store new metadata
        self._store_metadata(new_metadata)
        
        logger.info(f"Copied model {source_name} v{source_version} to {dest_name} v{dest_version}")
        return new_metadata
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Total models
            total_models = conn.execute('SELECT COUNT(*) FROM models').fetchone()[0]
            
            # Models by status
            status_cursor = conn.execute('SELECT status, COUNT(*) FROM models GROUP BY status')
            status_counts = dict(status_cursor.fetchall())
            
            # Models by environment
            env_cursor = conn.execute('SELECT environment, COUNT(*) FROM models GROUP BY environment')
            env_counts = dict(env_cursor.fetchall())
            
            # Models by type
            type_cursor = conn.execute('SELECT model_type, COUNT(*) FROM models GROUP BY model_type')
            type_counts = dict(type_cursor.fetchall())
            
            # Total storage size
            size_result = conn.execute('SELECT SUM(model_size_bytes) FROM models WHERE model_size_bytes IS NOT NULL').fetchone()
            total_size = size_result[0] if size_result[0] else 0
        
        return {
            "total_models": total_models,
            "status_counts": status_counts,
            "environment_counts": env_counts,
            "type_counts": type_counts,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
        }
    
    def export_registry(self, export_path: Union[str, Path]):
        """Export registry to JSON file"""
        
        export_path = Path(export_path)
        models = self.list_models()
        
        export_data = {
            "metadata": {
                "export_date": datetime.now(timezone.utc).isoformat(),
                "total_models": len(models),
            },
            "models": [model.to_dict() for model in models],
            "stats": self.get_registry_stats(),
        }
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported registry to {export_path}")
    
    @contextmanager
    def transaction(self):
        """Database transaction context manager"""
        
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# Global registry instance
_global_registry: Optional[ModelRegistry] = None


def get_registry(**kwargs) -> ModelRegistry:
    """Get global model registry instance"""
    global _global_registry
    
    if _global_registry is None:
        _global_registry = ModelRegistry(**kwargs)
    
    return _global_registry


def set_registry(registry: ModelRegistry):
    """Set global model registry instance"""
    global _global_registry
    _global_registry = registry
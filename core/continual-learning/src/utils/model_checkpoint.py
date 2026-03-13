"""
Model Checkpoint Manager for Continual Learning in Crypto Trading Bot v5.0

Enterprise-grade system management checkpoint' models
with integration for production deployment.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import pickle
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import gzip
import threading
from concurrent.futures import ThreadPoolExecutor


class CheckpointType(Enum):
    """Types checkpoint'"""
    MODEL_STATE = "model_state" # State model
    OPTIMIZER_STATE = "optimizer_state" # State optimizer
    TRAINING_STATE = "training_state" # State training
    FULL_SYSTEM = "full_system" # Full state system
    TASK_SPECIFIC = "task_specific" # for tasks
    EMERGENCY_BACKUP = "emergency_backup" #


class CompressionType(Enum):
    """Types compression"""
    NONE = "none"
    GZIP = "gzip"
    TORCH_JIT = "torch_jit"


@dataclass
class CheckpointMetadata:
    """Metadata checkpoint'"""
    # Main information
    checkpoint_id: str
    checkpoint_type: CheckpointType
    created_at: datetime
    file_path: str
    
    # Information model
    model_architecture: str
    model_parameters_count: int
    model_size_bytes: int
    
    # Information task
    task_id: Optional[int] = None
    task_name: Optional[str] = None
    market_regime: Optional[str] = None
    
    # Metrics performance
    performance_metrics: Dict[str, float] = None
    validation_accuracy: Optional[float] = None
    training_loss: Optional[float] = None
    
    # Technical
    pytorch_version: str = ""
    compression_type: CompressionType = CompressionType.NONE
    file_hash: str = ""
    
    # enterprise
    environment: str = "development"
    version_tag: str = "1.0.0"
    git_commit_hash: Optional[str] = None
    deployment_target: str = "local"
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}


class ModelCheckpointManager:
    """
    Manager checkpoint' for continual training
    
    enterprise Features:
    - Automatic checkpoint lifecycle management
    - Compression and deduplication
    - Performance-based checkpoint retention
    - Distributed checkpoint storage
    - Rollback and recovery mechanisms
    - Production deployment support
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 10,
        auto_cleanup: bool = True,
        compression: CompressionType = CompressionType.GZIP
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.auto_cleanup = auto_cleanup
        self.compression = compression
        
        # Create directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.checkpoint_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger("ModelCheckpointManager")
        
        # Index checkpoint'
        self.checkpoint_index: Dict[str, CheckpointMetadata] = {}
        self.load_checkpoint_index()
        
        # enterprise settings
        self.enable_deduplication = True
        self.enable_integrity_checks = True
        self.backup_retention_days = 30
        self.performance_based_retention = True
        
        # Threading for
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.Lock()
    
    def save_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        checkpoint_name: Optional[str] = None,
        checkpoint_type: CheckpointType = CheckpointType.MODEL_STATE,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save checkpoint'
        
        Args:
            checkpoint_data: Data for saving
            checkpoint_name: Name checkpoint'
            checkpoint_type: Type checkpoint'
            metadata: Additional
            
        Returns:
            Path to saved checkpoint'
        """
        # Generation if not
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"{checkpoint_type.value}_{timestamp}"
        
        checkpoint_id = f"{checkpoint_name}_{int(datetime.now().timestamp())}"
        
        # Determine paths to file
        file_extension = self._get_file_extension()
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}{file_extension}"
        
        try:
            # Save data
            self._save_data_to_file(checkpoint_data, checkpoint_file)
            
            # Computation hash' file
            file_hash = self._calculate_file_hash(checkpoint_file)
            
            # Check on
            if self.enable_deduplication:
                existing_checkpoint = self._find_duplicate_checkpoint(file_hash)
                if existing_checkpoint:
                    # Removing and use
                    checkpoint_file.unlink()
                    self.logger.info(f"Duplicate checkpoint detected, using existing: {existing_checkpoint}")
                    return str(self.checkpoint_index[existing_checkpoint].file_path)
            
            # Create metadata
            checkpoint_metadata = self._create_checkpoint_metadata(
                checkpoint_id=checkpoint_id,
                checkpoint_type=checkpoint_type,
                file_path=str(checkpoint_file),
                checkpoint_data=checkpoint_data,
                file_hash=file_hash,
                additional_metadata=metadata
            )
            
            # Save metadata
            self._save_checkpoint_metadata(checkpoint_metadata)
            
            # Update index
            with self._lock:
                self.checkpoint_index[checkpoint_id] = checkpoint_metadata
            
            #
            if self.auto_cleanup:
                self.cleanup_old_checkpoints()
            
            self.logger.info(f"Checkpoint saved: {checkpoint_id}")
            return str(checkpoint_file)
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint {checkpoint_name}: {e}")
            # Cleanup files
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            raise
    
    def load_checkpoint(self, checkpoint_identifier: str) -> Dict[str, Any]:
        """
        Load checkpoint'
        
        Args:
            checkpoint_identifier: ID checkpoint' or path to file
            
        Returns:
             data checkpoint'
        """
        # Search checkpoint'
        checkpoint_metadata = self._find_checkpoint(checkpoint_identifier)
        if not checkpoint_metadata:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_identifier}")
        
        checkpoint_file = Path(checkpoint_metadata.file_path)
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        try:
            # Check integrity
            if self.enable_integrity_checks:
                if not self._verify_checkpoint_integrity(checkpoint_metadata):
                    raise ValueError(f"Checkpoint integrity check failed: {checkpoint_identifier}")
            
            # Load data
            checkpoint_data = self._load_data_from_file(checkpoint_file)
            
            self.logger.info(f"Checkpoint loaded: {checkpoint_identifier}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint {checkpoint_identifier}: {e}")
            raise
    
    def _get_file_extension(self) -> str:
        """Get file on basis type compression"""
        if self.compression == CompressionType.GZIP:
            return ".pt.gz"
        elif self.compression == CompressionType.TORCH_JIT:
            return ".jit"
        else:
            return ".pt"
    
    def _save_data_to_file(self, data: Dict[str, Any], file_path: Path) -> None:
        """Save data in file with consideration compression"""
        if self.compression == CompressionType.GZIP:
            with gzip.open(file_path, 'wb') as f:
                torch.save(data, f)
        elif self.compression == CompressionType.TORCH_JIT:
            # For TorchScript serialization
            if 'model_state_dict' in data and hasattr(data.get('model'), 'save'):
                # If model supports TorchScript
                model = data.get('model')
                if model is not None:
                    torch.jit.save(torch.jit.script(model), file_path)
                else:
                    # Fallback to
                    torch.save(data, file_path)
            else:
                torch.save(data, file_path)
        else:
            torch.save(data, file_path)
    
    def _load_data_from_file(self, file_path: Path) -> Dict[str, Any]:
        """Load data from file with consideration compression"""
        if self.compression == CompressionType.GZIP and file_path.suffix == '.gz':
            with gzip.open(file_path, 'rb') as f:
                return torch.load(f, map_location='cpu')
        elif self.compression == CompressionType.TORCH_JIT and file_path.suffix == '.jit':
            # Load TorchScript model
            return {'model': torch.jit.load(file_path)}
        else:
            return torch.load(file_path, map_location='cpu')
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Computation SHA-256 hash' file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _find_duplicate_checkpoint(self, file_hash: str) -> Optional[str]:
        """Search duplicates checkpoint' by hash'"""
        for checkpoint_id, metadata in self.checkpoint_index.items():
            if metadata.file_hash == file_hash:
                return checkpoint_id
        return None
    
    def _create_checkpoint_metadata(
        self,
        checkpoint_id: str,
        checkpoint_type: CheckpointType,
        file_path: str,
        checkpoint_data: Dict[str, Any],
        file_hash: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> CheckpointMetadata:
        """Create metadata checkpoint'"""
        
        # Get size file
        file_size = Path(file_path).stat().st_size
        
        # Analysis architecture model
        model_info = self._analyze_model_architecture(checkpoint_data)
        
        # Extraction metrics performance
        performance_metrics = checkpoint_data.get('performance_metrics', {})
        
        # Additional information
        extra_metadata = additional_metadata or {}
        
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            created_at=datetime.now(),
            file_path=file_path,
            model_architecture=model_info.get('architecture', 'unknown'),
            model_parameters_count=model_info.get('parameters', 0),
            model_size_bytes=file_size,
            task_id=extra_metadata.get('task_id'),
            task_name=extra_metadata.get('task_name'),
            market_regime=extra_metadata.get('market_regime'),
            performance_metrics=performance_metrics,
            validation_accuracy=performance_metrics.get('accuracy'),
            training_loss=performance_metrics.get('loss'),
            pytorch_version=torch.__version__,
            compression_type=self.compression,
            file_hash=file_hash,
            environment=extra_metadata.get('environment', 'development'),
            version_tag=extra_metadata.get('version_tag', '1.0.0'),
            git_commit_hash=extra_metadata.get('git_commit_hash'),
            deployment_target=extra_metadata.get('deployment_target', 'local')
        )
        
        return metadata
    
    def _analyze_model_architecture(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analysis architecture model from data checkpoint'"""
        model_info = {
            'architecture': 'unknown',
            'parameters': 0
        }
        
        # Attempt extract model
        if 'model_state_dict' in checkpoint_data:
            state_dict = checkpoint_data['model_state_dict']
            model_info['parameters'] = sum(
                p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor)
            )
            
            # Simple determination architecture on basis layers
            layer_names = list(state_dict.keys())
            if any('conv' in name.lower() for name in layer_names):
                model_info['architecture'] = 'CNN'
            elif any('lstm' in name.lower() or 'gru' in name.lower() for name in layer_names):
                model_info['architecture'] = 'RNN'
            elif any('transformer' in name.lower() or 'attention' in name.lower() for name in layer_names):
                model_info['architecture'] = 'Transformer'
            else:
                model_info['architecture'] = 'MLP'
        
        return model_info
    
    def _save_checkpoint_metadata(self, metadata: CheckpointMetadata) -> None:
        """Save metadata checkpoint'"""
        metadata_file = self.metadata_dir / f"{metadata.checkpoint_id}.json"
        
        with open(metadata_file, 'w') as f:
            # Convert in dictionary with datetime
            metadata_dict = asdict(metadata)
            metadata_dict['created_at'] = metadata.created_at.isoformat()
            metadata_dict['checkpoint_type'] = metadata.checkpoint_type.value
            metadata_dict['compression_type'] = metadata.compression_type.value
            
            json.dump(metadata_dict, f, indent=2)
    
    def _find_checkpoint(self, identifier: str) -> Optional[CheckpointMetadata]:
        """Search checkpoint' by ID or paths"""
        # Search by ID
        if identifier in self.checkpoint_index:
            return self.checkpoint_index[identifier]
        
        # Search by paths to file
        for metadata in self.checkpoint_index.values():
            if metadata.file_path == identifier or Path(metadata.file_path).name == identifier:
                return metadata
        
        return None
    
    def _verify_checkpoint_integrity(self, metadata: CheckpointMetadata) -> bool:
        """Check integrity checkpoint'"""
        checkpoint_file = Path(metadata.file_path)
        
        if not checkpoint_file.exists():
            return False
        
        # Check hash'
        current_hash = self._calculate_file_hash(checkpoint_file)
        return current_hash == metadata.file_hash
    
    def load_checkpoint_index(self) -> None:
        """Load index checkpoint' from files metadata"""
        self.checkpoint_index.clear()
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                
                # Convert back in object
                metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                metadata_dict['checkpoint_type'] = CheckpointType(metadata_dict['checkpoint_type'])
                metadata_dict['compression_type'] = CompressionType(metadata_dict['compression_type'])
                
                metadata = CheckpointMetadata(**metadata_dict)
                self.checkpoint_index[metadata.checkpoint_id] = metadata
                
            except Exception as e:
                self.logger.warning(f"Error loading metadata from {metadata_file}: {e}")
        
        self.logger.info(f"Loaded {len(self.checkpoint_index)} checkpoint metadata entries")
    
    def list_checkpoints(
        self,
        checkpoint_type: Optional[CheckpointType] = None,
        task_id: Optional[int] = None,
        sort_by: str = "created_at"
    ) -> List[CheckpointMetadata]:
        """
        Get list checkpoint' with
        
        Args:
            checkpoint_type: Filter by checkpoint'
            task_id: Filter by ID tasks
            sort_by: for
            
        Returns:
              checkpoint'
        """
        checkpoints = list(self.checkpoint_index.values())
        
        #
        if checkpoint_type:
            checkpoints = [cp for cp in checkpoints if cp.checkpoint_type == checkpoint_type]
        
        if task_id is not None:
            checkpoints = [cp for cp in checkpoints if cp.task_id == task_id]
        
        # Sort
        if sort_by in ['created_at', 'model_size_bytes', 'model_parameters_count']:
            reverse = sort_by != 'created_at' # New first for
            checkpoints.sort(key=lambda cp: getattr(cp, sort_by), reverse=reverse)
        elif sort_by == 'validation_accuracy':
            checkpoints.sort(key=lambda cp: cp.validation_accuracy or 0, reverse=True)
        
        return checkpoints
    
    def cleanup_old_checkpoints(self) -> int:
        """
        Cleanup old checkpoint'
        
        Returns:
            Number removed checkpoint'
        """
        if len(self.checkpoint_index) <= self.max_checkpoints:
            return 0
        
        # Sort checkpoint' by saving
        checkpoints = list(self.checkpoint_index.values())
        
        if self.performance_based_retention:
            # Saving best checkpoint'
            checkpoints.sort(key=self._calculate_checkpoint_priority, reverse=True)
        else:
            # Saving checkpoint'
            checkpoints.sort(key=lambda cp: cp.created_at, reverse=True)
        
        # Checkpoint' for removal
        checkpoints_to_remove = checkpoints[self.max_checkpoints:]
        removed_count = 0
        
        for checkpoint in checkpoints_to_remove:
            try:
                self.delete_checkpoint(checkpoint.checkpoint_id)
                removed_count += 1
            except Exception as e:
                self.logger.error(f"Error removing checkpoint {checkpoint.checkpoint_id}: {e}")
        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} old checkpoints")
        
        return removed_count
    
    def _calculate_checkpoint_priority(self, checkpoint: CheckpointMetadata) -> float:
        """
        Calculation checkpoint' for retention policy
        
        Args:
            checkpoint: Metadata checkpoint'
            
        Returns:
            Priority (more = above priority)
        """
        priority = 0.0
        
        # Priority on basis performance
        if checkpoint.validation_accuracy:
            priority += checkpoint.validation_accuracy * 100
        
        # Priority on basis type
        type_priorities = {
            CheckpointType.EMERGENCY_BACKUP: 50,
            CheckpointType.FULL_SYSTEM: 40,
            CheckpointType.TASK_SPECIFIC: 30,
            CheckpointType.MODEL_STATE: 20,
            CheckpointType.TRAINING_STATE: 10,
            CheckpointType.OPTIMIZER_STATE: 5
        }
        priority += type_priorities.get(checkpoint.checkpoint_type, 0)
        
        # Priority on basis recency (decreasing with time)
        days_old = (datetime.now() - checkpoint.created_at).days
        freshness_bonus = max(0, 30 - days_old) # up to 30 days
        priority += freshness_bonus
        
        return priority
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Remove checkpoint'
        
        Args:
            checkpoint_id: ID checkpoint'
            
        Returns:
            True if checkpoint
        """
        if checkpoint_id not in self.checkpoint_index:
            return False
        
        metadata = self.checkpoint_index[checkpoint_id]
        
        try:
            # Remove file checkpoint'
            checkpoint_file = Path(metadata.file_path)
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            
            # Remove file metadata
            metadata_file = self.metadata_dir / f"{checkpoint_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            # Remove from index
            with self._lock:
                del self.checkpoint_index[checkpoint_id]
            
            self.logger.info(f"Deleted checkpoint: {checkpoint_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting checkpoint {checkpoint_id}: {e}")
            return False
    
    def create_backup(self, backup_dir: Union[str, Path]) -> str:
        """
        Create backup copies all checkpoint'
        
        Args:
            backup_dir: Directory for backup copies
            
        Returns:
            Path to created backup copies
        """
        backup_dir = Path(backup_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"checkpoint_backup_{timestamp}"
        
        try:
            # entire directory checkpoint'
            shutil.copytree(self.checkpoint_dir, backup_path)
            
            self.logger.info(f"Created backup at: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            raise
    
    def restore_from_backup(self, backup_path: Union[str, Path]) -> bool:
        """
        Restore from backup copies
        
        Args:
            backup_path: Path to backup copies
            
        Returns:
            True if restoration successfully
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            self.logger.error(f"Backup path does not exist: {backup_path}")
            return False
        
        try:
            # Create backup' current state
            current_backup = self.create_backup(self.checkpoint_dir.parent / "backup_before_restore")
            
            # Remove current directory
            shutil.rmtree(self.checkpoint_dir)
            
            # Restore from backup'
            shutil.copytree(backup_path, self.checkpoint_dir)
            
            # index
            self.load_checkpoint_index()
            
            self.logger.info(f"Restored from backup: {backup_path}")
            self.logger.info(f"Previous state backed up to: {current_backup}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error restoring from backup: {e}")
            return False
    
    def get_checkpoint_statistics(self) -> Dict[str, Any]:
        """
        Get statistics checkpoint'
        
        Returns:
            Statistics checkpoint'
        """
        if not self.checkpoint_index:
            return {"total_checkpoints": 0}
        
        checkpoints = list(self.checkpoint_index.values())
        
        # Grouping by
        by_type = {}
        for cp in checkpoints:
            cp_type = cp.checkpoint_type.value
            by_type[cp_type] = by_type.get(cp_type, 0) + 1
        
        # Sizes files
        total_size = sum(cp.model_size_bytes for cp in checkpoints)
        avg_size = total_size / len(checkpoints) if checkpoints else 0
        
        # Performance
        accuracies = [cp.validation_accuracy for cp in checkpoints if cp.validation_accuracy]
        
        stats = {
            "total_checkpoints": len(checkpoints),
            "by_type": by_type,
            "total_size_mb": total_size / (1024 * 1024),
            "average_size_mb": avg_size / (1024 * 1024),
            "oldest_checkpoint": min(cp.created_at for cp in checkpoints).isoformat(),
            "newest_checkpoint": max(cp.created_at for cp in checkpoints).isoformat(),
            "compression_type": self.compression.value
        }
        
        if accuracies:
            stats.update({
                "best_accuracy": max(accuracies),
                "average_accuracy": sum(accuracies) / len(accuracies),
                "accuracy_std": (sum((acc - stats["average_accuracy"]) ** 2 for acc in accuracies) / len(accuracies)) ** 0.5
            })
        
        return stats
    
    def __del__(self):
        """Cleanup at object"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
    
    def __repr__(self) -> str:
        return (
            f"ModelCheckpointManager("
            f"checkpoints={len(self.checkpoint_index)}, "
            f"dir='{self.checkpoint_dir}', "
            f"max={self.max_checkpoints})"
        )
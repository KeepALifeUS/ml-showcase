"""
Main AutoML Pipeline for Crypto Trading Bot v5.0
Orchestrates the complete machine learning workflow with enterprise patterns
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import pickle
import json
import time
from pathlib import Path
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt
import seaborn as sns

from ..feature_engineering.auto_feature_generator import AutoFeatureGenerator, FeatureGenerationResult
from ..feature_engineering.feature_selector import AdvancedFeatureSelector, FeatureSelectionResult
from ..optimization.bayesian_optimizer import CryptoMLHyperparameterOptimizer, OptimizationResult
from ..model_selection.model_selector import ModelSelector, ModelSelectionResult
from ..model_selection.ensemble_builder import EnsembleBuilder, EnsembleResult
from ..evaluation.model_evaluator import ModelEvaluator, EvaluationResult
from ..utils.config_manager import AutoMLConfig
from ..utils.data_preprocessor import DataPreprocessor


class PipelineStage(Enum):
    """Stages AutoML pipeline"""
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_GENERATION = "feature_generation"
    FEATURE_SELECTION = "feature_selection"
    MODEL_SELECTION = "model_selection"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    ENSEMBLE_BUILDING = "ensemble_building"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_DEPLOYMENT = "model_deployment"


@dataclass
class PipelineResult:
    """Result execution AutoML pipeline"""
    best_model: Any
    best_model_name: str
    best_score: float
    feature_generation_result: Optional[FeatureGenerationResult]
    feature_selection_result: Optional[FeatureSelectionResult]
    optimization_results: Dict[str, OptimizationResult]
    ensemble_result: Optional[EnsembleResult]
    evaluation_result: EvaluationResult
    pipeline_metadata: Dict[str, Any]
    total_time: float
    stages_completed: List[str]


class AutoMLPipeline:
    """
    Main class AutoML pipeline for crypto trading
    Implements enterprise patterns for scalable ML systems
    """
    
    def __init__(self, config: Optional[AutoMLConfig] = None, output_dir: Optional[str] = None):
        self.config = config or AutoMLConfig()
        self.output_dir = Path(output_dir or "automl_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.preprocessor = DataPreprocessor(self.config)
        self.feature_generator = AutoFeatureGenerator(self.config)
        self.feature_selector = AdvancedFeatureSelector(self.config)
        self.hyperparameter_optimizer = CryptoMLHyperparameterOptimizer(self.config)
        self.model_selector = ModelSelector(self.config)
        self.ensemble_builder = EnsembleBuilder(self.config)
        self.evaluator = ModelEvaluator(self.config)
        
        # State pipeline
        self.pipeline_state = {}
        self.console = Console()
        
        logger.info("🚀 AutoML Pipeline initialized")
    
    def run(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        time_series_split: bool = True,
        stages: Optional[List[str]] = None
    ) -> PipelineResult:
        """
        Launch full AutoML pipeline
        
        Args:
            data: Source data
            target_column: Name columns with target variable
            test_size: Size test set
            validation_size: Size validation set
            time_series_split: Use temporal splitting
            stages: List stages for execution (by default all)
        """
        start_time = time.time()
        
        self.console.print(
            Panel.fit(
                "🤖 [bold blue]CRYPTO TRADING AUTOML PIPELINE v5.0[/bold blue] 🚀\n"
                f"📊 Data: {len(data)} records, {len(data.columns)} features\n"
                f"🎯 Target variable: {target_column}",
                title="Launch AutoML Pipeline"
            )
        )
        
        if stages is None:
            stages = [stage.value for stage in PipelineStage]
        
        stages_completed = []
        pipeline_metadata = {
            'start_time': start_time,
            'data_shape': data.shape,
            'target_column': target_column,
            'config': self.config.dict() if hasattr(self.config, 'dict') else str(self.config)
        }
        
        try:
            # === Stage 1: Preprocessing Data ===
            if PipelineStage.DATA_PREPROCESSING.value in stages:
                logger.info("🔧 Stage 1: Preprocessing data")
                
                X, y = self._preprocess_data(data, target_column)
                X_train, X_test, y_train, y_test = self._split_data(
                    X, y, test_size, validation_size, time_series_split
                )
                
                stages_completed.append(PipelineStage.DATA_PREPROCESSING.value)
                
                self.console.print("✅ [green]Preprocessing data completed[/green]")
            else:
                logger.info("⏭️ Skipping preprocessing data")
                X = data.drop(columns=[target_column])
                y = data[target_column]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            
            # === Stage 2: Generation Features ===
            feature_generation_result = None
            if PipelineStage.FEATURE_GENERATION.value in stages:
                logger.info("🎨 Stage 2: Generation features")
                
                feature_generation_result = self._generate_features(X_train)
                
                if feature_generation_result and not feature_generation_result.features.empty:
                    # Apply generated features to training set
                    X_train_enhanced = pd.concat([X_train, feature_generation_result.features], axis=1)
                    
                    # Apply to test set
                    test_features = self.feature_generator.generate_features(X_test)
                    X_test_enhanced = pd.concat([X_test, test_features.features], axis=1)
                    
                    X_train = X_train_enhanced
                    X_test = X_test_enhanced
                
                stages_completed.append(PipelineStage.FEATURE_GENERATION.value)
                self.console.print("✅ [green]Generation features completed[/green]")
            
            # === Stage 3: Selection Features ===
            feature_selection_result = None
            if PipelineStage.FEATURE_SELECTION.value in stages:
                logger.info("🎯 Stage 3: Select features")
                
                feature_selection_result = self._select_features(X_train, y_train)
                
                if feature_selection_result and feature_selection_result.selected_features:
                    X_train = X_train[feature_selection_result.selected_features]
                    X_test = X_test[feature_selection_result.selected_features]
                
                stages_completed.append(PipelineStage.FEATURE_SELECTION.value)
                self.console.print("✅ [green]Select features completed[/green]")
            
            # === Stage 4: Selection Models ===
            model_selection_result = None
            if PipelineStage.MODEL_SELECTION.value in stages:
                logger.info("🤖 Stage 4: Select models")
                
                model_selection_result = self._select_models(X_train, y_train)
                
                stages_completed.append(PipelineStage.MODEL_SELECTION.value)
                self.console.print("✅ [green]Select models completed[/green]")
            
            # === Stage 5: Optimization Hyperparameters ===
            optimization_results = {}
            if PipelineStage.HYPERPARAMETER_OPTIMIZATION.value in stages:
                logger.info("⚙️ Stage 5: Optimization hyperparameters")
                
                # Determine models for optimization
                models_to_optimize = []
                if model_selection_result:
                    # Take top-3 model
                    sorted_models = sorted(
                        model_selection_result.model_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    models_to_optimize = [model[0] for model in sorted_models[:3]]
                else:
                    # By default optimize base model
                    models_to_optimize = ['xgboost', 'random_forest', 'lightgbm']
                
                optimization_results = self._optimize_hyperparameters(
                    X_train, y_train, models_to_optimize
                )
                
                stages_completed.append(PipelineStage.HYPERPARAMETER_OPTIMIZATION.value)
                self.console.print("✅ [green]Optimization hyperparameters completed[/green]")
            
            # === Stage 6: Construction ENSEMBLE ===
            ensemble_result = None
            if PipelineStage.ENSEMBLE_BUILDING.value in stages:
                logger.info("🤝 Stage 6: Build ensemble")
                
                ensemble_result = self._build_ensemble(
                    X_train, y_train, optimization_results
                )
                
                stages_completed.append(PipelineStage.ENSEMBLE_BUILDING.value)
                self.console.print("✅ [green]Build ensemble completed[/green]")
            
            # === Stage 7: Evaluation Models ===
            evaluation_result = None
            if PipelineStage.MODEL_EVALUATION.value in stages:
                logger.info("📊 Stage 7: Evaluate models")
                
                # Determine best model
                best_model, best_model_name, best_score = self._select_best_model(
                    optimization_results, ensemble_result
                )
                
                evaluation_result = self._evaluate_models(
                    X_train, y_train, X_test, y_test,
                    best_model, best_model_name
                )
                
                stages_completed.append(PipelineStage.MODEL_EVALUATION.value)
                self.console.print("✅ [green]Evaluate models completed[/green]")
            
            # === Creation FINAL RESULT ===
            total_time = time.time() - start_time
            pipeline_metadata.update({
                'end_time': time.time(),
                'stages_completed': stages_completed,
                'final_features_count': X_train.shape[1] if 'X_train' in locals() else 0
            })
            
            result = PipelineResult(
                best_model=best_model if 'best_model' in locals() else None,
                best_model_name=best_model_name if 'best_model_name' in locals() else "unknown",
                best_score=best_score if 'best_score' in locals() else 0.0,
                feature_generation_result=feature_generation_result,
                feature_selection_result=feature_selection_result,
                optimization_results=optimization_results,
                ensemble_result=ensemble_result,
                evaluation_result=evaluation_result,
                pipeline_metadata=pipeline_metadata,
                total_time=total_time,
                stages_completed=stages_completed
            )
            
            # Save results
            self._save_pipeline_results(result)
            
            # Final report
            self._print_final_report(result)
            
            logger.info(f"🎉 AutoML Pipeline completed for {total_time:.2f}with")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in AutoML Pipeline: {e}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocessing data"""
        logger.info("🔧 Preprocessing data...")
        
        # Split on features and target variable
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Preprocessing with DataPreprocessor
        X_processed = self.preprocessor.preprocess(X)
        y_processed = self.preprocessor.preprocess_target(y)
        
        logger.info(f"✅ Data processed: {X_processed.shape[0]} records, {X_processed.shape[1]} features")
        
        return X_processed, y_processed
    
    def _split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float,
        validation_size: float,
        time_series_split: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data on training and test set"""
        logger.info("✂️ Split data...")
        
        if time_series_split:
            # Temporal split (without shuffling)
            split_idx = int(len(X) * (1 - test_size))
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        logger.info(f"✅ Data split: training={len(X_train)}, test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def _generate_features(self, X: pd.DataFrame) -> Optional[FeatureGenerationResult]:
        """Generation features"""
        logger.info("🎨 Generation features...")
        
        try:
            result = self.feature_generator.generate_features(X, parallel=True)
            
            logger.info(f"✅ Generated {len(result.feature_names)} features")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generation features: {e}")
            return None
    
    def _select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Optional[FeatureSelectionResult]:
        """Select features"""
        logger.info("🎯 Select features...")
        
        try:
            result = self.feature_selector.select_features(
                X, y,
                ensemble_selection=True
            )
            
            logger.info(f"✅ {len(result.selected_features)} features")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error selection features: {e}")
            return None
    
    def _select_models(self, X: pd.DataFrame, y: pd.Series) -> Optional[ModelSelectionResult]:
        """Select models"""
        logger.info("🤖 Select models...")
        
        try:
            result = self.model_selector.select_best_models(
                X, y,
                models=['xgboost', 'random_forest', 'lightgbm', 'ridge', 'elasticnet'],
                cv_folds=3 # Fewer folds for speed
            )
            
            logger.info(f"✅ Tested {len(result.model_scores)} models")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error selection models: {e}")
            return None
    
    def _optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: List[str]
    ) -> Dict[str, OptimizationResult]:
        """Optimization hyperparameters"""
        logger.info(f"⚙️ Optimization {len(models)} models...")
        
        try:
            results = self.hyperparameter_optimizer.optimize_multiple_models(
                X, y,
                models=models,
                optimizer_method='optuna_tpe',
                n_calls=50 # Fewer iterations for speed
            )
            
            logger.info(f"✅ Optimized hyperparameters for {len(results)} models")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error optimization: {e}")
            return {}
    
    def _build_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        optimization_results: Dict[str, OptimizationResult]
    ) -> Optional[EnsembleResult]:
        """Build ensemble"""
        logger.info("🤝 Build ensemble...")
        
        try:
            # Create models with optimal parameters
            models = {}
            for model_name, result in optimization_results.items():
                model = self.hyperparameter_optimizer._get_model(model_name, result.best_params)
                models[model_name] = model
            
            if not models:
                logger.warning("⚠️ No models for ensemble")
                return None
            
            result = self.ensemble_builder.build_ensemble(
                X, y,
                models,
                ensemble_methods=['voting', 'stacking']
            )
            
            logger.info(f"✅ Ensemble built with {len(models)} models")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error building ensemble: {e}")
            return None
    
    def _select_best_model(
        self,
        optimization_results: Dict[str, OptimizationResult],
        ensemble_result: Optional[EnsembleResult]
    ) -> Tuple[Any, str, float]:
        """Select best model"""
        logger.info("🏆 Select best model...")
        
        best_model = None
        best_model_name = "unknown"
        best_score = float('inf')
        
        # Check optimized models
        for model_name, result in optimization_results.items():
            if result.best_score < best_score:
                best_score = result.best_score
                best_model_name = model_name
                best_model = self.hyperparameter_optimizer._get_model(
                    model_name, result.best_params
                )
        
        # Check ensemble
        if ensemble_result and ensemble_result.best_ensemble_score < best_score:
            best_score = ensemble_result.best_ensemble_score
            best_model_name = f"ensemble_{ensemble_result.best_ensemble_method}"
            best_model = ensemble_result.ensembles[ensemble_result.best_ensemble_method]
        
        logger.info(f"✅ Best model: {best_model_name} (score: {abs(best_score):.4f})")
        
        return best_model, best_model_name, best_score
    
    def _evaluate_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        best_model: Any,
        best_model_name: str
    ) -> EvaluationResult:
        """Evaluate models"""
        logger.info("📊 Evaluate best model...")
        
        try:
            # Training best model
            best_model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)
            
            result = self.evaluator.evaluate_model(
                best_model,
                X_train, y_train,
                X_test, y_test,
                model_name=best_model_name
            )
            
            logger.info(f"✅ Model : R² = {result.test_r2:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error evaluation model: {e}")
            # Return empty result
            return EvaluationResult(
                model_name=best_model_name,
                train_mse=0.0, train_mae=0.0, train_r2=0.0,
                test_mse=0.0, test_mae=0.0, test_r2=0.0,
                cross_val_scores=[], feature_importance={},
                evaluation_metadata={}, evaluation_time=0.0
            )
    
    def _save_pipeline_results(self, result: PipelineResult):
        """Save results pipeline"""
        logger.info("💾 Save results...")
        
        try:
            # Save best model
            if result.best_model:
                model_path = self.output_dir / "best_model.pkl"
                joblib.dump(result.best_model, model_path)
                logger.info(f"💾 Model saved: {model_path}")
            
            # Save metadata
            metadata_path = self.output_dir / "pipeline_metadata.json"
            with open(metadata_path, 'w') as f:
                # Converting results in serializable format
                serializable_metadata = {
                    'best_model_name': result.best_model_name,
                    'best_score': result.best_score,
                    'total_time': result.total_time,
                    'stages_completed': result.stages_completed,
                    'pipeline_metadata': result.pipeline_metadata
                }
                json.dump(serializable_metadata, f, indent=2)
            
            logger.info(f"✅ Metadata saved: {metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving results: {e}")
    
    def _print_final_report(self, result: PipelineResult):
        """Output report"""
        
        # Create table with
        table = Table(title="🎯 Results AUTOML PIPELINE")
        
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        table.add_row("🏆 Best model", result.best_model_name)
        table.add_row("📊 Best score", f"{abs(result.best_score):.4f}")
        table.add_row("⏱️ Time execution", f"{result.total_time:.2f}with")
        table.add_row("🎯 Stages completed", f"{len(result.stages_completed)}/8")
        
        if result.feature_generation_result:
            table.add_row("🎨 Features ", str(len(result.feature_generation_result.feature_names)))
        
        if result.feature_selection_result:
            table.add_row("🔍 Features ", str(len(result.feature_selection_result.selected_features)))
        
        if result.optimization_results:
            table.add_row("⚙️ Models ", str(len(result.optimization_results)))
        
        if result.evaluation_result:
            table.add_row("📈 R² on test", f"{result.evaluation_result.test_r2:.4f}")
            table.add_row("📉 MSE on test", f"{result.evaluation_result.test_mse:.4f}")
        
        self.console.print(table)
        
        # Details stages
        stages_panel = Panel(
            " → ".join(result.stages_completed),
            title="🔄 Completed stages"
        )
        self.console.print(stages_panel)


if __name__ == "__main__":
    # Example use AutoML Pipeline
    
    # Create test data (simulation cryptocurrency data)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=2000, freq='1H')
    
    # Base OHLCV data
    data = pd.DataFrame({
        'open': np.random.randn(2000).cumsum() + 50000,
        'high': np.random.randn(2000).cumsum() + 50100,
        'low': np.random.randn(2000).cumsum() + 49900,
        'close': np.random.randn(2000).cumsum() + 50000,
        'volume': np.random.exponential(1000, 2000)
    }, index=dates)
    
    # Target variable (future return)
    data['future_return'] = data['close'].shift(-1) / data['close'] - 1
    data = data.dropna()
    
    # Create and launch pipeline
    config = AutoMLConfig()
    pipeline = AutoMLPipeline(config, output_dir="test_automl_output")
    
    result = pipeline.run(
        data=data,
        target_column='future_return',
        test_size=0.2,
        time_series_split=True
    )
    
    print(f"\n🎉 AutoML Pipeline completed!")
    print(f"🏆 Best model: {result.best_model_name}")
    print(f"📊 Best score: {abs(result.best_score):.4f}")
    print(f"⏱️ Time execution: {result.total_time:.2f}with")
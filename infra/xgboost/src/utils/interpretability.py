"""
Model Interpretability Utilities
===============================

SHAP-based interpretability and feature importance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
from loguru import logger


class SHAPExplainer:
    """SHAP-based model explainer"""
    
    def __init__(self, model: Any):
        self.model = model
        self.explainer_ = None
        
        if SHAP_AVAILABLE:
            try:
                self.explainer_ = shap.TreeExplainer(model)
                logger.info("SHAP TreeExplainer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize SHAP explainer: {e}")
        else:
            logger.warning("SHAP not available. Install with: pip install shap")
    
    def get_shap_values(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Calculate SHAP values"""
        
        if not self.explainer_:
            logger.warning("SHAP explainer not available")
            return None
        
        try:
            shap_values = self.explainer_.shap_values(X)
            return shap_values
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return None
    
    def plot_summary(
        self, 
        X: pd.DataFrame, 
        max_display: int = 20,
        save_path: Optional[str] = None
    ):
        """Plot SHAP summary"""
        
        shap_values = self.get_shap_values(X)
        
        if shap_values is None:
            logger.warning("Cannot plot SHAP summary - values not available")
            return
        
        try:
            shap.summary_plot(
                shap_values, X, 
                max_display=max_display,
                show=save_path is None
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {save_path}")
                
        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {e}")


class FeatureImportanceAnalyzer:
    """Feature importance analysis"""
    
    def __init__(self, model: Any):
        self.model = model
    
    def get_feature_importance(
        self, 
        feature_names: List[str],
        importance_type: str = "gain"
    ) -> pd.DataFrame:
        """Get feature importance from model"""
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_values = self.model.feature_importances_
            elif hasattr(self.model, 'get_booster'):
                booster = self.model.get_booster()
                importance_dict = booster.get_score(importance_type=importance_type)
                importance_values = [
                    importance_dict.get(f'f{i}', 0) 
                    for i in range(len(feature_names))
                ]
            else:
                logger.warning("Cannot extract feature importance from model")
                return pd.DataFrame()
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()
    
    def plot_importance(
        self,
        feature_names: List[str],
        top_k: int = 20,
        importance_type: str = "gain",
        save_path: Optional[str] = None
    ):
        """Plot feature importance"""
        
        importance_df = self.get_feature_importance(feature_names, importance_type)
        
        if importance_df.empty:
            logger.warning("No feature importance data to plot")
            return
        
        # Take top k features
        top_features = importance_df.head(top_k)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        bars = plt.barh(
            range(len(top_features)),
            top_features['importance'],
            color='skyblue',
            alpha=0.7
        )
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel(f'Feature Importance ({importance_type})')
        plt.title(f'Top {top_k} Feature Importance')
        plt.grid(True, alpha=0.3)
        
        # Add values to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()


class ModelInterpreter:
    """Comprehensive model interpretation"""
    
    def __init__(self, model: Any):
        self.model = model
        self.shap_explainer = SHAPExplainer(model)
        self.importance_analyzer = FeatureImportanceAnalyzer(model)
    
    def interpret_prediction(
        self,
        X_sample: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Interpret a single prediction"""
        
        # Get prediction
        prediction = self.model.predict(X_sample)[0]
        
        # Get SHAP values
        shap_values = self.shap_explainer.get_shap_values(X_sample)
        
        interpretation = {
            'prediction': float(prediction),
            'feature_values': X_sample.iloc[0].to_dict(),
            'shap_values': {},
        }
        
        if shap_values is not None:
            shap_dict = dict(zip(feature_names, shap_values[0]))
            interpretation['shap_values'] = shap_dict
            
            # Top positive and negative contributors
            sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            interpretation['top_contributors'] = sorted_shap[:5]
        
        return interpretation
    
    def generate_explanation(
        self,
        interpretation: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation"""
        
        explanation = f"Prediction: {interpretation['prediction']:.4f}\n\n"
        
        if 'top_contributors' in interpretation:
            explanation += "Top Contributing Features:\n"
            explanation += "-" * 30 + "\n"
            
            for feature, shap_value in interpretation['top_contributors']:
                feature_value = interpretation['feature_values'].get(feature, 'N/A')
                direction = "increases" if shap_value > 0 else "decreases"
                explanation += f"• {feature}: {feature_value} ({direction} prediction by {abs(shap_value):.4f})\n"
        
        return explanation


class ExplanationGenerator:
    """Generate explanations for model predictions"""
    
    def __init__(self):
        pass
    
    def generate_report(
        self,
        model: Any,
        X_sample: pd.DataFrame,
        feature_names: List[str],
        save_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive explanation report"""
        
        interpreter = ModelInterpreter(model)
        
        # Interpret prediction
        interpretation = interpreter.interpret_prediction(X_sample, feature_names)
        
        # Generate explanation
        explanation = interpreter.generate_explanation(interpretation)
        
        # Create full report
        report = f"""
Model Interpretation Report
==========================

{explanation}

Feature Values:
--------------
"""
        
        for feature, value in interpretation['feature_values'].items():
            report += f"{feature}: {value}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Explanation report saved to {save_path}")
        
        return report
import pickle
import numpy as np
import pandas as pd
import shap
from typing import Dict, Tuple, Any, Union
from .features import FeatureProcessor
from .model import GradientBoostingPredictor

class LaunchPredictor:
    def __init__(self, model_path: str):
        """Initialize predictor by loading the saved GradientBoostingPredictor."""
        print(f"Loading predictor from {model_path}")
        with open(model_path, 'rb') as f:
            self.predictor = pickle.load(f)
        self.explainer = None

        # Verify predictor attributes and print detailed information
        print("\nLoaded predictor attributes and details:")
        print(f"- Has gb_model: {hasattr(self.predictor, 'gb_model')}")
        if hasattr(self.predictor, 'gb_model'):
            print(f"  Model type: {type(self.predictor.gb_model)}")
            print(f"  Number of features used: {len(self.predictor.feature_names)}")

        print(f"- Has scaler: {hasattr(self.predictor, 'scaler')}")
        if hasattr(self.predictor, 'scaler'):
            print(f"  Scaler type: {type(self.predictor.scaler)}")
            print(f"  Scaler feature range: {self.predictor.scaler.feature_range}")

        print(f"- Has feature_names: {hasattr(self.predictor, 'feature_names')}")
        if hasattr(self.predictor, 'feature_names'):
            print("  Feature names:")
            for name in self.predictor.feature_names:
                print(f"    - {name}")

    def initialize_explainer(self):
        """Initialize SHAP explainer for the model."""
        if self.explainer is None:
            # Use the LightGBM model from the predictor
            model = self.predictor.gb_model
            print(f"Initializing SHAP explainer with model type: {type(model)}")
            self.explainer = shap.TreeExplainer(model)

    def predict(self, features: Union[Dict[str, float], pd.DataFrame]) -> Tuple[bool, float, Dict[str, Any]]:
        """Make prediction and return GO/NOGO decision with confidence."""
        print("\n=== New Prediction ===")
        print("Input features:", features)

        # Create DataFrame with single sample if input is a dictionary
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features  # Input is already a DataFrame

        # Get prediction probabilities using the predictor object
        try:
            pred_proba = self.predictor.predict_proba(df)[0]
            print("Prediction probabilities obtained successfully")
        except Exception as e:
            print(f"Error getting prediction probabilities: {str(e)}")
            raise

        go_confidence = pred_proba[0]  # First element is GO probability
        decision = go_confidence >= 0.5

        print(f"Prediction: {'GO' if decision else 'NOGO'}, GO Confidence: {go_confidence*100:.4f}%")
        print("Raw probabilities: GO={:.4f}, NOGO={:.4f}".format(pred_proba[0], pred_proba[1]))

        # Get SHAP values
        try:
            self.initialize_explainer()
            X_scaled = self.predictor.scaler.transform(df)
            shap_values = self.explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Use GO class values
            print("SHAP values calculated successfully")
        except Exception as e:
            print(f"Error calculating SHAP values: {str(e)}")
            raise

        result = {
            'prediction': 'GO' if decision else 'NOGO',
            'confidence': go_confidence,
            'raw_probabilities': pred_proba,
            'input_features': df.to_dict('records')[0],
            'shap_values': shap_values,
            'feature_importance': self._get_feature_importance(df, shap_values)
        }

        return decision, go_confidence, result

    def _get_feature_importance(self, X: pd.DataFrame, shap_values: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance from SHAP values."""
        return dict(zip(X.columns, np.abs(shap_values).mean(0)))
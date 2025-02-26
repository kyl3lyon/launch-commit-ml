from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from .config import ALL_FEATURES, WARNING_THRESHOLDS

class FeatureProcessor:
    def __init__(self):
        # Define the exact feature order from training
        self.feature_names = [
            'MAX_SHEAR_SURFACE_T-0',
            'MAX_SHEAR_LOW_T-0',
            'MAX_SHEAR_ASCENT_T-0',
            'MAX_SHEAR_UPPER_T-0',
            'PRECIPITATION_15MIN_T-0',
            'RAIN_15MIN_T-0',
            'FREEZING_LEVEL_HEIGHT_T-0',
            'LIFTED_INDEX_T-0',
            'CONVECTIVE_INHIBITION_T-0',
            'CAPE_15MIN_T-0',
            'UV_INDEX_T-0',
            'WEATHER_CODE_15MIN_T-0',
            'VISIBILITY_15MIN_T-0',
            'MAX_TEMPERATURE_SURFACE_T0',
            'MAX_TEMPERATURE_LOW_T0',
            'MAX_TEMPERATURE_ASCENT_T0',
            'MAX_TEMPERATURE_UPPER_T0',
            'MAX_WIND_SPEED_SURFACE_T0',
            'MAX_WIND_SPEED_LOW_T0',
            'MAX_WIND_SPEED_ASCENT_T0',
            'MAX_WIND_SPEED_UPPER_T0',
            'MAX_CLOUD_COVER_SURFACE_T0',
            'MAX_CLOUD_COVER_LOW_T0',
            'MAX_CLOUD_COVER_ASCENT_T0',
            'MAX_CLOUD_COVER_UPPER_T0',
            'MAX_RELATIVE_HUMIDITY_SURFACE_T0',
            'MAX_RELATIVE_HUMIDITY_LOW_T0',
            'MAX_RELATIVE_HUMIDITY_ASCENT_T0',
            'MAX_RELATIVE_HUMIDITY_UPPER_T0'
        ]

    def validate_features(self, features: Dict[str, float]) -> List[str]:
        """Validate feature values and return list of warnings."""
        warnings = []

        for feature, value in features.items():
            min_val, max_val, _ = ALL_FEATURES[feature]

            if value < min_val or value > max_val:
                warnings.append(f"{feature} value {value} is outside valid range [{min_val}, {max_val}]")

            if feature in WARNING_THRESHOLDS:
                threshold = WARNING_THRESHOLDS[feature]
                if value > threshold:
                    warnings.append(f"{feature} value {value} exceeds warning threshold {threshold}")

        return warnings

    def process_features(self, features: Dict[str, float]) -> pd.DataFrame:
        """Convert feature dictionary to DataFrame in correct order."""
        # Ensure all features are present and in correct order
        processed_features = {name: features.get(name, ALL_FEATURES[name][2]) 
                            for name in self.feature_names}

        # Create DataFrame with features in exact order
        df = pd.DataFrame([processed_features])

        # Reorder columns to match training order
        ordered_columns = [
            'MAX_SHEAR_SURFACE_T-0',
            'MAX_SHEAR_LOW_T-0',
            'MAX_SHEAR_ASCENT_T-0',
            'MAX_SHEAR_UPPER_T-0',
            'PRECIPITATION_15MIN_T-0',
            'RAIN_15MIN_T-0',
            'FREEZING_LEVEL_HEIGHT_T-0',
            'LIFTED_INDEX_T-0',
            'CONVECTIVE_INHIBITION_T-0',
            'CAPE_15MIN_T-0',
            'UV_INDEX_T-0',
            'WEATHER_CODE_15MIN_T-0',
            'VISIBILITY_15MIN_T-0',
            'MAX_TEMPERATURE_SURFACE_T0',
            'MAX_TEMPERATURE_LOW_T0',
            'MAX_TEMPERATURE_ASCENT_T0',
            'MAX_TEMPERATURE_UPPER_T0',
            'MAX_WIND_SPEED_SURFACE_T0',
            'MAX_WIND_SPEED_LOW_T0',
            'MAX_WIND_SPEED_ASCENT_T0',
            'MAX_WIND_SPEED_UPPER_T0',
            'MAX_CLOUD_COVER_SURFACE_T0',
            'MAX_CLOUD_COVER_LOW_T0',
            'MAX_CLOUD_COVER_ASCENT_T0',
            'MAX_CLOUD_COVER_UPPER_T0',
            'MAX_RELATIVE_HUMIDITY_SURFACE_T0',
            'MAX_RELATIVE_HUMIDITY_LOW_T0',
            'MAX_RELATIVE_HUMIDITY_ASCENT_T0',
            'MAX_RELATIVE_HUMIDITY_UPPER_T0'
        ]

        df = df[ordered_columns]

        print("\nFeature Order Check:")
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")

        return df

    def get_feature_ranges(self) -> Dict[str, tuple]:
        """Return dictionary of feature ranges."""
        return {name: (min_val, max_val) 
                for name, (min_val, max_val, _) in ALL_FEATURES.items()}

    def get_default_values(self) -> Dict[str, float]:
        """Return dictionary of default feature values."""
        return {name: default 
                for name, (_, _, default) in ALL_FEATURES.items()}

    @staticmethod
    def format_feature_name(feature_name: str) -> str:
        """Convert feature name to display format."""
        name = feature_name.replace('_T-0', '').replace('_T0', '')
        name = name.replace('_', ' ').title()
        return name 
from src.predictor import LaunchPredictor
import pandas as pd

def test_model():
    # Initialize predictor
    predictor = LaunchPredictor('model/gb_predictor.pkl')

    # Get the feature names in the correct order
    feature_names = predictor.predictor.feature_names
    print("Using features in order:", feature_names)

    # Create a sample input with some reasonable values
    base_values = {
        'MAX_CLOUD_COVER_SURFACE_T0': 0.5,
        'MAX_CLOUD_COVER_LOW_T0': 0.3,
        'MAX_CLOUD_COVER_ASCENT_T0': 0.4,
        'MAX_CLOUD_COVER_UPPER_T0': 0.2,
        'MAX_RELATIVE_HUMIDITY_SURFACE_T0': 60.0,
        'MAX_RELATIVE_HUMIDITY_LOW_T0': 55.0,
        'MAX_RELATIVE_HUMIDITY_ASCENT_T0': 50.0,
        'MAX_RELATIVE_HUMIDITY_UPPER_T0': 45.0,
        'MAX_TEMPERATURE_SURFACE_T0': 25.0,
        'MAX_TEMPERATURE_LOW_T0': 20.0,
        'MAX_TEMPERATURE_ASCENT_T0': 15.0,
        'MAX_TEMPERATURE_UPPER_T0': 10.0,
        'MAX_WIND_SPEED_SURFACE_T0': 5.0,
        'MAX_WIND_SPEED_LOW_T0': 7.0,
        'MAX_WIND_SPEED_ASCENT_T0': 10.0,
        'MAX_WIND_SPEED_UPPER_T0': 15.0,
        'VISIBILITY_15MIN_T-0': 8000.0,
        'WEATHER_CODE_15MIN_T-0': 0.0,
        'UV_INDEX_T-0': 5.0,
        'CAPE_15MIN_T-0': 500.0,
        'CONVECTIVE_INHIBITION_T-0': -50.0,
        'LIFTED_INDEX_T-0': -2.0,
        'MAX_SHEAR_SURFACE_T-0': 10.0,
        'MAX_SHEAR_LOW_T-0': 15.0,
        'MAX_SHEAR_ASCENT_T-0': 20.0,
        'MAX_SHEAR_UPPER_T-0': 25.0,
        'FREEZING_LEVEL_HEIGHT_T-0': 3000.0,
        'PRECIPITATION_15MIN_T-0': 0.0,
        'RAIN_15MIN_T-0': 0.0
    }

    # Create ordered input dictionary
    sample_input = {name: base_values[name] for name in feature_names}

    # Make a prediction
    try:
        decision, confidence, result = predictor.predict(sample_input)
        print("\nTest Results:")
        print(f"Decision: {decision}")
        print(f"Confidence: {confidence:.4f}")
        print("\nTop 5 Most Important Features:")
        sorted_importance = sorted(result['feature_importance'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feature, importance in sorted_importance:
            print(f"{feature}: {importance:.4f}")
        print("\nTest passed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise

if __name__ == '__main__':
    test_model() 
from typing import Dict, List, Tuple

# Feature Groups and Ranges
SHEAR_FEATURES = {
    'MAX_SHEAR_SURFACE_T-0': (0.0, 150.0, 0.0),    # (min, max, default)
    'MAX_SHEAR_LOW_T-0': (0.0, 100.0, 0.0),
    'MAX_SHEAR_ASCENT_T-0': (0.0, 50.0, 0.0),
    'MAX_SHEAR_UPPER_T-0': (0.0, 100.0, 0.0)
}

PRECIPITATION_FEATURES = {
    'PRECIPITATION_15MIN_T-0': (0.0, 1.0, 0.0),
    'RAIN_15MIN_T-0': (0.0, 1.0, 0.0)
}

TEMPERATURE_FEATURES = {
    'MAX_TEMPERATURE_SURFACE_T0': (30.0, 100.0, 74.3),
    'MAX_TEMPERATURE_LOW_T0': (25.0, 80.0, 63.1),
    'MAX_TEMPERATURE_ASCENT_T0': (15.0, 60.0, 44.6),
    'MAX_TEMPERATURE_UPPER_T0': (-60.0, 0.0, -31.9)
}

WIND_FEATURES = {
    'MAX_WIND_SPEED_SURFACE_T0': (0.0, 100.0, 0.0),
    'MAX_WIND_SPEED_LOW_T0': (0.0, 100.0, 0.0),
    'MAX_WIND_SPEED_ASCENT_T0': (0.0, 150.0, 0.0),
    'MAX_WIND_SPEED_UPPER_T0': (0.0, 200.0, 0.0)
}

CLOUD_FEATURES = {
    'MAX_CLOUD_COVER_SURFACE_T0': (0.0, 100.0, 0.0),
    'MAX_CLOUD_COVER_LOW_T0': (0.0, 100.0, 0.0),
    'MAX_CLOUD_COVER_ASCENT_T0': (0.0, 100.0, 0.0),
    'MAX_CLOUD_COVER_UPPER_T0': (0.0, 100.0, 0.0)
}

HUMIDITY_FEATURES = {
    'MAX_RELATIVE_HUMIDITY_SURFACE_T0': (0.0, 100.0, 51.0),
    'MAX_RELATIVE_HUMIDITY_LOW_T0': (0.0, 100.0, 40.0),
    'MAX_RELATIVE_HUMIDITY_ASCENT_T0': (0.0, 100.0, 25.0),
    'MAX_RELATIVE_HUMIDITY_UPPER_T0': (0.0, 100.0, 24.0)
}

OTHER_FEATURES = {
    'FREEZING_LEVEL_HEIGHT_T-0': (0.0, 25000.0, 10993.0),
    'LIFTED_INDEX_T-0': (-20.0, 30.0, -20.0),
    'CONVECTIVE_INHIBITION_T-0': (-1000.0, 0.0, 0.0),
    'CAPE_15MIN_T-0': (0.0, 7000.0, 0.0),
    'UV_INDEX_T-0': (0.0, 10.0, 3.8),
    'WEATHER_CODE_15MIN_T-0': (0.0, 100.0, 1.0),
    'VISIBILITY_15MIN_T-0': (0.0, 300000.0, 300000.0)
}

# Combine all features
ALL_FEATURES = {
    **SHEAR_FEATURES,
    **PRECIPITATION_FEATURES,
    **TEMPERATURE_FEATURES,
    **WIND_FEATURES,
    **CLOUD_FEATURES,
    **HUMIDITY_FEATURES,
    **OTHER_FEATURES
}

# Feature groups for UI organization
FEATURE_GROUPS = {
    "Shear Measurements": SHEAR_FEATURES,
    "Precipitation": PRECIPITATION_FEATURES,
    "Temperature": TEMPERATURE_FEATURES,
    "Wind Speed": WIND_FEATURES,
    "Cloud Cover": CLOUD_FEATURES,
    "Humidity": HUMIDITY_FEATURES,
    "Other Conditions": OTHER_FEATURES
}

# Warning thresholds
WARNING_THRESHOLDS = {
    'MAX_WIND_SPEED_SURFACE_T0': 40.0,
    'PRECIPITATION_15MIN_T-0': 0.5,
    'MAX_CLOUD_COVER_SURFACE_T0': 80.0
}

# Preset scenarios
PRESET_SCENARIOS = {
    "Ideal Conditions": {
        # Shear - Very light and consistent
        'MAX_SHEAR_SURFACE_T-0': 0.5,            # Minimal shear
        'MAX_SHEAR_LOW_T-0': 2.0,                # Very light shear
        'MAX_SHEAR_ASCENT_T-0': 3.0,             # Very stable conditions
        'MAX_SHEAR_UPPER_T-0': 4.0,              # Light upper shear

        # Precipitation - None
        'PRECIPITATION_15MIN_T-0': 0.0,          # No precipitation
        'RAIN_15MIN_T-0': 0.0,                   # No rain

        # Freezing Level - High
        'FREEZING_LEVEL_HEIGHT_T-0': 4500,       # High freezing level

        # Stability - Very Stable
        'LIFTED_INDEX_T-0': 4.0,                 # Very stable
        'CONVECTIVE_INHIBITION_T-0': -25.0,      # Light inhibition
        'CAPE_15MIN_T-0': 200.0,                 # Very stable conditions

        # Surface Conditions
        'UV_INDEX_T-0': 6,                       # Moderate UV
        'WEATHER_CODE_15MIN_T-0': 0,             # Clear skies
        'VISIBILITY_15MIN_T-0': 80000,           # Excellent visibility

        # Temperature - Moderate
        'MAX_TEMPERATURE_SURFACE_T0': 72.0,      # Perfect temperature
        'MAX_TEMPERATURE_LOW_T0': 65.0,          # Cool and stable
        'MAX_TEMPERATURE_ASCENT_T0': 55.0,       # Cool ascent
        'MAX_TEMPERATURE_UPPER_T0': 45.0,        # Cold upper atmosphere

        # Wind Speed - Light
        'MAX_WIND_SPEED_SURFACE_T0': 5,          # Light breeze
        'MAX_WIND_SPEED_LOW_T0': 8,              # Light winds
        'MAX_WIND_SPEED_ASCENT_T0': 12,          # Moderate winds
        'MAX_WIND_SPEED_UPPER_T0': 15,           # Moderate upper winds

        # Cloud Cover - Minimal
        'MAX_CLOUD_COVER_SURFACE_T0': 10,        # Almost clear
        'MAX_CLOUD_COVER_LOW_T0': 5,             # Very clear
        'MAX_CLOUD_COVER_ASCENT_T0': 0,          # Clear
        'MAX_CLOUD_COVER_UPPER_T0': 0,           # Clear

        # Relative Humidity - Low
        'MAX_RELATIVE_HUMIDITY_SURFACE_T0': 45,   # Comfortable
        'MAX_RELATIVE_HUMIDITY_LOW_T0': 40,       # Dry
        'MAX_RELATIVE_HUMIDITY_ASCENT_T0': 35,    # Very dry
        'MAX_RELATIVE_HUMIDITY_UPPER_T0': 30,     # Very dry

    },
    "Marginal Conditions": {
        # Shear - Moderate
        'MAX_SHEAR_SURFACE_T-0': 8.0,            # Moderate shear
        'MAX_SHEAR_LOW_T-0': 12.0,               # Moderate shear
        'MAX_SHEAR_ASCENT_T-0': 15.0,            # Significant shear
        'MAX_SHEAR_UPPER_T-0': 18.0,             # Strong upper shear

        # Precipitation - Light
        'PRECIPITATION_15MIN_T-0': 0.02,         # Very light precipitation
        'RAIN_15MIN_T-0': 0.01,                  # Drizzle

        # Freezing Level
        'FREEZING_LEVEL_HEIGHT_T-0': 3500,       # Moderate freezing level

        # Stability - Moderately Stable
        'LIFTED_INDEX_T-0': 1.0,                 # Marginally stable
        'CONVECTIVE_INHIBITION_T-0': -100.0,     # Moderate inhibition
        'CAPE_15MIN_T-0': 1000.0,                # Some instability

        # Surface Conditions
        'UV_INDEX_T-0': 4,                       # Moderate UV
        'WEATHER_CODE_15MIN_T-0': 51,            # Light drizzle
        'VISIBILITY_15MIN_T-0': 7000,            # Decent visibility

        # Temperature - Warmer
        'MAX_TEMPERATURE_SURFACE_T0': 80.0,      # Warm
        'MAX_TEMPERATURE_LOW_T0': 75.0,          # Warm
        'MAX_TEMPERATURE_ASCENT_T0': 65.0,       # Moderate
        'MAX_TEMPERATURE_UPPER_T0': 55.0,        # Cool upper atmosphere

        # Wind Speed - Moderate
        'MAX_WIND_SPEED_SURFACE_T0': 15,         # Breezy
        'MAX_WIND_SPEED_LOW_T0': 20,             # Moderate winds
        'MAX_WIND_SPEED_ASCENT_T0': 25,          # Strong winds
        'MAX_WIND_SPEED_UPPER_T0': 30,           # Very strong upper winds

        # Cloud Cover - Scattered
        'MAX_CLOUD_COVER_SURFACE_T0': 45,        # Partly cloudy
        'MAX_CLOUD_COVER_LOW_T0': 40,            # Scattered clouds
        'MAX_CLOUD_COVER_ASCENT_T0': 30,         # Some clouds
        'MAX_CLOUD_COVER_UPPER_T0': 20,          # Light clouds

        # Relative Humidity - Moderate to High
        'MAX_RELATIVE_HUMIDITY_SURFACE_T0': 70,   # Humid
        'MAX_RELATIVE_HUMIDITY_LOW_T0': 65,       # Humid
        'MAX_RELATIVE_HUMIDITY_ASCENT_T0': 60,    # Moderate
        'MAX_RELATIVE_HUMIDITY_UPPER_T0': 55,     # Moderate

    },
    "Stormy Weather": {
        # Shear - Severe
        'MAX_SHEAR_SURFACE_T-0': 25.0,           # Strong shear
        'MAX_SHEAR_LOW_T-0': 30.0,               # Severe shear
        'MAX_SHEAR_ASCENT_T-0': 35.0,            # Very severe shear
        'MAX_SHEAR_UPPER_T-0': 40.0,             # Extreme upper shear

        # Precipitation - Heavy
        'PRECIPITATION_15MIN_T-0': 0.5,          # Heavy precipitation
        'RAIN_15MIN_T-0': 0.4,                   # Heavy rain

        # Freezing Level
        'FREEZING_LEVEL_HEIGHT_T-0': 2500,       # Low freezing level

        # Stability - Unstable
        'LIFTED_INDEX_T-0': -4.0,                # Very unstable
        'CONVECTIVE_INHIBITION_T-0': -5.0,       # Little inhibition
        'CAPE_15MIN_T-0': 2500.0,                # Severe instability

        # Surface Conditions
        'UV_INDEX_T-0': 1,                       # Low UV (cloudy)
        'WEATHER_CODE_15MIN_T-0': 95,            # Thunderstorm
        'VISIBILITY_15MIN_T-0': 2000,            # Poor visibility

        # Temperature - Variable
        'MAX_TEMPERATURE_SURFACE_T0': 85.0,      # Hot
        'MAX_TEMPERATURE_LOW_T0': 78.0,          # Warm
        'MAX_TEMPERATURE_ASCENT_T0': 70.0,       # Warm aloft
        'MAX_TEMPERATURE_UPPER_T0': 60.0,        # Moderate upper temps

        # Wind Speed - Strong
        'MAX_WIND_SPEED_SURFACE_T0': 35,         # Very strong winds
        'MAX_WIND_SPEED_LOW_T0': 45,             # Severe winds
        'MAX_WIND_SPEED_ASCENT_T0': 55,          # Storm-force winds
        'MAX_WIND_SPEED_UPPER_T0': 65,           # Hurricane-force upper winds

        # Cloud Cover - Overcast with storms
        'MAX_CLOUD_COVER_SURFACE_T0': 100,       # Complete overcast
        'MAX_CLOUD_COVER_LOW_T0': 100,           # Complete coverage
        'MAX_CLOUD_COVER_ASCENT_T0': 90,         # Nearly complete
        'MAX_CLOUD_COVER_UPPER_T0': 80,          # Heavy clouds

        # Relative Humidity - Very High
        'MAX_RELATIVE_HUMIDITY_SURFACE_T0': 95,   # Nearly saturated
        'MAX_RELATIVE_HUMIDITY_LOW_T0': 90,       # Very humid
        'MAX_RELATIVE_HUMIDITY_ASCENT_T0': 85,    # Very humid
        'MAX_RELATIVE_HUMIDITY_UPPER_T0': 80,     # Humid
    }
}

# Model paths
MODEL_PATH = 'model/gb_predictor.pkl'
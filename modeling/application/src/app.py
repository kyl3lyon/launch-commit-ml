import streamlit as st
import pandas as pd
from .predictor import LaunchPredictor
from .features import FeatureProcessor
from .utils import (
    create_feature_importance_plot,
    create_shap_waterfall_plot,
    display_warning_indicators,
    create_confidence_gauge
)
from .config import (
    FEATURE_GROUPS,
    MODEL_PATH,
    PRESET_SCENARIOS,
    ALL_FEATURES
)

def initialize_session_state():
    """Initialize session state variables."""
    if 'predictor' not in st.session_state:
        st.session_state.predictor = LaunchPredictor(MODEL_PATH)
    if 'feature_processor' not in st.session_state:
        st.session_state.feature_processor = FeatureProcessor()

def update_prediction():
    """Update prediction based on current feature values."""
    features = {}

    # Get values from sliders if they exist, otherwise use defaults
    for name, (min_val, max_val, default) in ALL_FEATURES.items():
        slider_key = f"slider_{name}"
        features[name] = st.session_state.get(slider_key, default)

    print("\n=== Update Prediction Called ===")
    print("Current features:", features)

    # Process features using feature processor
    processed_features = st.session_state.feature_processor.process_features(features)
    decision, confidence, results = st.session_state.predictor.predict(processed_features)
    print(f"Decision: {decision}, Confidence: {confidence*100:.1f}%")

    # Store results in session state
    st.session_state.current_decision = decision
    st.session_state.current_confidence = confidence
    st.session_state.current_results = results

def create_sidebar():
    """Create sidebar with feature inputs."""
    st.sidebar.title("Weather Inputs")

    # Add preset scenario selector
    selected_preset = st.sidebar.selectbox(
        "Select Preset Scenario",
        ["Custom"] + list(PRESET_SCENARIOS.keys()),
        key="preset_selector"
    )

    # Initialize features dictionary
    features = {}

    # If preset selected, use those values
    if selected_preset != "Custom":
        features = PRESET_SCENARIOS[selected_preset]

    # Create input widgets for each feature group
    for group_name, feature_dict in FEATURE_GROUPS.items():
        st.sidebar.subheader(group_name)

        for feature_name, feature_config in feature_dict.items():
            # Unpack the configuration tuple
            min_val, max_val, default = feature_config

            # Use preset value if available, otherwise use default
            current_value = features.get(feature_name, default)

            # Create slider with appropriate step size and callback
            step = 0.1 if max_val - min_val <= 10 else 1.0
            features[feature_name] = st.sidebar.slider(
                st.session_state.feature_processor.format_feature_name(feature_name),
                min_value=float(min_val), # should be a float
                max_value=float(max_val),
                value=float(current_value),
                step=step,
                key=f"slider_{feature_name}",
                on_change=update_prediction,
                help=f"Range: [{min_val}, {max_val}]"
            )

    return features

def main():
    st.set_page_config(
        page_title="Launch Wx Status Classifier",
        page_icon="🚀",
        layout="wide"
    )

    # Initialize session state
    initialize_session_state()

    # Create sidebar first to initialize sliders
    features = create_sidebar()

    # Then update prediction if needed
    if 'current_decision' not in st.session_state:
        update_prediction()

    # Title and description
    st.title("🚀 Launch Wx Status Classifier")
    st.markdown("""
    This application predicts whether current weather conditions are suitable for launch.
    Adjust the parameters in the sidebar to see how different conditions affect the launch decision.
    """)

    # Create columns for layout
    col1, col2 = st.columns([2, 1])

    # Display prediction and confidence
    with col1:
        st.header("Prediction")
        prediction_color = "green" if st.session_state.current_decision else "red"
        raw_probabilities = st.session_state.current_results['raw_probabilities']
        go_prob = raw_probabilities[0]
        nogo_prob = raw_probabilities[1]

        st.markdown(
            f"""
            <h2 style='color: {prediction_color};'>
                {st.session_state.current_results['prediction']} 
            </h2>
            <p>The probability of a GO decision is {go_prob*100:.1f}% and the probability of a NOGO decision is {nogo_prob*100:.1f}%. The GO threshold is 50%.</p>
            """,
            unsafe_allow_html=True
        )

    # Display confidence gauge in second column
    with col2:
        st.plotly_chart(
            create_confidence_gauge(st.session_state.current_confidence),
            use_container_width=True
        )

    # Display warnings
    st.header("Warnings")
    st.markdown("""
    This section displays any warnings that are triggered by the weather conditions.
    """)
    warnings = st.session_state.feature_processor.validate_features(features)
    display_warning_indicators(warnings)

    # Display analysis
    st.header("Analysis")
    st.markdown("""
    This section provides detailed analysis of the weather conditions that affect the launch decision.
    """)

    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["SHAP Analysis", "Feature Importance"])

    with tab1:
        st.plotly_chart(
            create_shap_waterfall_plot(
                st.session_state.current_results['shap_values'][0],  # SHAP values for current prediction
                st.session_state.feature_processor.feature_names  # Use feature names from processor
            ),
            use_container_width=True
        )

    with tab2:
        st.plotly_chart(
            create_feature_importance_plot(st.session_state.current_results['feature_importance']),
            use_container_width=True
        )

    # Add debug information
    with st.expander("Debug Information"):
        st.write("Current Features:", features)

if __name__ == "__main__":
    main() 
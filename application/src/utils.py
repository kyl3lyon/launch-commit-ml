import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Any

def create_feature_importance_plot(importance_dict: Dict[str, float]) -> go.Figure:
    """Create a horizontal bar plot of feature importance."""
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1])
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h'
    ))

    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=600
    )

    return fig

def create_shap_waterfall_plot(shap_values: np.ndarray, feature_names: List[str]) -> go.Figure:
    """Create a waterfall plot of SHAP values for the current prediction."""
    # Sort SHAP values by absolute magnitude
    indices = np.argsort(np.abs(shap_values))

    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="h",
        measure=["relative"] * len(indices),
        x=shap_values[indices],
        y=[feature_names[i] for i in indices],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title="SHAP Value Impact",
        xaxis_title="Impact on Prediction",
        yaxis_title="Feature",
        height=600
    )

    return fig

def display_warning_indicators(warnings: List[str]):
    """Display warning messages in the Streamlit UI."""
    if warnings:
        st.warning("⚠️ Warnings:")
        for warning in warnings:
            st.write(f"- {warning}")

def create_confidence_gauge(confidence: float) -> go.Figure:
    """Create a gauge chart for prediction confidence."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "red"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))

    fig.update_layout(
        title="Launch Probability",
        height=300
    )

    return fig 
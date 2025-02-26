# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Functions ---
def visualize_class_distribution(df: pd.DataFrame, target_column: str):
    """
    Visualize class distribution for imbalanced data.
    """
    plt.figure(figsize=(10, 6))
    df[target_column].value_counts().plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()


def identify_outliers(df: pd.DataFrame, columns: List[str], target_column: str, n_cols: int = 2):
    """
    Identify and plot outliers using rainfall plots for multiple columns, side by side with target variable.
    
    Args:
    df (pd.DataFrame): The input DataFrame
    columns (List[str]): List of column names to plot
    target_column (str): Name of the target variable column (multi-class: 2, 1, 0 for WX_UNSUITABLE)
    n_cols (int): Number of columns in the subplot grid (default: 2)
    """
    n_rows = len(columns)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    
    # Define colors for each class
    class_colors = {0: 'green', 1: 'orange', 2: 'red'}
    
    for i, column in enumerate(columns):
        # Plot without target variable, but color dots by class
        for class_value, color in class_colors.items():
            class_data = df[df[target_column] == class_value]
            sns.stripplot(x=class_data[column], ax=axes[i, 0], jitter=True, alpha=0.4, color=color, label=f'Class {class_value}')
        axes[i, 0].set_title(f'{i+1}a. {column}')
        axes[i, 0].set_xlabel('')
        axes[i, 0].legend()
        
        # Plot with target variable, split by class
        for class_value, color in class_colors.items():
            class_data = df[df[target_column] == class_value]
            sns.stripplot(x=class_data[target_column], y=class_data[column], ax=axes[i, 1], 
                          jitter=True, alpha=0.4, color=color, label=f'Class {class_value}')
        
        axes[i, 1].set_title(f'{i+1}b. {column} by {target_column}')
        axes[i, 1].set_xlabel(f'{target_column} (2: Unsuitable, 1: Marginal, 0: Suitable)')
        axes[i, 1].legend()

    # Ensure axes labels are unique
    for ax in axes.flat:
        ax.set_xlabel(ax.get_xlabel(), labelpad=20)  # Add padding to separate labels
        ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability

    plt.tight_layout()
    plt.show()


def visualize_missing_data(df: pd.DataFrame):
    """
    Detect and visualize missing data.
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels by 45 degrees
    plt.yticks(rotation=45)  # Rotate y-axis labels by 45 degrees
    plt.tight_layout()  # Adjust the layout to prevent label cutoff
    
    # Add legend
    legend_labels = ['Present', 'Missing']
    legend_colors = ['#440154', '#fde725']  # Dark purple and yellow from viridis colormap
    legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in legend_colors]
    plt.legend(legend_patches, legend_labels, loc='upper right', title='Data Status')
    
    plt.show()


def generate_correlation_matrix(df: pd.DataFrame):
    """
    Generate and visualize a correlation matrix.
    """
    corr = df.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, 
                annot_kws={'size': 8}, fmt='.2f', square=True)
    plt.title('Correlation Matrix', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def identify_disparate_scales(df: pd.DataFrame):
    """
    Identify features with different scales.
    """
    feature_ranges = df.max() - df.min()
    
    plt.figure(figsize=(12, 6))
    feature_ranges.sort_values(ascending=False).plot(kind='bar')
    plt.title('Feature Ranges')
    plt.xlabel('Feature')
    plt.ylabel('Range')
    plt.yscale('log')
    plt.show()


def select_important_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 10,
    task: str = 'classification'
) -> Dict[str, Any]:
    """
    Select the most important features using multiple methods.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        n_features (int): Number of features to select.
        task (str): 'classification' or 'regression'.

    Returns:
        Dict[str, Any]: A dictionary containing selected features and importance scores.
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Initialize importance scores dictionary
    importance_scores = {}

    # L1 regularization (Lasso) or Logistic Regression with L1 penalty
    if task == 'classification':
        lasso = LogisticRegression(
            penalty='l1', solver='liblinear', C=1.0, random_state=42
        )
        lasso.fit(X_scaled, y)
        lasso_coef = np.abs(lasso.coef_[0])  # Use [0] to get the first (and only) row
    else:
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_scaled, y)
        lasso_coef = np.abs(lasso.coef_)
    
    lasso_importance = pd.Series(lasso_coef, index=X.columns)
    importance_scores['Lasso'] = lasso_importance

    # L2 regularization (Ridge) or Logistic Regression with L2 penalty
    if task == 'classification':
        ridge = LogisticRegression(
            penalty='l2', solver='lbfgs', C=1.0, random_state=42, max_iter=1000
        )
        ridge.fit(X_scaled, y)
        ridge_coef = np.abs(ridge.coef_[0])  # Use [0] to get the first (and only) row
    else:
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled, y)
        ridge_coef = np.abs(ridge.coef_)
    
    ridge_importance = pd.Series(ridge_coef, index=X.columns)
    importance_scores['Ridge'] = ridge_importance

    # Random Forest feature importance
    if task == 'classification':
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        mi_func = mutual_info_classif
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        mi_func = mutual_info_regression
    
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
    importance_scores['Random Forest'] = rf_importance

    # Mutual Information
    mi_importance = pd.Series(mi_func(X, y, random_state=42), index=X.columns)
    importance_scores['Mutual Information'] = mi_importance

    # Calculate importance scores
    combined_ranks = pd.DataFrame()
    for method, scores in importance_scores.items():
        combined_ranks[method] = scores.rank(ascending=False)
    
    # Average the ranks
    combined_importance = combined_ranks.mean(axis=1)
    
    # Select top features
    selected_features = combined_importance.nsmallest(n_features).index.tolist()
    
    # Visualize feature importance for each method
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    fig.suptitle('Feature Importance by Different Methods', fontsize=24)
    
    for (method, scores), ax in zip(importance_scores.items(), axes.flatten()):
        sorted_scores = scores.sort_values(ascending=True)
        ax.barh(sorted_scores.index, sorted_scores.values)
        ax.set_title(f'{method} Importance', fontsize=20)
        ax.set_xlabel('Importance Score', fontsize=16)
        ax.set_ylabel('Feature', fontsize=16)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=12)
        ax.invert_yaxis()
        
        # Adjust the number of y-ticks to prevent overlapping
        ax.set_yticks(ax.get_yticks()[::3])
        
        # Rotate y-axis labels for better readability
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
        
        # Adjust subplot to make room for labels
        plt.subplots_adjust(left=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Visualize combined feature importance
    plt.figure(figsize=(18, 14))
    importance_sorted = combined_importance.sort_values(ascending=True)
    plt.barh(importance_sorted.index, importance_sorted.values)
    plt.title('Combined Feature Importance (Average Rank)', fontsize=24)
    plt.xlabel('Average Rank', fontsize=18)
    plt.ylabel('Feature', fontsize=18)
    plt.tick_params(axis='y', labelsize=12)
    plt.tick_params(axis='x', labelsize=14)
    
    # Adjust the number of y-ticks to prevent overlapping
    plt.yticks(plt.gca().get_yticks()[::3])
    
    # Rotate y-axis labels for better readability
    plt.gca().set_yticklabels(plt.gca().get_yticklabels(), rotation=0, ha='right')
    
    # Adjust subplot to make room for labels
    plt.subplots_adjust(left=0.3)
    
    plt.tight_layout()
    plt.show()

    # Return a dictionary with more information
    return {
        'selected_features': selected_features,
        'importance_scores': importance_scores,
        'combined_importance': combined_importance
    }


def analyze_collinearity(
    df: pd.DataFrame, 
    features: List[str], 
    vif_threshold: float = 5.0
) -> Dict[str, List[str]]:
    """
    Analyze collinearity among features and suggest which to keep.

    Args:
        df (pd.DataFrame): The dataframe containing the features.
        features (List[str]): List of feature names to analyze.
        vif_threshold (float): Threshold for high VIF.

    Returns:
        Dict[str, List[str]]: Suggested features to keep and remove.
    """
    X = df[features].dropna()
    features_to_keep = features.copy()
    dropped_features = []

    while True:
        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["feature"] = features_to_keep
        vif_data["VIF"] = [
            variance_inflation_factor(X[features_to_keep].values, i) 
            for i in range(len(features_to_keep))
        ]
        
        max_vif = vif_data["VIF"].max()
        if max_vif > vif_threshold:
            # Drop the feature with the highest VIF
            feature_to_drop = vif_data.sort_values("VIF", ascending=False)["feature"].iloc[0]
            features_to_keep.remove(feature_to_drop)
            dropped_features.append(feature_to_drop)
            print(f"Dropped '{feature_to_drop}' with VIF: {max_vif}")
        else:
            break

    print("\nFinal VIF for each feature:")
    print(vif_data.sort_values("VIF", ascending=False))

    return {
        "keep": features_to_keep,
        "remove": dropped_features
    }
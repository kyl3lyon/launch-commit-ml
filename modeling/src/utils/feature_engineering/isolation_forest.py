# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from typing import Tuple, List
import matplotlib.pyplot as plt

# --- Functions ---

def prepare_data(df: pd.DataFrame, target_column: str, predictor_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the data by separating unsuitable and suitable conditions.
    
    Args:
        df (pd.DataFrame): The input dataset.
        target_column (str): Name of the target column.
        predictor_columns (List[str]): List of predictor column names.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Unsuitable and suitable data subsets.
    """
    # Use .copy() to avoid SettingWithCopyWarning
    unsuitable = df[df[target_column] == 1][predictor_columns].copy()
    suitable = df[df[target_column] == 0][predictor_columns].copy()
    print(f"Unsuitable conditions: {len(unsuitable)}")
    print(f"Suitable conditions: {len(suitable)}")
    return unsuitable, suitable

def optimize_isolation_forest(X_train: pd.DataFrame, predictor_columns: List[str]) -> dict:
    """
    Optimize Isolation Forest parameters using GridSearchCV.
    
    Args:
        X_train (pd.DataFrame): Training data.
        predictor_columns (List[str]): List of predictor column names.
    
    Returns:
        dict: Best parameters for Isolation Forest.
    """
    param_grid = {
        'contamination': [0.1, 0.15, 0.2],
        'n_estimators': [100, 150, 200],
    }
    
    grid_search = GridSearchCV(IsolationForest(random_state=42), param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train[predictor_columns])
    
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    return best_params

def train_isolation_forest(unsuitable_data: pd.DataFrame, contamination: float = 0.1, n_estimators: int = 100) -> Tuple[IsolationForest, StandardScaler]:
    """
    Train an Isolation Forest on the unsuitable weather data.
    
    Args:
        unsuitable_data (pd.DataFrame): Unsuitable weather data.
        contamination (float): The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
        n_estimators (int): The number of base estimators in the ensemble.
    
    Returns:
        Tuple[IsolationForest, StandardScaler]: Trained Isolation Forest model and fitted StandardScaler.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(unsuitable_data)
    
    iso_forest = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
    iso_forest.fit(scaled_data)
    
    print(f"Isolation Forest trained on {len(unsuitable_data)} unsuitable conditions")
    return iso_forest, scaler

def find_potential_unsuitable(all_weather_data: pd.DataFrame, iso_forest: IsolationForest, 
                              scaler: StandardScaler, predictor_columns: List[str], 
                              threshold: float = -0.3) -> pd.DataFrame:
    """
    Find potentially unsuitable weather conditions in the larger dataset.
    
    Args:
        all_weather_data (pd.DataFrame): Larger weather dataset to analyze.
        iso_forest (IsolationForest): Trained Isolation Forest model.
        scaler (StandardScaler): Fitted StandardScaler.
        predictor_columns (List[str]): List of predictor column names.
        threshold (float): Anomaly score threshold for considering a point as unsuitable.
    
    Returns:
        pd.DataFrame: Subset of all_weather_data with potentially unsuitable conditions.
    """
    # Use .copy() to avoid SettingWithCopyWarning
    clean_data = all_weather_data.dropna(subset=predictor_columns).copy()
    print(f"Dropped {len(all_weather_data) - len(clean_data)} rows with NaN values")
    
    scaled_data = scaler.transform(clean_data[predictor_columns])
    
    anomaly_scores = iso_forest.decision_function(scaled_data)
    
    potentially_unsuitable = clean_data[anomaly_scores <= threshold].copy()
    potentially_unsuitable['anomaly_score'] = anomaly_scores[anomaly_scores <= threshold]
    
    print(f"Found {len(potentially_unsuitable)} potential unsuitable conditions")
    return potentially_unsuitable

def calculate_similarity(unsuitable_data: pd.DataFrame, potentially_unsuitable: pd.DataFrame) -> np.ndarray:
    """
    Calculate similarity between known unsuitable conditions and potentially unsuitable conditions.
    
    Args:
        unsuitable_data (pd.DataFrame): Known unsuitable weather data.
        potentially_unsuitable (pd.DataFrame): Potentially unsuitable weather data.
    
    Returns:
        np.ndarray: Array of similarity scores.
    """
    distances = euclidean_distances(unsuitable_data, potentially_unsuitable)
    similarities = 1 / (1 + distances)
    # Use np.max instead of .max() for numpy array
    return np.max(similarities, axis=0)

def augment_training_set(X_train: pd.DataFrame, y_train: pd.Series, 
                         new_unsuitable: pd.DataFrame, target_ratio: float = 0.3) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Augment the training set with new unsuitable conditions.
    
    Args:
        X_train (pd.DataFrame): Original training features.
        y_train (pd.Series): Original training labels.
        new_unsuitable (pd.DataFrame): Newly identified unsuitable conditions.
        target_ratio (float): Desired ratio of unsuitable to suitable conditions.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Augmented training features and labels.
    """
    n_suitable = sum(y_train == 0)
    n_unsuitable = sum(y_train == 1)
    n_new_unsuitable_needed = int(n_suitable * target_ratio / (1 - target_ratio)) - n_unsuitable
    
    # Ensure n is not larger than the DataFrame length
    n_new_unsuitable_needed = min(n_new_unsuitable_needed, len(new_unsuitable))
    new_unsuitable_sample = new_unsuitable.sample(n=n_new_unsuitable_needed, random_state=42)
    
    X_train_augmented = pd.concat([X_train, new_unsuitable_sample[X_train.columns]], ignore_index=True)
    y_train_augmented = pd.concat([y_train, pd.Series([1] * len(new_unsuitable_sample))], ignore_index=True)
    
    print(f"Added {len(new_unsuitable_sample)} new unsuitable conditions")
    return X_train_augmented, y_train_augmented

def visualize_data_distribution(X_train: pd.DataFrame, y_train: pd.Series, predictor_columns: List[str]):
    """
    Visualize data distribution using PCA.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        predictor_columns (List[str]): List of predictor column names.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train[predictor_columns])

    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis')
    plt.title('PCA of Weather Data')
    plt.colorbar(label='Weather Suitability')
    plt.show()

def visualize_model_output(iso_forest: IsolationForest, X_pca: np.ndarray):
    """
    Visualize Isolation Forest decision boundary.
    
    Args:
        iso_forest (IsolationForest): Trained Isolation Forest model.
        X_pca (np.ndarray): PCA-transformed data.
    """
    xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    Z = iso_forest.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.YlOrRd)
    plt.colorbar()
    plt.title("Isolation Forest Decision Boundary")
    plt.show()

def analyze_feature_importance(X_train: pd.DataFrame, y_train: pd.Series, predictor_columns: List[str]):
    """
    Analyze feature importance using Random Forest.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        predictor_columns (List[str]): List of predictor column names.
    """
    rf = RandomForestClassifier()
    rf.fit(X_train[predictor_columns], y_train)
    importances = rf.feature_importances_

    for feature, importance in zip(predictor_columns, importances):
        print(f"{feature}: {importance}")

def visualize_feature_distributions(X_train: pd.DataFrame, y_train: pd.Series, predictor_columns: List[str]):
    """
    Visualize feature distributions for suitable and unsuitable conditions.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        predictor_columns (List[str]): List of predictor column names.
    """
    for feature in predictor_columns:
        plt.figure(figsize=(10, 6))
        plt.hist(X_train[y_train == 0][feature], alpha=0.5, label='Suitable')
        plt.hist(X_train[y_train == 1][feature], alpha=0.5, label='Unsuitable')
        plt.title(f'Distribution of {feature}')
        plt.legend()
        plt.show()

def augment_unsuitable_weather_data(X_train: pd.DataFrame, y_train: pd.Series, 
                                    all_weather_data: pd.DataFrame, 
                                    predictor_columns: List[str], 
                                    target_ratio: float = 0.3,
                                    contamination: float = 0.15,
                                    n_estimators: int = 150,
                                    threshold: float = -0.3,
                                    similarity_threshold: float = 0.7) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Main function to augment the training set with potential unsuitable weather conditions.
    
    Args:
        X_train (pd.DataFrame): Original training features.
        y_train (pd.Series): Original training labels.
        all_weather_data (pd.DataFrame): Larger weather dataset.
        predictor_columns (List[str]): List of predictor column names.
        target_ratio (float): Desired ratio of unsuitable to suitable conditions.
        contamination (float): Proportion of outliers in the dataset for Isolation Forest.
        n_estimators (int): Number of trees in the Isolation Forest.
        threshold (float): Threshold for identifying potential unsuitable conditions.
        similarity_threshold (float): Threshold for selecting the most similar conditions.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]: Augmented training features and labels,
        potentially unsuitable conditions, and filtered potentially unsuitable conditions.
    """
    print("Step 1: Preparing data")
    print("-"*80)
    # Combine X_train and y_train before calling prepare_data
    train_df = pd.concat([X_train, y_train], axis=1)
    unsuitable_data, _ = prepare_data(train_df, 'Wx_Unsuitable', predictor_columns)
    
    print("\nStep 2: Training Isolation Forest")
    print("-"*80)
    iso_forest, scaler = train_isolation_forest(unsuitable_data, contamination=contamination, n_estimators=n_estimators)
    
    print("\nStep 3: Finding potential unsuitable conditions")
    print("-"*80)
    potentially_unsuitable = find_potential_unsuitable(all_weather_data, iso_forest, scaler, predictor_columns, threshold=threshold)
    print(f"Number of potential unsuitable conditions: {len(potentially_unsuitable)}")

    print("\nStep 4: Calculating similarity to known unsuitable conditions")
    print("-"*80)
    similarities = calculate_similarity(unsuitable_data, potentially_unsuitable[predictor_columns])
    potentially_unsuitable['similarity_score'] = similarities
    print(f"Similarity score range: {similarities.min():.2f} to {similarities.max():.2f}")
    
    print("\nDistribution of similarity scores:")
    print("-"*80)
    print(potentially_unsuitable['similarity_score'].describe())

    print("\nStep 5: Sorting and selecting top similar conditions")
    print("-"*80)
    potentially_unsuitable_sorted = potentially_unsuitable.sort_values('similarity_score', ascending=False)
    potentially_unsuitable_filtered = potentially_unsuitable_sorted[potentially_unsuitable_sorted['similarity_score'] >= similarity_threshold]
    print(f"Number of conditions after similarity filtering: {len(potentially_unsuitable_filtered)}")

    print("\nStep 6: Augmenting the training set")
    # Pass y_train as pd.Series
    X_train_augmented, y_train_augmented = augment_training_set(X_train, y_train.squeeze(), potentially_unsuitable_filtered, target_ratio)
    
    print(f"\nOriginal training set shape: {X_train.shape}")
    print("-"*80)
    print(f"Augmented training set shape: {X_train_augmented.shape}")
    print("-"*80)
    print(f"New unsuitable ratio: {sum(y_train_augmented == 1) / len(y_train_augmented):.2f}")
    
    return X_train_augmented, y_train_augmented, potentially_unsuitable, potentially_unsuitable_filtered
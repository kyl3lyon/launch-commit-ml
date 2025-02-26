from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

class GradientBoostingPredictor:
    def __init__(self, flat_data=None, target_column=None):
        """Initialize the predictor. flat_data and target_column are only needed for training."""
        self.flat_data = flat_data
        self.target_column = target_column
        self.gb_model = None
        self.scaler = MinMaxScaler()
        self.X_test = None
        self.y_test = None
        self.feature_names = None

    def train(self):
        """Train gradient boosting model using flattened data"""
        if self.flat_data is None or self.target_column is None:
            raise ValueError("flat_data and target_column must be provided for training")

        print("Preparing data for gradient boosting model...")
        # Prepare data
        X = self.flat_data.drop(columns=[self.target_column])
        # Only keep T-0 columns
        X = X[[col for col in X.columns if 'T-0' in col or not any(f'T-{i}' in col for i in range(3))]]
        y = self.flat_data[self.target_column]

        # Store feature names
        self.feature_names = X.columns.tolist()
        print("Feature names:", self.feature_names)
        print(f"Total features being used: {len(X.columns)}")

        print("Splitting data into train and test sets...")
        # Train-test split
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42
        )

        # Scale the features
        print("Scaling features...")
        X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"Training set size: {len(X_train)}, Test set size: {len(self.X_test)}")

        print("Setting up monotonic constraints...")
        # Define monotonic constraints
        monotonic_constraints = {
            'MAX_CLOUD_COVER_SURFACE_T0': 1,
            'MAX_CLOUD_COVER_LOW_T0': 1,
            'MAX_CLOUD_COVER_ASCENT_T0': 1,
            'MAX_CLOUD_COVER_UPPER_T0': 1,
            'MAX_RELATIVE_HUMIDITY_SURFACE_T0': 1,
            'MAX_RELATIVE_HUMIDITY_LOW_T0': 1,
            'MAX_RELATIVE_HUMIDITY_ASCENT_T0': 1,
            'MAX_RELATIVE_HUMIDITY_UPPER_T0': 1,
            'MAX_TEMPERATURE_SURFACE_T0': 0,
            'MAX_TEMPERATURE_LOW_T0': 0,
            'MAX_TEMPERATURE_ASCENT_T0': 0,
            'MAX_TEMPERATURE_UPPER_T0': 0,
            'MAX_WIND_SPEED_SURFACE_T0': 1,
            'MAX_WIND_SPEED_LOW_T0': 1,
            'MAX_WIND_SPEED_ASCENT_T0': 1,
            'MAX_WIND_SPEED_UPPER_T0': 1,
            'VISIBILITY_15MIN_T-0': -1,
            'VISIBILITY_THRESHOLD_MET_T-0': -1,
            'WEATHER_CODE_15MIN_T-0': 0,
            'UV_INDEX_T-0': 0,
            'CAPE_15MIN_T-0': 1,
            'CONVECTIVE_INHIBITION_T-0': -1,
            'LIFTED_INDEX_T-0': -1,
            'MAX_SHEAR_SURFACE_T-0': 1,
            'MAX_SHEAR_LOW_T-0': 1,
            'MAX_SHEAR_ASCENT_T-0': 1,
            'MAX_SHEAR_UPPER_T-0': 1,
            'FREEZING_LEVEL_HEIGHT_T-0': 1,
            'PRECIPITATION_15MIN_T-0': 1,
            'RAIN_15MIN_T-0': 1
        }

        print("Creating feature names and constraints lists...")
        monotone_constraints = [monotonic_constraints.get(col, 0) 
                              for col in self.feature_names]

        print("Setting up model with known best parameters...")
        # Initialize LightGBM classifier with optimized parameters
        self.gb_model = LGBMClassifier(
            objective='binary',    # Binary classification task (GO/NOGO)
            random_state=42,       # Set random seed for reproducibility
            n_jobs=-1,             # Use all available CPU cores
            boosting_type='gbdt',  # Use gradient boosting decision trees
            subsample=0.8,         # Use 80% of data for each tree to reduce overfitting
            colsample_bytree=0.8,  # Use 80% of features for each tree to reduce overfitting
            learning_rate=0.1,     # Learning rate controls how much we adjust the predictions
            max_depth=7,           # Maximum depth of trees - prevents overly complex trees
            min_child_samples=20,  # Minimum number of samples required to create a leaf
            n_estimators=300,      # Total number of trees to build
            num_leaves=63,         # Maximum number of leaves in each tree
            reg_alpha=0.01,        # L1 regularization term to prevent overfitting
            reg_lambda=1.0,        # L2 regularization term to prevent overfitting
            seed=42                # This was an accident, but I think I have to keep it now unless I want to re-train the model??
        )

        print("Fitting model...")
        self.gb_model.fit(X_train, y_train)
        print("Model training complete!")

        # Evaluate on test set
        test_score = self.gb_model.score(self.X_test, self.y_test)
        print(f"Test set accuracy: {test_score:.4f}")

        # Get feature importances
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.gb_model.feature_importances_
        })
        importances = importances.sort_values('importance', ascending=False)
        print("\nFeature Importances:")
        print(importances.to_string(index=False))

        return self.gb_model

    def predict(self, new_data):
        """Make predictions on new data."""
        if self.gb_model is None:
            raise ValueError("Model not trained or loaded")

        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(new_data.columns)
        if missing_features:
            raise ValueError(f"Missing features in new data: {missing_features}")

        # Reorder columns to match training data
        X = new_data[self.feature_names]

        # Scale the features using the same scaler used during training
        X_scaled = self.scaler.transform(X)

        # Make predictions
        return self.gb_model.predict(X_scaled)

    def predict_proba(self, new_data):
        """Make probability predictions on new data."""
        if self.gb_model is None:
            raise ValueError("Model not trained or loaded")

        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(new_data.columns)
        if missing_features:
            raise ValueError(f"Missing features in new data: {missing_features}")

        # Reorder columns to match training data
        X = new_data[self.feature_names]

        # Scale the features using the same scaler used during training
        X_scaled = self.scaler.transform(X)

        # Make probability predictions
        return self.gb_model.predict_proba(X_scaled) 
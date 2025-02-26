import pickle
import pandas as pd
from src.model import GradientBoostingPredictor

def train_and_save_model(data_path: str, output_path: str = 'model/gb_predictor.pkl'):
    """Train and save the model using the provided training data."""
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)

    print("Initializing predictor...")
    predictor = GradientBoostingPredictor(
        flat_data=data,
        target_column='NOGO'
    )

    print("Training model...")
    predictor.train()

    print(f"Saving trained model to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(predictor, f)
    print("Model saved successfully!")

if __name__ == '__main__':
    data_path = '/Users/kylelyon/Git/sdataplab/application/data/model_data.csv'
    train_and_save_model(data_path) 
# Launch Conditions Predictor

A Streamlit application for predicting launch conditions (GO/NOGO) using a trained LightGBM model.

## Features

- Interactive input widgets for all weather conditions
- Real-time GO/NOGO predictions with confidence scores
- Model explainability with SHAP values
- Feature importance visualization
- Preset weather scenarios
- Warning indicators for critical thresholds

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd launch_predictor
   ```

2. Create and activate a virtual environment:
   
   Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   
   macOS/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:s
   ```bash
   streamlit run app.py
   ```

## Project Structure

+ ```
  launch_predictor/
  ├── app.py              # Entry point
  ├── requirements.txt
  ├── README.md
  ├── models/
  │   ├── gb_predictor.pkl
  │   └── gb_model.pkl
  └── src/
      ├── __init__.py
      ├── app.py          # Main application logic
      ├── config.py
      ├── features.py
      ├── predictor.py
      └── utils.py
+ ```


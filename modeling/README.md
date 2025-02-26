# Launch Commit ML

**Problem Statement:** Using publicly available weather data, predict if weather conditions will satisfy space launch commit criteria automatically.

**Solution:** Develop a machine learning model to predict weather suitability for space launches, aiming to reduce delays, enhance safety, and infer launch criteria for various entities. This model will provide insights into operational capabilities and limitations, potentially offering valuable intelligence.
## Model Details

### Authors
- Kyle Lyon, `kyle.lyon@siliconmtn.com`

### Basic Information

- **Model Date:** September 2024
- **Model Version:** 0.1
- **License:** Apache 2.0

### Workflow

- Tree-based boosting model optimizes predictions on a weather dataset, addressing class imbalance, multicollinearity, and logical constraints.
- Feature engineering aligns with launch criteria, refined through cross-validation and importance analysis.

## Intended Use

- **Primary Intended Use:**
  - Provide an alternative to subjective, rules-based models for launch weather prediction.
  - Serve as a redundancy for current domestic launch verification, offering additional confidence.
  - Infer foreign launch weather go/no-go criteria.
  - Enhance collaboration within the kill chain, particularly with launch prep and launch detect teams.

- **Primary Intended Users:**
  - U.S. Space Domain Awareness (SDA) professionals
  - U.S. Threat Assessment and Prioritization (TAP) analysts
  - U.S. Launch Assessment Personnel (LAP)
  - Members of the SDA TAP LAP Cohort 4 Kill Chain

- **Out-of-scope Use Cases:**
  - Replacing human decision-making in critical launch decisions
  - Predicting non-weather-related launch criteria

## Data Dictionaries
### **Launch Criteria Data Readiness:**

| Launch Criteria | Shared Requirement | Data Availability | Notes |
|---|---|---|---|
| Wind Speed at Launch Pad | < 61 km/h (38 mph) | `wind_speed_10m (mp/h)`, `wind_speed_100m (mp/h)` | Available but at different heights. May need adjustment. |
| Wind Gusts | Not specified | `wind_gusts_10m (mp/h)` | Available, could be relevant. |
| Ceiling | > 1800 m (6000 ft) | No direct measure | Could potentially be derived from cloud cover data. |
| Visibility | > 6.4 km (4 mi) | Missing |  |
| Upper-level Wind Shear | Avoid conditions leading to control problems | No direct measure | Missing, might be derivable from wind speeds at different altitudes. |
| Cloud Layer Thickness | < 1400 m (4500 ft) if extending into freezing temps | `cloud_cover (%)`, `cloud_cover_low (%)`, `cloud_cover_mid (%)`, `cloud_cover_high (%)`, `temperature_2m (°F)` | Available but need to derive thickness and freezing level. |
| Cumulus Cloud Proximity | 5-10 miles if tops extend into freezing temps | `cloud_cover (%)`, `temperature_2m (°F)` | Partial data available; need to derive cloud type and proximity. |
| Lightning | No launch within 19 km (10 nmi) of lightning for 30 min after last strike | Missing | Crucial feature, requires external data. |
| Electric Field | No launch if field mill readings exceed +/- 1500 V/m within 9.3 km (5 nmi) | Missing | Crucial feature, requires external data. |
| Thunderstorm Anvil Cloud | Not specified | No direct measure | Missing. |
| Thunderstorm Debris Cloud | No launch within 5.6 km (3 nmi) | No direct measure | Missing. |
| Disturbed Weather | No launch through or within 9.3 km (5 nmi) if extending into freezing temps with moderate precipitation | `precipitation (inch)`, `temperature_2m (°F)` | Partial data available; need to define "disturbed weather" more precisely. |
| Smoke Plumes | No launch through cumulus clouds attached to smoke plume | No direct measure | Missing. |
| Temperature | Not directly specified but related to freezing conditions | `temperature_2m (°F)`, `apparent_temperature (°F)` | Available. |
| Precipitation | Mentioned in context of disturbed weather | `precipitation (inch)`, `rain (inch)`, `snowfall (inch)` | Available. |
| Humidity | Not directly specified | `relative_humidity_2m (%)` | Available, could be relevant for cloud formation. |
| Pressure | Not directly specified | `pressure_msl (hPa)`, `surface_pressure (hPa)` | Available, could be relevant for overall weather patterns. |
| Solar Radiation | Not directly specified | Multiple radiation measurements | Available, could be indirectly relevant. |
| Soil Conditions | Not specified | Multiple soil temperature and moisture measurements | Available but likely not directly relevant. |

### **Cape Canaveral Launch Data:**
| Name                | Description                                           | Measurement Level |
|---------------------|-------------------------------------------------------|-------------------|
| Launch Vehicle      | Type of launch vehicle used                           | string            |
| Payload             | Name of the Payload                                   | string            |
| Countdown           | Indicates if countdown occurred (1) or not (0)        | binary            |
| S/L                 | Indicates if successful launch occurred (1) or not (0)| binary            |
| Scrub N-Wx          | Indicates if non-weather scrub occurred (1) or not (0)| binary            |
| Scrub Wx            | Indicates if weather scrub occurred (1) or not (0)    | binary            |
| Delay N-Wx          | Indicates if non-weather delay occurred (1) or not (0)| binary            |
| Delay Wx            | Indicates if weather delay occurred (1) or not (0)    | binary            |
| Scrub/Del Reason LCC| Indicates if LCC was reason for scrub/delay (1 or 0)  | binary            |
| Scrub/Del Reason User Wx| Indicates if user weather was reason for scrub/delay (1 or 0) | binary |
| Remarks             | Additional comments or explanations                   | string            |

### **Training Data:**

| Feature Name | Modeling Role | Measurement Level | Description | 
|---|---|---|---|
| `temperature_2m (°F)` | Input | Float | Temperature at 2 meters above ground level in degrees Fahrenheit. |
| `apparent_temperature (°F)` | Input | Float | Perceived temperature considering humidity and wind chill. |
| `relative_humidity_2m (%)` | Input | Float | Relative humidity at 2 meters above ground level as a percentage. |
| `pressure_msl (hPa)` | Input | Float | Atmospheric pressure at mean sea level in hectopascals. |
| `surface_pressure (hPa)` | Input | Float | Atmospheric pressure at the surface in hectopascals. | 
| `cloud_cover (%)` | Input | Float | Total cloud cover as a percentage. |
| `cloud_cover_low (%)` | Input | Float | Cloud cover at low altitudes as a percentage. |
| `cloud_cover_mid (%)` | Input | Float | Cloud cover at mid altitudes as a percentage. |
| `cloud_cover_high (%)` | Input | Float | Cloud cover at high altitudes as a percentage. |
| `cloud_thickness (composite)` | Input | Float | Derived feature representing the total thickness of cloud layers (in km or other units). |
| `wind_speed_10m (mp/h)` | Input | Float | Wind speed at 10 meters above ground level in miles per hour. |
| `wind_speed_100m (mp/h)` | Input | Float | Wind speed at 100 meters above ground level in miles per hour. |
| `precipitation (inch)` | Input | Float | Total precipitation in inches. |
| `rain (inch)` | Input | Float | Rain accumulation in inches. | 

- **Source of Training Data:** [Information about the source of training data]
- **Training and Validation Data Division:** [Description of how the training data was divided]
- **Number of Rows in Training and Validation Datasets:** [Number of rows]

## Test Data

- **Source of Training Data:** [Information about the source of training data]
- **Number of rows in test data:** [Description of how the training data was divided]
- **Differences in columns between training and test data:** [Number of rows]

## Model Details

- **Input Columns:** [List of columns used as inputs in the final model]
- **Target Columns:** [List of columns used as targets in the final model]
- **Model Types:** [Types of models used]
- **Implementation Software:** [Software used to implement the model]
- **Model Implementation:** Random Forest and Gradient Boosting using Skcit-learn library

## Quantitative Analysis

### Clustering Analysis

#### Elbow Method

<img src="src/plots/elbow.png" width="600" alt="Elbow Method">

- Commentary on Elbow Method plot:
  - As k increases, the inertia decreases rapidly at first, then more slowly.
  - There's no clear "elbow" point where the rate of decrease changes sharply.
  - The curve is smoothly decreasing, making it challenging to pinpoint an optimal k.
  - Due to the lack of a clear elbow, determining the optimal number of clusters is not straightforward from this dataset and plot alone.

#### PCA Clustring Visualization
<img src="src/plots/cluster_class_pca.png" width="600" alt="Clustering">

- Commentary on Clustering plot:
  - The classes appear to be somewhat mixed, with no clear separation between them.
  - Most data points are concentrated in the center, with some spread along both axes.
  - There is significant overlap between different clusters, suggesting that the features used for clustering may not strongly differentiate between classes.
  - The visualization only shows the first two principal components, which may not capture all relevant information for classification.
  - The lack of distinct cluster boundaries indicates that the problem may be more complex than can be represented in two dimensions.
 
### Model Performance

#### Confusion Matrix
<img src="src/plots/confusion_matrix.png" width="600" alt="Feature Importance">

- Commentary on Confusion Matrix:
  - The model demonstrates high accuracy for negative cases, correctly identifying 455 out of 460 instances.
  - However, it struggles significantly with positive cases, correctly identifying only 3 out of 34 instances.
  - This indicates a strong bias towards predicting negative cases, which is likely due to:
    1. Class imbalance in the training data
    2. Model's inability to capture the characteristics of positive cases
    3. Potentially inadequate feature selection or engineering

#### ROC Curve
<img src="src/plots/roc.png" width="600" alt="ROC Curve">

- Commentary on ROC Curve:
  - The ROC curve is positioned above the diagonal line, indicating that the model performs better than random chance.
  - With an Area Under the Curve (AUC) of 0.71, the model demonstrates moderate discriminative ability, however, there is a class imbalance issue.
  - The stepped appearance of the curve suggests that the dataset may be discrete or relatively small in size.

#### Learning Curve
<img src="src/plots/learning_curve.png" width="600" alt="Confusion Matrix">

- Commentary on Learning Curve:
  - Significant overfitting observed: large gap between high training score and lower cross-validation score.
  - Training score decreases as dataset grows, indicating reduced memorization.
  - Cross-validation score generally improves with more data, showing slightly better generalization.
  - Curves converge slightly but gap remains at 1600 examples; more data may help.
  - Class imbalance can lead to misleading accuracy metrics.

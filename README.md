---

# Project Name: Agricultural Yield Prediction

This project aims to predict agricultural yield based on a range of environmental and pollinator variables using a machine learning model.

---

### Project Overview

The goal of this project is to leverage machine learning techniques to predict yield, focusing on factors like pollinator presence, environmental conditions, and other crop-specific metrics. With efficient feature engineering and model training, the project aims to maximize prediction accuracy for practical agricultural use.

### Dataset

The primary dataset used in this project includes the following key columns:

- Pollinator Information: e.g., honeybee, bumbles, andrena, osmia
- Environmental Factors: Raining Days, Average Raining Days
- Yield-Related Metrics: fruitset, fruitmass, seeds
- Target: yield

The dataset is loaded and cleaned in the project notebook and prepared for feature engineering and model training.

### Feature Engineering

Feature engineering is handled in engineer.py. This script creates new features to improve the model's predictive performance:

- Mean Pollinator Effect (`mean_bee`): Averages the presence of various pollinators (honeybee, bumbles, andrena, osmia).
- Fruit Mass to Seed Ratio (`fruitmass_to_seeds`): Computes the ratio of fruit mass to seed count.
- Enhanced Fruit Mass Effect (`fruitmass_to_p`): A power-based feature derived from fruitmass_to_seeds.
- Adjusted Fruit Set Metrics (`round_fruit_set`): Ratio of fruit set to seeds.
- Fruit Set & Pollinator Interaction (`fruitset * mean_bee`): Multiplication of fruit set with the pollinator mean.
- Fruit Set to Mass Ratio (`fruitset / fruitmass`): Ratio of fruit set to fruit mass.

These engineered features are intended to capture complex interactions and contribute to the model’s performance.

### Modeling

The machine learning model (model.pkl) is a pre-trained model that utilizes the engineered features to make yield predictions. Model details, including the training process, evaluation metrics, and hyperparameters, can be explored in model.ipynb and analys.ipynb.

### Installation

To set up the project environment, install the required dependencies:

pip install -r requirements.txt

The requirements include packages for data processing (pandas, numpy), machine learning (sklearn), and other necessary tools.

### Usage

1. Data Preprocessing:
   - Use engineer.py to apply feature transformations on the dataset.
   - FeatureTransformer class is used to engineer new features and prepare the data for modeling.

2. Model Prediction:
   - Load the trained model (model.pkl) to make predictions on the processed data.
   - from engineer import FeatureTransformer
```python
import pandas as pd
from joblib import load

# Load and preprocess data
data = pd.read_csv('your_data.csv')
transformer = FeatureTransformer()
processed_data = transformer.transform(data)

# Load model and predict
model = load('model.pkl')
predictions = model.predict(processed_data)

from engineer import FeatureTransformer
import pandas as pd
from joblib import load

# Load and preprocess data
data = pd.read_csv('your_data.csv')
transformer = FeatureTransformer()
processed_data = transformer.transform(data)

# Load model and predict
model = load('model.pkl')
predictions = model.predict(processed_data)
```
### Results

Results, model performance metrics, and visualizations can be found in the analys.ipynb and model.ipynb notebooks. These notebooks provide insights into feature importance, model evaluation (e.g., accuracy, RMSE), and analysis of prediction accuracy.

### Contributors

- Xamidullo Muratqulov - Data Scientist and Project Lead

### License

This project is licensed under the MIT License.

--- 

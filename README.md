
# California House Price Prediction 🏠

A machine learning project to predict California house prices using various features.

## Project Overview

This project builds a **regression model** to predict house prices based on available features.

The workflow includes:

- Exploratory Data Analysis (EDA)
- Data Preprocessing & Feature Engineering
- Model Building & Training
- Model Evaluation

## Dataset

- **Source**: Not specified
- **Target**: House Price

## Models Used

- Linear Regression

## Evaluation Metrics

- *(Add your metrics here: Mean Squared Error, Mean Absolute Error, R² Score, etc.)*

## Project Structure

```
california_house_price/
├── House.ipynb              # Main Jupyter Notebook
├── data/                    # (Optional) Data files
├── models/                  # Saved models (e.g. house_price_model.joblib)
├── requirements.txt         # Python dependencies
└── README.md                # Project description
```

## How to Run

1. Clone this repository:

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. Install required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Launch the Jupyter Notebook:

    ```bash
    jupyter notebook House.ipynb
    ```

## How to Save the Model

You can save the trained model using `joblib`:

```python
import joblib

# After training your model (example: reg = LinearRegression().fit(X_train, y_train))
joblib.dump(reg, 'models/house_price_model.joblib')
```

## How to Load the Model and Use It

If you've saved the trained model as `house_price_model.joblib`, you can load and use it like this:

```python
import joblib
import numpy as np

# Load the saved model
reg = joblib.load('models/house_price_model.joblib')

# Example input: array with shape (1, number_of_features)
# Replace the values below with actual values in correct order
sample_input = np.array([[value1, value2, value3, ..., valueN]])

# Make prediction
predicted_price = reg.predict(sample_input)

print("Predicted house price:", predicted_price[0])
```

## Results

- **Best MSE / MAE / R² Score**: 0.68

## Future Improvements

- Hyperparameter tuning
- Feature engineering
- Try advanced models like XGBoost or LightGBM
- Deploy with Streamlit or Flask

## Author

**Manish Bastola**  
[GitHub](https://github.com/ManishBastola)

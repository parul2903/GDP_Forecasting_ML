# GDP_Forecasting_ML

# GDP Forecasting Project

## Introduction
This project aims to forecast the Gross Domestic Product (GDP) for different countries using historical economic data. Time series forecasting techniques, specifically the ARIMA model, are employed to predict future GDP values based on past trends. This analysis helps in understanding economic trends and making informed decisions for economic planning and policy-making.

## Steps
1. **Data Preprocessing**:
   - Load the dataset containing various economic indicators.
   - Filter the data for a specific country.
   - Convert the 'Year' column to datetime format and set it as the index.

2. **Train-Test Split**:
   - Split the data into training (80%) and testing (20%) sets to evaluate the model's performance.

3. **Model Selection and Fitting**:
   - Use the ARIMA model to fit the training data. The model parameters (p, d, q) are selected based on initial analysis and optimization.

4. **Forecasting**:
   - Generate forecasts for the test period using the fitted ARIMA model.

5. **Evaluation**:
   - Evaluate the model's performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and Symmetric Mean Absolute Percentage Error (sMAPE).

6. **Visualization**:
   - Create plots to visualize the actual vs. predicted GDP values and key economic indicators.

## Working
The project is implemented using Python with libraries such as pandas, numpy, sklearn, statsmodels, and matplotlib. Below is an example function for forecasting GDP for a specific country:

```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

def forecast_gdp_by_country(data, country_name):
    country_data = data[data['ISO3'] == country_name]
    country_data['Year'] = pd.to_datetime(country_data['Year'], format='%Y')
    country_data.set_index('Year', inplace=True)
    
    train_data = country_data.iloc[:int(0.8 * len(country_data))]
    test_data = country_data.iloc[int(0.8 * len(country_data)):]

    model = ARIMA(train_data['gdp_real_gwt_next'], order=(1, 1, 1))
    arima_model = model.fit()
    
    forecast = arima_model.forecast(steps=len(test_data))
    
    mae = mean_absolute_error(test_data['gdp_real_gwt_next'], forecast)
    mse = mean_squared_error(test_data['gdp_real_gwt_next'], forecast)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(test_data['gdp_real_gwt_next'], forecast)
    smape = symmetric_mean_absolute_percentage_error(test_data['gdp_real_gwt_next'], forecast)
    
    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAPE: {mape:.2f}%')
    print(f'sMAPE: {smape:.2f}%')
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data['gdp_real

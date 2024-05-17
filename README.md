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


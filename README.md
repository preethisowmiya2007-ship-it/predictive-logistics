Week 4 – Predictive Modeling and Optimization in Logistics Systems

Project Overview

This project applies machine learning techniques to predict delivery time in a logistics system using Python. A hypothetical logistics dataset is used to study the relationship between shipment volume, transportation cost, distance, fuel cost, and delay hours.

Two regression models are implemented and compared:

- Linear Regression
- Decision Tree Regression

The models are evaluated using MAE, RMSE, and R² metrics. Five-fold cross-validation is also performed to check model reliability.

Objectives

- Predict logistics delivery time using machine learning.
- Compare Linear Regression and Decision Tree Regression.
- Evaluate model performance using MAE, RMSE, and R².
- Perform five-fold cross-validation.
- Suggest optimization strategies for logistics operations.

Dataset

The dataset contains 30 hypothetical logistics records.

Input Features

- Shipment_Volume
- Transportation_Cost
- Distance_km
- Fuel_Cost
- Delay_Hours

Target Variable

- Delivery_Time_Days

Machine Learning Models

1. Linear Regression

Linear Regression is used as a baseline model to understand the relationship between logistics features and delivery time.

2. Decision Tree Regression

Decision Tree Regression is used to capture non-linear relationships between logistics features and delivery time.

Model Results

Test Set Results

Model| MAE| RMSE| R²
Linear Regression| 0.2095| 0.2475| 0.9839
Decision Tree Regression| 0.1667| 0.4082| 0.9562

Cross-Validation Results

Model| Mean MAE| Mean RMSE| Mean R²
Linear Regression| 0.2613| 0.3208| 0.9503
Decision Tree Regression| 0.1000| 0.3162| 0.9543

Optimization Strategies

The following strategies are proposed based on the predictive analysis:

- Optimize route planning to reduce unnecessary travel distance.
- Reduce delivery delays through predictive monitoring.
- Control transportation and fuel costs.
- Improve resource allocation based on shipment volume.
- Use predicted delivery time for better logistics planning.

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Google Colab
- GitHub

Files

- "week4_predictive_model.py" – Python implementation of the predictive models.
- "README.md" – Project documentation.

Conclusion

This project demonstrates how predictive modeling can support logistics decision-making. Linear Regression and Decision Tree Regression were evaluated using multiple performance metrics and cross-validation.

The results show that machine learning can help estimate delivery time and support optimization decisions such as route planning, delay reduction, cost control, and resource allocation.

The dataset used in this project is hypothetical. Real-world logistics data can be used in future work to improve model reliability and practical deployment.
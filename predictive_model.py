import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create hypothetical logistics dataset
data = {
    "Shipment_Volume": [120,250,180,300,100,350,275,160,400,140,
                        320,240,190,360,130,280,210,170,420,110,
                        260,310,150,380,200,230,290,125,340,185],

    "Transportation_Cost": [1200,2400,1800,3100,1000,3500,2700,1600,4200,1400,
                            3300,2500,1900,3700,1300,2900,2200,1700,4400,1100,
                            2600,3200,1500,3900,2100,2400,3000,1250,3500,1950],

    "Distance_km": [100,200,150,260,80,300,220,130,350,110,
                    280,210,160,310,90,240,180,140,370,70,
                    230,270,120,330,170,200,250,100,290,155],

    "Fuel_Cost": [400,800,600,1000,300,1200,900,500,1400,450,
                  1100,850,650,1250,350,950,700,550,1500,280,
                  900,1050,400,1350,650,800,1000,350,1150,620],

    "Delay_Hours": [2,5,3,7,1,8,6,2,10,1,
                    7,4,3,9,2,6,5,3,11,1,
                    4,7,2,8,3,5,6,2,9,4],

    "Delivery_Time_Days": [3,5,4,6,2,7,5,4,8,3,
                           6,5,4,7,3,6,5,4,8,2,
                           5,6,3,7,4,5,6,3,7,4]
}

df = pd.DataFrame(data)

# Input features and target
features = [
    "Shipment_Volume",
    "Transportation_Cost",
    "Distance_km",
    "Fuel_Cost",
    "Delay_Hours"
]

X = df[features]
y = df["Delivery_Time_Days"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regression": DecisionTreeRegressor(
        max_depth=4, random_state=42
    )
}

# Train and evaluate models
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n" + name)
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2  :", round(r2, 4))

# Five-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n----- CROSS VALIDATION -----")

for name, model in models.items():

    mae = -cross_val_score(
        model, X, y, cv=kf,
        scoring="neg_mean_absolute_error"
    ).mean()

    rmse = np.sqrt(
        -cross_val_score(
            model, X, y, cv=kf,
            scoring="neg_mean_squared_error"
        ).mean()
    )

    r2 = cross_val_score(
        model, X, y, cv=kf,
        scoring="r2"
    ).mean()

    print("\n" + name)
    print("Mean MAE :", round(mae, 4))
    print("Mean RMSE:", round(rmse, 4))
    print("Mean R2  :", round(r2, 4))
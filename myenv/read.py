import pandas as pd
import numpy as np

# Load the CSV
cs = pd.read_csv("/home/maxi/Desktop/car_fuel_efficiency.csv")

# Display the first five rows
print(cs.head())

print(len(cs))        # number of records (rows)
print(cs.shape)       # (rows, columns)

print(cs["fuel_type"].unique())

#check for missing values in colums
no_of_missing_colms = cs.isnull().any().sum()

print(no_of_missing_colms)

#Maximum efficiency of cars from Asia

asia_cars = cs[cs["origin"] == "Asia"]

maxi_efficiency = asia_cars["fuel_efficiency_mpg"].max()
print(maxi_efficiency)

X = asia_cars[["vehicle_weight", "model_year"]].iloc[:7].to_numpy()
print("X shape:", X.shape)
XTX = X.T @ X       # matrix multiplication
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv @ X.T @ y
print("w:", w)
sum_w = np.sum(w)
print("Sum of w elements:", sum_w)


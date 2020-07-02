import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

# Load our data set
df = pd.read_csv("usedvehicles.csv")

# Create the X and y arrays
X = df[["year", "odometer"]]
y = df[["price"]]

X_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

X[X.columns] = X_scaler.fit_transform(X[X.columns])
y[y.columns] = y_scaler.fit_transform(y[y.columns])

X = X.to_numpy()
y = y.to_numpy()

# Split the data set in a training set (75%) and a test set (25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it to make predictions later
joblib.dump(model, 'used_car_value_model.pkl')

joblib.dump(X_scaler, "X_scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")

# Report how well the model is performing
print("Model training results:")

# Report an error rate on the training set
mse_train = mean_absolute_error(y_scaler.inverse_transform(y_train), y_scaler.inverse_transform(model.predict(X_train)))
print(f" - Training Set Error: {mse_train}")

# Report an error rate on the test set
mse_test = mean_absolute_error(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(model.predict(X_test)))
print(f" - Test Set Error: {mse_test}")
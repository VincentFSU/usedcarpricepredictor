import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

# Load our data set
df = pd.read_csv("usedvehicles.csv")

# Create the X and y arrays
X = df[["year", "odometer"]]
y = df["price"]

# Split the data set in a training set (75%) and a test set (25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it to make predictions later
joblib.dump(model, 'used_car_value_model.pkl')

# Report how well the model is performing
print("Model training results:")

# Report an error rate on the training set
mse_train = mean_absolute_error(y_train, model.predict(X_train))
print(f" - Training Set Error: {mse_train}")

# Report an error rate on the test set
mse_test = mean_absolute_error(y_test, model.predict(X_test))
print(f" - Test Set Error: {mse_test}")
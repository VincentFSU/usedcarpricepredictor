import joblib
from tkinter import *

root = Tk()
# Load our trained model
model = joblib.load('used_car_value_model.pkl')
X_scaler = joblib.load('X_scaler.pkl')
y_scaler = joblib.load('y_scaler.pkl')

year_in_label = Label(root, text='Vehicle Year:').grid(row=0)
odometer_in_label = Label(root, text='Odometer(mi)').grid(row=1)

e_year = Entry(root)
e_year.grid(row=0,column=1)
e_odometer = Entry(root)
e_odometer.grid(row=1,column=1)


def predict_click():
    car_1 = [int(e_year.get()), float(e_odometer.get())]
    cars = [car_1]
    scaled_car_data = X_scaler.transform(cars)
    car_values = model.predict(scaled_car_data)
    unscaled_car_values = y_scaler.inverse_transform(car_values)
    predicted_value = unscaled_car_values[0][0]
    details_label = Label(root, text="Car details:", anchor=W).grid(row=3)
    year_label = Label(root, text=f"vehicle year: {car_1[0]}").grid(row=4)
    odometer_label = Label(root, text=f"{car_1[1]} miles on odometer").grid(row=5)
    price_label = Label(root, text=f"Estimated price: ${predicted_value:,.2f}").grid(row=6)
    formatting_label = Label(root, text="").grid(row=7)


predictButton = Button(root, bg="gray", text="Predict Price", padx=50, pady=10, command=predict_click).grid(row=2)


root.mainloop()

from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        is_female = int(request.form["is_female"])
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        is_smoker = int(request.form["is_smoker"])
        region_southeast = int(request.form["region_southeast"])
        bmi_category_Obese = int(request.form["bmi_category_Obese"])

        input_data = pd.DataFrame([[
            age,
            is_female,
            bmi,
            children,
            is_smoker,
            region_southeast,
            bmi_category_Obese
        ]], columns=[
            "age",
            "is_female",
            "bmi",
            "children",
            "is_smoker",
            "region_southeast",
            "bmi_category_Obese"
        ])

        prediction = model.predict(input_data)
        predicted_cost = round(float(prediction[0]), 2)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Insurance Cost: ₹ {predicted_cost}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model("heart_disease_dl_model.h5")
scaler = joblib.load("scaler.save")

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        # Get data from form
        data = [
            float(request.form["age"]),
            float(request.form["sex"]),
            float(request.form["cp"]),
            float(request.form["trestbps"]),
            float(request.form["chol"]),
            float(request.form["fbs"]),
            float(request.form["restecg"]),
            float(request.form["thalach"]),
            float(request.form["exang"]),
            float(request.form["oldpeak"]),
            float(request.form["slope"]),
            float(request.form["ca"]),
            float(request.form["thal"])
        ]

        # Convert & scale
        input_data = np.array(data).reshape(1, -1)
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)
        result = "Heart Disease Detected" if prediction[0][0] > 0.5 else "No Heart Disease"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

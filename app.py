import os
from flask import Flask, request, render_template
import joblib

from feature import (
    get_all_query,
    remove_punctuation_stopwords_lemma
)

# Load trained model
pipeline = joblib.load("./pipeline.sav")

# Create Flask app
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api", methods=["POST"])
def predict():

    try:

        # Read user inputs
        title = request.form.get("title", "")
        author = request.form.get("author", "")
        text = request.form.get("maintext", "")

        # Combine all inputs
        query = get_all_query(title, author, text)

        # Preprocess text
        cleaned_query = [
            remove_punctuation_stopwords_lemma(query[0])
        ]

        # Predict
        prediction = pipeline.predict(cleaned_query)

        # Prediction confidence (if supported)
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(cleaned_query)
            confidence = round(max(probabilities[0]) * 100, 2)
        else:
            confidence = 100.00

        # Prediction mapping
        if prediction[0] == 1:
            result = "Real News"
            color = "#28a745"
            icon = "✅"
        else:
            result = "Fake News"
            color = "#dc3545"
            icon = "❌"

        return f"""
<!DOCTYPE html>
<html>

<head>

    <title>Fake News Detector | Prediction Result</title>

    <style>

        body {{
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            background: linear-gradient(135deg, #0f172a, #2563eb);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }}

        .card {{
            background: white;
            width: 500px;
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        }}

        h1 {{
            color: {color};
            margin-bottom: 20px;
        }}

        h2 {{
            color: #333;
            margin-bottom: 10px;
        }}

        .confidence {{
            font-size: 32px;
            font-weight: bold;
            color: #2563eb;
            margin: 20px 0;
        }}

        button {{
            margin-top: 25px;
            padding: 15px 30px;
            background: #2563eb;
            color: white;
            font-size: 18px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: 0.3s;
        }}

        button:hover {{
            background: #1d4ed8;
        }}

    </style>

</head>

<body>

    <div class="card">

        <h1>{icon} {result}</h1>

        <h2>Prediction Confidence</h2>

        <div class="confidence">
            {confidence}%
        </div>

        <form action="/">

            <button type="submit">
                🔍 Check Another News
            </button>

        </form>

    </div>

</body>

</html>
"""

    except Exception as e:

        return f"""
<!DOCTYPE html>
<html>

<head>

    <title>Fake News Detector | Error</title>

</head>

<body style="font-family:Arial; text-align:center; margin-top:100px;">

    <h2>Something went wrong!</h2>

    <p>{str(e)}</p>

    <br>

    <form action="/">

        <button
            type="submit"
            style="
                padding:12px 25px;
                font-size:18px;
                background:#dc3545;
                color:white;
                border:none;
                border-radius:5px;
                cursor:pointer;">

            Go Back

        </button>

    </form>

</body>

</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
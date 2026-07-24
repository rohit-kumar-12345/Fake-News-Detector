from flask import Flask, request, render_template
import joblib

from feature import (
    get_all_query,
    remove_punctuation_stopwords_lemma
)

# Load trained pipeline
pipeline = joblib.load("./pipeline.sav")

# Create Flask application
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

        # Combine all text
        query = get_all_query(title, author, text)

        # Apply preprocessing
        cleaned_query = [
            remove_punctuation_stopwords_lemma(query[0])
        ]

        # Prediction
        prediction = pipeline.predict(cleaned_query)

        # Prediction probabilities
        probabilities = pipeline.predict_proba(cleaned_query)

        # Highest probability
        confidence = max(probabilities[0]) * 100

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

            <title>Prediction Result</title>

            <style>

                body {{
                    font-family: Arial, Helvetica, sans-serif;
                    background: linear-gradient(135deg,#0f172a,#2563eb);
                    margin:0;
                    display:flex;
                    justify-content:center;
                    align-items:center;
                    height:100vh;
                }}

                .card {{

                    background:white;

                    width:500px;

                    padding:40px;

                    border-radius:15px;

                    text-align:center;

                    box-shadow:0 15px 35px rgba(0,0,0,.3);

                }}

                h1 {{

                    color:{color};

                    margin-bottom:20px;

                }}

                h2 {{

                    color:#333;

                }}

                p {{

                    font-size:22px;

                    color:#555;

                }}

                .confidence {{

                    font-size:30px;

                    color:#2563eb;

                    font-weight:bold;

                    margin-top:15px;

                }}

                button {{

                    margin-top:30px;

                    padding:15px 30px;

                    font-size:18px;

                    background:#2563eb;

                    color:white;

                    border:none;

                    border-radius:8px;

                    cursor:pointer;

                }}

                button:hover {{

                    background:#1d4ed8;

                }}

            </style>

        </head>

        <body>

            <div class="card">

                <h1>{icon} {result}</h1>

                <h2>Prediction Confidence</h2>

                <div class="confidence">

                    {confidence:.2f}%

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
        <html>

        <body style="font-family:Arial; text-align:center; margin-top:100px;">

            <h2>Something went wrong!</h2>

            <p>{str(e)}</p>

            <br>

            <form action="/">

                <button type="submit">

                    Go Back

                </button>

            </form>

        </body>

        </html>
        """


if __name__ == "__main__":
    app.run(debug=True, port=8080)
